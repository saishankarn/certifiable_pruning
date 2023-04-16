import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchprune as tp

class Sensitivity:
    def __init__(self, module): 
        self.module = module 
        self.weight = self.module.weight.data    
        idx_plus = self.weight > 0.0
        idx_minus = self.weight < 0.0
        weight_plus = torch.zeros_like(self.weight)
        weight_minus = torch.zeros_like(self.weight)
        weight_plus[idx_plus] = self.weight[idx_plus]
        weight_minus[idx_minus] = self.weight[idx_minus]

        self.module_plus = copy.deepcopy(module)
        self.module_plus.weight.data = weight_plus
        self.module_plus.state_dict()["bias"] = torch.zeros(self.module_plus.bias.shape).to(self.weight.device)
        self.module_minus = copy.deepcopy(module)
        self.module_minus.weight.data = weight_minus
        self.module_minus.state_dict()["bias"] = torch.zeros(self.module_minus.bias.shape).to(self.weight.device)

        self.sensitivity = torch.zeros(self.weight.shape).to(self.weight.device)
        self.sensitivity_in = torch.zeros(self.weight.shape[1]).to(self.weight.device)

        self.num_patches = None
            
    def compute_sensitivity_for_batch(self, inp, out):
        pos = inp >= 0.0
        if torch.all(pos):
            inp_processed = inp 
        else:
            inp_pos = torch.zeros_like(inp)
            inp_neg = torch.zeros_like(inp)
            inp_pos[pos] = inp[pos]
            inp_neg[~pos] = inp[~pos]

            inp_processed = torch.cat((inp_pos, -inp_neg))

        if self.module.bias is not None:
            shape = np.array(out.shape)
            shape_div = copy.deepcopy(shape)
            if isinstance(self.module, nn.Linear):
                shape_div[-1] = 1
            else:
                shape_div[-3] = 1
            shape_bias = (shape / shape_div).astype(int).tolist()
            out_no_bias = out - self.module.bias.view(shape_bias)
        else:
            out_no_bias = out.clone().detach()

        # print(inp_processed.shape, out_no_bias.shape, out.shape)
        g_sens, g_sens_in = self._update_g_sens(inp_processed, out_no_bias, out)

        self._update_sensitivity(g_sens, g_sens_in)

    def _process_denominator(self, z_values):
        # processing
        # replace values smaller than eps with infinity
        eps = torch.Tensor([np.finfo(np.float32).eps]).to(z_values.device)
        mask = torch.le(torch.abs(z_values), eps)
        z_values.masked_fill_(mask, np.Inf)
        return z_values

    def _get_g_sens_f(self, weight_f, activations, z_values_f):
        # compute g
        g_sens_f = weight_f.unsqueeze(0).unsqueeze(-1) * activations
        g_sens_f /= z_values_f.unsqueeze(1)

        return g_sens_f.clone().detach()
    
    def flatten_all_but_last(self, tensor):
        """Flatten all dimensions except last one and return tensor."""
        return tensor.view(tensor[..., 0].numel(), tensor.shape[-1])

    def _reshape_z(self, z_values):
        # flatten all batch dimensions first if it's linear...
        if isinstance(self.module, nn.Linear):
            z_values = self.flatten_all_but_last(z_values)
        z_values = z_values.view(z_values.shape[0], z_values.shape[1], -1)
        return z_values

    def _compute_g_sens_f(
        self, idx_f, w_unfold_plus, w_unfold_minus, a_unfold, z_plus, z_minus
    ):

        g_sens_f = torch.max(
            self._get_g_sens_f(
                w_unfold_plus[idx_f], a_unfold, z_plus[:, idx_f]
            ),
            self._get_g_sens_f(
                w_unfold_minus[idx_f], a_unfold, z_minus[:, idx_f]
            ),
        )
        return g_sens_f
    
    def unfold(self, x):
        if isinstance(self.module, nn.Conv2d):
            return nn.functional.unfold(
                x,
                kernel_size=self.module.kernel_size,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
            )
        else:
            # flatten all batch dimensions, then unsqueeze last dim
            return self.flatten_all_but_last(x).unsqueeze(-1)
        
    def _reduction(self, g_sens_f, dim):
        return torch.max(g_sens_f, dim=dim)[0]

    def _update_g_sens(self, activations, z_no_bias, out):
        weight_plus = self.module_plus.weight.data
        weight_minus = self.module_minus.weight.data
        w_unfold_plus = weight_plus.view((weight_minus.shape[0], -1))
        w_unfold_minus = weight_minus.view((weight_minus.shape[0], -1))
        #print(self.weight.shape, weight_plus.shape, weight_minus.shape, w_unfold_plus.shape, w_unfold_minus.shape)

        z_plus = self.module_plus(activations)
        z_minus = self.module_minus(activations)
        # print(z_plus.shape, z_minus.shape, activations.shape)
        # out = self._reshape_z(out)
        z_plus = self._reshape_z(z_plus)
        z_minus = self._reshape_z(z_minus)
        z_no_bias = self._reshape_z(z_no_bias)
        #print(outs.shape, z_plus.shape, z_minus.shape, z_no_bias.shape)

        #self._num_points_processed += activations.shape[0]

        #self._current_batch += 1
        z_plus = self._process_denominator(z_plus)
        z_minus = self._process_denominator(z_minus)

        # shape = (batchSize, filterSize, outExamples) as above
        a_unfolded = self.unfold(activations)

        if self.num_patches is None:
            self.num_patches = a_unfolded.shape[-1]

        # preallocate g
        batch_size = a_unfolded.shape[0]
        device = self.sensitivity.device
        # print(self.sensitivity.shape, self.sensitivity_in.shape, batch_size, a_unfolded.shape)
        g_sens = torch.zeros((batch_size,) + self.sensitivity.shape).to(device)
        g_sens_in = torch.zeros((batch_size,) + self.sensitivity_in.shape).to(
            device
        )

        # populate g for this batch
        for idx_f in range(w_unfold_plus.shape[0]):
            # compute g
            g_sens_f = self._compute_g_sens_f(
                idx_f,
                w_unfold_plus,
                w_unfold_minus,
                a_unfolded,
                z_plus,
                z_minus,
            )

            # Reduction over outExamples
            g_sens_f = self._reduction(g_sens_f, dim=-1)

            # Reduction over outputChannels
            g_sens_in_f = self._reduction(
                g_sens_f.view((g_sens_f.shape[0], weight_plus.shape[1], -1)),
                dim=-1,
            )

            # store results
            g_sens[:, idx_f] = g_sens_f.view_as(g_sens[:, idx_f])
            g_sens_in = torch.max(g_sens_in, g_sens_in_f)

        # return g
        return g_sens, g_sens_in

    def _update_sensitivity(self, g_sens, g_sens_in):
        # get a quick reference
        sens = self.sensitivity
        sens_in = self.sensitivity_in

        # Max over this batch
        sens_batch = torch.max(g_sens, dim=0)[0]
        sens_in_batch = torch.max(g_sens_in, dim=0)[0]

        # store sensitivity
        self.sensitivity.copy_(torch.max(sens, sens_batch.view(sens.shape)))

        # store sensitivity over input channels
        self.sensitivity_in.copy_(torch.max(sens_in, sens_in_batch))


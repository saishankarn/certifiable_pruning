import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torchprune as tp
from torchprune.method.pfp.pfp_tracker import PFPTracker
from torchprune.util import tensor
from scipy import optimize
from torch.utils.data import SubsetRandomSampler

from get_sensitivity import * 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
"""
network architecture
"""
class LeNet300(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)


    def forward(self, x):
        in_0 = x.view(-1, 784)
        in_0 = nn.Dropout(0.25)(in_0)
        out_0 = self.fc1(in_0) # Apply Fully Connected Layer 1 and ReLU activation
        in_1 = F.relu(out_0)
        in_1 = nn.Dropout(0.25)(in_1)
        out_1 = self.fc2(in_1) # Apply Fully Connected Layer 2 and ReLU activation
        in_2 = F.relu(out_1)
        in_2 = nn.Dropout(0.25)(in_2)
        out_2 = self.fc3(in_2) # Apply Fully Connected Layer 3 (Output Layer)
        return out_2

    def forward_mod(self, x):
        in_0 = x.view(-1, 784)
        in_0 = nn.Dropout(0.25)(in_0)
        out_0 = self.fc1(in_0) # Apply Fully Connected Layer 1 and ReLU activation
        in_1 = F.relu(out_0)
        in_1 = nn.Dropout(0.25)(in_1)
        out_1 = self.fc2(in_1) # Apply Fully Connected Layer 2 and ReLU activation
        in_2 = F.relu(out_1)
        in_2 = nn.Dropout(0.25)(in_2)
        out_2 = self.fc3(in_2) # Apply Fully Connected Layer 3 (Output Layer)
        return [(in_0, in_1, in_2), (out_0, out_1, out_2)]


"""
data loader
"""

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

size_s = 128
batch_size = 128
testset, set_s = torch.utils.data.random_split(
    testset, [len(testset) - size_s, size_s]
)

loader_s = torch.utils.data.DataLoader(set_s, batch_size=32, shuffle=False)
loader_test = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False
)
loader_train = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False
)

"""
Define and train the model
"""

def train(net):
    for epoch in range(40):
        running_loss = 0.0
        for i, data in enumerate(loader_train, 0):
            # Get the inputs and move them to the GPU if available
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), model_path)

    return net

def test(net_, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net_(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 1-correct/total

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
name = 'lenet300_mnist_pfp'
net = LeNet300(num_classes=10).to(device)

model_path = '../checkpoints/' + name + '.pth'
if os.path.isfile(model_path):
    checkpt = torch.load(model_path)
    net.load_state_dict(checkpt)
    net.eval()
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net = train(net)
    print('Finished training')
    torch.save(net.state_dict(), model_path)

loss = test(net, loader_test)
print('Accuracy of the network on the 10000 test images: %f %%' % loss)

backup_net = copy.deepcopy(net)

"""
Let's start with Provable Filter Pruning
Some basic definitions and declarations first
"""
# modules = [module for module in net.modules() if module != net and isinstance(module, nn.Module)]

# sens_trackers = nn.ModuleList()

def get_original_size(modules):
    nonzeros = 0
    for module in modules:
        for param in module.parameters():
            if param is not None:
                nonzeros += (param != 0.0).sum().item()
    return nonzeros

def get_compressible_size(modules):
    nonzeros = 0
    for module in modules:
        nonzeros += (module.weight != 0.0).sum().item()
    return nonzeros

def get_compressible_layers(modules):
    compressible_layers = []
    num_weights = []
    for module in modules:
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if hasattr(module, "groups") and module.groups > 1:
            continue
        compressible_layers.append(module)
        num_weights.append(module.weight.data.numel())

    return compressible_layers, num_weights

class CrossEntropyLossWithAuxiliary(nn.CrossEntropyLoss):

    def forward(self, input, target):
        if isinstance(input, dict):
            loss = super().forward(input["out"], target)
            if "aux" in input:
                loss += 0.5 * super().forward(input["aux"], target)
        else:
            loss = super().forward(input, target)
        return loss


def nan_to_minint(x):
    if math.isnan(x):
        return -9223372036854775808
    else:
        return x

def get_num_features(tensor, dim):
    dims_to_sum = [i for i in range(tensor.dim()) if i is not dim]
    return (torch.abs(tensor).sum(dims_to_sum) != 0).sum()

def get_resulting_layer_budget(eps):
    k_constant = 3.0
    c_constant = 3.0
    
    resulting_layer_budget = {}
    m_budget = []
    for ell in range(len(modules)):
        tracker = sens_trackers[ell]
        num_patches = tracker.num_patches
        weight = tracker.module.weight

        sens_in = tracker.sensitivity_in
        nnz = (sens_in != 0.0).sum().view(-1)
        sum_sens = sens_in.sum().view(-1)
        probs = sens_in / sum_sens

        k_size = weight[0, 0].numel()
        log_numerator = torch.tensor(8.0 * (num_patches + 1) * k_size).to(nnz.device)
        log_term = c_constant * torch.log(log_numerator / delta_failure)
        sens_tilde = sum_sens * 2.0 * log_term 

        sc = k_constant * (6 + 2 * eps) * sens_tilde / (eps ** 2)
        sc = sc.ceil()
        if sc == 0:
            sc = 1
        sc = nan_to_minint(sc)
        m_budget.append(sc)

        vals = 1 - (1 - probs) ** sc
        expectation = torch.sum(torch.as_tensor(vals), dim=-1)
        per_layer_budget = torch.ceil(expectation)
        resulting_layer_budget[ell] = per_layer_budget
    #print(m_budget)
    return resulting_layer_budget

def get_resulting_size_per_eps(eps):
    
    layerwise_budget = get_resulting_layer_budget(eps)
    #print('******************')
    #print(layerwise_budget)

    total_in_features = []
    in_feat_reduction = []
    for ell in range(len(modules)):
        tracker = sens_trackers[ell]
        weight = tracker.module.weight
        layer_budget = layerwise_budget[ell]

        in_features = get_num_features(weight, 1)
        if layer_budget < in_features:
            in_features = layer_budget
        if in_features < 1:
            in_features = 1

        total_in_features.append(in_features)
        in_feat_reduction.append(get_num_features(weight, 1) - in_features)

    resulting_size = 0
    for ell in range(len(modules)):
        tracker = sens_trackers[ell]
        weight = tracker.module.weight

        out_features = float(get_num_features(weight, 0))
        if ell < len(modules)-1:
            sub = in_feat_reduction[ell+1] 
            out_features -= float(sub) 

        if out_features < 1:
            out_features = 1

        in_features = total_in_features[ell]
        k_size = weight[0, 0].numel()
        size_total = float(in_features * out_features * k_size)
        resulting_size += size_total
        #print(out_features, in_features, k_size)
    return resulting_size


def get_layerwise_size_per_eps(eps):
    
    layerwise_budget = get_resulting_layer_budget(eps)

    total_in_features = []
    in_feat_reduction = []
    for ell in range(len(modules)):
        tracker = sens_trackers[ell]
        weight = tracker.module.weight
        layer_budget = layerwise_budget[ell]

        in_features = get_num_features(weight, 1)
        if layer_budget < in_features:
            in_features = layer_budget
        if in_features < 1:
            in_features = 1

        total_in_features.append(int(in_features.data.cpu().numpy()))
        in_feat_reduction.append(get_num_features(weight, 1) - in_features)

    total_out_features = []
    for ell in range(len(modules)):
        tracker = sens_trackers[ell]
        weight = tracker.module.weight

        out_features = float(get_num_features(weight, 0))
        if ell < len(modules)-1:
            sub = in_feat_reduction[ell+1] 
            out_features -= float(sub) 
        if out_features < 1:
            out_features = 1
        total_out_features.append(int(out_features))
    
    return total_in_features, total_out_features

def find_opt_eps(budget):
    eps_min = 1e-300
    eps_max = 1e+150

    def f_opt(arg):
        size_resulting = get_resulting_size_per_eps(arg)
        return budget - size_resulting
    
    f_value_min = f_opt(eps_min)
    f_value_max = f_opt(eps_max)
    print(f_value_min, f_value_max)
    if f_value_min * f_value_max > 0:
        print("pruning not possible")
        return 0

    eps_opt = optimize.brentq(f_opt, eps_min, eps_max, maxiter=1000, xtol=10e-250, disp=False)
        
    return eps_opt

"""
now let's compress
"""

def prune(size_pruned, probs):
    mask = torch.zeros_like(probs, dtype=torch.bool)
    if size_pruned > 0:
        size_pruned = int(size_pruned)
        idx_top = np.argpartition(probs.view(-1).cpu().numpy(), -size_pruned)[-size_pruned:]
        mask.view(-1)[idx_top] = True
    masked_features = mask.view(mask.shape[0], -1).sum(dim=-1)
        
    return masked_features

def sparsify(masked_features, weight_original):
    gammas = copy.deepcopy(masked_features).float()
    gammas = (gammas > 0).float()
    gammas = gammas.unsqueeze(0).unsqueeze(-1)
    weight_hat = (gammas * weight_original.view(weight_original.shape[0], weight_original.shape[1], -1)).view_as(weight_original)

    return nn.Parameter(weight_hat)

def get_dummy_net(compressed_modules):
    compressed_net = LeNet300(num_classes=10)
    compressed_net.fc1 = compressed_modules[0]
    compressed_net.fc2 = compressed_modules[1]
    compressed_net.fc3 = compressed_modules[2]
    
    return compressed_net

def propagate_compression(pruned_net):
    def zero_grad(net):
        for param in net.parameters():
            param.grad = None

    def parameters_for_grad_prune(pruned_net):
        pruned_net_modules = [module for module in pruned_net.modules() \
               if module != pruned_net and isinstance(module, nn.Module)]
        for module in pruned_net_modules:
            if module != net and isinstance(module, nn.Module):
                yield module.weight

    def get_prune_mask_from_grad(grad):
        out_is_const = grad.view(grad.shape[0], -1).abs().sum(-1) == 0.0
        mask = out_is_const.view((-1,) + (1,) * (grad.dim() - 1))
        return mask
    
    zero_grad(pruned_net)


    pruned_net.eval()
    device = modules[0].weight.device

    at_least_one_batch = False
    with torch.enable_grad():
        for images, targets in loader_s:
            if len(images) < 2:
                continue
            at_least_one_batch = True
            images = tensor.to(images, device, non_blocking=True)
            targets = tensor.to(targets, device, non_blocking=True)
            outs = pruned_net(images)
            loss = loss_handle(outs, targets)
            loss.backward()

    # post-process gradients to set respective weights to zero
    some_grad_none = False
    with torch.no_grad():
        for param in parameters_for_grad_prune(pruned_net):
            grad = param.grad
            if grad is None:
                some_grad_none = True
                continue
            # mask anything at machine precision or below.
            prune_mask = get_prune_mask_from_grad(grad)
            param.masked_fill_(prune_mask, 0.0)

    # issue warning in case some gradients were None
    if some_grad_none:
        print("Some parameters did not received gradients"
              " while propagating compression!")

    zero_grad(pruned_net)

    return pruned_net

def compress_once(keep_ratio):
    compressed_modules = copy.deepcopy(modules)
    budget_per_layer = [(module.weight != 0.0).sum().item() for module in compressed_modules]
    #print(budget_per_layer)

    budget_total = int(keep_ratio * original_size)
    budget_available = budget_total - uncompressible_size

    budget_available = min(budget_available, compressible_size)
    budget_available = max(0, budget_available)

    eps_opt = find_opt_eps(budget_available)
    pruned_input_size, pruned_output_size = get_layerwise_size_per_eps(eps_opt)
    #print(pruned_input_size, pruned_output_size)

    # left_inp_features = {ell:[] for ell in range(len(compressed_modules))}
    for ell in reversed(range(len(compressed_modules))):
        module = compressed_modules[ell]
        #print(module.weight.shape)
        #print(module.bias.shape)
        tracker = sens_trackers[ell]
        sens_in = tracker.sensitivity_in
        sum_sens = sens_in.sum().view(-1)
        probs = sens_in / sum_sens
        size_pruned = pruned_input_size[ell]

        masked_features = prune(size_pruned, probs)
        #print(size_pruned)
        weight_hat = sparsify(masked_features, module.weight)
        module.weight.data = weight_hat
        
    compressed_net = get_dummy_net(compressed_modules)
    compressed_net = propagate_compression(compressed_net)
    compressed_net_modules = [module for module in compressed_net.modules() \
               if module != compressed_net and isinstance(module, nn.Module)]
    compressed_net_size = get_original_size(compressed_net_modules)
    #print(compressed_net_size, original_size)
    print("achieved compression : ", compressed_net_size/original_size)
    #accuracy = test(compressed_net, loader_test)*100
    #print('Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)
    #print(modules[-1].weight, compressed_modules[-1].weight)
    return compressed_net_size, compressed_net

# compress_once(0.5)




def compress(keep_ratio):
    kr_min = 0.4 * keep_ratio
    kr_max = max(keep_ratio, 0.999 * 1.0)
    f_opt_lookup = {}

    def _f_opt(kr_compress):
        if kr_compress in f_opt_lookup:
            return f_opt_lookup[kr_compress]
        
        compressed_net_size = compress_once(kr_compress)[0]
        kr_actual = compressed_net_size / original_size
        kr_diff = kr_actual - keep_ratio

        #print(f"Current diff in keep ratio is: {kr_diff * 100.0:.2f}%")
        
        if abs(kr_diff) < 0.0005 * keep_ratio:
            kr_diff = 0.0
    
        f_opt_lookup[kr_compress] = kr_diff
        return f_opt_lookup[kr_compress]
    
    try:
        kr_diff_nominal = _f_opt(keep_ratio)
        if kr_diff_nominal == 0.0:
            return compress_once(keep_ratio)[1]
        elif kr_diff_nominal > 0.0:
            kr_max = keep_ratio
        else:
            kr_min = keep_ratio
    except (ValueError, RuntimeError):
        pass
    try:
        kr_opt = optimize.brentq(
            lambda kr: _f_opt(kr),
            kr_min,
            kr_max,
            maxiter=20,
            xtol=5e-3,
            rtol=5e-3,
            disp=True,
        )
    except (ValueError, RuntimeError):
        kr_diff_opt = float("inf")
        kr_opt = None
        #print(f_opt_lookup)
        for kr_compress in list(f_opt_lookup.items()):
            kr_diff = f_opt_lookup[kr_compress]
            if abs(kr_diff) < abs(kr_diff_opt):
                kr_diff_opt = kr_diff
                kr_opt = kr_compress
        print(
            "Cannot approximate keep ratio. "
            f"Picking best available keep ratio {kr_opt * 100.0:.2f}% "
            f"with actual diff {kr_diff_opt * 100.0:.2f}%."
        )

    return compress_once(kr_opt)[1]

modules = [module for module in net.modules() if module != net and isinstance(module, nn.Module)]
sens_trackers = nn.ModuleList()

loss_handle = CrossEntropyLossWithAuxiliary()

original_size = get_original_size(modules) # contains the number of non-zero parameters
compressible_layers, num_weights = get_compressible_layers(modules) 
compressible_size = get_compressible_size(modules) # contains the number of non-zero parameters belonging to the weight and the bias is excluded
uncompressible_size = original_size - compressible_size # number of parameters that belong to the bias in different layers

for ell, module in enumerate(modules):
    sens_trackers.append(PFPTracker(module))
    sens_trackers[ell].enable_tracker()

# get a loader with mini-batch size 1
loader_mini = tensor.MiniDataLoader(loader_s, 1)
num_batches = len(loader_mini)

for i_batch, (images, _) in enumerate(loader_mini):
    print(i_batch, len(loader_mini))
    images = images.to(device)
    outputs = net.forward_mod(images)
    for ell in range(len(modules)):
        module = sens_trackers[ell].module
        sens_trackers[ell]._hook(module, (outputs[0][ell],), outputs[1][ell]) 

for ell in range(len(modules)):
    sens_trackers[ell].probability = torch.zeros(sens_trackers[ell].sensitivity_in.shape)
    nnz = (sens_trackers[ell].sensitivity_in != 0.0).sum().view(-1)
    sum_sens = sens_trackers[ell].sensitivity_in.sum().view(-1)
    sens_trackers[ell].probability = sens_trackers[ell].sensitivity_in / sum_sens

delta_failure = 1e-12
keep_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
error_percent_replication = []

for keep_ratio in keep_ratios:
    print("---------------------------------------------------------------------------")
    final_compressed_net = compress(keep_ratio)
    print(final_compressed_net)
    # Evaluate the model on the test set
    loss = test(final_compressed_net, loader_test)
    print('Keep ratio : %0.2f, loss of the network on the 10000 test images: %f %%' % (keep_ratio, loss))
    error_percent_replication.append(loss)

# Plot the data
plt.plot(keep_ratios, error_percent_replication, '-o', label='replication')

loss = test(net, loader_test)
plt.plot(keep_ratios, [loss]*9, '-o', label='unpruned loss')

# Set the grid
plt.grid(True)

# Set the axes labels
plt.xlabel('Retained Parameters ratio')
plt.ylabel('Error')

# Set the title
plt.title('LeNet-300 pruning statistics - replication')

# Set the legend
plt.legend()

# Show the plot
plt.savefig('lenet300_prune_stats.png')


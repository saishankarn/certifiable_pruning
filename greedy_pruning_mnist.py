import os
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

from ptflops import get_model_complexity_info

eps = 1e-40 # subgradient of ReLU in pytorch = 0
 

# mask layer
class Mask(nn.Module):
    def __init__(self, D_in, layer_num=-1):
        super(Mask, self).__init__()

        '''
        [(a_i + gamma_i)(1 + u_i * gamma) + w_i * gamma] * neuron
        '''
        self.prune_a = nn.Parameter(1./D_in * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_gamma = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        #self.prune_u = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_w = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_lsearch = nn.Parameter(0. * torch.tensor(1.), requires_grad=False)
        self.scale = D_in # output size of a particular filter

        self.layer_num = layer_num
        self.D_in = D_in
        self.device = 'cuda'
        self.mode = 'train'
        self.zeros = nn.Parameter(torch.zeros(1,1,1,1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(1, self.D_in, 1, 1), requires_grad=False)

    def forward(self, x):
        return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w))

    def pforward(self, x, chosen_layer):
        if self.layer_num == chosen_layer:
            return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (
                    1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w)), x
        else:
            return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (
                1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w)), self.zeros

    def turn_off(self, src_param, is_lsearch = False):
        if not is_lsearch:
            tar_param = nn.Parameter(torch.zeros(1, self.D_in, 1, 1), requires_grad=False)
        else:
            tar_param = nn.Parameter(torch.tensor(1.), requires_grad=False)
        tar_param.data = src_param.data.clone()

        return tar_param
    def switch_mode(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.prune_gamma = self.turn_off(self.prune_gamma)
            self.prune_lsearch = self.turn_off(self.prune_lsearch, True)
            self.prune_a = self.turn_off(self.prune_a)

        elif mode == 'prune':
            self.prune_gamma.requires_grad = True
            self.prune_lsearch.requires_grad = True
            self.prune_a = self.turn_off(self.prune_a)

        elif mode == 'adjust_a':
            self.prune_gamma = self.turn_off(self.prune_gamma)
            self.prune_lsearch = self.turn_off(self.prune_lsearch, True)
            self.prune_a.requires_grad = True
        else:
            raise NotImplementedError

    def empty_all_eps(self):
        self.prune_a.data = -eps * self.prune_a.data

    def init_lsearch(self, neuron_index):
        self.prune_gamma.data = 0. * self.prune_gamma.data
        self.prune_w.data = 0.* self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        if neuron_index >= 0:
            self.prune_w[:, neuron_index, :, :] += 1.

    def update_alpha(self, neuron_index, lsearch):
        self.prune_a.data *= (1. - lsearch)
        self.prune_a[:, neuron_index, : ,:] += lsearch
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data
    
    # def update_alpha_back(self, neuron_index, lsearch):
    #     #self.prune_a.data *= (1. - lsearch)
    #     self.prune_a[:, neuron_index, : ,:] *= 0
    #     self.prune_w.data = 0. * self.prune_w.data
    #     self.prune_lsearch.data = 0. * self.prune_lsearch.data
    #     self.prune_gamma.data = 0. * self.prune_gamma.data

    def set_alpha_to_init(self, prunable_neuron):
        if len(prunable_neuron) != self.prune_a.shape[1]:
            print('dim of prunable_neuron error!')
            raise ValueError

        self.prune_a.data = 0. * self.prune_a.data

        num_prunable_neuron = prunable_neuron.sum()
        for _ in range(len(prunable_neuron)):
            if prunable_neuron[_] > 0:
                self.prune_a.data[0, _, 0, 0] += 1. / num_prunable_neuron

        # self.prune_a.data += 1./self.D_in * torch.ones(1, self.D_in, 1, 1).to(self.device)
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

    # def assign_alpha(self, alpha):
    #     self.prune_a.data = 0. * self.prune_a.data
    #     self.prune_a.data += alpha
    #     self.prune_w.data = 0. * self.prune_w.data
    #     self.prune_lsearch.data = 0. * self.prune_lsearch.data
    #     self.prune_gamma.data = 0. * self.prune_gamma.data


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 4 * 4)
        x = self.classifier(x)
        return x

    
class LeNet5_prescreen(nn.Module):
    def __init__(self, net):
        super(LeNet5_prescreen, self).__init__()
        
        self.features = nn.Sequential(
            net.features[0],
            net.features[1],
            Mask(D_in = net.features[0].weight.shape[0], layer_num = 0),
            net.features[2],
            net.features[3],
            net.features[4],
            Mask(D_in = net.features[3].weight.shape[0], layer_num = 1),
            net.features[5],
        )
        self.classifier = net.classifier

        self.mask_features = [2,6]
        self.conv_features = [0,4]

    def pforward(self, x, chosen_layer = -1):
        score = 0.
        for idx, block in enumerate(self.features):
            if idx in self.mask_features:
                x, l_score, chosen_layer = block.pforward(x, score, chosen_layer)
                score += l_score
            else:
                x = block(x)
        x = x.view(x.size(0), 16 * 4 * 4)
        x = self.classifier(x)
        return x, score, chosen_layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 4 * 4)
        x = self.classifier(x)
        return x


def decide_candidate_set(m, prunable_neuron, num_evaluate=50, isfirst=False):
    # randomly picks up num_evaluate number of neurons

    # only randomly pickup num_evaluate number of neurons to form the candidate set
    candidate_plus = []
    
    tem_a = m.prune_a.data.squeeze().cpu().numpy()

    if isfirst:
        eps_ = eps
    else:
        eps_ = 0.

    tem_a = np.where(tem_a <= eps_)[0] # randomly pick up outside neuron to add
    # print(tem_a)
    np.random.shuffle(tem_a)
    tem_a = set(tem_a)
    # print(tem_a)
    prunable_neuron = set(np.where(prunable_neuron.astype(float) > 0)[0])
    tem_a = list(tem_a & prunable_neuron)
    
    candidate_plus = tem_a[:num_evaluate]
    # print(candidate_plus)

    return candidate_plus

def decide_candidate(m, candidate_plus, isfirst=False):
    opt_index = -1
    opt_acc = 0.0
    opt_stepsize = 0.

    current_num_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))

    for candidate in candidate_plus:
        #print(candidate)
        m.init_lsearch(candidate) # increases the particular index of prune_w by 1
        #print(torch.where(m.prune_w))
        m.prune_lsearch.data += 1. / (current_num_neuron + 1)
        #print(m.prune_lsearch.data)

        acc = test_with_data(masked_net, validation_data)

        if acc > opt_acc:
            opt_index = candidate
            opt_acc = acc
            opt_stepsize = 1. / (current_num_neuron + 1)

    if isfirst:
        m.prune_a *= 0.
        m.prune_a[:, opt_index, :, :] += 1.
        m.prune_w.data = 0. * m.prune_w.data
        m.prune_lsearch.data = 0. * m.prune_lsearch.data
        m.prune_gamma.data = 0. * m.prune_gamma.data
    else:
        m.update_alpha(opt_index, opt_stepsize)




def prune_a_layer(m):

    isalladd = 0 
    num_layer = m.layer_num
    #print(m.prune_a.shape, m.prune_gamma.shape, m.prune_w.shape, m.prune_lsearch.shape)

    unpruned_accuracy = test_with_data(masked_net, validation_data)
    
    print('Unpruned Accuracy : %.2f' % unpruned_accuracy)
    
    m.switch_mode('prune')    
    
    # prunable neuron list; only consider the neuron that is inside at initial
    prunable_neuron = (m.prune_a.cpu().data.squeeze().numpy() > 0)
    all_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
    
    m.empty_all_eps()
    #print(masked_net.features[2].prune_a)
    #print(all_neuron)
    
    is_first_neuron = 1

    while 1:
        
        candidate_plus = decide_candidate_set(m, prunable_neuron, num_evaluate=num_evaluate, isfirst=is_first_neuron)        
        decide_candidate(m, candidate_plus, is_first_neuron)
        if is_first_neuron:
            is_first_neuron = 0
        
        pruned_accuracy = test_with_data(masked_net, validation_data)
        # print('Pruned Accuracy : ', pruned_accuracy)
        if pruned_accuracy >= epsilon * unpruned_accuracy:
            # evaluate whether converged
            pruned_accuracy = test_with_data(masked_net, validation_data)
            cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            
            # print('Cur_neuron/ All neuron', cur_neuron/m.scale)
            

            if pruned_accuracy >= epsilon * unpruned_accuracy: break  

        else:
            cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            # print('Cur_neuron/ All neuron', cur_neuron / all_neuron)
        
        if cur_neuron >= all_neuron:
            # print('all the neurons are added')
            m.set_alpha_to_init(prunable_neuron)
            isalladd = 1
            break


    print("This layer's Neuron", cur_neuron)
    pruned_accuracy = test(masked_net, val_loader)
    print("Accuracy : ", pruned_accuracy)

    a_para = m.prune_a.data
    a_num = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
    m.set_alpha_to_init(prunable_neuron)
    
    return a_para, a_num, pruned_accuracy, isalladd


def net_prune(masked_net):
    # net.eval()

    full_cfg = []
    cur_cfg = []

    for block_idx, block in enumerate(masked_net.features):
        if isinstance(block, Mask):
            full_cfg.append(block.prune_a.shape[1])
            cur_cfg.append(block.prune_a.shape[1])
    
    num_layer = -1
    # total_start = time.time()
    for block_idx, block in enumerate(masked_net.features):
        print("pruning layer number : ", block_idx)
        mask_count = -1
        if isinstance(block, Mask):
            #print(m)
            num_layer += 1
            isalladd = 0

            
            a_para, a_num, global_cur_top1, isalladd = prune_a_layer(block)
        
            block.prune_a.data = a_para
            cur_neuron=a_num
            mb2_prune_ratio(masked_net)

    fullflops, pruneflops, fullparams, pruneparams = mb2_prune_ratio(masked_net)
    print("Full Flops, Prune Flops, Full Params, Prune Params")
    print(fullflops, pruneflops, fullparams, pruneparams)


    return masked_net

class tem_block(nn.Module):
    def __init__(self, layers, identity, ind, convd, outd):
        super(tem_block, self).__init__()
        self.conv = nn.Sequential(*layers)
        self.identity = identity
        self.ind = ind
        self.convd = convd
        self.outd = outd  # ind, convd <= outd

    def forward(self, x):
        xx = self.conv(x)
        if self.identity:
            if self.ind < self.outd:
                x = torch.cat([x, torch.zeros(x.size(0), self.outd - self.ind, x.size(2), x.size(3))], 1)
            if self.convd < self.outd:
                xx = torch.cat([xx, torch.zeros(xx.size(0), self.outd - self.convd, xx.size(2), xx.size(3))], 1)
            return x + xx
        else:
            return xx


def get_pruned_conv(conv, mask, inp_channels):
    out_channels = int((mask.prune_a.cpu().squeeze().numpy() > 0.).sum())
    kernel_size, stride, padding = conv.kernel_size, conv.stride, conv.padding
    pruned_conv = nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, \
                            stride=stride, padding=padding, bias=False)
    return pruned_conv, out_channels
    

class tem_LeNet(nn.Module):
    def __init__(self, net):
        super(tem_LeNet, self).__init__()
        blocks = []
        inp_channels = 1
        out_features = 10

        self.features = []
        mask_idx = 0
        for m in net.features:
            if isinstance(m, nn.Conv2d):
                mask = net.features[net.mask_features[mask_idx]]
                mask_idx += 1
                m, inp_channels = get_pruned_conv(m, mask, inp_channels)
            #elif isinstance(m, nn.ReLU):

            #elif isinstance(m, nn.MaxPool2d):
            if isinstance(m, Mask):
                continue
            self.features.append(m)

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(in_features=inp_channels*4*4, out_features=out_features, bias=True)
        self.feat_channels = inp_channels

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.feat_channels * 4 * 4)
        x = self.classifier(x)
        return x


def mb2_prune_ratio(masked_net):
    net = LeNet5()
    
    net.eval()
    fullflops, fullparams = get_model_complexity_info(net, (1, 28, 28), as_strings=False,
                                                      print_per_layer_stat=False)
    copy_masked_net = copy.deepcopy(masked_net)
    pruned_network = tem_LeNet(copy_masked_net).cpu()
    pruned_network.eval()
    # print(pruned_network)

    pruneflops, pruneparams = get_model_complexity_info(pruned_network, (1, 28, 28), as_strings=False,
                                                        print_per_layer_stat=False)

    # print('calculation pruned finish')

    # print('flops% = ', pruneflops/fullflops)
    # print('parameters% = ', pruneparams/fullparams)
    return fullflops, pruneflops, fullparams, pruneparams


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

size_s = 64
batch_size = 64
testset, set_s = torch.utils.data.random_split(
    testset, [len(testset) - size_s, size_s]
)

loader_s = torch.utils.data.DataLoader(set_s, batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False
)

eval_train_loader = trainloader

# Instantiate the AlexNet model and move it to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet5().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Train the model
def train(net):
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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



def test(masked_net, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = masked_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def test_with_data(net, data):
    correct = 0
    total = 0
    with torch.no_grad():
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct/total



model_path = '../../checkpoints/lenet5_mnist_pfp.pth'
if os.path.isfile(model_path):
    checkpt = torch.load(model_path)
    net.load_state_dict(checkpt)
else:
    net = train(net)

for validation_data in val_loader: break

net.eval()
loss = test(net, val_loader)
print('Loss of the network on the 10000 test images: %f' % loss)

epsilon_vals = [1*i/50 for i in range(50)] 
# epsilon_vals = [0.99]
num_evaluate = 10000

pr_vals = []
acc_vals = []

for eps_idx in range(len(epsilon_vals)):
    print("------------------------------------------")
    epsilon = epsilon_vals[eps_idx]
    masked_net = LeNet5_prescreen(net).to(device)
    masked_net.eval()
    masked_net = net_prune(masked_net)
    acc = test(masked_net, val_loader)
    _, _, full_params, prune_params = mb2_prune_ratio(masked_net)
    pr = prune_params / full_params
    pr_vals.append(prune_params)
    acc_vals.append(acc)

print(pr_vals)
print(acc_vals)
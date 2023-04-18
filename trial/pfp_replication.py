import os
import copy
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

from get_sensitivity import * 

"""
network architecture
"""

class LeNet5(nn.Module):
    def __init__(self, num_classes, num_in_channels):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channels, 6, 5)  # Convolutional Layer 1: input channels=1, output channels=6, kernel size=5x5
        self.conv2 = nn.Conv2d(6, 16, 5) # Convolutional Layer 2: input channels=6, output channels=16, kernel size=5x5
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # Fully Connected Layer 1: input features=16*4*4 (output size of conv2), output features=120
        self.fc2 = nn.Linear(120, 84) # Fully Connected Layer 2: input features=120, output features=84
        self.fc3 = nn.Linear(84, num_classes) # Fully Connected Layer 3 (Output Layer): input features=84, output features=10 (number of classes in MNIST)

    def forward(self, x):
        in_0 = x
        out_0 = self.conv1(in_0) # Apply Convolutional Layer 1 and ReLU activation
        #print(in_0.shape, out_0.shape)
        in_1 = F.relu(out_0)
        in_1 = F.max_pool2d(in_1, 2) # Apply Max Pooling Layer 1
        out_1 = self.conv2(in_1) # Apply Convolutional Layer 2 and ReLU activation
        #print(in_1.shape, out_1.shape)
        in_2 = F.relu(out_1)
        in_2 = F.max_pool2d(in_2, 2) # Apply Max Pooling Layer 2
        in_2 = in_2.view(-1, 16 * 4 * 4) # Flatten the feature maps
        out_2 = self.fc1(in_2) # Apply Fully Connected Layer 1 and ReLU activation
        #print(in_2.shape, out_2.shape)
        in_3 = F.relu(out_2)
        out_3 = self.fc2(in_3) # Apply Fully Connected Layer 2 and ReLU activation
        #print(in_3.shape, out_3.shape)
        in_4 = F.relu(out_3)
        out_4 = self.fc3(in_4) # Apply Fully Connected Layer 3 (Output Layer)
        #print(in_4.shape, out_4.shape)
        return [(in_0, in_1, in_2, in_3, in_4), (out_0, out_1, out_2, out_3, out_4)]

name = 'lenet5_mnist'
net = LeNet5(num_classes=10, num_in_channels=1)

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
Train the model
"""

model_path = 'checkpoints/' + name + '.pth'
if os.path.isfile(model_path):
    checkpt = torch.load(model_path)
    net.load_state_dict(checkpt)
    net.eval()
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader_train, 0):
            inputs, labels = data

            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)[-1][-1]
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch [%d], Loss: %.4f' % (epoch + 1, running_loss / (i + 1)))

    print('Finished training')
    torch.save(net.state_dict(), model_path)

# Evaluate the model on the test set

correct = 0
total = 0
with torch.no_grad():
    for data in loader_test:
        inputs, labels = data
        outputs = net(inputs)[-1][-1]
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print('Accuracy on test set: %.2f %%' % (100 * correct / total))

backup_net = copy.deepcopy(net)

"""
Let's start with Provable Filter Pruning
Some basic definitions and declarations first
"""
modules = [module for module in net.modules() if module != net and isinstance(module, nn.Module)]
sens_trackers = nn.ModuleList()

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

original_size = get_original_size(modules) # contains the number of non-zero parameters
compressible_layers, num_weights = get_compressible_layers(modules) 
compressible_size = get_compressible_size(modules) # contains the number of non-zero parameters belonging to the weight and the bias is excluded
uncompressible_size = original_size - compressible_size # number of parameters that belong to the bias in different layers


keep_ratio = 0.5
delta_failure = 1e-6
kr_min = 0.4 * keep_ratio
kr_max = max(keep_ratio, 0.999)

"""
Now let's calculate sensitivity
"""

for ell, module in enumerate(modules):
    sens_trackers.append(PFPTracker(module))
    sens_trackers[ell].enable_tracker()

# get a loader with mini-batch size 1
loader_mini = tensor.MiniDataLoader(loader_s, 1)
num_batches = len(loader_mini)
       
for i_batch, (images, _) in enumerate(loader_mini):
    outputs = net(images)
    #print(outputs[-1][-1])
    #print(i_batch)
    #print(images.shape)
    for ell in range(len(modules)):
        module = sens_trackers[ell].module
        #print(outputs[0][ell].shape, outputs[1][ell].shape)
        sens_trackers[ell]._hook(module, (outputs[0][ell],), outputs[1][ell]) 

# for ell in range(len(modules)):
#     print(sens_trackers[ell].sensitivity_in)

"""
obtaining the probability
"""

for ell in range(len(modules)):
    sens_trackers[ell].probability = torch.zeros(sens_trackers[ell].sensitivity_in.shape)
    nnz = (sens_trackers[ell].sensitivity_in != 0.0).sum().view(-1)
    sum_sens = sens_trackers[ell].sensitivity_in.sum().view(-1)
    sens_trackers[ell].probability = sens_trackers[ell].sensitivity_in / sum_sens

# """
# let's find the optimal compression rate possible that is close to the user requested compression
# """

def get_num_features(tensor, dim):
    dims_to_sum = [i for i in range(tensor.dim()) if i is not dim]
    return (torch.abs(tensor).sum(dims_to_sum) != 0).sum()

def get_resulting_layer_budget(eps):
    k_constant = 3.0
    c_constant = 3.0
    
    resulting_layer_budget = {}
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

        vals = 1 - (1 - probs) ** sc
        expectation = torch.sum(torch.as_tensor(vals), dim=-1)
        per_layer_budget = torch.ceil(expectation)
        resulting_layer_budget[ell] = per_layer_budget

    return resulting_layer_budget

def get_resulting_size_per_eps(eps):
    
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
        out_features = 1

        in_features = total_in_features[ell]
        k_size = weight[0, 0].numel()
        size_total = float(in_features * out_features * k_size)
        resulting_size += size_total

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

        total_in_features.append(in_features)
        in_feat_reduction.append(get_num_features(weight, 1) - in_features)

    total_out_features = []
    for ell in range(len(modules)):
        tracker = sens_trackers[ell]
        weight = tracker.module.weight

        out_features = get_num_features(weight, 0)
        if ell < len(modules)-1:
            sub = in_feat_reduction[ell+1] 
            out_features -= sub 
        out_features = 1
        total_out_features.append(out_features)
    
    return total_in_features, total_out_features

def find_opt_eps(budget):
    eps_min = 1e-300
    eps_max = 1e150

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

find_opt_eps(25000)

"""
now let's compress
"""


def compress(keep_ratio):
        
    budget_total = int(keep_ratio * original_size) # the total number of parameters we can have
    budget_available = budget_total - uncompressible_size # few of them cannot be compressed like the bias.
    
    # some sanity checks for the budget
    budget_available = min(budget_available, compressible_size)
    budget_available = max(0, budget_available)

    eps_opt = find_opt_eps(budget_available)
    req_in_features, req_out_features = get_layerwise_size_per_eps(eps_opt)






class CrossEntropyLossWithAuxiliary(nn.CrossEntropyLoss):

    def forward(self, input, target):
        if isinstance(input, dict):
            loss = super().forward(input["out"], target)
            if "aux" in input:
                loss += 0.5 * super().forward(input["aux"], target)
        else:
            loss = super().forward(input, target)
        return loss

# get a loss handle
loss_handle = CrossEntropyLossWithAuxiliary()

net = tp.util.net.NetHandle(net, name)
net_filter_pruned = tp.PFPNet(net, loader_s, loss_handle)
print(
    f"The network has {net_filter_pruned.size()} parameters and "
    f"{net_filter_pruned.flops()} FLOPs left."
)
#net_filter_pruned.cuda()
net_filter_pruned.compress(keep_ratio=0.5)
#net_filter_pruned.cpu()


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

loader_s = torch.utils.data.DataLoader(set_s, batch_size=1, shuffle=False)
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
First, we calculate sensitivity
"""
modules = [module for module in net.modules() if module != net and isinstance(module, nn.Module)]
sens_trackers = nn.ModuleList()

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

for ell in range(len(modules)):
    print(sens_trackers[ell].sensitivity_in)


def get_optimal_compression_ratio(kr):
    # check for look-up
    if kr_compress in f_opt_lookup:
        return f_opt_lookup[kr_compress]

    # compress
    b_per_layer = self._compress_once(kr_compress, backup_net)

    # check resulting keep ratio
    kr_actual = (
        self.compressed_net.size() / self.original_net[0].size()
    )
    kr_diff = kr_actual - keep_ratio
    print(f"Current diff in keep ratio is: {kr_diff * 100.0:.2f}%")

    # set to zero if we are already close and stop
    if abs(kr_diff) < 0.005 * keep_ratio:
        kr_diff = 0.0

    # store look-up
    f_opt_lookup[kr_compress] = (kr_diff, b_per_layer)

    return f_opt_lookup[kr_compress]


keep_ratio = 0.5
kr_min = 0.4 * keep_ratio
kr_max = max(keep_ratio, 0.999)

kr_opt = optimize.brentq(lambda keep_ratio: get_optimal_compression_ratio(keep_ratio)[0], kr_min, kr_max, maxiter=20, xtol=5e-3, rtol=5e-3, disp=True)

b_per_layer, compressed_net = compress(kr_opt, backup_net) # define compress below
compression = compressed_net.size() / backup_net.size()
diff = compression - keep_ratio
print("Current diff in keep ratio is: ", diff * 100)

# set to zero if we are already close and stop
if abs(diff) < 0.005 * keep_ratio:
    diff = 0.0



"""
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
"""


# boundaries for binary search over potential keep_ratios

        # wrapper for root finding and look-up to speed it up.
        f_opt_lookup = {}

        def _f_opt(kr_compress):
            # check for look-up
            if kr_compress in f_opt_lookup:
                return f_opt_lookup[kr_compress]

            # compress
            b_per_layer = self._compress_once(kr_compress, backup_net)

            # check resulting keep ratio
            kr_actual = (
                self.compressed_net.size() / self.original_net[0].size()
            )
            kr_diff = kr_actual - keep_ratio
            print(f"Current diff in keep ratio is: {kr_diff * 100.0:.2f}%")

            # set to zero if we are already close and stop
            if abs(kr_diff) < 0.005 * keep_ratio:
                kr_diff = 0.0

            # store look-up
            f_opt_lookup[kr_compress] = (kr_diff, b_per_layer)

            return f_opt_lookup[kr_compress]

        # some times the keep ratio is pretty accurate
        # so let's try with the correct keep ratio first
        try:
            # we can either run right away or update the boundaries for the
            # binary search to make it faster.

            kr_diff_nominal, b_per_layer = _f_opt(keep_ratio)
            if kr_diff_nominal == 0.0:
                return b_per_layer
            elif kr_diff_nominal > 0.0:
                kr_max = keep_ratio
            else:
                kr_min = keep_ratio

        except (ValueError, RuntimeError):
            pass

        # run the root search
        # if it fails we simply pick the best value from the look-up table
        try:
            kr_opt = optimize.brentq(
                lambda kr: _f_opt(kr)[0],
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
            for kr_compress, kr_diff_b_per_layer in f_opt_lookup.items():
                kr_diff = kr_diff_b_per_layer[0]
                if abs(kr_diff) < abs(kr_diff_opt):
                    kr_diff_opt = kr_diff
                    kr_opt = kr_compress
            print(
                "Cannot approximate keep ratio. "
                f"Picking best available keep ratio {kr_opt * 100.0:.2f}% "
                f"with actual diff {kr_diff_opt * 100.0:.2f}%."
            )

        # now run the compression one final time
        return self._compress_once(kr_opt, backup_net)

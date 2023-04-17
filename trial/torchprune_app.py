# %% import the required packages
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
        x = F.relu(self.conv1(x)) # Apply Convolutional Layer 1 and ReLU activation
        x = F.max_pool2d(x, 2) # Apply Max Pooling Layer 1
        x = F.relu(self.conv2(x)) # Apply Convolutional Layer 2 and ReLU activation
        x = F.max_pool2d(x, 2) # Apply Max Pooling Layer 2
        x = x.view(-1, 16 * 4 * 4) # Flatten the feature maps
        x = F.relu(self.fc1(x)) # Apply Fully Connected Layer 1 and ReLU activation
        x = F.relu(self.fc2(x)) # Apply Fully Connected Layer 2 and ReLU activation
        x = self.fc3(x) # Apply Fully Connected Layer 3 (Output Layer)
        return x

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

size_s = 1
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
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch [%d], Loss: %.4f' % (epoch + 1, running_loss / (i + 1)))

    print('Finished training')
    torch.save(net.state_dict(), model_path)

class CrossEntropyLossWithAuxiliary(nn.CrossEntropyLoss):
    """Cross-entropy loss that can add auxiliary loss if present."""

    def forward(self, input, target):
        """Return cross-entropy loss and add auxiliary loss if possible."""
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
net_filter_pruned.cuda()
net_filter_pruned.compress(keep_ratio=0.5)
net_filter_pruned.cpu()

correct = 0
total = 0
with torch.no_grad():
    for data in loader_test:
        inputs, labels = data
        outputs = net_filter_pruned(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print('Accuracy on test set: %.2f %%' % (100 * correct / total))

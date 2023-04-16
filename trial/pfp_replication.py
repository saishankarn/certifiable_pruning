import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchprune as tp

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

        in_1 = F.relu(out_0)
        in_1 = F.max_pool2d(in_1, 2) # Apply Max Pooling Layer 1
        out_1 = self.conv2(in_1) # Apply Convolutional Layer 2 and ReLU activation
        
        in_2 = F.relu(out_1)
        in_2 = F.max_pool2d(in_2, 2) # Apply Max Pooling Layer 2
        in_2 = in_2.view(-1, 16 * 4 * 4) # Flatten the feature maps
        out_2 = self.fc1(in_2) # Apply Fully Connected Layer 1 and ReLU activation
        
        in_3 = F.relu(out_2)
        out_3 = self.fc2(in_3) # Apply Fully Connected Layer 2 and ReLU activation

        in_4 = F.relu(out_3)
        out_4 = self.fc3(in_4) # Apply Fully Connected Layer 3 (Output Layer)
        
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
sens_list = [Sensitivity(module) for module in modules]

for i_batch, (images, _) in enumerate(loader_s):
    outputs = net(images)
    for midx, sens in enumerate(sens_list):
        print("processing batch %d for module %d" % (i_batch+1, midx))
        sens.compute_sensitivity_for_batch(outputs[0][midx].data, outputs[-1][midx].data)
    
for midx, sens in enumerate(sens_list):
    print(sens.sensitivity_in.shape)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        in_0 = x
        out_0 = self.conv1(in_0)
        
        in_1 = F.relu(out_0)
        in_1 = F.max_pool2d(in_1, 2)
        out_1 = self.conv2(in_1)

        in_2 = F.relu(out_1)
        in_2 = F.max_pool2d(in_2, 2)
        out_2 = self.conv3(in_2)
        
        in_3 = F.relu(out_2)
        out_3 = self.conv4(in_3)
        
        in_4 = F.relu(out_3)
        out_4 = self.conv5(in_4)
        
        in_5 = F.relu(out_4)
        in_5 = F.max_pool2d(in_5, 2)
        in_5 = in_5.view(in_5.size(0), 256*2*2)
        out_5 = self.fc1(in_5)

        in_6 = F.relu(out_5)
        out_6 = self.fc2(in_6)

        in_7 = F.relu(out_6)
        out_7 = self.fc3(in_7)

        return out_7
    
    def forward_mod(self, x):
        in_0 = x
        out_0 = self.conv1(in_0)
        
        in_1 = F.relu(out_0)
        in_1 = F.max_pool2d(in_1, 2)
        out_1 = self.conv2(in_1)

        in_2 = F.relu(out_1)
        in_2 = F.max_pool2d(in_2, 2)
        out_2 = self.conv3(in_2)
        
        in_3 = F.relu(out_2)
        out_3 = self.conv4(in_3)
        
        in_4 = F.relu(out_3)
        out_4 = self.conv5(in_4)
        
        in_5 = F.relu(out_4)
        in_5 = F.max_pool2d(in_5, 2)
        in_5 = in_5.view(in_5.size(0), 256*2*2)
        out_5 = self.fc1(in_5)

        in_6 = F.relu(out_5)
        out_6 = self.fc2(in_6)

        in_7 = F.relu(out_6)
        out_7 = self.fc3(in_7)

        return [(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7), \
                (out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7)]

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

cifar_tran_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
cifar_tran_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

data_root='../../Network-Pruning-Greedy-Forward-Selection/dataroot'
batch_size = 128
n_worker = 1

transform_train = transforms.Compose(cifar_tran_train)
transform_test = transforms.Compose(cifar_tran_test)
trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, pin_memory=True, sampler=None)
    

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=n_worker, pin_memory=True)

# Initialize the LeNet-300 model, loss function, and optimizer
net = AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

keep_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
name = 'alexnet_cifar_pfp'
errors = []
for i, kr in enumerate(keep_ratios):
    net = AlexNet(num_classes=10).to(device)
    model_path = '../checkpoints/' + name + 'pr_ratio_' + str(i) + '_.pth'
    checkpt = torch.load(model_path)
    net.load_state_dict(checkpt)
    net.eval()
    loss = test(net, val_loader)
    print('Keep ratio : %0.2f, loss of the network on the 10000 test images: %f %%' % (kr, loss))
    errors.append(loss)

# Plot the data
plt.plot(keep_ratios, errors, '-o', label='pruned loss')

plt.plot(keep_ratios, [0.27]*9, '-o', label='unpruned loss')

# Set the grid
plt.grid(True)

# Set the axes labels
plt.xlabel('Retained Parameters ratio')
plt.ylabel('Error')

# Set the title
plt.title('AlexNet pruning statistics - replication')

# Set the legend
plt.legend()

# Show the plot
plt.savefig('alexnet_prune_stats.png')
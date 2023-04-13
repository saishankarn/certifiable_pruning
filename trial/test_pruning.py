import os 

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torchprune as tp
from torchprune.util import nn_loss

class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784) # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet300()
model_path = 'models/lenet300.pth'
model_checkpt = torch.load(model_path)
model.load_state_dict(model_checkpt)

net = tp.util.net.NetHandle(model, 'lenet300')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)


loss_handle = getattr(nn_loss, "CrossEntropyLoss")(**{"reduction": "mean"})

net_filter_pruned = tp.PFPNet(net, testloader, loss_handle)
# keep_ratio = 0.5  # Ratio of parameters to keep
# net_filter_pruned.cuda()
# net_filter_pruned.compress(keep_ratio=keep_ratio)


# # Evaluate the model on the test set
# net_filter_pruned.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         outputs = net_filter_pruned(inputs)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

# print('Accuracy on test set: %.2f %%' % (100 * correct / total))





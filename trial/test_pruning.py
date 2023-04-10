import os 
import torch 
import torch.nn as nn
import torchprune as tp

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

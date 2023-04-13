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
        self.fc1_in = x.view(-1, 784).T # Flatten the input
        self.fc1_out = torch.matmul(self.fc1.weight, self.fc1_in)
        self.fc2_in = torch.relu(self.fc1_out + self.fc1.bias.unsqueeze(axis=-1))
        self.fc2_out = torch.matmul(self.fc2.weight, self.fc2_in)
        self.fc3_in = torch.relu(self.fc2_out + self.fc2.bias.unsqueeze(axis=-1))
        self.fc3_out = torch.matmul(self.fc3.weight, self.fc3_in) + self.fc3.bias.unsqueeze(axis=-1)
        return {'fc1_in' : self.fc1_in, \
                'fc1_out' : self.fc1_out, \
                'fc2_in' : self.fc2_in, \
                'fc2_out' : self.fc2_out, \
                'fc3_in' : self.fc3_in, \
                'fc3_out' : self.fc3_out}

model = LeNet300()
model_path = 'models/lenet300.pth'
model_checkpt = torch.load(model_path)
model.load_state_dict(model_checkpt)
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)



# Evaluate the model on the test set

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)['fc3_out'].T
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print('Accuracy on test set: %.2f %%' % (100 * correct / total))

prunable_layers = [key.split('.')[0] for key in model.state_dict().keys() \
                   if 'weight' in key]

# for player in prunable_layers:
#     print("looking into the layer %s" % player)
#     print("the layer size is %d x %d" % (model.state_dict()[player].shape[0], \
#                                          model.state_dict()[player].shape[1]))
    
"""
storing the inputs and the preactivation outputs for each layer
"""

path = 'results/sensitivity_inputs.pth'
sensitivity_inputs = {player:{'in' : torch.tensor([]), \
                              'out' : torch.tensor([])} for player in prunable_layers}
if os.path.exists(path):
    sensitivity_inputs = torch.load(path)
else:
    for data in testloader:
        inputs, labels = data 
        outputs = model(inputs)
        for player in prunable_layers:
            sensitivity_inputs[player]['in'] = torch.cat((sensitivity_inputs[player]['in'], \
                                                outputs[player.split('.')[0] + '_in']), axis=-1) 
            sensitivity_inputs[player]['out'] = torch.cat((sensitivity_inputs[player]['out'], \
                                                outputs[player.split('.')[0] + '_out']), axis=-1) 

    torch.save(sensitivity_inputs, path)

"""
calculating sensitivity for each weight column in all the layers
"""
sensitivity = {player : [] for player in prunable_layers}
for player in prunable_layers:
    print('considering pruning the layer - ', player)
    inputs = sensitivity_inputs[player]['in']
    outputs = sensitivity_inputs[player]['out']
    
    num_features = inputs.shape[0]
    num_data_points = inputs.shape[1]
    num_next_features = outputs.shape[0]
    weight = model.state_dict()[player + '.weight']
    bias = model.state_dict()[player + '.bias']

    print("number of features for which we need to calculate the sensitivity : %d" % num_features)
 
    ## now we can calculate the sensitivity of each feature 
    numerator_list = []
    for fidx in range(num_features):
        print(fidx)
        num = torch.matmul(weight[:, fidx].unsqueeze(1), inputs[fidx].unsqueeze(0))
        print(num.shape)
        numerator_list.append(num)

    positive_denominator = torch.zeros(numerator_list[0].shape)
    negative_denominator = torch.zeros(numerator_list[0].shape)
    for mat in numerator_list:
        pos_mat = torch.clamp(mat, min=0)
        neg_mat = torch.clamp(mat, max=0)
        positive_denominator += pos_mat 
        negative_denominator += neg_mat

    for fidx in range(num_features):
        num = numerator_list[fidx]
        mat1 = torch.div(num, positive_denominator)
        mat2 = torch.div(num, negative_denominator)
        mat = torch.cat((mat1.unsqueeze(-1), mat2.unsqueeze(-1)), axis=-1)
        mat, _ = torch.max(mat, dim=-1)
        print(mat.shape)
        
    

#     for fidx in range(num_features):
#         num = torch.matmul(weight[:, fidx].unsqueeze(1), inputs[fidx].unsqueeze(0))
#         den_pos = 
#         den_neg =  
#         mat1 = torch.div(num, den_pos)
#         mat2 = torch.div(num, den_neg)
        
#         sen = torch.max(mat).item()
#         sensitivity[player].append(sen)

# """
# calculating the importance sampling distribution from sensitivity scores
# """

# importance_sampling_dist = {player : [] for player in prunable_layers}
# for player in prunable_layers:
#     player_sensitivities = sensitivity[player]
#     norm = sum(player_sensitivities)
#     player_dist = [player_sensitivities[i] / norm \
#                    for i in range(len(player_sensitivities))]
    
#     importance_sampling_dist[player] = player_dist


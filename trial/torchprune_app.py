# %% import the required packages
import os
import copy
import torch
import torchvision
import torchprune as tp

net_name = "lenet300_mnist"
net = tp.util.models.lenet300_mnist(num_classes=10)

net = tp.util.net.NetHandle(net, net_name)

n_idx = 0  # network index 0
keep_ratio = 0.5  # Ratio of parameters to keep
s_idx = 0  # keep ratio's index
r_idx = 0  # repetition index


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


train_params = {
    # any loss and corresponding kwargs for __init__ from tp.util.nn_loss
    "loss": "CrossEntropyLoss",
    "lossKwargs": {"reduction": "mean"},
    # exactly two metrics with __init__ kwargs from tp.util.metrics
    "metricsTest": [
        {"type": "TopK", "kwargs": {"topk": 1}},
        {"type": "TopK", "kwargs": {"topk": 5}},
    ],
    # any optimizer from torch.optim with corresponding __init__ kwargs
    "optimizer": "SGD",
    "optimizerKwargs": {
        "lr": 0.1,
        "weight_decay": 1.0e-4,
        "nesterov": False,
        "momentum": 0.9,
    },
    # batch size
    "batchSize": batch_size,
    # desired number of epochs
    "startEpoch": 0,
    "retrainStartEpoch": -1,
    "earlyStopEpoch":100,
    "numEpochs": 10,  # 182
    # any desired combination of lr schedulers from tp.util.lr_scheduler
    "lrSchedulers": [
        {
            "type": "MultiStepLR",
            "stepKwargs": {"milestones": [91, 136]},
            "kwargs": {"gamma": 0.1},
        },
        {"type": "WarmupLR", "stepKwargs": {"warmup_epoch": 5}, "kwargs": {}},
    ],
    "enableAMP": False,
    # output size of the network
    "outputSize": 10,
    # directory to store checkpoints
    "dir": os.path.realpath("./checkpoints"),
}

# Setup retraining parameters (just copy train-parameters)
retrain_params = copy.deepcopy(train_params)

# Setup trainer
trainer = tp.util.train.NetTrainer(
    train_params=train_params,
    retrain_params=retrain_params,
    train_loader=loader_train,
    test_loader=loader_test,
    valid_loader=loader_s,
    num_gpus=1,
)

# get a loss handle
loss_handle = trainer.get_loss_handle()

trainer.train(net, n_idx)
net_filter_pruned = tp.PFPNet(net, loader_s, loss_handle)
print(
    f"The network has {net_filter_pruned.size()} parameters and "
    f"{net_filter_pruned.flops()} FLOPs left."
)
net_filter_pruned.cuda()
net_filter_pruned.compress(keep_ratio=keep_ratio)
net_filter_pruned.cpu()
#print(net_filter_pruned.deterministic)

#net_filter_pruned = net_filter_pruned.cuda()
#trainer.retrain(net_filter_pruned, n_idx, keep_ratio, s_idx, r_idx)
loss, acc1, acc5 = trainer.test(net_filter_pruned)
print(f"Loss: {loss:.4f}, Top-1 Acc: {acc1*100:.2f}%, Top-5: {acc5*100:.2f}%")
print("\nTesting on test data set:")

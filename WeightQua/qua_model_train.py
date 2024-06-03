# 用于测试将只有卷积和全连接的SNN网络模型中的权重量化为4bit
# 采用的模型结构是简单的5层卷积+1层全连接
# 先对网络进行训练
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from spikingjelly.clock_driven import functional
from tqdm import tqdm
import time
from spikingjelly.datasets.n_mnist import NMNIST
from models.DVSEventNet import DVSEventNet

_seed_ = 2022
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

data_path = '/home/hp/DataSet/N-MNIST'
distributed = False
batch_size = 64
workers = 4
T = 16
train_data_loader = torch.utils.data.DataLoader(
    dataset=NMNIST(data_path, train=True, frames_number=16, data_type='frame', split_by='number'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    drop_last=True,
    pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(
    dataset=NMNIST(data_path, train=False, frames_number=16, data_type='frame', split_by='number'),
    batch_size=batch_size * 4,
    shuffle=False,
    num_workers=workers,
    drop_last=False,
    pin_memory=True)

# %%
max_epoch = 10
targetnum = 10

device = torch.device('cuda:0')
net = DVSEventNet()
net = net.to(device)

# %%
max_test_acc = 0.
maxaccepo = 0.
min_loss = 100.
minlossepo = 0.
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0 - step / max_epoch), last_epoch=-1)
for epoch in range(max_epoch):
    start_time = time.time()
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0

    for data1, label in tqdm(train_data_loader):
        # print(label)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data1 = data1.float()
        label = label.to(device)
        label_onehot = F.one_hot(label, targetnum).float()
        out_fr = net(data1)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(net)

    train_loss /= train_samples
    train_acc /= train_samples

    lr_scheduler.step()
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        matix = torch.zeros((targetnum, targetnum))
        for data1, label in tqdm(test_data_loader):
            data1 = data1.to(device)
            data1 = data1.float()
            label = label.to(device)
            label_onehot = F.one_hot(label, targetnum).float()

            out_fr = net(data1)
            loss = F.mse_loss(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

    test_loss /= test_samples
    test_acc /= test_samples

    if test_acc >= max_test_acc:
        max_test_acc = test_acc
        maxaccepo = epoch
        torch.save(net, "./model_pth/max_acc.pkl")

    if test_loss <= min_loss:
        min_loss = test_loss
        minlossepo = epoch
        torch.save(net, "./model_pth/min_loss.pkl")

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print(
        f'epoch={epoch}, train_acc={train_acc}, train_loss={train_loss}, test_acc={test_acc}, test_loss={test_loss}, max_test_acc={max_test_acc}, min_loss={min_loss}, total_time={time.time() - start_time}, LR={lr}')
    torch.save(net, "model_pth/final.pkl")

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from spikingjelly.clock_driven import neuron, surrogate
from scipy.io import savemat


class DVSEventNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda')
        self.write = False
        self.num = 0

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.ner1 = neuron.IFNode()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.ner2 = neuron.IFNode()

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.ner3 = neuron.IFNode()

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.ner4 = neuron.IFNode()

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.ner5 = neuron.IFNode()

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.ner6 = neuron.IFNode()

        self.linear1 = nn.Linear(64 * 1 * 1, 10)
        self.ner7 = neuron.IFNode()

    def forward(self, x_all):
        # x_all.shape=[bs,T,2,34,34]
        T = x_all.shape[1]
        out = torch.zeros([x_all.shape[0], T, 10]).to(self.device)

        for t in range(T):
            self.num += 1
            x = x_all[:, t, :, :, :].squeeze(1)  # x.shape=[B,2,34,34]
            x = self.conv1(x)  # x.shape=[B,16,17,17]
            x = self.ner1(x)
            x = self.conv2(x)  # x.shape=[B,32,9,9]
            x = self.ner2(x)
            x = self.conv3(x)  # x.shape=[B,32,5,5]
            x = self.ner3(x)
            x = self.conv4(x)  # x.shape=[B,32,3,3]
            x = self.ner4(x)
            x = self.conv5(x)  # x.shape=[B,64,2,2]
            x = self.ner5(x)
            x = self.conv6(x)  # x.shape=[B,64,1,1]
            x = self.ner6(x)
            x = x.view(x.size(0), -1)  # x.shape=[B,64]
            x = self.linear1(x)  # x.shape=[B,10]
            x = self.ner7(x)
            out[:, t, :] = x

        out_spike = torch.sum(out, dim=1)
        out_spike = out_spike.squeeze(dim=1)
        return out_spike / T

import torch
import torch.nn as nn
from models.LIFNeuron import LIF


class Conv(nn.Module):

    def __init__(self, in_planes, planes, spike):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            spike
        )

    def forward(self, x):
        return self.conv(x)


class VGGnet(nn.Module):
    def __init__(self, num_classes):
        super(VGGnet, self).__init__()
        self.ch = 64
        self.spike_func = LIF()
        self.feature = nn.Sequential(
            Conv(2, self.ch, self.spike_func),
            Conv(self.ch, self.ch, self.spike_func),
            nn.AvgPool2d(2),

            Conv(self.ch, self.ch * 2, self.spike_func),
            Conv(self.ch * 2, self.ch * 2, self.spike_func),
            nn.AvgPool2d(2),

            Conv(self.ch * 2, self.ch * 4, self.spike_func),
            Conv(self.ch * 4, self.ch * 4, self.spike_func),
            Conv(self.ch * 4, self.ch * 4, self.spike_func),
            nn.AvgPool2d(2),

            Conv(self.ch * 4, self.ch * 8, self.spike_func),
            Conv(self.ch * 8, self.ch * 8, self.spike_func),
            Conv(self.ch * 8, self.ch * 8, self.spike_func),
            nn.AvgPool2d(2),

            Conv(self.ch * 8, self.ch * 8, self.spike_func),
            Conv(self.ch * 8, self.ch * 8, self.spike_func),
            Conv(self.ch * 8, self.ch * 8, self.spike_func),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.ch * 4 * 4 * 8, 512),
            self.spike_func,
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            self.spike_func,
            nn.Dropout(0.25),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out


def VGG16(num_classes):
    return VGGnet(num_classes)

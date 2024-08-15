import torch
import torch.nn as nn
from models.Neurons import LIF, DropBlockLIFS, TAB_Layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, modified=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.spike_func = LIF()
        # self.spike_func1 = LIF()
        # self.spike_func = TAB_Layer(planes)
        # self.spike_func1 = TAB_Layer(planes)
        self.spike_func = DropBlockLIFS()
        self.spike_func1 = DropBlockLIFS()

        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.spike_func(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        if self.modified:
            out = self.spike_func1(out)
            out += self.shortcut(x)  # Equivalent to union of all spikes  先转换为脉冲再相加，即DS-ResNet
        else:
            out += self.shortcut(x)
            out = self.spike_func1(out)
        return out


class BLock_Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_block, downsample, modified):
        super(BLock_Layer, self).__init__()
        layers = []
        if downsample:
            layers.append(block(in_planes=in_planes, planes=planes, stride=2, modified=modified))
        else:
            layers.append(block(in_planes=in_planes, planes=planes, stride=1, modified=modified))
        for _ in range(1, num_block):
            layers.append(block(in_planes=planes, planes=planes, stride=1, modified=modified))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class ResNet18Two(nn.Module):
    # 网络结构来自于 原始ResNet
    def __init__(self, num_classes, useAvg):
        super(ResNet18Two, self).__init__()
        self.useAvg = useAvg
        self.avgpool_input = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = BLock_Layer(BasicBlock, 64, 64, 2, False, modified=True)
        self.layer2 = BLock_Layer(BasicBlock, 64, 128, 2, True, modified=True)
        self.layer3 = BLock_Layer(BasicBlock, 128, 256, 2, True, modified=True)
        self.layer4 = BLock_Layer(BasicBlock, 256, 512, 2, True, modified=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        # self.spike_func = LIF()
        # self.spike_func = TAB_Layer(64)
        self.spike_func = DropBlockLIFS()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if self.useAvg:
            x = self.avgpool_input(x)
        # print('-----x.shape----')
        # print(x.shape)
        out = self.spike_func(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

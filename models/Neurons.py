import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors

Vth = 1
a = 1.0
TimeStep = 5
tau = 0.5


# Vth = 0.6
# Vth2 = 1.6
# Vth3 = 2.6


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = torch.gt(input, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth) < (a / 2)) / a
        return grad_input * hu


spikefunc = SpikeFunction.apply


class SpikeFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, Vth2)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth2) < (a / 2)) / a
        return grad_input * hu


spikefunc2 = SpikeFunction2.apply


class SpikeFunction3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, Vth3)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth3) < (a / 2)) / a
        return grad_input * hu


spikefunc3 = SpikeFunction3.apply


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.ge(input, 0)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input) < (a / 2)) / a
        # hu = torch.where(abs(input) < 1/2, 1, 0)
        return grad_input * hu


step = StepFunction.apply


# dropblock
class DropBlockLIF(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DropBlockLIF, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            # 硬重置
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
            o[t * bs:(t + 1) * bs, ...] = spike_mask
        return o


# 软重置
class DropBlockLIFS(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DropBlockLIFS, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        # mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        # 生成一个与原始形状匹配的张量，但每个元素为 gamma
        bernoulli_prob = torch.full((x.shape[0], *x.shape[2:]), gamma)

        # 使用伯努利分布生成掩码
        mask = torch.bernoulli(bernoulli_prob).float()
        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]

        # print('-----------------torch.sum(block_mask==0)--------------------')
        # print(torch.sum(block_mask == 0))
        # print('-----------------(torch.sum(block_mask==0)+torch.sum(block_mask==1)--------------------')
        # print((torch.sum(block_mask == 0) + torch.sum(block_mask == 1)))
        # print('----block_mask的比例-----')
        # print(torch.sum(block_mask == 0) / (torch.sum(block_mask == 0) + torch.sum(block_mask == 1)))
        for t in range(TimeStep):
            # 软重置
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
            # print('------------------torch.sum(spike == 1)----------------')
            # print(torch.sum(spike == 1))
            # print('------------------torch.sum(spike_mask == 1)----------------')
            # print(torch.sum(spike_mask == 1))
            # print('------------------spike_mask后剩下的1的比例----------------')
            # print(torch.sum(spike_mask == 1) / torch.sum(spike == 1))
            o[t * bs:(t + 1) * bs, ...] = spike_mask
            u = u - spike
        # NSAR
        b, T, c, w, h = int(o.shape[0] / TimeStep), TimeStep, o.shape[1], o.shape[2], o.shape[3]
        output = o.view(b, T, c, w, h)

        # 提取第二个样本的所有时间步的数据
        second_sample_output = output[1]

        # 计算第二个样本每个时间步的脉冲总数
        spike_counts_per_timestep = second_sample_output.sum(dim=[1, 2, 3])  # 在c, w, h维度上求和

        # 输出第二个样本每个时间步的脉冲总数
        print("Spike counts per timestep for the second sample:", spike_counts_per_timestep)

        # 计算神经元总数
        total_neurons = c * w * h
        print("Total number of neurons:", total_neurons)

        return o


# 按照论文计算gama 使用软重置
class DBLIF(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DBLIF, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def calculate_gamma(self, x) -> float:
        invalid = (1 - self.drop_rate) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        gamma = self.calculate_gamma(x)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        block_mask = 1 - F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=(self.block_size // 2, self.block_size // 2),
        )
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spike_mask
        return o


#     硬重置
class DBLIFH(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DBLIFH, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def calculate_gamma(self, x) -> float:
        invalid = (1 - self.drop_rate) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        gamma = self.calculate_gamma(x)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        block_mask = 1 - F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=(self.block_size // 2, self.block_size // 2),
        )
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spike_mask
        return o


# 按照论文计算gama
class DBLIF1(nn.Module):
    def __init__(self, drop_rate=0.9, block_size=7):
        super(DBLIF, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def calculate_gamma(self, x) -> float:
        invalid = (1 - self.drop_rate) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        # 训练过程需要对其进行掩码，测试不需要
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            block_mask = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            if self.block_size % 2 == 0:
                # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
                block_mask = block_mask[:, :, :-1, :-1]
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            if self.training:
                spike_mask = spike * block_mask[t * bs:(t + 1) * bs, ...]
                o[t * bs:(t + 1) * bs, ...] = spike_mask
            else:
                o[t * bs:(t + 1) * bs, ...] = spike
        return o


# 使用winner-take-all方法生成mask
def reparameterization_winner_take_all(input_tensor):
    beta = 0.2
    # 计算保留的特征数 K
    P = input_tensor.size(1) * input_tensor.size(2) * input_tensor.size(3)  # 输入特征的总数量
    K = int(beta * P)  # 根据屏蔽比例计算保留的特征数

    # 将输入特征展平为一维向量，并计算最大的前 K 个值及其对应的索引
    flattened_input = input_tensor.view(input_tensor.size(0), -1)
    topk_values, topk_indices = flattened_input.topk(K, dim=1)

    # 创建输出的掩码向量，保留最大的前 K 个值对应的位置
    mask = torch.zeros_like(flattened_input)
    mask.scatter_(1, topk_indices, 1)

    # 将掩码向量恢复成输入特征的形状
    mask = mask.view(input_tensor.size())

    return mask


class WTALIF(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(WTALIF, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        block_mask = reparameterization_winner_take_all(x)
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, :, :]
            o[t * bs:(t + 1) * bs, ...] = spike_mask
        return o


class VEILIF(nn.Module):
    # VLIF神经元，基于发射的脉冲决定阈值
    def __init__(self, inchannel=128):
        super(VEILIF, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.vth = nn.Parameter(torch.zeros(inchannel))

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        vth = torch.sigmoid(self.vth).view(1, self.vth.shape[0], 1, 1).expand_as(u)  # 每个channel都有一个阈值
        for t in range(TimeStep):
            # x.shape=torch.size([128*TimeStep,64,32,32])
            # u.shape=torch.size([128,64,32,32])
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
            # vth.shape=torch.Size([128, 64, 32, 32]) 与u一致的shape
            spike = step(u - vth)
            o[t * bs:(t + 1) * bs, ...] = spike
            # 软重置
            u = u - spike * vth

        return o

    def extra_repr(self):
        return f'VEILIF神经元，tau={tau}, 第一个时刻每个channel一个自适应阈值'


# 卷积学习mask
class CMLIF(nn.Module):
    def __init__(self, inchannel, dropout_p=0.3, threshold=0.5):
        super(CMLIF, self).__init__()
        self.dropout_p = dropout_p
        self.mask_conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1)
        self.threshold = threshold

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        # 生成自适应掩码
        mask = torch.sigmoid(self.mask_conv(torch.ones((bs,) + x.shape[1:], device=x.device)))

        # 二值化掩码
        mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike_output = spikefunc(u)
            # 使用自适应掩码
            o[t * bs:(t + 1) * bs, ...] = spike_output * mask

        return o


# 上一个时刻的spike来学习mask+自适应的阈值
class CMSVLIF(nn.Module):
    def __init__(self, inchannel, threshold=0.5):
        super(CMSVLIF, self).__init__()
        self.mask_conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1)
        self.threshold = threshold
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.vth = nn.Parameter(torch.zeros(inchannel))

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        # 生成阈值
        vth = torch.sigmoid(self.vth).view(1, self.vth.shape[0], 1, 1).expand_as(u)

        prev_spike_output = torch.zeros((bs,) + x.shape[1:], device=x.device)  # 存储上一个时刻的spike_output

        for t in range(TimeStep):
            if t != 0:
                # 生成自适应掩码，依赖于上一个时刻的spike_output
                mask = torch.sigmoid(self.mask_conv(prev_spike_output))

                # 二值化掩码
                mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))
                vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
                spike = step(u - vth)
                o[t * bs:(t + 1) * bs, ...] = spike * mask
                u = u - spike * vth

                # 更新prev_spike_output为当前时刻的spike_output，用于下一个时刻的mask生成
                prev_spike_output = spike * mask
            else:
                vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
                spike = step(u - vth)
                o[t * bs:(t + 1) * bs, ...] = spike
                u = u - spike * vth
                prev_spike_output = spike

        return o


# 通过上一个时刻t输出的spike来进行学习mask
class CMSLIF(nn.Module):
    def __init__(self, inchannel, threshold=0.5):
        super(CMSLIF, self).__init__()
        self.mask_conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1)
        self.threshold = threshold

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        prev_spike_output = torch.zeros((bs,) + x.shape[1:], device=x.device)  # 存储上一个时刻的spike_output

        for t in range(TimeStep):
            if t != 0:
                # 生成自适应掩码，依赖于上一个时刻的spike_output
                mask = torch.sigmoid(self.mask_conv(prev_spike_output))

                # 二值化掩码
                mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

                u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
                spike_output = spikefunc(u)
                # 使用自适应掩码
                o[t * bs:(t + 1) * bs, ...] = spike_output * mask

                # 更新prev_spike_output为当前时刻的spike_output，用于下一个时刻的mask生成
                prev_spike_output = spike_output
                # prev_spike_output = o[t * bs:(t + 1) * bs, ...]
            else:
                u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
                spike_output = spikefunc(u)
                # 使用自适应掩码
                o[t * bs:(t + 1) * bs, ...] = spike_output
                prev_spike_output = spike_output

        return o


# 注意力机制去学习mask
class AttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=4):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class SAMLIF(nn.Module):
    def __init__(self, inchannel, threshold=0.5):
        super(SAMLIF, self).__init__()
        self.threshold = threshold
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.attention = AttentionModule(input_dim=inchannel)  # 初始化时不设置input_dim

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        abs_x = torch.abs(x)
        globalAvg = self.globalAvg(abs_x).view(x.shape[0], -1)
        mask = self.attention(globalAvg).unsqueeze(-1).unsqueeze(-1)
        print("------x.shape------")
        print(x.shape)
        print("------------mask.shape-------")
        print(mask.shape)
        print('-----------mask------')
        print(mask)
        # 二值化掩码
        mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))
        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike_output = spikefunc(u)
            # 使用自适应掩码
            o[t * bs:(t + 1) * bs, ...] = spike_output * mask[t * bs:(t + 1) * bs, ...]
            print("mask[t * bs:(t + 1) * bs, ...]:")
            print(mask[t * bs:(t + 1) * bs, ...])
            print("----------o[t * bs:(t + 1) * bs, ...]-------------------")
            print(o[t * bs:(t + 1) * bs, ...])
            return;

        return o


# 改进的mask
class CMILIF(nn.Module):
    def __init__(self, inchannel, threshold=0.5):
        super(CMILIF, self).__init__()
        self.mask_conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1)
        self.threshold = threshold

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        # 生成自适应掩码，同时依赖于输入x
        mask = torch.sigmoid(self.mask_conv(x))

        # 二值化掩码
        mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike_output = spikefunc(u)
            # 使用自适应掩码
            o[t * bs:(t + 1) * bs, ...] = spike_output * mask[t * bs:(t + 1) * bs, ...]

        return o


# 改进的mask +自适应的阈值
class CMIVLIF(nn.Module):
    def __init__(self, inchannel, threshold=0.5):
        super(CMIVLIF, self).__init__()
        self.mask_conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1)
        self.threshold = threshold
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.vth = nn.Parameter(torch.zeros(inchannel))

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        # 生成阈值
        vth = torch.sigmoid(self.vth).view(1, self.vth.shape[0], 1, 1).expand_as(u)

        # 生成自适应掩码，同时依赖于输入x
        mask = torch.sigmoid(self.mask_conv(x))

        # 二值化掩码
        mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        for t in range(TimeStep):
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
            spike = step(u - vth)
            o[t * bs:(t + 1) * bs, ...] = spike * mask[t * bs:(t + 1) * bs, ...]
            u = u - spike * vth

        return o


# 自适应阈值+卷积学习的mask
class CMVLIF(nn.Module):
    def __init__(self, inchannel, threshold=0.5):
        super(CMVLIF, self).__init__()
        self.threshold = threshold
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.vth = nn.Parameter(torch.zeros(inchannel))
        self.mask_conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1)

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        # 生成阈值
        vth = torch.sigmoid(self.vth).view(1, self.vth.shape[0], 1, 1).expand_as(u)

        # 生成自适应掩码
        mask = torch.sigmoid(self.mask_conv(torch.ones((bs,) + x.shape[1:], device=x.device)))
        # 二值化掩码
        mask = torch.where(mask > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        for t in range(TimeStep):
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
            spike = step(u - vth)
            o[t * bs:(t + 1) * bs, ...] = spike
            u = u - spike * vth * mask  # 使用自适应阈值和自适应掩码

        return o

    def extra_repr(self):
        return f'CMVLIF神经元，tau={tau}, 第一个时刻每个channel一个自适应阈值和自适应掩码'


# 自适应阈值+mask
class MVLIF(nn.Module):
    # VLIF神经元，基于发射的脉冲决定阈值
    def __init__(self, inchannel=128, dropout_p=0.3):
        super(MVLIF, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.vth = nn.Parameter(torch.zeros(inchannel))
        self.dropout_p = dropout_p

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        dropout_mask = torch.bernoulli(torch.full((bs,) + x.shape[1:], 1 - self.dropout_p, device=x.device))
        vth = torch.sigmoid(self.vth).view(1, self.vth.shape[0], 1, 1).expand_as(u)  # 每个channel都有一个阈值
        for t in range(TimeStep):
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
            spike = step(u - vth)
            o[t * bs:(t + 1) * bs, ...] = spike * dropout_mask
            u = u - spike * vth

        return o

    def extra_repr(self):
        return f'VEILIF神经元，tau={tau}, 第一个时刻每个channel一个自适应阈值'


# 自适应阈值+可学习的dropout_p
class MVALIF(nn.Module):
    # VLIF神经元，基于发射的脉冲决定阈值
    def __init__(self, inchannel=128, dropout_p=0.3):
        super(MVALIF, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.vth = nn.Parameter(torch.zeros(inchannel))
        self.dropout_p = nn.Parameter(torch.tensor(dropout_p))

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        dropout_mask = torch.bernoulli(
            torch.full((bs,) + tuple(x.shape[1:]), 1 - self.dropout_p.item(), device=x.device))

        vth = torch.sigmoid(self.vth).view(1, self.vth.shape[0], 1, 1).expand_as(u)  # 每个channel都有一个阈值
        for t in range(TimeStep):
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
            spike = step(u - vth)
            o[t * bs:(t + 1) * bs, ...] = spike * dropout_mask
            u = u - spike * vth

        return o


# 可学习的dropout_p
class MALIF(nn.Module):
    def __init__(self, dropout_p=0.3):
        super(MALIF, self).__init__()
        self.dropout_p = nn.Parameter(torch.tensor(dropout_p))

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        dropout_mask = torch.bernoulli(
            torch.full((bs,) + tuple(x.shape[1:]), 1 - self.dropout_p.item(), device=x.device))

        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike_output = spikefunc(u)
            o[t * bs:(t + 1) * bs, ...] = spike_output * dropout_mask  # 应用dropout掩码

        return o


# 使用决策网络选择阈值
class PVLIF(nn.Module):
    # PVLIF神经元，基于决策网络根据每个样本选择对应的阈值
    def __init__(self, max_threshold):
        super(PVLIF, self).__init__()
        self.vth = max_threshold

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        # print('------self.vth---')
        # print(self.vth)
        for t in range(TimeStep):
            vth_slice = self.vth[t * bs:(t + 1) * bs]  # 从self.vth中取出对应阈值范围
            vth_slice = vth_slice.view(bs, 1, 1, 1)  # 调整阈值范围的shape与u相匹配
            vth_slice = vth_slice.to(u.device)
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            spike = step(u - vth_slice)  # 使用self.vth作为阈值
            o[t * bs:(t + 1) * bs, ...] = spike
            # 软重置
            u = u - spike * vth_slice
        # 输出每个神经元在每个时间步长的脉冲总数和神经元数目的总数
        b, T, c, w, h = int(o.shape[0] / TimeStep), TimeStep, o.shape[1], o.shape[2], o.shape[3]
        output = o.view(b, T, c, w, h)

        # 提取第二个样本的所有时间步的数据
        second_sample_output = output[1]

        # 计算第二个样本每个时间步的脉冲总数
        spike_counts_per_timestep = second_sample_output.sum(dim=[1, 2, 3])  # 在c, w, h维度上求和

        # 输出第二个样本每个时间步的脉冲总数
        print("Spike counts per timestep for the second sample:", spike_counts_per_timestep)

        # 计算神经元总数
        total_neurons = c * w * h
        print("Total number of neurons:", total_neurons)
        return o

    def extra_repr(self):
        return f'PVLIF神经元，tau={tau}, 每个样本根据策略网络得到一个自适应阈值'


# 基本的LIF
class LIF(nn.Module):
    def __init__(self):
        super(LIF, self).__init__()

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        last_timestep = TimeStep
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        for t in range(last_timestep):
            # 软重置：发放脉冲的话就减去阈值
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spikefunc(u)
            u = u - spikefunc(u)
            # max_num = 0
            # max_i = 0
            # max_j = 0
            # for i in range(128):
            #     for j in range(64):
            #         spikes_t = torch.sum(spikefunc(u)[i][j] == 1)
            #         if spikes_t > max_num:
            #             max_num = spikes_t
            #             max_i = i
            #             max_j = j
            # print(max_i, max_j)
            # #     可视化一个脉冲的发放
            # # 创建一个自定义的颜色映射
            # data = spikefunc(u)[98][47].cpu().numpy()
            # cmap = mcolors.ListedColormap(['#EAF0FC', '#FF5196'])
            #
            # plt.figure(figsize=(6, 6))
            # plt.pcolormesh(data, cmap=cmap, edgecolors='white', linewidth=2)  # 使用 pcolormesh 并设置边框颜色和宽度
            # plt.title('Pulse Matrix Visualization with Grids')
            # plt.axis('on')  # 显示坐标轴
            # plt.gca().invert_yaxis()  # 调整Y轴，使得0在上方
            # plt.show()
            # return;
        # # 重新整理数据的形状
        # c, w, h = o.shape[1], o.shape[2], o.shape[3]
        # input_data_reshaped = o.view(bs, TimeStep, c, w, h)
        # visualization_data_cpu = input_data_reshaped.cpu()
        #
        # # 对时间步长 T 和通道 c 进行平均
        # spike_rates = visualization_data_cpu.mean(dim=1).mean(dim=1)  # 最终形状为 [bs, w, h]
        #
        # # 选择要可视化的批次
        # batch_index = 1
        # spike_rate = spike_rates[batch_index].numpy()  # 将 tensor 转换为 NumPy 数组用于可视化
        #
        # fig, ax = plt.subplots()
        # cax = ax.imshow(spike_rate, cmap='coolwarm', interpolation='nearest',vmin=0, vmax=0.15)
        # fig.colorbar(cax)
        #
        # # 设置刻度标签的字体大小
        # tick_font_size = 10  # 可以调整这个值以适应你的具体需求
        # plt.xticks(fontsize=tick_font_size)
        # plt.yticks(fontsize=tick_font_size)
        #
        # # 如果标签仍然重叠，可以考虑只显示每第二个刻度标签
        # ax.set_xticks(np.arange(spike_rate.shape[1])[::2])  # 每隔一个刻度显示一个标签
        # ax.set_yticks(np.arange(spike_rate.shape[0]))
        #
        # # 在热力图的每个单元格中添加数值标签
        # font_size = min(fig.get_size_inches()) * 6 / max(spike_rate.shape)
        # for i in range(spike_rate.shape[0]):
        #     for j in range(spike_rate.shape[1]):
        #         text_color = 'white' if spike_rate[i, j] < 0.15 else 'black'
        #         ax.text(j, i, f'{spike_rate[i, j]:.2f}', ha='center', va='center', color=text_color,
        #                 fontsize=font_size)
        #
        # ax.set_title('Conv-3')
        # ax.set_xlabel('Column Index')
        # ax.set_ylabel('Row Index')
        # plt.show()
        # return;
        # return;
        # visualization_data = o[4 * 128 + 1]
        # visualization_data_cpu = visualization_data.cpu()
        # w = visualization_data_cpu.shape[1]
        # h = visualization_data_cpu.shape[2]
        #
        # rgb_image = np.zeros((w, h, 3))
        #
        # # 将两个通道映射到RGB图像的前两个通道
        # rgb_image[:, :, 0] = visualization_data_cpu[0, :, :]  # Red
        # rgb_image[:, :, 1] = visualization_data_cpu[1, :, :]  # Green
        # rgb_image[:, :, 2] = visualization_data_cpu[2, :, :]  # Green
        #
        # # 显示RGB图像
        # plt.imshow(rgb_image)
        # plt.title('LIF after RGB Image')
        # plt.axis('off')
        # plt.show()
        # return 0;
        # c, w, h = o.shape[1], o.shape[2], o.shape[3]
        # tmp = o.view(bs, TimeStep, c, w, h)
        # 计算每个神经元在所有时间步上的平均发放频率
        # spike_rates = tmp.mean(dim=1)  # 在时间维度上取平均，形状变为 [bs, c, w, h]
        # 选择要可视化的批次和通道
        # batch_index = 1
        # channel_index = 0
        # spike_rates = tmp.cpu().float()  # 假设原始数据是0和1，转换为浮点数，然后转换为布尔型
        # neuron_spike_data = spike_rates[batch_index, :, channel_index, w // 2, h // 2].bool()
        # print("----------neuron_spike_data---------")
        # print(neuron_spike_data)
        # # 创建时间轴
        # time_steps = torch.arange(TimeStep)
        #
        # # 绘制图形
        # plt.figure(figsize=(10, 2))
        # plt.eventplot(time_steps[neuron_spike_data], orientation='horizontal', colors='black', linelengths=1)
        # plt.xlabel('Time Step')
        # plt.yticks([])
        # plt.title('Spike Raster Plot of a Single Neuron Over Time')
        # plt.tight_layout()
        # plt.show()
        # return;
        # 重塑输出数据为 [b, T, c, w, h]
        # 输出每个神经元在每个时间步长的脉冲总数和神经元数目的总数
        b, T, c, w, h = int(o.shape[0] / TimeStep), TimeStep, o.shape[1], o.shape[2], o.shape[3]
        output = o.view(b, T, c, w, h)

        # 提取第二个样本的所有时间步的数据
        second_sample_output = output[1]

        # 计算第二个样本每个时间步的脉冲总数
        spike_counts_per_timestep = second_sample_output.sum(dim=[1, 2, 3])  # 在c, w, h维度上求和

        # 输出第二个样本每个时间步的脉冲总数
        print("Spike counts per timestep for the second sample:", spike_counts_per_timestep)

        # 计算神经元总数
        total_neurons = c * w * h
        print("Total number of neurons:", total_neurons)

        return o

    def extra_repr(self):
        return f'LIF神经元，自适应batchsize和timestep，tau={tau}'


# 使用决策网络选择阈值+BlockMask
# class PVBMLIF(nn.Module):
#     # PVBMLIF神经元，基于决策网络根据每个样本选择对应的阈值,并使用block mask的方法来进行mask脉冲输出
#     def __init__(self, max_threshold, drop_rate=0.1, block_size=7):
#         super(PVBMLIF, self).__init__()
#         self.vth = max_threshold
#         self.drop_rate = drop_rate
#         self.block_size = block_size
#
#     def forward(self, x):
#         gamma = self.drop_rate / (self.block_size ** 2)
#         # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
#         mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
#
#         mask = mask.to(x.device)
#
#         # compute block mask
#         block_mask = F.max_pool2d(input=mask[:, None, :, :],
#                                   kernel_size=(self.block_size,
#                                                self.block_size),
#                                   stride=(1, 1),
#                                   padding=self.block_size // 2)
#         if self.block_size % 2 == 0:
#             # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
#             block_mask = block_mask[:, :, :-1, :-1]
#         block_mask = 1 - block_mask.squeeze(1)
#         bs = int(x.shape[0] / TimeStep)
#         u = torch.zeros((bs,) + x.shape[1:], device=x.device)
#         o = torch.zeros(x.shape, device=x.device)
#         for t in range(TimeStep):
#             vth_slice = self.vth[t * bs:(t + 1) * bs]  # 从self.vth中取出对应阈值范围
#             vth_slice = vth_slice.view(bs, 1, 1, 1)  # 调整阈值范围的shape与u相匹配
#             vth_slice = vth_slice.to(u.device)
#             u = tau * u + x[t * bs:(t + 1) * bs, ...]
#             spike = step(u - vth_slice)  # 使用self.vth作为阈值
#             # 进行mask
#             spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
#             o[t * bs:(t + 1) * bs, ...] = spike_mask
#             u = u - spike * vth_slice
#
#         return o
#
#     def extra_repr(self):
#         return f'PVBMLIF神经元，tau={tau}, 每个样本根据策略网络得到一个自适应阈值，并使用block mask的方式来对脉冲进行稀疏化'


# 使用决策网络选择阈值
class PVBMLIF(nn.Module):
    # PVBMLIF神经元，基于决策网络根据每个样本选择对应的阈值,并使用block mask的方法来进行mask脉冲输出
    def __init__(self, max_threshold, drop_rate=0.1, block_size=7):
        super(PVBMLIF, self).__init__()

        self.vth = max_threshold
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        # bs = int(x.shape[0] / 8)
        # c, w, h = x.shape[1], x.shape[2], x.shape[3]
        # input_data_reshaped = x.view(bs, TimeStep, c, w, h)
        # visualization_data_cpu = input_data_reshaped.cpu()
        #
        # # 对时间步长 T 和通道 c 进行平均
        # spike_rates = visualization_data_cpu.mean(dim=1).mean(dim=1)  # 最终形状为 [bs, w, h]
        #
        # # 选择要可视化的批次
        # batch_index = 1
        # spike_rate = spike_rates[batch_index].numpy()  # 将 tensor 转换为 NumPy 数组用于可视化
        #
        # fig, ax = plt.subplots()
        # cax = ax.imshow(spike_rate, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=0.3)
        # fig.colorbar(cax)
        #
        # # 设置刻度标签的字体大小
        # tick_font_size = 10  # 可以调整这个值以适应你的具体需求
        # plt.xticks(fontsize=tick_font_size)
        # plt.yticks(fontsize=tick_font_size)
        #
        # # 如果标签仍然重叠，可以考虑只显示每第二个刻度标签
        # ax.set_xticks(np.arange(spike_rate.shape[1])[::2])  # 每隔一个刻度显示一个标签
        # ax.set_yticks(np.arange(spike_rate.shape[0]))
        #
        # # 在热力图的每个单元格中添加数值标签
        # font_size = min(fig.get_size_inches()) * 6 / max(spike_rate.shape)
        # for i in range(spike_rate.shape[0]):
        #     for j in range(spike_rate.shape[1]):
        #         text_color = 'white' if spike_rate[i, j] < 0.15 else 'black'
        #         ax.text(j, i, f'{spike_rate[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=font_size)
        #
        # ax.set_title('Conv-3')
        # ax.set_xlabel('Column Index')
        # ax.set_ylabel('Row Index')
        # plt.show()
        # return;

        # visualization_data = x[4 * 128 + 1]
        # visualization_data_cpu = visualization_data.cpu()
        # w = visualization_data_cpu.shape[1]
        # h = visualization_data_cpu.shape[2]
        #
        # rgb_image = np.zeros((w, h, 3))
        #
        # # 将两个通道映射到RGB图像的前两个通道
        # rgb_image[:, :, 0] = visualization_data_cpu[0, :, :]  # Red
        # rgb_image[:, :, 1] = visualization_data_cpu[1, :, :]  # Green
        #
        # # 显示RGB图像
        # plt.imshow(rgb_image)
        # plt.title('RGB1 Image')
        # plt.axis('off')
        # plt.show()
        # return ;
        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        # mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        # 生成一个与原始形状匹配的张量，但每个元素为 gamma
        bernoulli_prob = torch.full((x.shape[0], *x.shape[2:]), gamma)

        # 使用伯努利分布生成掩码
        mask = torch.bernoulli(bernoulli_prob).float()
        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        # o_nomask = torch.zeros(x.shape, device=x.device)
        # print('-----self.vth---')
        # print(self.vth)
        for t in range(TimeStep):
            vth_slice = self.vth[t * bs:(t + 1) * bs]  # 从self.vth中取出对应阈值范围
            vth_slice = vth_slice.view(bs, 1, 1, 1)  # 调整阈值范围的shape与u相匹配
            vth_slice = vth_slice.to(u.device)
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            spike = step(u - vth_slice)  # 使用self.vth作为阈值
            # spike_sum = torch.sum(spike == 1)
            # 进行mask
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
            # spike_mask_sum = torch.sum(spike_mask == 1)

            # print("----------------------spike_sum-------------------")
            # print(spike_sum)
            # print("----------------------spike_mask_sum-------------------")
            # print(spike_mask_sum)
            # print('------------------------mask的比例--------------')
            # print((spike_sum - spike_mask_sum) / spike_sum)
            o[t * bs:(t + 1) * bs, ...] = spike_mask
            # o_nomask[t * bs:(t + 1) * bs, ...] = spike
            # 软重置
            u = u - spike * vth_slice
            # tmp = block_mask[t * bs:(t + 1) * bs, None, :, :]
            # print(tmp.shape)
            # # data = block_mask[t * bs:(t + 1) * bs, None, :, :][6][0].cpu().numpy()
            # data = spikefunc(u)[3][0].cpu().numpy()
            # cmap = mcolors.ListedColormap(['#EAF0FC', '#FF5196'])
            # # cmap = mcolors.ListedColormap(['black', '#EAF0FC'])
            #
            # plt.figure(figsize=(6, 6))
            # plt.pcolormesh(data, cmap=cmap, edgecolors='white', linewidth=2)  # 使用 pcolormesh 并设置边框颜色和宽度
            # plt.title('Pulse Matrix Visualization with Grids')
            # plt.axis('on')  # 显示坐标轴
            # plt.gca().invert_yaxis()  # 调整Y轴，使得0在上方
            # plt.show()
            # return;
        # # 对TC维度进行平均，再对平均脉冲发放率进行可视化
        # # 重新整理数据的形状
        # c, w, h = o.shape[1], o.shape[2], o.shape[3]
        # input_data_reshaped = o.view(bs, TimeStep, c, w, h)
        # visualization_data_cpu = input_data_reshaped.cpu()
        #
        # # 对时间步长 T 和通道 c 进行平均
        # spike_rates = visualization_data_cpu.mean(dim=1).mean(dim=1)  # 最终形状为 [bs, w, h]
        #
        # # 选择要可视化的批次
        # batch_index = 1
        # spike_rate = spike_rates[batch_index].numpy()  # 将 tensor 转换为 NumPy 数组用于可视化
        #
        # fig, ax = plt.subplots()
        # cax = ax.imshow(spike_rate, cmap='coolwarm', interpolation='nearest',vmin=0, vmax=0.15)
        # fig.colorbar(cax)
        #
        # # 设置刻度标签的字体大小
        # tick_font_size = 10  # 可以调整这个值以适应你的具体需求
        # plt.xticks(fontsize=tick_font_size)
        # plt.yticks(fontsize=tick_font_size)
        #
        # # 如果标签仍然重叠，可以考虑只显示每第二个刻度标签
        # ax.set_xticks(np.arange(spike_rate.shape[1])[::2])  # 每隔一个刻度显示一个标签
        # ax.set_yticks(np.arange(spike_rate.shape[0]))
        #
        # # 在热力图的每个单元格中添加数值标签
        # font_size = min(fig.get_size_inches()) * 6 / max(spike_rate.shape)
        # for i in range(spike_rate.shape[0]):
        #     for j in range(spike_rate.shape[1]):
        #         text_color = 'white' if spike_rate[i, j] < 0.15 else 'black'
        #         ax.text(j, i, f'{spike_rate[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=font_size)
        #
        # # ax.set_title('Conv-3')
        # ax.set_xlabel('Column Index')
        # ax.set_ylabel('Row Index')
        # plt.show()
        # return;
        # visualization_data = o[4 * 128 + 1]
        # visualization_data_cpu = visualization_data.cpu()
        # w = visualization_data_cpu.shape[1]
        # h = visualization_data_cpu.shape[2]
        #
        # rgb_image = np.zeros((w, h, 3))
        #
        # # 将两个通道映射到RGB图像的前两个通道
        # rgb_image[:, :, 0] = visualization_data_cpu[0, :, :]  # Red
        # rgb_image[:, :, 1] = visualization_data_cpu[1, :, :]  # Green
        # rgb_image[:, :, 2] = visualization_data_cpu[2, :, :]  # Green
        #
        # # # 标准化图像数据到[0, 1]范围以供显示
        # # rgb_image -= rgb_image.min()
        # # rgb_image /= rgb_image.max()
        #
        # # Converting tensor to numpy array for visualization
        # # rgb_image_np = rgb_image.numpy()
        #
        # # Use matplotlib to display the image
        # plt.imshow(rgb_image)
        # plt.title('RGB Image from Averaged Channels')
        # plt.axis('off')  # Turn off axis labels
        # plt.show()
        # return ;
        # visualization_data = o_nomask[4 * 128 + 1]
        # visualization_data_cpu = visualization_data.cpu()
        # w = visualization_data_cpu.shape[1]
        # h = visualization_data_cpu.shape[2]
        #
        # rgb_image = torch.zeros((32, 32, 3), dtype=torch.float32)
        #
        # # 将两个通道映射到RGB图像的前两个通道
        # rgb_image[:, :, 0] = visualization_data[0:21].mean(dim=0)  # Red channel
        # rgb_image[:, :, 1] = visualization_data[21:42].mean(dim=0)  # Green channel
        # rgb_image[:, :, 2] = visualization_data[42:64].mean(dim=0)  # Blue channel
        #
        # # 标准化图像数据到[0, 1]范围以供显示
        # rgb_image -= rgb_image.min()
        # rgb_image /= rgb_image.max()
        #
        # # Converting tensor to numpy array for visualization
        # rgb_image_np = rgb_image.numpy()
        #
        # # Use matplotlib to display the image
        # plt.imshow(rgb_image_np)
        # plt.title('RGB Image from Averaged Channels')
        # plt.axis('off')  # Turn off axis labels
        # plt.show()
        # return;
        # c, w, h = o.shape[1], o.shape[2], o.shape[3]
        # tmp = o.view(bs, TimeStep, c, w, h)
        # # 计算每个神经元在所有时间步上的平均发放频率
        # spike_rates = tmp.mean(dim=1)  # 在时间维度上取平均，形状变为 [bs, c, w, h]
        # # 选择要可视化的批次和通道
        # batch_index = 1
        # channel_index = 0
        # spike_rates = spike_rates.cpu()
        # spike_rate = spike_rates[batch_index, channel_index].numpy()  # 转换为 NumPy 数组用于可视化
        #
        # # 绘制热力图
        # plt.figure(figsize=(8, 6))
        # plt.imshow(spike_rate, cmap='hot', interpolation='nearest')
        # plt.colorbar(label='Average Firing Rate')
        # plt.title(f'Heatmap of Firing Rates for Batch {batch_index} and Channel {channel_index}')
        # plt.xlabel('Width')
        # plt.ylabel('Height')
        # plt.show()
        # return;
        # NSAR
        # b, T, c, w, h = int(o.shape[0] / TimeStep), TimeStep, o.shape[1], o.shape[2], o.shape[3]
        # output = o.view(b, T, c, w, h)
        #
        # # 提取第二个样本的所有时间步的数据
        # second_sample_output = output[1]
        #
        # # 计算第二个样本每个时间步的脉冲总数
        # spike_counts_per_timestep = second_sample_output.sum(dim=[1, 2, 3])  # 在c, w, h维度上求和
        #
        # # 输出第二个样本每个时间步的脉冲总数
        # print("Spike counts per timestep for the second sample:", spike_counts_per_timestep)
        #
        # # 计算神经元总数
        # total_neurons = c * w * h
        # print("Total number of neurons:", total_neurons)

        return o

    def extra_repr(self):
        return f'PVBMLIF神经元，tau={tau}, 每个样本根据策略网络得到一个自适应阈值，并使用block mask的方式来对脉冲进行稀疏化'


# 使用决策网络选择阈值+通过计算图像的显著性来进行选择哪些patch被屏蔽(硬重置）
class PVBMLIFH(nn.Module):
    # PVBMLIFH神经元，基于决策网络根据每个样本选择对应的阈值,并使用block mask的方法来进行mask脉冲输出
    def __init__(self, max_threshold, drop_rate=0.1, block_size=7):
        super(PVBMLIFH, self).__init__()
        self.vth = max_threshold
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(TimeStep):
            vth_slice = self.vth[t * bs:(t + 1) * bs]  # 从self.vth中取出对应阈值范围
            vth_slice = vth_slice.view(bs, 1, 1, 1)  # 调整阈值范围的shape与u相匹配
            vth_slice = vth_slice.to(u.device)
            # 硬重置
            u = tau * u * (1 - step(u - vth_slice)) + x[t * bs:(t + 1) * bs, ...]
            spike = step(u - vth_slice)  # 使用self.vth作为阈值
            # spike_sum = torch.sum(spike == 1)
            # 进行mask
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
            # spike_mask_sum = torch.sum(spike_mask == 1)

            # print("----------------------spike_sum-------------------")
            # print(spike_sum)
            # print("----------------------spike_mask_sum-------------------")
            # print(spike_mask_sum)
            # print('------------------------mask的比例--------------')
            # print((spike_sum - spike_mask_sum) / spike_sum)
            o[t * bs:(t + 1) * bs, ...] = spike_mask
            # # 软重置
            # u = u - spike * vth_slice

        return o

    def extra_repr(self):
        return f'PVBMLIF神经元，tau={tau}, 每个样本根据策略网络得到一个自适应阈值，并使用block mask的方式来对脉冲进行稀疏化'


# 对脉冲的平均输出进行mask+基于显著性去判断哪些地方应该mask
class AVGMaskLIF(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7, inchannel=128):
        super(AVGMaskLIF, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        # gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        # mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            # 硬重置
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            o[t * bs:(t + 1) * bs, ...] = spike
        # 计算TimeStep的平均脉冲发射率
        # 重新整形数据为 [bs, t, c, w, h]
        data_reshaped = o.view(bs, TimeStep, *x.shape[1:])

        # 沿着时间维度 t 计算平均值
        avg_spike = data_reshaped.mean(dim=1)
        avg_spike = avg_spike.float()
        # 将avg_spike的数据类型转换为float32
        # 根据平均脉冲发射率去计算显著性？
        # 先要使得shape变为[bs,3,w,h]
        # 使用卷积层将 c 个通道转换为 3 个通道

        # 应用卷积层
        data_tmp = self.conv(avg_spike)
        # 转换为 [w, h, c] 并转换为0-255的uint8类型
        # 转换为 NumPy 数组前确保调用 .cpu() 将张量移动到 CPU
        # 转换为 [bs, w, h, c] 并转换为0-255的uint8类型
        # 使用.detach()来确保我们不在计算图中
        image_numpy = data_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
        image_numpy = (image_numpy * 255).astype(np.uint8)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        masks = []
        for image_cv in image_numpy:
            # 计算显著性图
            (success, saliency_map) = saliency.computeSaliency(image_cv)
            # 根据阈值生成mask
            mask = (saliency_map > 0.5).astype(np.uint8)
            masks.append(mask)
        # 如果需要将masks转换为Tensor并移动到原始设备 shape为[bs,w,h]
        masks_tensor = torch.tensor(masks, dtype=torch.uint8).to(x.device)  # 将Numpy数组转换为Tensor并移至设备
        # 添加一个通道维度，使形状从 [bs, w, h] 变为 [bs, 1, w, h]
        masks_tensor = masks_tensor.unsqueeze(1)
        masks_tensor = masks_tensor.float()
        # compute block mask
        block_mask = F.max_pool2d(input=masks_tensor,
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        # block_mask = 1 - block_mask

        for t in range(TimeStep):
            # 再将学习到的mask应用到每个时间步长的脉冲输出中
            o[t * bs:(t + 1) * bs, ...] = o[t * bs:(t + 1) * bs, ...] * block_mask
        # 有些时候mask的比例为0，有些时候mask的比例高达98%
        return o


# 对脉冲的平均输出进行mask+基于一个简单的卷积网络
class AVGConMLIF(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7, inchannel=128):
        super(AVGConMLIF, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        # gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        # mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            # 硬重置
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike = spikefunc(u)
            o[t * bs:(t + 1) * bs, ...] = spike
        # 计算TimeStep的平均脉冲发射率
        # 重新整形数据为 [bs, t, c, w, h]
        data_reshaped = o.view(bs, TimeStep, *x.shape[1:])

        # 沿着时间维度 t 计算平均值
        avg_spike = data_reshaped.mean(dim=1)
        avg_spike = avg_spike.float()
        mask_tensor = torch.sigmoid(self.conv(avg_spike))
        mask_tensor = torch.where(mask_tensor > 0.51, torch.ones_like(mask_tensor), torch.zeros_like(mask_tensor))

        # compute block mask
        block_mask = F.max_pool2d(input=mask_tensor,
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        # block_mask = 1 - block_mask
        # 计算一下mask掉的脉冲的比例
        # before_sum = torch.sum(o == 1)
        for t in range(TimeStep):
            # 再将学习到的mask应用到每个时间步长的脉冲输出中
            o[t * bs:(t + 1) * bs, ...] = o[t * bs:(t + 1) * bs, ...] * block_mask
        # after_sum = torch.sum(o == 1)
        # print("---------before_sum---")
        # print(before_sum)
        # print("-----after_sum---")
        # print(after_sum)
        # print('-------------mask的比例----------')
        # print((before_sum - after_sum) / before_sum)
        # 有些时候mask的比例为0，有些时候mask的比例高达98%
        return o


# 使用决策网络对每层的神经元选择阈值+通过计算图像的显著性来进行选择哪些patch被屏蔽(硬重置）
class PVBMLYLIFH(nn.Module):
    # PVBMLIFH神经元，基于决策网络根据每个样本选择对应的阈值,并使用block mask的方法来进行mask脉冲输出
    def __init__(self, max_threshold, drop_rate=0.2, block_size=7, layer=1):
        super(PVBMLYLIFH, self).__init__()
        self.vth = max_threshold
        self.drop_rate = drop_rate
        self.block_size = block_size
        self.layer = layer

    def forward(self, x):
        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        layer_vth = self.vth[:, self.layer - 1]
        # tensor_list = layer_vth.tolist()
        # # 使用Counter计算频次
        # counter = Counter(tensor_list)
        # # 打印结果
        # print(f"------------{self.layer}-----------------")
        # for element, count in counter.items():
        #     print(f"Element {element} appears {count} times")
        for t in range(TimeStep):
            vth_slice = layer_vth[t * bs:(t + 1) * bs]  # 从self.vth中取出对应阈值范围
            vth_slice = vth_slice.view(bs, 1, 1, 1)  # 调整阈值范围的shape与u相匹配
            vth_slice = vth_slice.to(u.device)
            # 硬重置
            u = tau * u * (1 - step(u - vth_slice)) + x[t * bs:(t + 1) * bs, ...]
            spike = step(u - vth_slice)  # 使用self.vth作为阈值
            # spike_sum = torch.sum(spike == 1)
            # 进行mask
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
            # spike_mask_sum = torch.sum(spike_mask == 1)
            # print("----------------------spike_sum-------------------")
            # print(spike_sum)
            # print("----------------------spike_mask_sum-------------------")
            # print(spike_mask_sum)
            # print('------------------------mask的比例--------------')
            # print((spike_sum - spike_mask_sum) / spike_sum)
            o[t * bs:(t + 1) * bs, ...] = spike_mask
            # # 软重置
            # u = u - spike * vth_slice

        return o

    def extra_repr(self):
        return f'PVBMLIF神经元，tau={tau}, 每个样本根据策略网络得到一个自适应阈值，并使用block mask的方式来对脉冲进行稀疏化'


# TAB
def _prob_check(p):
    p2 = p ** 2
    return p2


class TAB_Layer(nn.Module):
    def __init__(self, num_features, time_steps=TimeStep, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(TAB_Layer, self).__init__()
        self.time_steps = time_steps
        self.bn_list = nn.ModuleList(
            [nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for i in range(time_steps)])
        self.p = nn.Parameter(torch.ones(time_steps, 1, 1, 1, 1))

    def forward(self, input):
        # ## Shape of x is [NTCHW]
        # 转换输入的维度 从 [bs*T,C,H,W] 到 [bs,T,C,H,W]
        bs = int(input.shape[0] / self.time_steps)
        x = torch.zeros((bs, self.time_steps,) + input.shape[1:], device=input.device)
        for t in range(self.time_steps):
            x[:, t, :, :, :] = input[t * bs:(t + 1) * bs, ...]

        self.p = (self.p).to(x.device)
        self.bn_list = nn.ModuleList([self.bn_list[i].to(x.device) for i in range(self.time_steps)])
        pt = _prob_check(self.p)

        assert x.shape[
                   1] == self.time_steps, f"Time-steps not match input dimensions. x.shape: {x.shape}, x.shape[1]: {x.shape[1]} and self.time_steps: {self.time_steps}"
        # y_res = x.clone()
        y_res = []
        for t in range(self.time_steps):
            # xt = x[:,0:(t+1),...]
            y = x[:, 0:(t + 1), ...].clone().transpose(1, 2).contiguous()  # [N,T,C,H,W] ==> [N,C,T,H,W], put C in dim1.
            y = self.bn_list[t](y)
            y = y.contiguous().transpose(1, 2).contiguous()  # [N,C,T,H,W] to [N,T,C,H,W]
            # y_res[:,t,...] = y[:,t,...].clone()  # Only slice the t-th
            y_res.append(y[:, t, ...].clone())  # Only slice the t-th
        y_res = torch.stack(y_res, dim=1)
        # ### reshape the data and multipy the p[t] to each time-step [t]
        y = y_res.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        # y = y * self.p
        y = y * pt
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW

        out = torch.zeros(input.shape, device=input.device)
        for t in range(self.time_steps):
            out[t * bs:(t + 1) * bs, ...] = y[:, t, :, :, :]
        return out


# tdBN
class tdBatchNorm(nn.Module):
    def __init__(self, bn, alpha=1):
        super(tdBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha

    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.bn.momentum

        if self.training:
            mean = x.mean([0, 2, 3], keepdim=True)
            var = x.var([0, 2, 3], keepdim=True, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0] \
                                       + (1 - exponential_average_factor) * self.bn.running_mean
                self.bn.running_var = exponential_average_factor * var[0, :, 0, 0] * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.bn.running_var
        else:
            mean = self.bn.running_mean[None, :, None, None]
            var = self.bn.running_var[None, :, None, None]

        x = self.alpha * Vth * (x - mean) / (torch.sqrt(var) + self.bn.eps)

        if self.bn.affine:
            x = x * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]

        return x


# PLIF
class PLIF(nn.Module):
    # 基本的PLIF神经元，使用自适应膜电势时间常数
    def __init__(self):
        super(PLIF, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        # self.reset = reset

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(TimeStep):
            #     if self.reset == 'hard':
            #         u = torch.sigmoid(self.w) * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            #         o[t * bs:(t + 1) * bs, ...] = spikefunc(u)
            #     elif self.reset == 'soft':
            u = torch.sigmoid(self.w) * u + x[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spikefunc(u)
            u = u - spikefunc(u)
        return o

    # def extra_repr(self):
    #     return f'PLIF神经元，tau={torch.sigmoid(self.w)}，重置方式：soft}'


# MLF
class MLF_unit(nn.Module):
    """ MLF unit (K=3).
    MLF (K=2) can be got by commenting out the lines related to u3 and replace
    o[t*bs:(t+1)*bs, ...] = spikefunc(u) + spikefunc2(u2) + spikefunc3(u3) with
    o[t*bs:(t+1)*bs, ...] = spikefunc(u) + spikefunc2(u2).
    """

    def __init__(self):
        super(MLF_unit, self).__init__()

    def forward(self, x):
        if self.training:
            time = TimeStep
            # bs = int(x.shape[0] / TimeStep)
        else:
            time = TimeStep
        bs = int(x.shape[0] / time)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # x.shape[1:] = [32, 42, 42] [bs, 42, 42]
        u2 = torch.zeros((bs,) + x.shape[1:], device=x.device)
        u3 = torch.zeros((bs,) + x.shape[1:], device=x.device)  # comment this line if you want MLF (K=2)

        o = torch.zeros(x.shape, device=x.device)
        for t in range(time):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            u2 = tau * u2 * (1 - spikefunc2(u2)) + x[t * bs:(t + 1) * bs, ...]
            u3 = tau * u3 * (1 - spikefunc3(u3)) + x[t * bs:(t + 1) * bs,
                                                   ...]  # comment this line if you want MLF (K=2)
            o[t * bs:(t + 1) * bs, ...] = spikefunc(u) + spikefunc2(u2) + spikefunc3(
                u3)  # Equivalent to union of all spikes
        return o

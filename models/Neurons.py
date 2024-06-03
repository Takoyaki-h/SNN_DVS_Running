import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from collections import Counter
Vth = 1
a = 1.0
TimeStep = 5
tau = 0.5


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


# 按照论文计算gama
class DBLIF(nn.Module):
    def __init__(self, drop_rate=0.9, block_size=7):
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
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            vth = torch.mean(vth, dim=[2, 3], keepdim=True) * (1 + self.conv(x[t * bs:(t + 1) * bs, ...]))
            spike = step(u - vth)
            o[t * bs:(t + 1) * bs, ...] = spike
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
        for t in range(TimeStep):
            vth_slice = self.vth[t * bs:(t + 1) * bs]  # 从self.vth中取出对应阈值范围
            vth_slice = vth_slice.view(bs, 1, 1, 1)  # 调整阈值范围的shape与u相匹配
            vth_slice = vth_slice.to(u.device)
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            spike = step(u - vth_slice)  # 使用self.vth作为阈值
            o[t * bs:(t + 1) * bs, ...] = spike
            u = u - spike * vth_slice

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
        return o

    def extra_repr(self):
        return f'LIF神经元，自适应batchsize和timestep，tau={tau}'


# 使用决策网络选择阈值+BlockMask
class PVBMLIF(nn.Module):
    # PVBMLIF神经元，基于决策网络根据每个样本选择对应的阈值,并使用block mask的方法来进行mask脉冲输出
    def __init__(self, max_threshold, drop_rate=0.1, block_size=7):
        super(PVBMLIF, self).__init__()
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
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            spike = step(u - vth_slice)  # 使用self.vth作为阈值
            # 进行mask
            spike_mask = spike * block_mask[t * bs:(t + 1) * bs, None, :, :]
            o[t * bs:(t + 1) * bs, ...] = spike_mask
            u = u - spike * vth_slice

        return o

    def extra_repr(self):
        return f'PVBMLIF神经元，tau={tau}, 每个样本根据策略网络得到一个自适应阈值，并使用block mask的方式来对脉冲进行稀疏化'


# 使用决策网络选择阈值+通过计算图像的显著性来进行选择哪些patch被屏蔽
class PVBMLIF(nn.Module):
    # PVBMLIF神经元，基于决策网络根据每个样本选择对应的阈值,并使用block mask的方法来进行mask脉冲输出
    def __init__(self, max_threshold, drop_rate=0.1, block_size=7):
        super(PVBMLIF, self).__init__()

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
            # 软重置
            u = u - spike * vth_slice

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

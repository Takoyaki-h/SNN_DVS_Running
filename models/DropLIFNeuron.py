import torch
import torch.nn as nn
from math import log

Vth = 1
a = 1.0
TimeStep = 8
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


class DropLIF(nn.Module):
    def __init__(self, dropout_p=0.3):
        super(DropLIF, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        dropout_mask = torch.bernoulli(torch.full((bs,) + x.shape[1:], 1 - self.dropout_p, device=x.device))

        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            spike_output = spikefunc(u)
            o[t * bs:(t + 1) * bs, ...] = spike_output * dropout_mask  # 应用dropout掩码

        return o



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


class DDLIF(nn.Module):
    def __init__(self, dynamic=True, channel=512):
        super(DDLIF, self).__init__()
        self.dynamic = dynamic
        self.ratio = 0.3
        if self.dynamic:
            t = int(abs((log(channel, 2) + 1) / 2))
            k = t if t % 2 else t + 1
            self.channelConv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, padding=int(k / 2), stride=1,
                                         bias=False)
            self.spatialConv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2, stride=1, bias=False)
            self.temporalConv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=1, bias=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # x.shape[1:] = [32, 42, 42] = [bs, 42, 42]
        o = torch.zeros(x.shape, device=x.device)
        if self.dynamic:
            temporal_avgo = torch.zeros((bs, 1, TimeStep), device=x.device)  # [bs,1, Timestep]
            tem_attention = torch.zeros((bs * TimeStep), device=x.device)  # [bs*Timestep]

        for t in range(TimeStep):
            u = tau * u + x[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spikefunc(u)
            if self.dynamic:
                temporal_avgo[:, 0, t] = torch.mean(self.pool(spikefunc(u)).squeeze(-1).squeeze(-1), dim=1,
                                                    keepdim=False)
            u = u - spikefunc(u)
        # if self.training == False:
        #     c, h, w = o.shape[1],o.shape[2],o.shape[3]
        #     print(f'脉冲发射率：{torch.sum(o) / (bs * c * h * w)}')
        if self.dynamic:
            channel_avg = self.pool(o).squeeze(-1).transpose(-1, -2)  # [bs*Timestep,1,c]
            channel_attention = self.channelConv(channel_avg).transpose(-1, -2).unsqueeze(
                -1)  # [bs*Timestep,c,1,1] 通道注意力

            spatial_avg = torch.mean(o, dim=1, keepdim=True)  # [bs*Timestep,1,h,w]
            spatial_max, _ = torch.max(o, dim=1, keepdim=True)  # [bs*Timestep,1,h,w]
            spatial_attention = self.spatialConv(
                torch.concat([spatial_avg, spatial_max], dim=1))  # [bs*Timestep,1,h,w] 空间注意力

            temporal_attention = self.temporalConv(temporal_avgo)
            for t in range(TimeStep):
                tem_attention[t * bs:(t + 1) * bs] = temporal_attention[:, 0, t]
            mask = step(tem_attention.view(bs * TimeStep, 1, 1, 1).expand_as(x) * channel_attention.expand_as(
                x) * spatial_attention.expand_as(x))

        else:
            mask = (torch.rand(x.shape) > self.ratio).to(x.device)
        return o * mask

    def extra_repr(self):
        return f'drop脉冲的LIF神经元，tau={tau}，软重置，drop比例：{self.ratio}，动态drop：{self.dynamic}'

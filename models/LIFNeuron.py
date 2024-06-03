import torch
import torch.nn as nn

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
class LIF(nn.Module):
    def __init__(self):
        super(LIF, self).__init__()

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)  # x.shape[bs*T,c,h,w]
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)  # u.shape[bs,c,h,w]
        o = torch.zeros(x.shape, device=x.device)  # o.shape[bs*T,c,h,w]
        for t in range(TimeStep):
            # 硬重置
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            o[t * bs:(t + 1) * bs, ...] = spikefunc(u)
        return o

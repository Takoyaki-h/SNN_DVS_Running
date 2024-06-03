import torch
import torch.nn as nn
from models.DVSEventNet import DVSEventNet

def linear_quantize(tensor, num_bits=16):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    min_val, max_val = tensor.min(), tensor.max()

    # Scale and zero-point computation
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    # Quantize
    q_tensor = ((tensor / scale) + zero_point).round().clamp(qmin, qmax).to(torch.int16)

    return q_tensor, scale, zero_point

def dequantize(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)

model = DVSEventNet()
model.load_state_dict(torch.load('./model_pth/model_weights.pth'))

# 打印量化前模型的权重范围
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"量化前{name}: min={param.data.min().item()}, max={param.data.max().item()}")

# 对每层权重进行线性量化
quantized_params = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        q_weight, scale, zero_point = linear_quantize(param.data, num_bits=16)
        quantized_params[name] = (q_weight, scale, zero_point)
        param.data = q_weight.float()  # 保持量化后的整数值

# 保存量化后的模型权重
torch.save(model.state_dict(), './model_pth/linear_quantized_max_acc_signed_16bit.pth')

# 打印量化后模型的权重范围
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"量化后{name}: min={param.data.min().item()}, max={param.data.max().item()}")
        # print(f"量化后{name}的权重值: {param.data}")

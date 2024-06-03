from models.DVSEventNet import DVSEventNet
import torch
from sklearn.cluster import KMeans
import numpy as np

model = DVSEventNet()

model.load_state_dict(torch.load('./model_pth/model_weights.pth'))


def find_scale_factor(tensor, num_bits):
    # 计算量化的缩放因子，适用于有符号量化
    tensor = tensor.cpu().numpy()
    max_val = np.max(np.abs(tensor))  # 获取张量的最大绝对值
    # scale_factor 调整为适用于有符号量化。有符号16-bit的范围是-32768到32767
    scale_factor = (2 ** (num_bits - 1) - 1) / max_val  # 计算缩放因子
    return scale_factor


# 基于聚类的方法进行权值量化，量化为16-bit有符号数
def k_means_cpu(weight, n_clusters, init='k-means++', quantization_bits=16):
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)
    if n_clusters > weight.numel():
        n_clusters = weight.numel()  # 获取张量元素数量

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=50)
    k_means.fit(weight)

    centroids = torch.from_numpy(k_means.cluster_centers_).cuda().view(1, -1)
    labels = k_means.labels_
    labels = torch.from_numpy(labels.reshape(org_shape)).int().cuda()
    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()

    # Quantization
    scale_factor = find_scale_factor(weight, quantization_bits)
    qmin = -(2 ** (quantization_bits - 1))
    qmax = 2 ** (quantization_bits - 1) - 1
    weight = torch.round(weight * scale_factor).clamp(qmin, qmax).to(torch.int16)
    print('scale_factor:', scale_factor)
    return centroids, labels, weight


# 打印量化前模型的权重范围
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"量化前{name}: min={param.data.min().item()}, max={param.data.max().item()}")

# 对每层权重进行 k-means 量化
for name, param in model.named_parameters():
    if 'weight' in name:
        param.data = k_means_cpu(param.data, n_clusters=16, quantization_bits=16)[2].float()

# 保存量化后的模型权重
torch.save(model.state_dict(), './model_pth/quantized_max_acc_k-means_16bit.pth')

# 打印量化后模型的权重范围
# 有符号的16bit [-32768, 32767]
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"量化后{name}: min={param.data.min().item()}, max={param.data.max().item()}")

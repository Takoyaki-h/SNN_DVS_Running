import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 通过计算显著性，再根据某个阈值判断，超过该阈值就不mask
class DropBlockLIF(nn.Module):
    def __init__(self, block_size=10, threshold=0.3):
        super(DropBlockLIF, self).__init__()
        self.block_size = block_size
        self.threshold = threshold

    def forward(self, x, saliency):
        # 确保 saliency 至少有三个维度 (N, H, W)，如果不是则添加一个假的批次维度
        if saliency.dim() == 2:
            saliency = saliency.unsqueeze(0)

        # Generate initial mask where low saliency areas are set to 1
        initial_mask = (saliency < self.threshold).float().to(x.device)

        # Apply max pooling to create block effect
        block_mask = F.max_pool2d(input=initial_mask[:, None, :, :],  # Add channel dimension for pooling
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        # Invert the block_mask to have 0s where blocks are
        block_mask = 1 - block_mask.squeeze(1)

        # Adjust block_mask size if it does not match x size
        if block_mask.shape[2] != x.shape[2] or block_mask.shape[3] != x.shape[3]:
            # Resize block_mask to match x size
            block_mask = F.interpolate(block_mask.unsqueeze(1), size=x.shape[2:], mode='nearest').squeeze(1)

        # Ensure x has the same batch dimension as block_mask
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Apply block mask to the image
        x_masked = x * block_mask[:, None, :, :]  # Apply block_mask to each channel
        return x_masked


def compute_saliency(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = torch.tensor(saliencyMap, dtype=torch.float32)
    return saliencyMap


# Load image and compute saliency
img_path = '/home/hp/Projects/homework/test.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

saliency_map = compute_saliency(img_rgb)

# Normalize saliency map to [0, 1]
saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

# Prepare tensor
# 调整张量的维度顺序，将通道维度从最后一维移动到第一维 shape就变为[h,w,c]
img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
img_tensor = img_tensor.unsqueeze(0)  # Batch dimension

# Apply DropBlock
dropblock = DropBlockLIF(block_size=7)
img_masked = dropblock(img_tensor, saliency_map)
img_masked = img_masked.squeeze(0).permute(1, 2, 0).numpy()
# Visualize the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_masked)
plt.title('Masked Image')
plt.axis('off')

plt.show()

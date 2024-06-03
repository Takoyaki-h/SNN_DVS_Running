import torch
import cv2
import numpy as np

def tensor_to_cv2(image_tensor):
    # 假设输入Tensor为 [bs, c, w, h]，并且数据范围是[0, 1]
    # 如果通道数大于3，只取前三个通道
    if image_tensor.shape[1] > 3:
        image_tensor = image_tensor[:, :3, :, :]
    # 转换为 [w, h, c] 并转换为0-255的uint8类型
    image_numpy = image_tensor.permute(0, 2, 3, 1).numpy()  # 转换为 [bs, w, h, c]
    image_numpy = (image_numpy * 255).astype(np.uint8)
    return image_numpy

def compute_saliency_and_mask(batch_tensor, threshold=0.5):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    batch_cv = tensor_to_cv2(batch_tensor)
    masks = []
    for image_cv in batch_cv:
        # 计算显著性图
        (success, saliency_map) = saliency.computeSaliency(image_cv)
        # 根据阈值生成mask
        mask = (saliency_map > threshold).astype(np.uint8)
        masks.append(mask)
    return torch.from_numpy(np.array(masks))  # 将mask列表转换回Tensor

# 示例数据，假设批次大小为2，通道数为4
bs, c, w, h = 128, 64, 224, 224  # 批次大小、通道数、宽度、高度
image_tensor = torch.rand(bs, c, w, h)  # 生成随机数据

# 计算显著性和mask
mask_tensor = compute_saliency_and_mask(image_tensor)


# 显示结果
print(mask_tensor.shape)  # 应该是 [bs, w, h]，因为mask是单通道

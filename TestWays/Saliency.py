import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# 加载图像并计算显著性图
image = cv2.imread('/home/hp/Projects/homework/cat.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliency_map) = saliency.computeSaliency(image)

# 设定阈值
threshold = 0.5  # 可以根据需要调整这个值
mask = (saliency_map > threshold).astype(np.uint8)  # 创建二值掩码


# 应用掩码
masked_image = cv2.bitwise_and(image, image, mask=mask)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Saliency Map')
plt.imshow(saliency_map, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Masked Image')
plt.imshow(masked_image)
plt.axis('off')

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取示例图片，显示原始图片和计算出显著性后的图片
def compute_saliency(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return None, None  # 如果路径不正确或图像不可读，则返回空

    # OpenCV加载的图像默认是BGR格式，需要转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建显著性检测器并计算显著性图
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img_rgb)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    return img_rgb, saliencyMap


# 指定图像路径
image_path = '/home/hp/Projects/homework/test.png'

# 计算显著性图
img, saliency_map = compute_saliency(image_path)

if img is not None and saliency_map is not None:
    # 显示图像和显著性图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(saliency_map, cmap='gray')
    ax[1].set_title('Saliency Map')
    ax[1].axis('off')

    plt.show()
else:
    print("图像无法加载，请检查路径是否正确。")

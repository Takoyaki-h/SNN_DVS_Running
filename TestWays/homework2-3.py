import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corners(image):
    # 检查图像是否为彩色（即3通道）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # 如果已经是灰度图，则直接使用

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Threshold for an optimal value, it may vary depending on the image.
    corners = np.argwhere(harris > 0.01 * harris.max())
    corners = np.float32(corners[:, [1, 0]])  # x, y coordinates
    return corners

def ncc(patch1, patch2):
    # Assuming both patches are of the same size and are already in float32 format
    patch1_mean = patch1 - np.mean(patch1)
    patch2_mean = patch2 - np.mean(patch2)
    ncc_value = np.sum(patch1_mean * patch2_mean) / (np.sqrt(np.sum(patch1_mean**2)) * np.sqrt(np.sum(patch2_mean**2)))
    return ncc_value

def find_best_match(patch, keypoints, image):
    h, w = patch.shape
    best_ncc = -1
    best_coord = None
    for y, x in keypoints:
        if x < w // 2 or y < h // 2 or x > image.shape[1] - (w // 2) - 1 or y > image.shape[0] - (h // 2) - 1:
            continue  # Avoid corners where the patch would go out of bounds
        comp_patch = image[y - h // 2:y + h // 2 + 1, x - w // 2:x + w // 2 + 1]
        ncc_value = ncc(patch.astype(np.float32), comp_patch.astype(np.float32))
        if ncc_value > best_ncc:
            best_ncc = ncc_value
            best_coord = (x, y)
    return best_coord

def mark_matches(image1, corners1, image2, corners2, window_size=5):
    matches = []
    for corner in corners1:
        x1, y1 = corner.ravel()
        x1, y1 = int(x1), int(y1)  # 确保索引是整数
        if x1 < window_size or y1 < window_size or x1 > image1.shape[1] - window_size - 1 or y1 > image1.shape[0] - window_size - 1:
            continue  # Avoid corners where the window would go out of bounds
        patch1 = image1[y1 - window_size:y1 + window_size + 1, x1 - window_size:x1 + window_size + 1]
        for corner2 in corners2:
            x2, y2 = corner2.ravel()
            x2, y2 = int(x2), int(y2)  # 确保索引是整数
            if x2 < window_size or y2 < window_size or x2 > image2.shape[1] - window_size - 1 or y2 > image2.shape[0] - window_size - 1:
                continue
            patch2 = image2[y2 - window_size:y2 + window_size + 1, x2 - window_size:x2 + window_size + 1]
            ncc_value = ncc(patch1.astype(np.float32), patch2.astype(np.float32))
            if ncc_value > best_ncc:
                best_ncc = ncc_value
                best_match = (x2, y2)
        if best_match is not None:
            matches.append((x1, y1, best_match[0], best_match[1]))
    return matches


# Dummy implementation of loading images and finding Harris corners
# You need to load your actual images and call the harris_corners function
image1 = cv2.imread('/home/hp/Projects/homework/1.png')
image2 = cv2.imread('/home/hp/Projects/homework/2.png')
if image1 is None or image2 is None:
    raise ValueError("图像没有正确加载，请检查路径。")
corners1 = harris_corners(image1)
corners2 = harris_corners(image2)

# Find best matches and mark them
matches = mark_matches(image1, corners1, image2, corners2)

# Draw matches on the image
for x1, y1, x2, y2 in matches:
    # Draw on image1 for visualization
    cv2.circle(image1, (x1, y1), 5, (0, 255, 0), 1)
    # Draw on image2 for visualization
    cv2.circle(image2, (x2, y2), 5, (255, 0, 0), 1)

# Show the images with marked matches
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.imshow(image1, cmap='gray')
plt.title('待匹配图'), plt.axis('off')
plt.subplot(122), plt.imshow(image2, cmap='gray')
plt.title('匹配图'), plt.axis('off')
plt.show()

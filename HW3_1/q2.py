import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像 (灰階)
image = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)

# 2. 直方圖均衡化
equalized_image = cv2.equalizeHist(image)

# 3. 計算直方圖
hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
hist_eq_normalized = hist_eq / hist_eq.sum()

# 4. 顯示影像 + 直方圖（同一張圖）
plt.figure(figsize=(12,5))

# (左邊) 均衡化影像
plt.subplot(1,2,1)
plt.imshow(equalized_image, cmap="gray")
plt.title("Image After Equalization")
plt.axis("off")

# (右邊) 直方圖
plt.subplot(1,2,2)
plt.bar(range(256), hist_eq_normalized.flatten(), width=1.0, color='black')
plt.title("Normalized Histogram after Equalization")
plt.xlabel("Pixel Intensity (0~255)")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()

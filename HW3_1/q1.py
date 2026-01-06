import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像 (灰階)
image = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)

# 2. 計算直方圖
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_normalized = hist / hist.sum()

# 3. 同一張 figure 顯示原圖 & 直方圖
plt.figure(figsize=(10,4))

# 原始影像
plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# 直方圖
plt.subplot(1,2,2)
plt.bar(range(256), hist_normalized.flatten(), width=1.0, color='black')
plt.title("Normalized Histogram")
plt.xlabel("Pixel Intensity (0~255)")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()

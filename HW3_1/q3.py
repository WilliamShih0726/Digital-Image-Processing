import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像 (灰階)
image = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)

# 2. 計算常數 c
z = np.arange(256)
c = 1.0 / np.sum(z**0.4)
target_pdf = c * (z**0.4)
target_pdf[0] = 0  # 避免 z=0 出現 nan
target_pdf = target_pdf / target_pdf.sum()  # 正規化

# 3. 計算目標 CDF
target_cdf = np.cumsum(target_pdf)

# 4. 計算原始影像的直方圖 & CDF
orig_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
orig_pdf = orig_hist / orig_hist.sum()
orig_cdf = np.cumsum(orig_pdf)

# 5. 建立映射 (原始灰階 → 目標灰階)
mapping = np.zeros(256, dtype=np.uint8)
for i in range(256):
    diff = np.abs(target_cdf - orig_cdf[i])
    mapping[i] = np.argmin(diff)

# 6. 應用映射，生成新影像
matched_image = mapping[image]

# 7. 計算匹配後直方圖
matched_hist = cv2.calcHist([matched_image], [0], None, [256], [0, 256])
matched_hist = matched_hist / matched_hist.sum()

# 8. 顯示影像與直方圖（並排）
plt.figure(figsize=(12,5))

# (左) 規格化後影像
plt.subplot(1,2,1)
plt.imshow(matched_image, cmap="gray")
plt.title("Histogram Matched Image")
plt.axis("off")

# (右) 規格化後直方圖
plt.subplot(1,2,2)
plt.bar(range(256), matched_hist.flatten(), width=1.0, color='black')
plt.title("Histogram after Matching")
plt.xlabel("Pixel Intensity (0~255)")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg   # 新增：用來讀 GIF


def rgb_vector_gradient_edge_detection(img_path, save_path="Visual_resolution_gradient.png"):
    # 1. 讀取 GIF 影像，用 matplotlib 支援 GIF
    img = mpimg.imread(img_path)   # 可能是 uint8 或 float、RGB 或 RGBA

    # 如果有 alpha 通道 (RGBA)，只取前 3 個通道
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入影像必須是 RGB 彩色影像 (H x W x 3)")

    # 轉成 float64，範圍 [0,1]
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)

    # 2. 定義 Sobel 濾波器 (水平方向 Sx 與垂直方向 Sy)
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    Sy = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=float)

    # 建立 Gx, Gy 陣列
    Gx = np.zeros_like(img, dtype=float)
    Gy = np.zeros_like(img, dtype=float)

    # 3. 對 R、G、B 三個通道分別做卷積
    # 使用 cv2.filter2D
    for c in range(3):  # 0:R, 1:G, 2:B
        Gx[:, :, c] = cv2.filter2D(img[:, :, c], -1, Sx, borderType=cv2.BORDER_REFLECT)
        Gy[:, :, c] = cv2.filter2D(img[:, :, c], -1, Sy, borderType=cv2.BORDER_REFLECT)

    # 4. 在 RGB 向量空間計算梯度大小
    Gx_norm2 = np.sum(Gx ** 2, axis=2)   # H x W
    Gy_norm2 = np.sum(Gy ** 2, axis=2)   # H x W
    grad_mag = np.sqrt(Gx_norm2 + Gy_norm2)

    # 5. 正規化到 [0,1]
    grad_mag = grad_mag / (grad_mag.max() + 1e-12)

    # 6. 顯示結果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Visual resolution image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(grad_mag, cmap="gray")
    plt.title("RGB Vector Gradient Magnitude")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 7. 儲存結果
    grad_mag_uint8 = (grad_mag * 255).astype(np.uint8)
    cv2.imwrite(save_path, grad_mag_uint8)
    print(f"已將彩色梯度邊緣影像儲存為：{save_path}")


if __name__ == "__main__":
    # 跟 q2.py 在同一個資料夾的情況
    rgb_vector_gradient_edge_detection("Visual resolution.gif")

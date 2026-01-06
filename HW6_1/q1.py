import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage  # 如果沒有這個套件，請先: pip3 install scipy


def rgb_vector_gradient_edge_detection(img_path, save_path="lenna_RGB_gradient.png"):
    # 1. 用 OpenCV 讀取彩色影像 (BGR)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到影像檔案: {img_path}")

    # 轉成 RGB，並正規化到 [0,1] (float64)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.0

    # 檢查影像形狀
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入影像必須是 RGB 彩色影像 (H x W x 3)")

    # 2. 定義 Sobel 濾波器 (水平方向 Sx 與垂直方向 Sy)
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    Sy = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=float)

    # 預先建立 Gx, Gy 陣列，用來存放每個通道的梯度
    Gx = np.zeros_like(img, dtype=float)  # H x W x 3
    Gy = np.zeros_like(img, dtype=float)

    # 3. 對 R、G、B 三個通道分別做卷積，得到每一通道的 x 與 y 項梯度
    for c in range(3):  # c = 0(R),1(G),2(B)
        Gx[:, :, c] = ndimage.convolve(img[:, :, c], Sx, mode='reflect')
        Gy[:, :, c] = ndimage.convolve(img[:, :, c], Sy, mode='reflect')

    # 4. 在 RGB 向量空間計算梯度大小
    #    ||Gx||^2 = (dR/dx)^2 + (dG/dx)^2 + (dB/dx)^2
    #    ||Gy||^2 = (dR/dy)^2 + (dG/dy)^2 + (dB/dy)^2
    Gx_norm2 = np.sum(Gx ** 2, axis=2)   # H x W
    Gy_norm2 = np.sum(Gy ** 2, axis=2)   # H x W

    #    ||G|| = sqrt( ||Gx||^2 + ||Gy||^2 )
    grad_mag = np.sqrt(Gx_norm2 + Gy_norm2)

    # 5. 正規化到 [0,1]
    grad_mag = grad_mag / (grad_mag.max() + 1e-12)

    # 6. 顯示結果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original RGB image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(grad_mag, cmap="gray")
    plt.title("RGB Vector Gradient Magnitude")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 7. 儲存結果為 8-bit 灰階影像
    grad_mag_uint8 = (grad_mag * 255).astype(np.uint8)
    cv2.imwrite(save_path, grad_mag_uint8)
    print(f"已將彩色梯度邊緣影像儲存為：{save_path}")


if __name__ == "__main__":
    # lenna-RGB.tif 放在同一個資料夾就直接這樣寫
    rgb_vector_gradient_edge_detection("lenna-RGB.tif")

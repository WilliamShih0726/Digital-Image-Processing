import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------- 參數（可依作業微調） ---------------------
IMG_PATH = Path("Fig5.25.jpg")  # 題目影像
OUT_DIR  = Path("./q1_outputs")
k        = 0.0025     # 湍流強度（同學程式用 0.0025）
radius   = 66         # 頻域半徑遮罩（同學用 66）
eps      = 5e-3       # 逆濾波穩定項，避免 |H|≈0 時爆噪
K_wiener = 0.01       # Wiener 的 NSR（同學設 0.01）

# --------------------- 工具函式 ---------------------
def to_float_img(img_bgr):
    """轉成 float32, 範圍 0~255；並回傳 channel list（灰階時長度=1）"""
    if img_bgr.ndim == 2:
        return [img_bgr.astype(np.float32)]
    else:
        # OpenCV 讀進來是 BGR；改成 RGB 的順序只是視覺一致，頻域不受影響
        b, g, r = cv2.split(img_bgr)
        return [r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)]

def fft2c(x):  # centered FFT
    return np.fft.fftshift(np.fft.fft2(x))

def ifft2c(X): # centered iFFT
    return np.real(np.fft.ifft2(np.fft.ifftshift(X))).astype(np.float32)

def H_turbulence(shape, k):
    """H(u,v) = exp(-k*(u^2+v^2)^(5/6))，頻率座標已中心化"""
    M, N = shape
    u = np.arange(-M//2, M//2, dtype=np.float32)
    v = np.arange(-N//2, N//2, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    D2 = U**2 + V**2
    H = np.exp(-k * (D2 ** (5/6)))
    return H.astype(np.float32), D2

def apply_inverse(G, H, mask, eps):
    """F = G / H（|H|<eps 用 eps 以防爆炸），並套半徑遮罩"""
    Hs = H.copy().astype(np.complex64)
    Hs[np.abs(Hs) < eps] = eps
    F = G / Hs
    if mask is not None:
        F[~mask] = 0
    return F

def apply_wiener(G, H, mask, K):
    """F = (H* / (|H|^2 + K)) * G，並套半徑遮罩"""
    F = (np.conj(H) / (np.abs(H)**2 + K)) * G
    if mask is not None:
        F[~mask] = 0
    return F

def stack_channels(chs):
    """把 [R,G,B] 或 [gray] 合回 8-bit 影像"""
    chs_u8 = [np.clip(c, 0, 255).astype(np.uint8) for c in chs]
    if len(chs_u8) == 1:
        return chs_u8[0]
    # 回存成 RGB -> 轉回 BGR 方便 imwrite 顯示色彩一致
    r, g, b = chs_u8
    return cv2.merge([b, g, r])

# --------------------- 主流程 ---------------------
if __name__ == "__main__":
    assert IMG_PATH.exists(), f"找不到影像：{IMG_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(IMG_PATH)

    chans = to_float_img(img_bgr)
    M, N = chans[0].shape

    # 建 H 與半徑遮罩
    H, D2 = H_turbulence((M, N), k=k)
    radius_mask = D2 < (radius**2)  # 同學的 if(u^2+v^2) >= 66^2 -> 置 0

    # 分通道處理（與同學一致）
    inv_chs, wnr_chs = [], []
    for c in chans:
        G = fft2c(c)
        # Inverse
        F_inv = apply_inverse(G, H, radius_mask, eps=eps)
        f_inv = ifft2c(F_inv)
        # 轉回 0~255（對齊同學的 abs+uint8）
        f_inv_u8 = cv2.normalize(np.abs(f_inv), None, 0, 255, cv2.NORM_MINMAX)
        inv_chs.append(f_inv_u8)

        # Wiener
        F_w = apply_wiener(G, H, radius_mask, K=K_wiener)
        f_w  = ifft2c(F_w)
        f_w_u8 = cv2.normalize(np.abs(f_w), None, 0, 255, cv2.NORM_MINMAX)
        wnr_chs.append(f_w_u8)

    # 合通道並存檔
    inv_img  = stack_channels(inv_chs)
    wien_img = stack_channels(wnr_chs)
    cv2.imwrite(str(OUT_DIR/"inverse_result.png"), inv_img)
    cv2.imwrite(str(OUT_DIR/"wiener_result.png"),  wien_img)

    # 並排比較（非必要，但助檢視）
    fig = plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("original image"); plt.axis('off')
    plt.subplot(1,3,2); 
    plt.imshow(inv_img if inv_img.ndim==2 else cv2.cvtColor(inv_img, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title(f"Inverse (k={k}, radius={radius}, eps={eps})"); plt.axis('off')
    plt.subplot(1,3,3); 
    plt.imshow(wien_img if wien_img.ndim==2 else cv2.cvtColor(wien_img, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title(f"Wiener (k={k}, radius={radius}, K={K_wiener})"); plt.axis('off')
    plt.tight_layout(); plt.savefig(str(OUT_DIR/"comparison.png"), dpi=200, bbox_inches="tight"); plt.show()

    print("[Saved]", OUT_DIR/"inverse_result.png")
    print("[Saved]", OUT_DIR/"wiener_result.png")

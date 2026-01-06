import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ 讀圖 ------------------
IN_PATH = Path("book-cover-blurred.tif")   # 題目影像
OUT_DIR = Path("./q2_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(IN_PATH), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(IN_PATH)
g = img.astype(np.float32)              # 與你同學的 MATLAB 一樣保留 0~255 範圍
M, N = g.shape

# ------------------ 轉頻域並中心化 ------------------
G = np.fft.fftshift(np.fft.fft2(g))

# ------------------ 運動模糊頻域模型（依 MATLAB 公式） ------------------
# H(u,v) = sin(pi*(a*u + b*v)) / (pi*(a*u + b*v)) * exp(-j*pi*(a*u + b*v))
# 其中 (u,v) 採中心化的頻率座標
a = 0.1   # ← 可調：水平/垂直方向的模糊參數（和你同學的程式一致）
b = 0.1   # ← 可調
u = np.arange(-M//2, M//2, dtype=np.float32)
v = np.arange(-N//2, N//2, dtype=np.float32)
U, V = np.meshgrid(u, v, indexing="ij")
phi = np.pi * (a*U + b*V)

H = np.zeros((M, N), dtype=np.complex64)
eps = 1e-8
mask0 = np.abs(phi) < eps
H[mask0] = 1.0 + 0j
H[~mask0] = (np.sin(phi[~mask0]) / (phi[~mask0])) * np.exp(-1j * phi[~mask0])

# ------------------ (1) 逆濾波（Inverse Filter） ------------------
# 與同學一樣用 F = G./H，但加入穩定項避免 |H|~0 時放大雜訊
H_stab = H.copy()
H_stab[np.abs(H_stab) < 1e-3] = 1e-3 + 0j
F_inv = G / H_stab
inv_spatial = np.fft.ifft2(np.fft.ifftshift(F_inv))
inv_img = np.abs(inv_spatial)
# 轉為 0~255
inv_img = (255 * (inv_img - inv_img.min()) / (inv_img.ptp() + 1e-8)).astype(np.uint8)

# ------------------ (2) 維納濾波（Wiener Filter） ------------------
# F = (H* / (|H|^2 + K)) * G ，其中 K ~ NSR
K = 1e-4  # ← 可調（1e-4 ~ 1e-2 之間試）
F_w = (np.conj(H) / (np.abs(H)**2 + K)) * G
w_spatial = np.fft.ifft2(np.fft.ifftshift(F_w))
w_img = np.abs(w_spatial)
w_img = (255 * (w_img - w_img.min()) / (w_img.ptp() + 1e-8)).astype(np.uint8)

# ------------------ 輸出與對照 ------------------
p_orig = OUT_DIR/"original_input.png"
p_inv  = OUT_DIR/f"inverse_result_a{a}_b{b}.png"
p_wien = OUT_DIR/f"wiener_result_a{a}_b{b}_K{K}.png"
cv2.imwrite(str(p_orig), g.astype(np.uint8))
cv2.imwrite(str(p_inv),  inv_img)
cv2.imwrite(str(p_wien), w_img)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(g, cmap='gray');     plt.title("原始受損影像"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(inv_img, cmap='gray'); plt.title(f"Inverse (a={a}, b={b})"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(w_img, cmap='gray');   plt.title(f"Wiener (a={a}, b={b}, K={K})"); plt.axis('off')
plt.tight_layout(); plt.savefig(str(OUT_DIR/"comparison.png"), dpi=200, bbox_inches="tight")
plt.show()

print("[Saved]", p_inv)
print("[Saved]", p_wien)

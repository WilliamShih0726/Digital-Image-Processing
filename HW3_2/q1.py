# q1_show_matlab_equiv.py
# 等價重現 MATLAB：Global HE -> Local enhancement(3x3, k0..k3, c)
# 顯示三張二合一圖：Original / Equalization / Local Enhancement

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------- 讀檔（支援 'hidden_object.jpg' 或 'hidden object.jpg' 或 .tif） ----------
def resolve_input():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        # 常見三種命名
        cand = [
            Path("hidden_object.jpg"),
            Path("hidden object.jpg"),
            Path("hidden_object.tif"),
            Path("hidden object.tif"),
        ]
        p = next((c for c in cand if c.exists()), None)
        if p is None:
            p = Path(__file__).resolve().parent / "hidden object.jpg"
    if not p.exists():
        alt = p.with_suffix(".tif")
        if alt.exists(): p = alt
    if not p.exists():
        raise FileNotFoundError(f"找不到影像：{p}\n用法：python3 q1_show_matlab_equiv.py 'hidden_object.jpg'")
    return p

# ---------- 工具：直方圖（機率） ----------
def hist_prob(img_u8):
    h, _ = np.histogram(img_u8.ravel(), bins=256, range=(0, 255))
    return h / max(h.sum(), 1)

def show_pair(img_u8, title_img, title_hist):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_u8, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title(title_img); axs[0].axis('off')
    p = hist_prob(img_u8)
    axs[1].bar(np.arange(256), p, width=1.0)
    axs[1].set_xlim(0, 255)
    axs[1].set_xlabel('Gray level'); axs[1].set_ylabel('Probability')
    axs[1].set_title(title_hist)
    fig.tight_layout()
    plt.show()

# ---------- 全域直方圖等化（照 MATLAB 思路手寫 CDF） ----------
def global_hist_equalize(img_u8):
    hist, _ = np.histogram(img_u8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_scaled = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    lut = np.ma.filled(cdf_scaled, 0).astype('uint8')
    return lut[img_u8]

def main():
    in_path = resolve_input()
    # 讀 jpg/tif 為灰階
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cv2.imread 讀檔失敗，請確認檔名/副檔名")

    # 尺寸對齊 MATLAB：665×652（注意 OpenCV resize 是 (width,height)）
    H, W = 665, 652
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    # (1) 原圖 + 直方圖
    show_pair(img, "Original (Grayscale, 665×652)", "Histogram: Original")

    # (2) Global Histogram Equalization（直方圖統計法）
    img_eq = global_hist_equalize(img)
    show_pair(img_eq, "Image after Histogram Equalization", "Histogram: Equalization")

    # (3) Local Enhancement（3×3 視窗之局部統計，對齊 MATLAB 參數）
    # 參數（與 MATLAB 一致）
    k0, k1 = 0.0, 0.25
    k2, k3 = 0.0, 1.0
    c_gain  = 3.0

    # 全域 μ、σ：用「等化後」影像（與 MATLAB 一致）
    g = img_eq.astype(np.float32)
    mG = g.mean()
    sG = g.std() + 1e-12

    # 3×3 區域均值與標準差（用 box blur/blur 計算）
    g32 = g.astype(np.float32)
    mL  = cv2.blur(g32, (3, 3))                      # E[x]
    m2L = cv2.blur(g32 * g32, (3, 3))                # E[x^2]
    sL  = cv2.sqrt(cv2.max(m2L - mL * mL, 0))        # sqrt(Var)

    # 條件：k0*mG ≤ mL ≤ k1*mG 且 k2*sG ≤ sL ≤ k3*sG
    mask = (mL >= k0 * mG) & (mL <= k1 * mG) & (sL >= k2 * sG) & (sL <= k3 * sG)

    # 乘上增益 c（並截斷到 0..255）
    out = img_eq.copy().astype(np.float32)
    out[mask] = np.clip(c_gain * out[mask], 0, 255)
    out = out.astype(np.uint8)

    show_pair(out, f"Local Enhancement (3×3, k0={k0},k1={k1},k2={k2},k3={k3}, c={c_gain})",
              "Histogram: Local Enhancement")

if __name__ == "__main__":
    main()

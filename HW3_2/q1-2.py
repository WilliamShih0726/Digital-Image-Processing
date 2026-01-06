# q1_show_final.py
# 直接顯示三組二合一（左影像、右直方圖）：Original / Global HE / Local Enhancement (Dark-biased CLAHE)

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ====================== 可調參數 ======================
# 先做輕微中值濾波，抑制雜點（None/0 表示不做；建議用 3）
MEDIAN_BLUR_K = 3

# 暗部遮罩的門檻與過渡帶寬（越高越多區域走「hard」）
DARK_THRESH = 150         # 120~160 常用；想更像“很兇”的參考圖可設 150~160
SOFT_BAND   = 30.0        # 過渡帶寬，越大越柔和

# CLAHE 兩組參數：soft（非暗部）與 hard（暗部）
CLAHE_SOFT_CLIP = 2.5
CLAHE_SOFT_TILE = (8, 8)

CLAHE_HARD_CLIP = 7.0     # 6~10 越大越兇
CLAHE_HARD_TILE = (4, 4)  # (3,3) 或 (4,4) tile 越小區塊感越強
# =====================================================

def load_input_path():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        p = Path(__file__).resolve().parent / "hidden object.jpg"
    if not p.exists():
        alt = p.with_suffix(".tif")
        if alt.exists():
            p = alt
    if not p.exists():
        raise FileNotFoundError(f"找不到影像：{p}\n用法：python3 q1_show_final.py 'hidden object.jpg'")
    return p

def global_hist_equalize(img_u8):
    """全域直方圖等化（Histogram Statistics / HE）"""
    hist, _ = np.histogram(img_u8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_scaled = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    lut = np.ma.filled(cdf_scaled, 0).astype('uint8')
    return lut[img_u8]

def clahe_apply(img_u8, clip, tile):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img_u8)

def dark_biased_clahe(img_u8,
                      soft_clip=CLAHE_SOFT_CLIP, soft_tile=CLAHE_SOFT_TILE,
                      hard_clip=CLAHE_HARD_CLIP, hard_tile=CLAHE_HARD_TILE,
                      dark_T=DARK_THRESH, band=SOFT_BAND, median_k=MEDIAN_BLUR_K):
    """只對偏暗區域套用更兇的 CLAHE，其他區域用較溫和的 CLAHE，再做權重融合。"""
    base = img_u8.copy()
    if median_k and median_k >= 3:
        base = cv2.medianBlur(base, median_k)

    soft = clahe_apply(base, soft_clip, soft_tile)
    hard = clahe_apply(base, hard_clip, hard_tile)

    blur = cv2.GaussianBlur(base, (9, 9), 0)
    # 權重：越暗權重越接近 1（越用 hard）
    w = np.clip((dark_T - blur) / float(band), 0.0, 1.0).astype(np.float32)

    out = (w * hard + (1.0 - w) * soft).astype(np.uint8)
    return out

def show_pair(image_u8, title_img, title_hist):
    """顯示二合一：左影像、右直方圖（機率）"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(image_u8, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title(title_img)
    axs[0].axis('off')

    h, bins = np.histogram(image_u8.ravel(), bins=256, range=(0, 255))
    p = h / max(h.sum(), 1)
    axs[1].bar(bins[:-1], p, width=1.0)
    axs[1].set_xlim(0, 255)
    axs[1].set_xlabel('Gray level')
    axs[1].set_ylabel('Probability')
    axs[1].set_title(title_hist)

    fig.tight_layout()
    plt.show()

def main():
    in_path = load_input_path()
    print("[INFO] OpenCV:", cv2.__version__)
    print("[INFO] Input:", in_path.resolve())

    # 灰階讀檔（jpg/tif 都可）
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cv2.imread 讀檔失敗，請確認檔名/副檔名。")

    # (1) 原圖（可選：同樣做 median blur 以便公平）
    orig = img.copy()
    if MEDIAN_BLUR_K and MEDIAN_BLUR_K >= 3:
        orig = cv2.medianBlur(orig, MEDIAN_BLUR_K)
    
    # (2) 全域直方圖等化（Histogram Statistics）
    he = global_hist_equalize(orig)

    # (3) 區域增強：暗部偏重的 CLAHE 融合
    loc = dark_biased_clahe(img_u8=img)

    # 依序顯示三組
    show_pair(orig, "Original (with median blur)" if MEDIAN_BLUR_K else "Original",
              "Histogram: Original")
    show_pair(he, "Global Histogram Equalization (HE)", "Histogram: Global HE")
    show_pair(loc,
              f"Local Enhancement (Dark-biased CLAHE)\n"
              f"soft: clip={CLAHE_SOFT_CLIP}, tile={CLAHE_SOFT_TILE}; "
              f"hard: clip={CLAHE_HARD_CLIP}, tile={CLAHE_HARD_TILE}; "
              f"T={DARK_THRESH}, band={SOFT_BAND}, median={MEDIAN_BLUR_K}",
              "Histogram: Local (CLAHE blend)")

if __name__ == "__main__":
    main()

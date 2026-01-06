# -*- coding: utf-8 -*-
"""
Reproduce the attached figure style:
- Show 3 figures with titles: 'fft amp', 'mask result', 'output image'
- Do rectangular notch (manual coords) or Gaussian notch
- Coordinates are the ones that match the visible interference peaks.

Usage:
    python3 q1.py --mode rect     # 矩形 0/1 凹點（和 MATLAB 相同邏輯）
    python3 q1.py --mode gauss    # 高斯凹點（建議）
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到輸入影像: {path}")
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

def normalize_u8(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255.0
    return x.astype(np.uint8)

def build_rect_mask(shape, peaks_xy, half):
    H, W = shape
    cy, cx = H//2, W//2
    mask = np.ones(shape, dtype=np.float32)
    for (x, y) in peaks_xy:
        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W:
                    mask[yy, xx] = 0.0
                ys = (2*cy - yy) % H
                xs = (2*cx - xx) % W
                mask[ys, xs] = 0.0
    return mask

def build_gauss_mask(shape, peaks_xy, sigma):
    H, W = shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    cy, cx = H//2, W//2
    def notch(x0,y0):
        return 1.0 - np.exp(-((yy-y0)**2 + (xx-x0)**2)/(2*sigma*sigma))
    Hf = np.ones(shape, dtype=np.float32)
    for (x0, y0) in peaks_xy:
        Hf *= notch(x0, y0)
        y1 = (2*cy - y0) % H
        x1 = (2*cx - x0) % W
        Hf *= notch(x1, y1)
    return Hf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="astronaut-interference.tif")
    ap.add_argument("--mode", choices=["rect","gauss"], default="gauss")
    ap.add_argument("--half", type=int, default=8, help="rect mask half-width (pixels)")
    ap.add_argument("--sigma", type=float, default=7.0, help="gaussian notch sigma")
    ap.add_argument("--no-show", action="store_true", help="只存檔，不跳視窗")
    args = ap.parse_args()

    img = read_gray(args.input)
    H, W = img.shape

    # 2D FFT + center
    F  = np.fft.fft2(img)
    F0 = np.fft.fftshift(F)
    amp = np.abs(F0)

    # 兩個主峰位置（搭配共軛點）
    peaks_xy = [(475, 387), (525, 437)]  # (x=col, y=row)

    # 選擇濾波器
    if args.mode == "rect":
        Hf = build_rect_mask((H, W), peaks_xy, half=args.half)
        out_path = "astronaut_interference_filtered_manual_rect.png"
    else:
        Hf = build_gauss_mask((H, W), peaks_xy, sigma=args.sigma)
        out_path = "astronaut_interference_filtered_gauss_notch.png"

    # 濾波與 IFFT
    Ff   = F0 * Hf
    img2 = np.fft.ifft2(np.fft.ifftshift(Ff)).real
    img2_u8 = normalize_u8(img2)
    Image.fromarray(img2_u8).save(out_path)
    print(f"[Saved] result: {out_path}")

    # ====== 中間過程也另存檔，並可視化 ======
    
    out_dir = "resault_q1"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1) 原始輸入影像
    plt.figure()
    plt.title("Original")
    plt.imshow(img.astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "original.png"), dpi=300, bbox_inches="tight")
    print("[Saved] original.png")
    
    # 2) FFT amplitude 圖
    plt.figure()
    plt.title("fft amp")
    fft_amp_vis = np.log1p(amp) / np.max(np.log1p(amp))
    plt.imshow(fft_amp_vis, cmap="gray")
    plt.axis("on")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fft_amp.png"), dpi=300, bbox_inches="tight")
    print("[Saved] fft_amp.png")

    # 3) Mask 之後的頻譜圖
    plt.figure()
    plt.title("mask result")
    masked_vis = np.log1p(np.abs(Ff)) / np.max(np.log1p(np.abs(Ff)))
    plt.imshow(masked_vis, cmap="gray")
    plt.axis("on")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mask_result.png"), dpi=300, bbox_inches="tight")
    print("[Saved] mask_result.png")

    # 4) 輸出影像（僅顯示；檔案已在前面存好）
    plt.figure()
    plt.title("output image")
    plt.imshow(img2_u8, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "output_image.png"), dpi=300, bbox_inches="tight")
    print("[Saved] output_image.png")

    # === 直接顯示視窗 ===
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()

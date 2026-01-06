# -*- coding: utf-8 -*-
"""
Remove moiré pattern in 'car-moire-pattern.tif' via FFT Gaussian notch filtering (auto-detect peaks).
在運行時顯示所有中間結果，並自動存下四張圖片：
original.png, fft_amp.png, filter_mask.png, output_image.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Utils ----------
def read_gray(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.float32)

def normalize_to_0_255(img: np.ndarray) -> np.ndarray:
    m, M = np.min(img), np.max(img)
    if M - m < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - m) / (M - m) * 255.0
    return out.astype(np.uint8)

# ---------- FFT helpers ----------
def fft2_shift(img: np.ndarray):
    F = np.fft.fft2(img)
    return np.fft.fftshift(F)

def ifft2_from_shift(Fs: np.ndarray) -> np.ndarray:
    F = np.fft.ifftshift(Fs)
    return np.real(np.fft.ifft2(F))

# ---------- Peak detection ----------
def detect_strong_peaks(logmag: np.ndarray, center_radius: int = 22,
                        k: int = 24, min_separation: int = 8):
    H, W = logmag.shape
    cy, cx = H // 2, W // 2

    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]

    dc_mask = (yy - cy)**2 + (xx - cx)**2 <= center_radius**2
    border = 3
    border_mask = np.zeros_like(dc_mask, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    cand = logmag.copy()
    cand[dc_mask | border_mask] = -np.inf
    flat = cand.ravel()
    K = min(k * 4, flat.size)
    idx = np.argpartition(flat, -K)[-K:]
    r, c = np.unravel_index(idx, logmag.shape)
    pts = np.stack([r, c], axis=1)
    vals = flat[idx]
    order = np.argsort(-vals)

    selected = []
    for i in order:
        rr, cc = pts[i]
        ok = True
        for (r0, c0) in selected:
            if (rr - r0)**2 + (cc - c0)**2 < min_separation**2:
                ok = False
                break
        if ok:
            selected.append((int(rr), int(cc)))
        if len(selected) >= k:
            break
    return selected

def build_gaussian_notch_filter(shape, peaks, sigma=8.0, pair_enforce=True):
    H, W = shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    cy, cx = H // 2, W // 2
    Hf = np.ones((H, W), dtype=np.float32)

    def notch(r0, c0):
        return 1.0 - np.exp(-((yy - r0)**2 + (xx - c0)**2) / (2.0 * sigma**2))

    used = set()
    for (r, c) in peaks:
        if (r, c) in used:
            continue
        if abs(r - cy) <= 1 and abs(c - cx) <= 1:
            continue
        Hf *= notch(r, c)
        used.add((r, c))
        if pair_enforce:
            rs = (2 * cy - r) % H
            cs = (2 * cx - c) % W
            Hf *= notch(rs, cs)
            used.add((rs, cs))
    return Hf

# ---------- Main ----------
def main():
    in_path = "car-moire-pattern.tif"
    out_path = "car_moire_filtered.png"

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到輸入影像：{in_path}")

    img = read_gray(in_path)
    Fs = fft2_shift(img)
    mag = np.abs(Fs)
    logmag = np.log1p(mag)

    peaks = detect_strong_peaks(logmag, center_radius=22, k=24, min_separation=8)
    Hf = build_gaussian_notch_filter(img.shape, peaks, sigma=8.0, pair_enforce=True)
    Ff = Fs * Hf
    rec = ifft2_from_shift(Ff)
    rec_u8 = normalize_to_0_255(rec)
    Image.fromarray(rec_u8).save(out_path)
    print(f"[Saved] result: {out_path}")
    print(f"Detected peaks = {len(peaks)}")

    # ====== 儲存並顯示每張過程圖 ======

    out_dir = "resault_q2"
    os.makedirs(out_dir, exist_ok=True)

    # 1) 原始輸入影像
    plt.figure()
    plt.title("Original")
    plt.imshow(img.astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "original.png"), dpi=300, bbox_inches="tight")
    print("[Saved] original.png")

    # 2) FFT amplitude
    plt.figure()
    plt.title("Log Magnitude Spectrum")
    fft_vis = logmag / np.max(logmag)
    plt.imshow(fft_vis, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fft_amp.png"), dpi=300, bbox_inches="tight")
    print("[Saved] fft_amp.png")

    # 3) 濾波器圖像
    plt.figure()
    plt.title("Gaussian Notch Filter")
    plt.imshow(Hf, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mask_result.png"), dpi=300, bbox_inches="tight")
    print("[Saved] filter_mask.png")

    # 4) 結果影像
    plt.figure()
    plt.title("Filtered (Result)")
    plt.imshow(rec_u8, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "output_image.png"), dpi=300, bbox_inches="tight")
    print("[Saved] output_image.png")

    plt.show()

if __name__ == "__main__":
    main()

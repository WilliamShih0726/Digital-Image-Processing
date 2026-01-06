# q1.py — 存檔 + 立即顯示結果圖（--show 可關閉）
import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_gray(path):
    p = Path(path)
    if not p.exists():
        here = Path.cwd()
        cand = list(here.glob("**/checkerboard1024-shaded.*")) + list(here.glob("**/N1.*"))
        msg = [f"[ERROR] File not found: {p}"]
        msg.append(f"  - Current working dir: {here}")
        if cand:
            msg.append("  - I found similarly named files:")
            msg += [f"    * {c}" for c in cand[:10]]
        else:
            msg.append("  - No similarly named files found under current dir.")
        raise FileNotFoundError("\n".join(msg))

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            import imageio.v2 as imageio
            img = imageio.imread(p)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception:
            try:
                from PIL import Image
                img = np.array(Image.open(p).convert("L"))
            except Exception as e:
                raise RuntimeError(f"[ERROR] Failed to load image by cv2/imageio/PIL: {p}\n{e}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)

def gaussian_kernel_params(h, w, K):
    sigma = min(h, w) / K
    ksize = int(2*np.ceil(3*sigma) + 1)
    return int(ksize), float(sigma)

def estimate_shading(img, ksize, sigma):
    if ksize > 0 and ksize % 2 == 1 and ksize <= 1001:
        return cv2.GaussianBlur(img, (ksize, ksize), sigma, sigma, borderType=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(img, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)

def flatfield_correction(img, shade, eps=1e-6):
    shade_norm = shade / (shade.mean() + eps)
    return img / (shade_norm + eps)

def rescale_to_uint8(im):
    im = im - im.min()
    if im.max() > 0: im = im / im.max()
    return (im*255.0 + 0.5).astype(np.uint8)

def run(path, K, tag, out_dir, show=True):
    img = read_gray(path)
    h, w = img.shape
    ksize, sigma = gaussian_kernel_params(h, w, K)
    shade = estimate_shading(img, ksize, sigma)
    corr  = flatfield_correction(img, shade)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    base = Path(path).stem
    p_shade = out/f"{base}_shade_{tag}.png"
    p_corr  = out/f"{base}_corrected_{tag}.png"
    p_panel = out/f"{base}_panel_{tag}.png"

    cv2.imwrite(str(p_shade), rescale_to_uint8(shade))
    cv2.imwrite(str(p_corr),  rescale_to_uint8(corr))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(rescale_to_uint8(img),   cmap="gray"); plt.title(f"{base}: original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(rescale_to_uint8(shade), cmap="gray"); plt.title(f"shading (ksize={ksize}, σ={sigma:.1f})"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(rescale_to_uint8(corr),  cmap="gray"); plt.title("corrected (divide by shading)"); plt.axis("off")
    plt.tight_layout(); plt.savefig(p_panel, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    print(f"[OK] {base}  size={h}x{w}  kernel≈{ksize}  sigma={sigma:.2f}")
    print(f"     shading   → {p_shade}")
    print(f"     corrected → {p_corr}")
    print(f"     panel     → {p_panel}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", default="./checkerboard1024-shaded.tif", help="path to checkerboard shaded image")
    ap.add_argument("--img2", default="./N1.bmp", help="path to N1 image")
    ap.add_argument("--out",  default="./outputs", help="output directory")
    ap.add_argument("--K1", type=float, default=18.0, help="K for image 1 (checkerboard)")
    ap.add_argument("--K2", type=float, default=28.0, help="K for image 2 (N1)")
    ap.add_argument("--no-show", action="store_true", help="do not display figures (still saves files)")
    args = ap.parse_args()

    run(args.img1, K=args.K1, tag="checkerboard", out_dir=args.out, show=(not args.no_show))
    run(args.img2, K=args.K2, tag="N1",          out_dir=args.out, show=(not args.no_show))

if __name__ == "__main__":
    main()

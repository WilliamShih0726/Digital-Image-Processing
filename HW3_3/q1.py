import argparse
from pathlib import Path
import numpy as np, cv2, matplotlib.pyplot as plt

def read_gray(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] File not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            import imageio.v2 as imageio
            img = imageio.imread(path)
            if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception:
            from PIL import Image
            img = np.array(Image.open(path).convert("L"))
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)

def gaussian_params(h, w, K):
    sigma = min(h, w)/K
    ksize = int(2*np.ceil(3*sigma)+1)
    return int(ksize), float(sigma)

def blur(img, ksize, sigma):
    if 0 < ksize <= 1001 and ksize % 2 == 1:
        return cv2.GaussianBlur(img, (ksize, ksize), sigma, sigma, borderType=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(img, (0, 0), sigma, sigma, borderType=cv2.BORDER_REFLECT)

def flatfield(img, shade, eps=1e-6):
    shade_n = shade/(shade.mean()+eps)
    return img/(shade_n+eps)

def to_u8(x):
    x = x - x.min()
    if x.max()>0: x = x/x.max()
    return (x*255.0+0.5).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(description="Q1: Gaussian lowpass shading correction on checkerboard")
    ap.add_argument("--img", default="./checkerboard1024-shaded.tif")
    ap.add_argument("--out", default="./outputs_q1")
    ap.add_argument("--K", type=float, default=18.0, help="sigma = min(H,W)/K (bigger blur when smaller K)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    img = read_gray(Path(args.img))
    h, w = img.shape
    ksize, sigma = gaussian_params(h, w, args.K)
    shade = blur(img, ksize, sigma)
    corr  = flatfield(img, shade)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = Path(args.img).stem
    p_shade = out/f"{base}_shade.png"
    p_corr  = out/f"{base}_corrected.png"
    p_panel = out/f"{base}_panel.png"

    cv2.imwrite(str(p_shade), to_u8(shade))
    cv2.imwrite(str(p_corr),  to_u8(corr))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(to_u8(img),   cmap="gray"); plt.title("original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(to_u8(shade), cmap="gray"); plt.title(f"shading (ksize={ksize}, σ={sigma:.1f})"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(to_u8(corr),  cmap="gray"); plt.title("corrected (divide)"); plt.axis("off")
    plt.tight_layout(); plt.savefig(p_panel, dpi=150, bbox_inches="tight")
    if not args.no_show: plt.show()
    else: plt.close()

    print(f"[Q1 OK] size={h}x{w}  kernel≈{ksize}  sigma={sigma:.2f}")
    print(f"  shading   → {p_shade}")
    print(f"  corrected → {p_corr}")
    print(f"  panel     → {p_panel}")

if __name__ == "__main__":
    main()

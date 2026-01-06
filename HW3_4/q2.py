import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- parameters (feel free to tweak) ----------------
GAUSS_SIGMA = 1.5      # mild denoise on L
ALPHA       = 1.2      # weight for Laplacian term
BETA        = 0.8      # weight for Sobel term
LAP_SIGN    = -1       # -1 => L - ALPHA*Lap, +1 => L + ALPHA*Lap
USE_FULL_GRADIENT = False  # True => (Gx+Gy)/2 ; False => sqrt(Gx^2+Gy^2)
# ----------------------------------------------------------------

def load_color(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                return img, p
    raise FileNotFoundError("fish.jpg not found. Tried:\n  - " + "\n  - ".join(candidates))

def percentile_stretch01(x, low=1.0, high=99.5, gamma=0.8):
    """For nicer visualization only (does not affect computation)."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, low), np.percentile(x, high)
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return np.power(y, gamma)

def main():
    # 1) Read color image
    bgr, used = load_color(["fish.jpg", "./fish.jpg", "/mnt/data/fish.jpg"])
    print(f"[INFO] Loaded: {used}, shape={bgr.shape}")

    # 2) Convert BGR->Lab, get L in [0,1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # 3) Mild denoise on L
    L_blur = cv2.GaussianBlur(L, (0, 0), GAUSS_SIGMA)

    # 4) Sobel on L
    gx = cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3)
    if USE_FULL_GRADIENT:
        sobel_term = 0.5 * (gx + gy)  # can be +/- ; we'll normalize for stability
        # scale to [0,1] for combination
        smin, smax = sobel_term.min(), sobel_term.max()
        sobel_term = (sobel_term - smin) / (smax - smin + 1e-8)
    else:
        sobel_mag = np.sqrt(gx * gx + gy * gy)
        sobel_term = sobel_mag / (sobel_mag.max() + 1e-8)  # [0,1]

    # 5) Laplacian on L (second derivative, +/-)
    lap = cv2.Laplacian(L_blur, cv2.CV_32F, ksize=3)

    # 6) High-boost on L
    L_enh = L + LAP_SIGN * ALPHA * lap + BETA * sobel_term
    L_enh = np.clip(L_enh, 0.0, 1.0)

    # 7) Merge back to color
    lab_out = np.dstack([(L_enh * 255.0).astype(np.uint8), a, b])
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

    # ----- Prepare colorized intermediate displays (for Sobel & Laplacian) -----
    # Sobel display (0..1 -> 0..255 as L) + original a/b to keep green tone
    sobel_L_disp = (np.clip(percentile_stretch01(sobel_term), 0, 1) * 255).astype(np.uint8)
    sobel_lab    = np.dstack([sobel_L_disp, a, b])
    sobel_bgr    = cv2.cvtColor(sobel_lab, cv2.COLOR_Lab2BGR)

    # Laplacian display (map to 0..1 -> 0..255 as L) + original a/b
    lap_norm = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
    lap_L_disp = (np.clip(percentile_stretch01(lap_norm), 0, 1) * 255).astype(np.uint8)
    lap_lab    = np.dstack([lap_L_disp, a, b])
    lap_bgr    = cv2.cvtColor(lap_lab, cv2.COLOR_Lab2BGR)

    # 8) Show three images
    plt.figure(figsize=(10, 12))

    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(sobel_bgr, cv2.COLOR_BGR2RGB))
    title_sobel = "The result of Sobel filter"
    title_sobel += " (full gradient)" if USE_FULL_GRADIENT else " (magnitude)"
    plt.title(title_sobel + " — colorized")
    plt.axis("off")

    plt.subplot(3, 2, 2)
    plt.imshow(cv2.cvtColor(lap_bgr, cv2.COLOR_BGR2RGB))
    plt.title("The result of Laplacian filter — colorized")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB))
    sign_str = "+" if LAP_SIGN > 0 else "-"
    grad_str = "Full_gradient" if USE_FULL_GRADIENT else "|∇L|"
    plt.title(f"Final enhanced image (color preserved)\n"
              f"L {sign_str} {abs(ALPHA)}*Lap(L) + {BETA}*{grad_str}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

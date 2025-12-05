import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10

def compute_psnr(img1, img2):
    """Compute PSNR (higher = better)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / np.sqrt(mse))

def compute_ssim(img1, img2):
    """Compute SSIM (0–1)."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def evaluate_folder(folder):
    # ---------- Load original ----------
    orig_path = os.path.join(folder, "46_GT.png")
    if not os.path.exists(orig_path):
        print("ERROR: 46_GT.png not found!")
        return
    
    original = cv2.imread(orig_path)
    H, W = original.shape[:2]

    print("\n=== PSNR & SSIM with respect to ORIGINAL IMAGE ===\n")
    print(f"{'Image':30}  {'PSNR':>10}  {'SSIM':>10}")
    print("-" * 60)

    # ---------- Loop over all images ----------
    for fname in sorted(os.listdir(folder)):
        if fname == "original_input.png":
            continue  # skip original itself
        
        if fname.endswith(".png") or fname.endswith(".jpg"):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)

            # resize to original size (safety)
            img = cv2.resize(img, (W, H))

            psnr_val = compute_psnr(original, img)
            ssim_val = compute_ssim(original, img)

            print(f"{fname:30}  {psnr_val:10.4f}  {ssim_val:10.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    folder = r"F:\Derain_code_and_github\results_RL\20251205_130412 copy"  # << YOUR FOLDER
    evaluate_folder(folder)

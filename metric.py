import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# === PATH TO YOUR FOLDER ===
folder = r"F:/results/RAIN_1400_native_desm"   # <-- change this if needed

psnr_list = []
ssim_list = []

print("\n=== PSNR & SSIM Evaluation ===\n")
print(f"{'Idx':<6} {'PSNR(dB)':<12} {'SSIM'}")

idx = 1

while True:
    fake_name = f"{str(idx).zfill(4)}_fake_B.png"
    real_name = f"{str(idx).zfill(4)}_real_B.png"

    fake_path = os.path.join(folder, fake_name)
    real_path = os.path.join(folder, real_name)

    # Stop when next pair does not exist
    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        break

    # Read images
    fake = cv2.imread(fake_path)
    real = cv2.imread(real_path)

    if fake is None or real is None:
        print(f"Skipping {idx}: Unable to read images.")
        idx += 1
        continue

    # Convert BGR → RGB
    fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
    real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

    # Ensure same size
    if fake.shape != real.shape:
        print(f"❌ Size mismatch at {idx}. Skipping.")
        idx += 1
        continue

    # Compute PSNR
    psnr_val = peak_signal_noise_ratio(real, fake, data_range=255)

    # Compute SSIM
    ssim_val = structural_similarity(real, fake, channel_axis=2)

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    print(f"{idx:<6} {psnr_val:<12.4f} {ssim_val:.4f}")

    idx += 1

# === FINAL AVERAGES ===
print("\n=== FINAL AVERAGE METRICS ===")

if len(psnr_list) == 0:
    print("❌ No valid image pairs found.")
else:
    print(f"Average PSNR : {np.mean(psnr_list):.4f} dB")
    print(f"Average SSIM : {np.mean(ssim_list):.4f}")
    # top 100 average
    if len(psnr_list) >= 10:
        top100_psnr = np.mean(sorted(psnr_list, reverse=True)[:10])
        top100_ssim = np.mean(sorted(ssim_list, reverse=True)[:10])
        print(f"Top 10 Average PSNR : {top100_psnr:.4f} dB")
        print(f"Top 10 Average SSIM : {top100_ssim:.4f}")

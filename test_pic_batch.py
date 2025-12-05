# import os
# import torch
# import numpy as np
# from options.test_options import TestOptions
# from models import create_model
# from util import util
# import cv2
# from datetime import datetime
# import shutil
# import time

# # === PATHS ===
# input_folder = r"F:\RAIN_1400\rainy_image_dataset\testing\testB"     # Input hazy images
# results_dir = r"F:/results/RAIN_1400_native_desm"    # Output folder
# original_dir = r"F:\RAIN_1400\rainy_image_dataset\testing\testA"      # Ground truth clear images

# os.makedirs(results_dir, exist_ok=True)

# # === MODEL SETTINGS ===
# model_name = "desmoke"
# model_type = "test"
# no_dropout = True

# # === TIMER ===
# total_time = 0.0

# # === DEVICE ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if __name__ == "__main__":

#     # Load TestOptions
#     opt = TestOptions().parse()
#     opt.num_threads = 0
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.no_flip = True
#     opt.display_id = -1

#     opt.results_dir = results_dir
#     opt.name = model_name
#     opt.model = model_type
#     opt.no_dropout = no_dropout

#     # Create & Setup Model
#     model = create_model(opt)
#     model.setup(opt)
#     if opt.eval:
#         model.eval()

#     # Counter for fake_B outputs
#     idx = 1  

#     print("\n=== Processing ALL IMAGES (native resolution) ===\n")

#     for filename in sorted(os.listdir(input_folder)):

#         if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue

#         image_path = os.path.join(input_folder, filename)
#         input_image = cv2.imread(image_path)

#         if input_image is None:
#             print(f"Skipping {filename}: Could not read.")
#             continue

#         # Start timer
#         start_time = time.time()

#         # Convert to tensor (native res)
#         rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#         rgb = np.asarray([rgb])
#         rgb = np.transpose(rgb, (0, 3, 1, 2))
#         data = {"A": torch.FloatTensor(rgb), "A_paths": [image_path]}

#         # Inference
#         model.set_input(data)
#         model.test()

#         # Get result image
#         result = model.get_current_visuals()["fake"]
#         result = util.tensor2im(result)
#         result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

#         # Save fake_B
#         fake_name = f"{str(idx).zfill(4)}_fake_B.png"
#         fake_path = os.path.join(results_dir, fake_name)
#         cv2.imwrite(fake_path, result)

#         # End timer
#         elapsed = time.time() - start_time
#         total_time += elapsed

#         print(f"[{idx}] {filename}  →  {fake_name}  |  Time: {elapsed:.3f} s")

#         idx += 1


#     # === SAVE REAL CLEAR IMAGES (GT) – ALSO IN NATIVE RES ===
#     print("\n=== Saving REAL testB images (native resolution) ===\n")

#     idx = 1  # reset

#     for filename in sorted(os.listdir(original_dir)):

#         if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue

#         src = os.path.join(original_dir, filename)
#         real_img = cv2.imread(src)

#         if real_img is None:
#             print(f"Skipping GT {filename}: Could not read.")
#             continue

#         # Preserve same conversion style
#         real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
#         real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)

#         dest_name = f"{str(idx).zfill(4)}_real_B.png"
#         dest_path = os.path.join(results_dir, dest_name)

#         cv2.imwrite(dest_path, real_img)
#         print(f"[{idx}] Saved GT {filename} → {dest_name}")

#         idx += 1


#     # === FINAL TIMING ===
#     num_images = idx - 1

#     print("\n=== COMPLETED SUCCESSFULLY ===")
#     print(f"Total processing time: {total_time:.3f} seconds")
#     print(f"Average per image: {total_time / num_images:.3f} seconds")



import os
import torch
import numpy as np
from options.test_options import TestOptions
from models import create_model
from util import util
import cv2
from datetime import datetime
import shutil
import time

total_time = 0.0

# === PATHS ===
input_folder = r"F:\RAIN_1400\rainy_image_dataset\testing\testB"     # Input hazy images
results_dir = r"F:/results/RAIN_1400_resized_desm"    # Output folder
original_dir = r"F:\RAIN_1400\rainy_image_dataset\testing\testA" 

os.makedirs(results_dir, exist_ok=True)

# === MODEL SETTINGS ===
model_name = "desmoke"
model_type = "test"
no_dropout = True

# === RESIZE TARGET ===
target_w = 256
target_h = 256

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load TestOptions
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    opt.results_dir = results_dir
    opt.name = model_name
    opt.model = model_type
    opt.no_dropout = no_dropout

    # Create model
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # Counter for output image numbering
    idx = 1

    print("\n=== Processing ALL IMAGES — resized to 256×256 ===\n")

    for filename in sorted(os.listdir(input_folder)):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(input_folder, filename)
        input_image = cv2.imread(image_path)

        if input_image is None:
            print(f"Skipping {filename}: unreadable.")
            continue

        start_time = time.time()

        # Resize BEFORE inference
        resized_input = cv2.resize(input_image, (target_w, target_h))

        # Prepare tensor
        rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)
        rgb = np.asarray([rgb], dtype=np.float32)
        rgb = np.transpose(rgb, (0, 3, 1, 2))
        data = {"A": torch.from_numpy(rgb).to(device), "A_paths": [image_path]}

        # Inference
        model.set_input(data)
        model.test()

        # Get result image
        result = model.get_current_visuals()["fake"]
        result = util.tensor2im(result)
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # SAVE GENERATED CLEAR IMAGE (fake_B)
        fake_name = f"{str(idx).zfill(4)}_fake_B.png"
        fake_path = os.path.join(results_dir, fake_name)
        cv2.imwrite(fake_path, result)

        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        print(f"[{idx}] {filename} → {fake_name}")

        idx += 1


    # === SAVE REAL CLEAR IMAGES (GROUND TRUTH) — ALSO RESIZED ===
    print("\n=== Saving REAL images (resized to 256×256) ===\n")

    idx = 1
    for filename in sorted(os.listdir(original_dir)):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        src = os.path.join(original_dir, filename)
        gt_image = cv2.imread(src)

        if gt_image is None:
            print(f"Skipping {filename}: unreadable.")
            continue

        # Resize GT image
        resized_gt = cv2.resize(gt_image, (target_w, target_h))

        real_name = f"{str(idx).zfill(4)}_real_B.png"
        real_path = os.path.join(results_dir, real_name)
        cv2.imwrite(real_path, resized_gt)

        print(f"[{idx}] GT {filename} → {real_name}")

        idx += 1

    print("\n=== COMPLETED SUCCESSFULLY ===")
    print(f"Total processing time (resized): {total_time:.2f} seconds")
    print(f"Average per image: {total_time/(idx-1):.2f} seconds")

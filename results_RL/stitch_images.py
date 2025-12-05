import os
import cv2
import numpy as np

def add_label(img, text):
    """Adds a black label bar at the top of the image."""
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(labeled, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def make_grid(folder_path, steps_to_use):
    print(f"[GRID] Building comparison grid from: {folder_path}")

    # ---------------------- LOAD MAIN IMAGES ----------------------
    orig_path = os.path.join(folder_path, "original_input.png")
    gan_path  = os.path.join(folder_path, "gan_output.png")
    final_path = os.path.join(folder_path, "rl_step_5.png")

    if not all(os.path.exists(p) for p in [orig_path, gan_path, final_path]):
        print("ERROR: Missing required images in folder!")
        return

    orig  = cv2.imread(orig_path)
    gan   = cv2.imread(gan_path)
    final = cv2.imread(final_path)

    H, W = orig.shape[:2]

    # ---------------------- LOAD SPECIFIED RL STEPS ----------------------
    step_images = []
    for s in steps_to_use:
        p = os.path.join(folder_path, f"rl_step_{s}.png")
        if os.path.exists(p):
            img = cv2.imread(p)
            img = cv2.resize(img, (W, H))
            step_images.append((s, img))
        else:
            print(f"[WARN] RL step {s} not found, skipping.")

    # ---------------------- RESIZE & LABEL ----------------------
    gan = cv2.resize(gan, (W, H))
    final = cv2.resize(final, (W, H))

    tiles = [
        add_label(orig, "Original"),
        add_label(gan, "GAN Output"),
    ]

    # Add steps in order
    for s, img in step_images:
        tiles.append(add_label(img, f"RL Step {s}"))

    tiles.append(add_label(final, "Final RL Output"))

    # ---------------------- BUILD 2-ROW GRID ----------------------
    mid = (len(tiles) + 1) // 2

    row1 = np.hstack(tiles[:mid])
    row2 = np.hstack(tiles[mid:])

    # Equalize widths
    max_width = max(row1.shape[1], row2.shape[1])

    if row1.shape[1] < max_width:
        pad = np.zeros((H + 40, max_width - row1.shape[1], 3), dtype=np.uint8)
        row1 = np.hstack((row1, pad))

    if row2.shape[1] < max_width:
        pad = np.zeros((H + 40, max_width - row2.shape[1], 3), dtype=np.uint8)
        row2 = np.hstack((row2, pad))

    grid = np.vstack((row1, row2))

    # ---------------------- SAVE OUTPUT ----------------------
    out_path = os.path.join(folder_path, "comparison_grid.png")
    cv2.imwrite(out_path, grid)
    print(f"[GRID] Saved at: {out_path}")


# ---------------------- RUN HERE ----------------------
if __name__ == "__main__":
    folder_path = r"results_RL\20251205_130412 copy" # <-- PUT YOUR FOLDER
    steps_to_use = [0, 1, 2, 3, 4]  # <-- LIST WHICH RL STEPS YOU WANT

    make_grid(folder_path, steps_to_use)

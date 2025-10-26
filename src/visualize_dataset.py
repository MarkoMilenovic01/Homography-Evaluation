# ============================================================
# visualize_dataset.py
# ============================================================
# Author: Marko Milenović
# Task: Ocenjevanje homografije (Homography Estimation)
#
# Description:
#   Visualizes and saves examples showing:
#     - Original image with both patch outlines
#     - Warped image where the displaced patch is rectified
#   Each visualization is saved to 'visualizations/' directory.
# ============================================================

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from generate_dataset import CONFIG, random_patch, corners, displaced_corners


# ------------------------------------------------------------
# Drawing Utilities
# ------------------------------------------------------------
def draw_quad(image: np.ndarray, points: np.ndarray, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw a quadrilateral (4-point polygon) on a grayscale image."""
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    points = points.astype(int)
    cv2.polylines(img_color, [points], isClosed=True, color=color, thickness=thickness)
    return img_color


# ------------------------------------------------------------
# Visualization of One Example (and save)
# ------------------------------------------------------------
def visualize_one(image: np.ndarray, save_dir: str, idx: int = 0) -> None:
    """Visualize one example and save it as an image file."""
    ps, mo = CONFIG["PATCH_SIZE"], CONFIG["MAX_OFFSET"]
    h, w = image.shape

    # 1. Select a random patch and its displaced version
    x, y = random_patch(image, ps, mo)
    src = corners(x, y, ps)
    dst, _ = displaced_corners(src, mo)

    # 2. Compute homography (dst → src) and warp the image
    H = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        image, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 3. Draw both quadrilaterals on the original image
    original_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(original_colored, [src.astype(int)], True, (255, 0, 0), 2)  # Blue: original patch
    cv2.polylines(original_colored, [dst.astype(int)], True, (0, 0, 255), 2)  # Red: displaced patch

    # 4. Draw the rectified (square) patch on the warped image
    warped_with_quad = draw_quad(warped, src, color=(255, 0, 0))

    # 5. Plot and save
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(cv2.cvtColor(original_colored, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image\nBlue: original patch | Red: displaced patch")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(warped_with_quad, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Warped Image\n(Displaced patch rectified to square)")
    axes[1].axis("off")

    plt.tight_layout()

    # Create save directory if missing
    os.makedirs(save_dir, exist_ok=True)

    # Build filename
    filename = f"visualization_{idx:03d}_{datetime.now().strftime('%H%M%S')}.png"
    save_path = os.path.join(save_dir, filename)

    # Save instead of show
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved visualization → {save_path}")


# ------------------------------------------------------------
# Visualization Across Multiple Images
# ------------------------------------------------------------
def visualize_random_images(num_examples: int = 10, save_dir: str = "visualizations") -> None:
    """Visualize several random COCO images and save them."""
    data_dir = CONFIG["DATA_PATH"]
    image_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".png"))
    ]

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_examples):
        path = random.choice(image_files)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            visualize_one(img, save_dir, idx=i)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    visualize_random_images(num_examples=10, save_dir="visualizations")

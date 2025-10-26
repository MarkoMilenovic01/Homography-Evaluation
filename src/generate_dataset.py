import os
import cv2
import random
import numpy as np
from tqdm import tqdm


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CONFIG = {
    "DATA_PATH": "data/mini_coco/images",
    "PATCH_SIZE": 64,
    "MAX_OFFSET": 16,
    "PATCHES_PER_IMAGE": 5,
    "NUM_BINS": 21,
    "NUM_IMAGES": 10000,
    "RANDOM_SEED": 42,
}

random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

os.makedirs(CONFIG["DATA_PATH"], exist_ok=True)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def random_patch(image: np.ndarray, patch_size: int, margin: int):
    """Select a random patch within the image, avoiding borders."""
    h, w = image.shape
    x = random.randint(margin, w - patch_size - margin - 1)
    y = random.randint(margin, h - patch_size - margin - 1)
    return x, y


def corners(x: int, y: int, size: int) -> np.ndarray:
    """Return the four corner coordinates of a square patch."""
    return np.array([
        [x, y],
        [x + size, y],
        [x + size, y + size],
        [x, y + size]
    ], dtype=np.float32)


def displaced_corners(src: np.ndarray, max_offset: int):
    """Apply random corner displacements within ±max_offset pixels."""
    d = np.random.randint(-max_offset, max_offset + 1, (4, 2)).astype(np.float32)
    return src + d, d


def quantize_displacements(disp: np.ndarray, max_offset: int, num_bins: int) -> np.ndarray:
    """Quantize continuous displacements into discrete integer bins."""
    norm = (disp + max_offset) / (2 * max_offset)
    return np.rint(norm * (num_bins - 1)).astype(np.int64)


def patch_pair(image: np.ndarray):
    """Generate a single (original, warped) patch pair and its displacement label."""
    ps, mo = CONFIG["PATCH_SIZE"], CONFIG["MAX_OFFSET"]
    h, w = image.shape

    x, y = random_patch(image, ps, mo)
    src = corners(x, y, ps)
    dst, disp = displaced_corners(src, mo)

    H = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_REFLECT)

    A = image[y:y + ps, x:x + ps]
    B = warped[y:y + ps, x:x + ps]
    X = np.stack([A, B], axis=0).astype(np.float32)
    return X, disp


# ------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------
def generate_dataset(mode: str = "train"):
    """Generate the dataset for training or testing."""
    assert mode in ["train", "test"], "mode must be 'train' or 'test'"

    if mode == "train":
        save_dir = "data/generated_train"
        num_images = CONFIG["NUM_IMAGES"]
        patches_per_img = CONFIG["PATCHES_PER_IMAGE"]
        seed = CONFIG["RANDOM_SEED"]
    else:
        save_dir = "data/generated_test"
        num_images = 100
        patches_per_img = 10
        seed = CONFIG["RANDOM_SEED"] + 1

    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    images = [
        os.path.join(CONFIG["DATA_PATH"], f)
        for f in sorted(os.listdir(CONFIG["DATA_PATH"]))[:num_images]
        if f.lower().endswith((".jpg", ".png"))
    ]

    X_all, y_reg_all, y_cls_all = [], [], []

    print(f"Generating {patches_per_img} patches × {len(images)} images ({mode})...")
    for path in tqdm(images):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        for _ in range(patches_per_img):
            X, disp = patch_pair(img)
            disp_flat = disp.flatten().astype(np.float32)
            disp_bins = quantize_displacements(disp_flat, CONFIG["MAX_OFFSET"], CONFIG["NUM_BINS"])

            X_all.append(X)
            y_reg_all.append(disp_flat)
            y_cls_all.append(disp_bins)

    X = np.array(X_all, np.float32)
    y_reg = np.array(y_reg_all, np.float32)
    y_cls = np.array(y_cls_all, np.int64)

    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y_reg.npy"), y_reg)
    np.save(os.path.join(save_dir, "y_cls.npy"), y_cls)

    print(f"Saved {mode} set → {save_dir}")
    print(f"Shapes: X={X.shape}, y_reg={y_reg.shape}, y_cls={y_cls.shape}")
    return X, y_reg, y_cls


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_dataset("train")
    generate_dataset("test")

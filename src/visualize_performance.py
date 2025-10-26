# ============================================================
#   Visualizes examples showing:
#     - White: original patch (vzorec)
#     - Blue:  ground-truth warped patch (popacenje)
#     - Red:   predicted patch (ocena)
#   Each figure also includes the original, warped, and rectified patches below.
# ============================================================

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import HomographyNet
from generate_dataset import CONFIG as GEN_CFG, random_patch, corners, displaced_corners


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = GEN_CFG["DATA_PATH"]
PATCH_SIZE = GEN_CFG["PATCH_SIZE"]
MAX_OFFSET = GEN_CFG["MAX_OFFSET"]
SAVE_DIR = "results"

os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def classification_to_disp(logits, num_bins=21, max_offset=16):
    """Convert class probabilities into continuous displacements."""
    probs = torch.softmax(logits, dim=2)
    bins = torch.linspace(-max_offset, max_offset, num_bins, device=probs.device)
    return torch.sum(probs * bins, dim=2)


def rectify_patch(patch, disp_pred):
    """Rectify a warped patch using predicted displacements."""
    ps = patch.shape[0]
    src = np.array([[0, 0], [ps, 0], [ps, ps], [0, ps]], np.float32)
    dst = src + disp_pred.reshape(4, 2).astype(np.float32)
    H = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(patch, H, (ps, ps), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------------
# Main Visualization
# ------------------------------------------------------------
def visualize_prediction(mode="regression", num_examples=3):
    """Visualize a few random examples of homography predictions."""
    model = HomographyNet(mode=mode, num_bins=GEN_CFG["NUM_BINS"]).to(DEVICE)
    ckpt_path = f"checkpoints/best_{mode}.pth"

    if not os.path.exists(ckpt_path):
        print(f"Model checkpoint not found: {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    img_files = [
        os.path.join(DATA_PATH, f)
        for f in os.listdir(DATA_PATH)
        if f.lower().endswith((".jpg", ".png"))
    ]

    for i in range(num_examples):
        path = np.random.choice(img_files)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        # 1. Sample random patch and ground-truth perturbation
        x, y = random_patch(img, PATCH_SIZE, MAX_OFFSET)
        pts_src = corners(x, y, PATCH_SIZE)
        pts_dst, disp_gt = displaced_corners(pts_src, MAX_OFFSET)

        # 2. Warp image to simulate perturbation
        H_gt = cv2.getPerspectiveTransform(pts_dst, pts_src)
        img_warp = cv2.warpPerspective(img, H_gt, (w, h), borderMode=cv2.BORDER_REFLECT)

        patch_A = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float32)
        patch_B = img_warp[y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float32)

        inp = np.stack([patch_A, patch_B])[np.newaxis, :]
        inp_t = torch.from_numpy(inp).to(DEVICE, dtype=torch.float32)

        # 3. Predict displacement
        with torch.no_grad():
            out = model(inp_t)
            if mode == "regression":
                disp_pred = out.cpu().numpy()[0]
            else:
                disp_pred = classification_to_disp(out).cpu().numpy()[0]

        pts_pred = pts_src + disp_pred.reshape(4, 2)

        # 4. Rectify the warped patch
        patch_rectified = rectify_patch(patch_B, disp_pred)

        # 5. Draw quadrilaterals
        img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_col, [pts_src.astype(int)], True, (255, 255, 255), 2)  # original (white)
        cv2.polylines(img_col, [pts_dst.astype(int)], True, (255, 0, 0), 2)      # ground truth (blue)
        cv2.polylines(img_col, [pts_pred.astype(int)], True, (0, 0, 255), 2)     # predicted (red)

        # Add small legend
        legend = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.putText(legend, "vzorec", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(legend, "popacenje", (170, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(legend, "ocena", (360, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        img_full = np.vstack([img_col, legend])

        # 6. Layout: top full image + bottom patches
        fig, axes = plt.subplots(2, 3, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
        fig.subplots_adjust(hspace=0.3)

        # Merge top row into one large image
        for j in range(3):
            axes[0, j].remove()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_top.imshow(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
        ax_top.axis("off")
        ax_top.set_title(f"Visualization ({mode})", fontsize=12, pad=10)

        patches = [patch_A, patch_B, patch_rectified]
        labels = ["original", "warped", "rectified"]

        for j, (p, label) in enumerate(zip(patches, labels)):
            axes[1, j].imshow(p, cmap="gray", vmin=0, vmax=255)
            axes[1, j].set_title(label, fontsize=10)
            axes[1, j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/triple_{mode}_{i:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved {num_examples} visualizations for {mode} mode in '{SAVE_DIR}/'.")


# ------------------------------------------------------------
# Run Both Modes
# ------------------------------------------------------------
if __name__ == "__main__":
    for mode in ["regression", "classification"]:
        visualize_prediction(mode=mode, num_examples=10)

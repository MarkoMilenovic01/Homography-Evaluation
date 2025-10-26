# ============================================================
# evaluate_classical.py
# ============================================================
# Author: Marko Milenović
# Task: Homography Estimation
#
# Description:
#   Evaluates traditional feature-based homography estimation
#   (SIFT, SURF, or ORB) on synthetic patch pairs.
#
#   Computes RMSE of corner displacements in the same way as
#   the neural network evaluation, for direct comparison.
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from generate_dataset import CONFIG as GEN_CFG, random_patch, corners, displaced_corners


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CONFIG = {
    "num_samples": 100,
    "patch_size": 256,
    "max_offset": 64,
    "detector": "SIFT",       # SIFT, SURF, or ORB
    "save_dir": "results"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def compute_rmse(pred_pts, true_pts):
    """Compute RMSE between predicted and ground-truth corner points."""
    return np.sqrt(np.mean((pred_pts - true_pts) ** 2))


# ------------------------------------------------------------------
# estimate_homography(imgA, imgB, method="SIFT")
# ------------------------------------------------------------------
# Estimates the 3×3 homography matrix H that best aligns two image
# patches, using classical feature-based matching.
#
# Steps:
#   1. Detect distinctive keypoints in both patches (e.g. corners,
#      blobs, edges) using SIFT, SURF, or ORB.
#   2. Extract feature descriptors around each keypoint that describe
#      the local image texture (e.g. gradient orientations for SIFT).
#   3. Match descriptors between both patches using a brute-force
#      nearest-neighbor search (BFMatcher). If there are fewer than
#      4 matches, the homography cannot be estimated — we return
#      the identity matrix in that case.
#   4. Use the RANSAC algorithm to robustly estimate the 3×3
#      projective transformation H from the matched points while
#      rejecting outliers.
#   5. Return H, which maps pixel coordinates from imgA to imgB.
#
# This approach is the traditional computer-vision method used
# in image stitching, panorama creation, and geometric alignment.
# ------------------------------------------------------------------
def estimate_homography(imgA, imgB, method="SIFT"):
    if method == "SIFT":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(2000)
        norm_type = cv2.NORM_HAMMING

    # Detect and compute local feature descriptors
    kptsA, descA = detector.detectAndCompute(imgA, None)
    kptsB, descB = detector.detectAndCompute(imgB, None)

    # Return identity if not enough valid keypoints
    if descA is None or descB is None or len(kptsA) < 4 or len(kptsB) < 4:
        return np.eye(3)

    # Match descriptors using brute-force matching
    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = matcher.match(descA, descB)
    if len(matches) < 4:
        return np.eye(3)

    # Extract matched keypoint coordinates
    ptsA = np.float32([kptsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kptsB[m.trainIdx].pt for m in matches])

    # Compute homography using RANSAC to reject outliers
    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
    if H is None or np.isnan(H).any() or np.isinf(H).any():
        return np.eye(3)

    return H


# ------------------------------------------------------------------
# evaluate_classical(cfg)
# ------------------------------------------------------------------
# Evaluates how accurately a classical feature-based method
# (SIFT, SURF, or ORB) can recover homographies on synthetic data.
#
# Process overview:
#   1. Load real images from the COCO subset used for data generation.
#   2. For each iteration, select a random image and sample a random
#      square patch of a chosen size (e.g. 256×256).
#   3. Randomly perturb the four corner points of this patch within
#      ±max_offset pixels to create a “warped” version.
#   4. Apply a known ground-truth homography to warp the full image,
#      then extract the original and warped patches.
#   5. Use `estimate_homography()` to recover H from these two patches.
#   6. Apply the predicted H to the original corner points and measure
#      how close they are to the true displaced corners using RMSE.
#   7. Repeat for many samples, then compute the mean and standard
#      deviation of RMSE over all examples.
#   8. Save both numeric results and a histogram of errors.
#
# This function quantitatively compares traditional geometric
# matching methods with learned (CNN-based) homography estimation.
# ------------------------------------------------------------------
def evaluate_classical(cfg):
    ps = cfg["patch_size"]
    max_offset = cfg["max_offset"]
    detector = cfg["detector"]

    print(f"\nEvaluating classical method: {detector}")
    rmses = []

    image_files = [
        os.path.join(GEN_CFG["DATA_PATH"], f)
        for f in os.listdir(GEN_CFG["DATA_PATH"])
        if f.lower().endswith((".jpg", ".png"))
    ]

    valid = 0
    attempt = 0
    pbar = tqdm(total=cfg["num_samples"], desc="Processing", ncols=100)

    # Generate synthetic patch pairs and measure estimation error
    while valid < cfg["num_samples"] and attempt < cfg["num_samples"] * 10:
        attempt += 1
        img_path = np.random.choice(image_files)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        if h < ps + 2 * max_offset or w < ps + 2 * max_offset:
            continue

        # Generate random source and warped patches
        x, y = random_patch(img, ps, max_offset)
        pts_src = corners(x, y, ps)
        pts_dst, _ = displaced_corners(pts_src, max_offset)

        # Create ground-truth warped image using known homography
        H_gt = cv2.getPerspectiveTransform(pts_dst, pts_src)
        img_warp = cv2.warpPerspective(img, H_gt, (w, h), borderMode=cv2.BORDER_REFLECT)

        patch_A = img[y:y + ps, x:x + ps]
        patch_B = img_warp[y:y + ps, x:x + ps]

        # Estimate homography using classical detector
        H_est = estimate_homography(patch_A, patch_B, method=detector)

        # Compute RMSE between predicted and true corner locations
        pts_src_local = pts_src - np.array([[x, y]])
        pts_dst_local = pts_dst - np.array([[x, y]])
        pts_pred_local = cv2.perspectiveTransform(
            pts_src_local.reshape(-1, 1, 2), H_est
        ).reshape(4, 2)
        rmse = compute_rmse(pts_pred_local, pts_dst_local) / 4.0
        rmses.append(rmse)

        valid += 1
        pbar.update(1)

    pbar.close()

    # Compute and save summary statistics
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(f"RMSE ({detector}): mean = {mean_rmse:.4f}, std = {std_rmse:.4f}")

    np.savetxt(
        os.path.join(cfg["save_dir"], f"rmse_classical_{detector.lower()}.txt"),
        [mean_rmse, std_rmse],
        fmt="%.4f"
    )

    # Save RMSE histogram plot
    plt.figure(figsize=(7, 4))
    plt.hist(rmses, bins=20, color="cornflowerblue", edgecolor="black")
    plt.title(f"RMSE Histogram ({detector})")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg["save_dir"], f"errors_classical_{detector.lower()}.png"),
        dpi=150
    )
    plt.close()

    print("Classical evaluation complete.\n")


# Run evaluation for chosen detector(s)
if __name__ == "__main__":
    for detector in ["SIFT"]:
        CONFIG["detector"] = detector
        evaluate_classical(CONFIG)

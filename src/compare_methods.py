# ============================================================
#   Loads RMSE results from different homography estimation
#   methods (regression, classification, and SIFT) and compares
#   them quantitatively and visually.
#
#   Produces summary statistics (mean ± std) and a boxplot
#   visualization for inclusion in evaluation reports.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RESULTS_DIR = r"C:\Users\marko\Desktop\Master\RV\Homography\results"
FILES = {
    "Regression": "rmse_regression.txt",
    "Classification": "rmse_classification.txt",
    "SIFT": "rmse_classical_sift.txt",
}

SAVE_BOX = os.path.join(RESULTS_DIR, "compare_boxplot.png")
SAVE_HIST = os.path.join(RESULTS_DIR, "compare_histogram.png")


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
def load_rmse(path):
    """
    Load numeric RMSE values from a file, ignoring any text or symbols.
    Returns a NumPy array of floats.
    """
    values = []
    with open(path, "r") as f:
        for line in f:
            for token in line.replace(":", " ").replace(",", " ").split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue
    return np.array(values, dtype=float)


# ------------------------------------------------------------
# Load RMSE data
# ------------------------------------------------------------
rmse_data, rmse_means, rmse_stds = {}, {}, {}

for name, file in FILES.items():
    path = os.path.join(RESULTS_DIR, file)
    vals = load_rmse(path)
    rmse_data[name] = vals

    # If only two values are stored, interpret them as mean and std
    if len(vals) == 2:
        rmse_means[name], rmse_stds[name] = vals[0], vals[1]
    else:
        rmse_means[name], rmse_stds[name] = np.mean(vals), np.std(vals)

    print(f"{name:<15} mean = {rmse_means[name]:.4f}, std = {rmse_stds[name]:.4f}")


# ------------------------------------------------------------
# Print RMSE summary
# ------------------------------------------------------------
print("\nRMSE SUMMARY")
print("────────────────────────────")
for n in FILES.keys():
    print(f"{n:<15} mean = {rmse_means[n]:7.4f}, std = {rmse_stds[n]:7.4f}")


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.title("RMSE Boxplot Comparison", fontsize=13)
plt.boxplot(
    [rmse_data[m] for m in FILES.keys()],
    tick_labels=list(FILES.keys()),
    patch_artist=True,
    boxprops=dict(facecolor="#d9e2f3", color="black"),
    medianprops=dict(color="red", linewidth=2),
)
plt.ylabel("RMSE", fontsize=11)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(SAVE_BOX, dpi=150)
plt.close()
print(f"Saved boxplot → {SAVE_BOX}")


# ------------------------------------------------------------
# Final comparison table
# ------------------------------------------------------------
print("\nFINAL COMPARISON TABLE")
print("────────────────────────────")
for n in FILES.keys():
    print(f"{n:<15} {rmse_means[n]:7.4f} ± {rmse_stds[n]:.4f}")

print("\nComparison complete. Results saved in 'results/' folder.\n")

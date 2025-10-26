# ============================================================
# compare_methods_fixed.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
RESULTS_DIR = r"C:\Users\marko\Desktop\Master\RV\Homography\results"
FILES = {
    "Regression": "rmse_regression.txt",
    "Classification": "rmse_classification.txt",
    "SIFT": "rmse_classical_sift.txt"
}

SAVE_BOX = os.path.join(RESULTS_DIR, "compare_boxplot.png")
SAVE_HIST = os.path.join(RESULTS_DIR, "compare_histogram.png")

# ------------------------------------------------------------
# LOAD RMSE FILES
# ------------------------------------------------------------
def load_rmse(path):
    """Load numeric values from file, ignoring text."""
    values = []
    with open(path, "r") as f:
        for line in f:
            for token in line.replace(":", " ").replace(",", " ").split():
                try:
                    values.append(float(token))
                except ValueError:
                    pass
    return np.array(values, dtype=float)

rmse_data, rmse_means, rmse_stds = {}, {}, {}

for name, file in FILES.items():
    path = os.path.join(RESULTS_DIR, file)
    vals = load_rmse(path)
    rmse_data[name] = vals

    # 🧠 New logic:
    if len(vals) == 2:
        rmse_means[name], rmse_stds[name] = vals[0], vals[1]
    else:
        rmse_means[name], rmse_stds[name] = np.mean(vals), np.std(vals)

    print(f"{name:<15} mean = {rmse_means[name]:.4f}, std = {rmse_stds[name]:.4f}")

# ------------------------------------------------------------
# PRINT SUMMARY
# ------------------------------------------------------------
print("\n📊 RMSE SUMMARY")
print("────────────────────────────")
for n in FILES.keys():
    print(f"{n:<15} mean = {rmse_means[n]:7.4f}, std = {rmse_stds[n]:7.4f}")

# ------------------------------------------------------------
# PLOTS
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
print(f"✅ Saved boxplot → {SAVE_BOX}")


# ------------------------------------------------------------
# FINAL TABLE
# ------------------------------------------------------------
print("\n📈 FINAL COMPARISON TABLE")
print("────────────────────────────")
for n in FILES.keys():
    print(f"{n:<15} {rmse_means[n]:7.4f} ± {rmse_stds[n]:.4f}")

print("\n🎯 Comparison complete! Results saved in 'results/' folder.\n")

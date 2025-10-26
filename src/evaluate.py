# ============================================================
#   Evaluates trained regression and classification models on
#   unseen test data. Computes RMSE and generates comparison plots.
# ============================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import HomographyNet
from generate_dataset import CONFIG as GEN_CFG


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CONFIG = {
    "test_dir": "data/generated_test",
    "num_bins": GEN_CFG["NUM_BINS"],
    "max_offset": GEN_CFG["MAX_OFFSET"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "save_dir": "results"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def bin_to_disp(bin_idx, num_bins, max_offset):
    """Convert predicted bin index to continuous displacement value."""
    return (bin_idx / (num_bins - 1)) * (2 * max_offset) - max_offset


def compute_rmse(pred, gt):
    """Compute RMSE per sample."""
    return np.sqrt(np.mean((pred - gt) ** 2, axis=1))


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate_model(mode, cfg):
    """Run evaluation for the given mode ('regression' or 'classification')."""
    device = cfg["device"]
    model_path = f"checkpoints/best_{mode}.pth"

    print(f"\nEvaluating {mode.upper()} model...")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    model = HomographyNet(mode=mode, num_bins=cfg["num_bins"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    X = np.load(os.path.join(cfg["test_dir"], "X.npy"))
    y_reg = np.load(os.path.join(cfg["test_dir"], "y_reg.npy"))

    preds_all = []
    X_tensor = torch.from_numpy(X).to(device)

    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), cfg["batch_size"])):
            batch = X_tensor[i:i + cfg["batch_size"]]
            out = model(batch)

            if mode == "regression":
                preds = out.cpu().numpy()
            else:
                logits = out.cpu().numpy()
                pred_bins = np.argmax(logits, axis=2)
                preds = bin_to_disp(pred_bins, cfg["num_bins"], cfg["max_offset"])

            preds_all.append(preds)

    preds_all = np.concatenate(preds_all, axis=0)
    rmse = compute_rmse(preds_all, y_reg)

    mean_rmse = float(np.mean(rmse))
    std_rmse = float(np.std(rmse))

    print(f"{mode.capitalize()} RMSE: mean = {mean_rmse:.2f}, std = {std_rmse:.2f}")

    np.savetxt(
        os.path.join(cfg["save_dir"], f"rmse_{mode}.txt"),
        [mean_rmse, std_rmse],
        fmt="%.4f"
    )

    return rmse


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_rmse_distributions(rmse_reg, rmse_cls, save_dir):
    """Create and save histograms, boxplots, and violin plots comparing RMSE."""
    # Individual histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(rmse_reg, bins=30, color="royalblue", edgecolor="black", alpha=0.7)
    plt.title("RMSE Histogram (Regression)")
    plt.xlabel("RMSE")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(rmse_cls, bins=30, color="tomato", edgecolor="black", alpha=0.7)
    plt.title("RMSE Histogram (Classification)")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histograms_individual.png"), dpi=150)
    plt.close()

    # Combined histogram
    plt.figure(figsize=(6, 4))
    plt.hist(rmse_reg, bins=30, alpha=0.6, label="Regression", color="royalblue")
    plt.hist(rmse_cls, bins=30, alpha=0.6, label="Classification", color="tomato")
    plt.legend()
    plt.title("RMSE Distribution Comparison")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histogram_combined.png"), dpi=150)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot([rmse_reg, rmse_cls],
                labels=["Regression", "Classification"],
                patch_artist=True,
                boxprops=dict(facecolor="lightgray", color="black"),
                medianprops=dict(color="red", linewidth=2))
    plt.ylabel("RMSE")
    plt.title("RMSE Boxplot Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "boxplot_comparison.png"), dpi=150)
    plt.close()

    # Violin plot
    plt.figure(figsize=(6, 5))
    plt.violinplot([rmse_reg, rmse_cls], showmeans=True, showextrema=False)
    plt.xticks([1, 2], ["Regression", "Classification"])
    plt.ylabel("RMSE")
    plt.title("RMSE Distribution (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "violin_comparison.png"), dpi=150)
    plt.close()

    print(f"Plots saved to: {os.path.abspath(save_dir)}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    rmse_reg = evaluate_model("regression", CONFIG)
    rmse_cls = evaluate_model("classification", CONFIG)

    if rmse_reg is not None and rmse_cls is not None:
        plot_rmse_distributions(rmse_reg, rmse_cls, CONFIG["save_dir"])
        print("\nEvaluation complete. RMSE plots and summaries saved.")

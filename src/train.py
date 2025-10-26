import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import HomographyNet


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CONFIG = {
    "train_dir": "data/generated_train",
    "num_bins": 21,
    "batch_size": 64,
    "epochs": 40,
    "lr": 1e-4,
    "save_dir": "checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "use_amp": True,
}


# ------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------
def loss_regression(pred, target):
    """Root Mean Square Error (RMSE) loss."""
    return torch.sqrt(F.mse_loss(pred, target))


def loss_classification(logits, target):
    """Cross-entropy loss applied to each of the 8 displacement components."""
    B, K, C = logits.shape
    return nn.CrossEntropyLoss()(logits.view(B * K, C), target.view(B * K))


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class HomographyDataset(Dataset):
    def __init__(self, data_dir):
        self.X = np.load(os.path.join(data_dir, "X.npy"))
        self.y_reg = np.load(os.path.join(data_dir, "y_reg.npy"))
        self.y_cls = np.load(os.path.join(data_dir, "y_cls.npy"))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])
        y_reg = torch.from_numpy(self.y_reg[idx])
        y_cls = torch.from_numpy(self.y_cls[idx])
        return X, y_reg, y_cls


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_model(mode, dataset, cfg):
    """Train the network in the selected mode."""
    device = cfg["device"]
    os.makedirs(cfg["save_dir"], exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = HomographyNet(mode=mode, num_bins=cfg["num_bins"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"])

    print(f"\nTraining HomographyNet in {mode.upper()} mode on {device}...")

    best_loss = float("inf")
    losses = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for X, y_reg, y_cls in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False):
            X = X.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg["use_amp"]):
                preds = model(X)
                if mode == "regression":
                    loss = loss_regression(preds, y_reg)
                else:
                    loss = loss_classification(preds, y_cls.long())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch:03d}: loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], f"best_{mode}.pth"))
            print(f"Best {mode} model saved (epoch {epoch}, loss={avg_loss:.4f})")

    # Save training loss plot
    plt.figure()
    plt.plot(losses, label=f"{mode} loss")
    plt.title(f"Training Loss ({mode})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(cfg["save_dir"], f"loss_{mode}.png"))
    plt.close()

    print(f"Training for {mode} mode finished. Final loss: {best_loss:.4f}\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cfg = CONFIG
    dataset = HomographyDataset(cfg["train_dir"])

    train_model("regression", dataset, cfg)
    train_model("classification", dataset, cfg)


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

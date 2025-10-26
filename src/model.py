# ============================================================
# Architecture:
#     Input (2×64×64) →
#     [2× ResBlock(64)] → MaxPool →
#     [2× ResBlock(64)] → MaxPool →
#     [2× ResBlock(128)] → MaxPool →
#     [2× ResBlock(128)] → MaxPool →
#     FC(2048 → 512) → BN → Dropout → ReLU
#
# Heads:
#   • Regression:      FC(512 → 8)
#   • Classification:  FC(512 → 8×num_bins) → reshape(B, 8, num_bins)
#
# Losses:
#   • Regression:      RMSE
#   • Classification:  CrossEntropy (per displacement)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Residual Block
# ------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block: Conv3×3 → BN → ReLU → Dropout → Conv3×3 → BN → Add(skip) → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_drop)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        return self.relu(out + identity)


# ------------------------------------------------------------
# CNN Backbone
# ------------------------------------------------------------
class Backbone(nn.Module):
    """CNN encoder shared between regression and classification heads."""

    def __init__(self, in_ch: int = 2, p_drop: float = 0.5):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                ResBlock(c_in, c_out),
                ResBlock(c_out, c_out),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential(
            block(in_ch, 64),    # 64×32×32
            block(64, 64),       # 64×16×16
            block(64, 128),      # 128×8×8
            block(128, 128)      # 128×4×4
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p_drop),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.fc(x)  # (B, 512)


# ------------------------------------------------------------
# Output Heads
# ------------------------------------------------------------
class RegressionHead(nn.Module):
    """Predicts 8 continuous corner displacements."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ClassificationHead(nn.Module):
    """Predicts 8×num_bins logits (8 displacement components, each quantized)."""

    def __init__(self, num_bins: int = 21):
        super().__init__()
        self.num_bins = num_bins
        self.fc = nn.Linear(512, 8 * num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)  # (B, 8*num_bins)
        return logits.view(x.size(0), 8, self.num_bins)  # (B, 8, num_bins)

    @staticmethod
    def softmax_eval(logits: torch.Tensor) -> torch.Tensor:
        """Apply softmax along bins for evaluation."""
        return F.softmax(logits, dim=2)


# ------------------------------------------------------------
# Main Model
# ------------------------------------------------------------
class HomographyNet(nn.Module):
    """Homography Estimation Network (regression or classification)."""

    def __init__(self, mode: str = "regression", num_bins: int = 21):
        super().__init__()
        self.backbone = Backbone()
        self.mode = mode

        if mode == "regression":
            self.head = RegressionHead()
        elif mode == "classification":
            self.head = ClassificationHead(num_bins)
        else:
            raise ValueError("Mode must be 'regression' or 'classification'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------
def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error loss for regression."""
    return torch.sqrt(F.mse_loss(pred, target))


def classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss across 8 displacement dimensions."""
    B, K, C = logits.shape
    return nn.CrossEntropyLoss()(logits.view(B * K, C), target.view(B * K))


# ------------------------------------------------------------
# Self-Test
# ------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(4, 2, 64, 64)

    print("=== REGRESSION TEST ===")
    model_r = HomographyNet(mode="regression")
    y_r = model_r(x)
    print(model_r)
    print("Output:", y_r.shape)

    print("\n=== CLASSIFICATION TEST ===")
    model_c = HomographyNet(mode="classification", num_bins=21)
    y_c = model_c(x)
    print(model_c)
    print("Output (logits):", y_c.shape)

    probs = ClassificationHead.softmax_eval(y_c)
    print("Sum of probabilities (sample 0):", probs[0].sum(dim=1))

    delta_x1_probs = probs[0, 0].detach().cpu().numpy()
    print("\n=== Probabilities for Δx₁ (first sample) ===")
    for i, p in enumerate(delta_x1_probs):
        print(f"Bin {i:02d}: {p:.4f}")

    print(f"Sum = {delta_x1_probs.sum():.4f}")
    pred_bin = delta_x1_probs.argmax()
    print(f"Predicted bin: {pred_bin}")

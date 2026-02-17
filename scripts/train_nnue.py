#!/usr/bin/env python3
"""
Train an NNUE evaluation network from Stockfish-labeled positions.

Architecture (dual-perspective, following Stockfish pattern):
    Input: 768 sparse binary features (per perspective)
    Feature transformer: Linear(768, 128) with shared weights
    Combine: concat(SCReLU(stm_acc), SCReLU(nstm_acc)) -> 256
    Output: Linear(256, 1)
    Parameters: ~100K

Loss: Sigmoid-scaled MSE + WDL cross-entropy

Usage:
    uv run --extra nn python scripts/train_nnue.py \
        --data data/training_data.npz \
        --output moonfish/models/nnue_v1.npz \
        --epochs 20
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("PyTorch is required for training. Install with: uv pip install torch")
    sys.exit(1)


NUM_FEATURES = 768
HIDDEN_SIZE = 128
SCORE_SCALE = 400.0  # Stockfish standard sigmoid scaling


class SCReLU(nn.Module):
    """Squared Clipped ReLU: screlu(x) = clamp(x, 0, 1)^2"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0) ** 2


class NNUEModel(nn.Module):
    """
    Dual-perspective NNUE model.

    The feature transformer is shared between both perspectives.
    During training, we construct both perspectives from the input features.
    """

    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(NUM_FEATURES, HIDDEN_SIZE)
        self.screlu = SCReLU()
        self.output = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        stm_is_white: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            white_features: (batch, 768) binary features from white's perspective
            black_features: (batch, 768) binary features from black's perspective
            stm_is_white: (batch,) bool tensor, True if side to move is white
        """
        white_acc = self.ft(white_features)
        black_acc = self.ft(black_features)

        white_out = self.screlu(white_acc)
        black_out = self.screlu(black_acc)

        # Assemble dual perspective: [stm, nstm]
        stm_mask = stm_is_white.unsqueeze(1).float()
        stm = white_out * stm_mask + black_out * (1 - stm_mask)
        nstm = black_out * stm_mask + white_out * (1 - stm_mask)

        hidden = torch.cat([stm, nstm], dim=1)
        return self.output(hidden).squeeze(1)


def feature_index(piece_type: int, color: int, square: int) -> int:
    """White-perspective feature index."""
    return (piece_type - 1) * 128 + color * 64 + square


def feature_index_flipped(piece_type: int, color: int, square: int) -> int:
    """Black-perspective feature index (flip colors and squares)."""
    flipped_color = 1 - color
    flipped_square = square ^ 56
    return (piece_type - 1) * 128 + flipped_color * 64 + flipped_square


class NNUEDataset(Dataset):
    """Dataset for NNUE training from generated .npz data."""

    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.feature_indices = data["feature_indices"]  # (N, max_features)
        self.feature_counts = data["feature_counts"]  # (N,)
        self.scores = data["scores"].astype(np.float32)  # (N,)
        self.wdl_probs = data["wdl_probs"].astype(np.float32)  # (N,)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int):
        # Build white-perspective and black-perspective feature vectors
        white_features = np.zeros(NUM_FEATURES, dtype=np.float32)
        black_features = np.zeros(NUM_FEATURES, dtype=np.float32)

        count = self.feature_counts[idx]
        indices = self.feature_indices[idx, :count]

        for w_idx in indices:
            if w_idx < 0:
                continue
            white_features[w_idx] = 1.0

            # Decode the feature index to get piece info
            pt_idx = w_idx // 128  # 0-5 (piece type - 1)
            remainder = w_idx % 128
            color = remainder // 64  # 0=white, 1=black
            square = remainder % 64

            # Compute black-perspective index
            b_idx = feature_index_flipped(pt_idx + 1, color, square)
            black_features[b_idx] = 1.0

        score = self.scores[idx]
        wdl = self.wdl_probs[idx]

        # Determine side to move from score perspective
        # The scores in training data are from white's perspective
        # For now, we treat all as white-to-move and the model learns
        # the perspective via the dual accumulator
        stm_is_white = True

        return (
            torch.from_numpy(white_features),
            torch.from_numpy(black_features),
            torch.tensor(stm_is_white, dtype=torch.bool),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(wdl, dtype=torch.float32),
        )


def sigmoid_scale(x: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid scaling with Stockfish standard factor."""
    return torch.sigmoid(x / SCORE_SCALE)


def compute_loss(
    predicted: torch.Tensor,
    target_score: torch.Tensor,
    target_wdl: torch.Tensor,
    wdl_lambda: float = 0.5,
) -> torch.Tensor:
    """
    Sigmoid-scaled MSE + WDL cross-entropy loss.

    pred_sigmoid = sigmoid(predicted / 400)
    target_sigmoid = sigmoid(target_score / 400)
    loss = (1 - lambda) * MSE(pred_sigmoid, target_sigmoid)
         + lambda * BCE(pred_sigmoid, target_wdl)
    """
    pred_sig = sigmoid_scale(predicted)
    target_sig = sigmoid_scale(target_score)

    # Sigmoid-scaled MSE
    mse_loss = torch.mean((pred_sig - target_sig) ** 2)

    # WDL binary cross-entropy (clamped to avoid log(0))
    eps = 1e-7
    pred_clamped = torch.clamp(pred_sig, eps, 1 - eps)
    wdl_loss = -torch.mean(
        target_wdl * torch.log(pred_clamped)
        + (1 - target_wdl) * torch.log(1 - pred_clamped)
    )

    return (1 - wdl_lambda) * mse_loss + wdl_lambda * wdl_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading data from {args.data}...")
    dataset = NNUEDataset(args.data)
    print(f"Total positions: {len(dataset)}")

    # Train/val split
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, args.workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.workers),
        pin_memory=True,
    )

    # Model
    model = NNUEModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # LR schedule: reduce by 3x every 3 epochs
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=1.0 / 3.0
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            white_feat, black_feat, stm, score, wdl = [
                b.to(device) for b in batch
            ]

            optimizer.zero_grad()
            predicted = model(white_feat, black_feat, stm)
            loss = compute_loss(predicted, score, wdl, args.wdl_lambda)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                white_feat, black_feat, stm, score, wdl = [
                    b.to(device) for b in batch
                ]
                predicted = model(white_feat, black_feat, stm)
                loss = compute_loss(predicted, score, wdl, args.wdl_lambda)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={lr:.6f}"
        )

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model weights in numpy format
            save_weights(model, args.output)
            print(f"  -> Best model saved to {args.output}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")


def save_weights(model: NNUEModel, output_path: str):
    """Export model weights to numpy .npz format for inference."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()

    # Feature transformer: w1 = (768, 128), b1 = (128,)
    w1 = state["ft.weight"].cpu().numpy().T  # PyTorch stores (out, in), we want (in, out)
    b1 = state["ft.bias"].cpu().numpy()

    # Output layer: w2 = (256, 1), b2 = (1,)
    w2 = state["output.weight"].cpu().numpy().T  # (256, 1)
    b2 = state["output.bias"].cpu().numpy()

    np.savez(
        str(output),
        w1=w1.astype(np.float32),
        b1=b1.astype(np.float32),
        w2=w2.astype(np.float32),
        b2=b2.astype(np.float32),
    )

    file_size = output.stat().st_size / 1024
    print(f"  Weights saved: w1={w1.shape}, b1={b1.shape}, w2={w2.shape}, b2={b2.shape}")
    print(f"  File size: {file_size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Train NNUE evaluation network")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training data .npz file",
    )
    parser.add_argument(
        "--output", type=str, default="moonfish/models/nnue_v1.npz",
        help="Output path for trained weights",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--wdl-lambda", type=float, default=0.5,
        help="Weight for WDL loss (0=pure MSE, 1=pure WDL)",
    )
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

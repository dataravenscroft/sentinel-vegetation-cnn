"""
Training script for EuroSAT vegetation / land-cover classification.

Usage (from the project root):
    python -m src.train
    python -m src.train --architecture resnet18 --epochs 20
    python -m src.train --data_dir /path/to/data --batch_size 128

The script:
  1. Downloads EuroSAT (if not already cached) via torchvision.
  2. Splits the dataset into train / val / test and saves the index file
     to results/split_indices.json for reproducibility.
  3. Trains the selected model with AdamW + cosine annealing LR schedule.
  4. Saves the best-val-accuracy checkpoint to models/<arch>_best.pt.
  5. Saves per-epoch train/val loss and accuracy to results/<arch>_history.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import get_data_loaders, save_split_indices
from .model import build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        # Gradient clipping prevents occasional instability with label smoothing.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)

    # Prefer GPU > Apple MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    models_dir  = Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True,  exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading EuroSAT dataset...")
    (
        train_loader, val_loader, test_loader,
        class_names,
        train_idx, val_idx, test_idx,
    ) = get_data_loaders(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
        image_size = args.image_size,
        train_frac = args.train_frac,
        val_frac   = args.val_frac,
        seed       = args.seed,
        download   = True,
    )

    split_path = results_dir / "split_indices.json"
    save_split_indices(train_idx, val_idx, test_idx, split_path)
    print(
        f"  Classes : {len(class_names)}\n"
        f"  Train   : {len(train_idx):,}\n"
        f"  Val     : {len(val_idx):,}\n"
        f"  Test    : {len(test_idx):,}\n"
        f"  Splits saved to {split_path}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        architecture = args.architecture,
        num_classes  = len(class_names),
        in_channels  = 3,
        dropout      = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.architecture}  ({n_params:,} trainable parameters)")

    # Label smoothing reduces overconfidence and has been shown to improve
    # generalisation on EuroSAT-scale problems.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing decays the learning rate smoothly to near-zero,
    # avoiding the need to manually tune step-decay schedules.
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }
    best_val_acc    = 0.0
    patience_counter = 0
    best_ckpt_path  = models_dir / f"{args.architecture}_best.pt"

    header = (
        f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}"
        f"  {'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(current_lr)

        improved = vl_acc > best_val_acc
        if improved:
            best_val_acc = vl_acc
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc":         best_val_acc,
                    "val_loss":        vl_loss,
                    "class_names":     class_names,
                    "args":            vars(args),
                },
                best_ckpt_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        marker = "  *" if improved else ""
        print(
            f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>10.4f}"
            f"  {vl_loss:>10.4f}  {vl_acc:>9.4f}  {current_lr:>10.2e}{marker}"
        )

        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no val-acc improvement for {args.early_stopping} epochs)."
            )
            break

    elapsed = time.time() - t_start
    print(
        f"\nTraining complete in {elapsed / 60:.1f} min  |  "
        f"Best val acc: {best_val_acc:.4f}  |  Checkpoint: {best_ckpt_path}"
    )

    # ── Save history ─────────────────────────────────────────────────────────
    history_path = results_dir / f"{args.architecture}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an EuroSAT vegetation / land-cover classifier."
    )
    # Paths
    p.add_argument("--data_dir",    default="data",    help="Root dir for dataset download")
    p.add_argument("--results_dir", default="results", help="Directory for metrics / history JSON")
    p.add_argument("--models_dir",  default="models",  help="Directory for model checkpoints")
    # Model
    p.add_argument(
        "--architecture", default="small_cnn",
        choices=["small_cnn", "resnet18"],
        help="Model architecture to train",
    )
    p.add_argument("--dropout",      type=float, default=0.4)
    # Training hyper-parameters
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--image_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=4)
    # Split
    p.add_argument("--train_frac",   type=float, default=0.70)
    p.add_argument("--val_frac",     type=float, default=0.15)
    p.add_argument("--seed",         type=int,   default=42)
    # Early stopping (0 = disabled)
    p.add_argument(
        "--early_stopping", type=int, default=10,
        help="Stop if val accuracy does not improve for this many epochs (0 = off)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

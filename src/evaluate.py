"""
Evaluation, visualisation, and error analysis for the EuroSAT classifier.

Usage (from the project root — run after src.train):
    python -m src.evaluate
    python -m src.evaluate --checkpoint models/resnet18_best.pt --run_baseline

Outputs written to figures/ and results/:
    training_curves.png        — train / val loss and accuracy vs epoch
    confusion_matrix.png       — row-normalised recall matrix
    per_class_metrics.png      — per-class precision, recall, F1
    error_examples.png         — example patches for the top confused class pairs
    test_metrics.json          — overall accuracy, macro F1, per-class report
    baseline_results.json      — accuracy / F1 for logistic regression & random forest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for scripted execution
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .data import (
    CLASS_DISPLAY_NAMES,
    EUROSAT_MEAN,
    EUROSAT_STD,
    VEGETATION_CLASSES,
    extract_mean_features,
    get_raw_transform,
    get_transforms,
    load_split_indices,
)
from .model import build_model


# ---------------------------------------------------------------------------
# Utility: image denormalisation for display
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised CHW float tensor back to an HWC array in [0, 1].
    Used to recover a displayable image from a normalised model input.
    """
    mean = torch.tensor(EUROSAT_MEAN).view(3, 1, 1)
    std  = torch.tensor(EUROSAT_STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, List[str], dict]:
    """Load a model and its metadata from a checkpoint file."""
    ckpt        = torch.load(checkpoint_path, map_location=device)
    saved_args  = ckpt["args"]
    class_names = ckpt["class_names"]

    model = build_model(
        architecture = saved_args["architecture"],
        num_classes  = len(class_names),
        in_channels  = 3,
        dropout      = saved_args.get("dropout", 0.4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names, saved_args


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(
    model:         nn.Module,
    data_dir:      str | Path,
    split_indices: List[int],
    device:        torch.device,
    image_size:    int = 64,
    batch_size:    int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a subset of EuroSAT.

    Returns
    -------
    y_true  : ground-truth labels  (n,)
    y_pred  : predicted labels     (n,)
    y_proba : class probabilities  (n, num_classes)
    """
    dataset = datasets.EuroSAT(
        root=str(data_dir),
        transform=get_transforms(train=False, image_size=image_size),
        download=False,
    )
    loader = DataLoader(
        Subset(dataset, split_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    all_true, all_pred, all_proba = [], [], []
    for images, labels in loader:
        images  = images.to(device)
        logits  = model(images)
        proba   = torch.softmax(logits, dim=1).cpu()
        all_true.append(labels.numpy())
        all_pred.append(proba.argmax(1).numpy())
        all_proba.append(proba.numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.vstack(all_proba),
    )


# ---------------------------------------------------------------------------
# Plot: training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", color="#1565C0", lw=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val",   color="#C62828", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", color="#1565C0", lw=1.8)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val",   color="#C62828", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle("Training and Validation Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    normalize:   bool = True,
) -> None:
    display_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in class_names]
    cm = confusion_matrix(y_true, y_pred)
    cm_plot = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) if normalize else cm.astype(float)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=(1.0 if normalize else None))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Per-class recall (fraction)" if normalize else "Count", fontsize=10)

    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    title = ("Confusion matrix — row-normalised recall" if normalize
             else "Confusion matrix — raw counts")
    ax.set_title(title, fontsize=12)

    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val   = f"{cm_plot[i,j]:.2f}" if normalize else f"{int(cm_plot[i,j])}"
            color = "white" if cm_plot[i, j] > thresh else "black"
            ax.text(j, i, val, ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: per-class precision / recall / F1
# ---------------------------------------------------------------------------

def plot_per_class_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: List[str],
    output_path: str | Path,
) -> None:
    report        = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    display_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in class_names]

    precision = [report[c]["precision"] for c in class_names]
    recall    = [report[c]["recall"]    for c in class_names]
    f1        = [report[c]["f1-score"]  for c in class_names]

    x     = np.arange(len(class_names))
    width = 0.26
    veg_c = "#2E7D32"
    oth_c = "#78909C"
    bar_colors = [veg_c if cn in VEGETATION_CLASSES else oth_c for cn in class_names]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, precision, width, label="Precision",
           color=bar_colors, alpha=0.60, edgecolor="white")
    ax.bar(x,          recall,   width, label="Recall",
           color=bar_colors, alpha=0.90, edgecolor="white")
    ax.bar(x + width,  f1,       width, label="F1",
           color=bar_colors, alpha=0.40, edgecolor=bar_colors, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Per-class Precision, Recall and F1   "
        "(dark green = vegetation classes,  grey = non-vegetation)",
        fontsize=11,
    )
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: error examples
# ---------------------------------------------------------------------------

def plot_error_examples(
    data_dir:      str | Path,
    split_indices: List[int],
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    class_names:   List[str],
    output_path:   str | Path,
    n_per_pair:    int = 4,
    top_k_pairs:   int = 5,
    image_size:    int = 64,
) -> None:
    """
    Visualise example patches for the most frequently confused class pairs.

    Each row shows one confusion pair (true class → predicted class) with
    n_per_pair example patches drawn from the test-set misclassifications.
    """
    dataset_raw = datasets.EuroSAT(
        root=str(data_dir),
        transform=get_raw_transform(image_size),
        download=False,
    )

    # Build ranked list of confusion pairs (off-diagonal cells)
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    pairs = [
        (cm[i, j], i, j)
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)
    top_pairs = [(i, j) for _, i, j in pairs[:top_k_pairs]]

    fig, axes = plt.subplots(
        top_k_pairs, n_per_pair,
        figsize=(n_per_pair * 2.4, top_k_pairs * 2.7),
    )
    if top_k_pairs == 1:
        axes = axes[np.newaxis, :]

    for row, (true_cls, pred_cls) in enumerate(top_pairs):
        mask         = (y_true == true_cls) & (y_pred == pred_cls)
        error_idxs   = np.where(mask)[0][:n_per_pair]
        true_name    = CLASS_DISPLAY_NAMES.get(class_names[true_cls], class_names[true_cls])
        pred_name    = CLASS_DISPLAY_NAMES.get(class_names[pred_cls], class_names[pred_cls])
        n_errors     = mask.sum()

        for col in range(n_per_pair):
            ax = axes[row, col]
            if col < len(error_idxs):
                global_idx      = split_indices[error_idxs[col]]
                img_tensor, _   = dataset_raw[global_idx]
                img             = img_tensor.permute(1, 2, 0).numpy()
                img             = np.clip(img, 0, 1)
                ax.imshow(img)
            else:
                ax.set_facecolor("#EEEEEE")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    f"True: {true_name}\nPred: {pred_name}\n({n_errors} errors)",
                    fontsize=7.5,
                    rotation=0,
                    labelpad=100,
                    va="center",
                )

    fig.suptitle(
        "Most Frequent Misclassifications — Test-Set Patch Examples",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Spectral baseline
# ---------------------------------------------------------------------------

def run_baseline(
    data_dir:   str | Path,
    train_idx:  List[int],
    test_idx:   List[int],
    results_dir: str | Path,
    image_size:  int = 64,
) -> dict:
    """
    Train and evaluate mean-spectral baseline classifiers.

    Features: mean RGB values per patch (3 dimensions).
    Classifiers: Logistic Regression and Random Forest.

    The accuracy of these models shows what a classifier can learn from
    aggregate spectral reflectance alone, without any spatial information.
    The CNN's improvement over this baseline quantifies the benefit of
    learning spatial structure from convolutional feature maps.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\nBaseline — extracting mean-spectral features...")
    X_train, y_train = extract_mean_features(data_dir, train_idx, image_size=image_size)
    X_test,  y_test  = extract_mean_features(data_dir, test_idx,  image_size=image_size)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = {}
    models  = [
        ("logistic_regression",
         LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
        ("random_forest",
         RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ]
    for name, clf in models:
        print(f"  Fitting {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        mac_f1 = f1_score(y_test, y_pred, average="macro")
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(mac_f1, 4)}
        print(f"    {name}:  accuracy={acc:.4f}  macro_f1={mac_f1:.4f}")

    out_path = Path(results_dir) / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Baseline results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate_model(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, class_names, saved_args = load_model_from_checkpoint(args.checkpoint, device)
    image_size  = saved_args.get("image_size", 64)
    architecture = saved_args["architecture"]

    # ── Load split indices ──────────────────────────────────────────────────
    split_path = results_dir / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split index file not found at {split_path}. "
            "Run src.train first to generate it."
        )
    train_idx, val_idx, test_idx = load_split_indices(split_path)
    print(f"Test set: {len(test_idx):,} samples")

    # ── Training curves ─────────────────────────────────────────────────────
    history_path = results_dir / f"{architecture}_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print("\nPlotting training curves...")
        plot_training_curves(history, figures_dir / "training_curves.png")
    else:
        print(f"\nWarning: history file not found at {history_path}, skipping curves.")

    # ── Test-set inference ───────────────────────────────────────────────────
    print("\nRunning test-set inference...")
    y_true, y_pred, y_proba = get_predictions(
        model, args.data_dir, test_idx, device, image_size
    )

    # ── Summary metrics ──────────────────────────────────────────────────────
    acc    = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Macro F1      : {mac_f1:.4f}")
    print("\nPer-class classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    metrics = {
        "architecture":            architecture,
        "test_accuracy":           round(acc, 4),
        "macro_f1":                round(mac_f1, 4),
        "class_names":             class_names,
        "classification_report":   classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
    }
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        figures_dir / "confusion_matrix.png",
    )
    plot_per_class_metrics(
        y_true, y_pred, class_names,
        figures_dir / "per_class_metrics.png",
    )
    plot_error_examples(
        args.data_dir, test_idx, y_true, y_pred, class_names,
        figures_dir / "error_examples.png",
        image_size=image_size,
    )

    # ── Baseline ────────────────────────────────────────────────────────────
    if args.run_baseline:
        run_baseline(args.data_dir, train_idx, test_idx, results_dir, image_size)

    print(f"\nEvaluation complete. All figures written to {figures_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained EuroSAT classifier."
    )
    p.add_argument("--checkpoint",   default="models/small_cnn_best.pt",
                   help="Path to the model checkpoint (.pt)")
    p.add_argument("--data_dir",     default="data",
                   help="Root directory containing the EuroSAT dataset")
    p.add_argument("--results_dir",  default="results",
                   help="Directory containing split_indices.json and for saving metrics")
    p.add_argument("--figures_dir",  default="figures",
                   help="Directory for output figures")
    p.add_argument("--run_baseline", action="store_true",
                   help="Also train and evaluate mean-spectral baseline classifiers")
    return p.parse_args()


if __name__ == "__main__":
    evaluate_model(parse_args())

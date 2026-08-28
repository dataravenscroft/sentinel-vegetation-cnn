"""
Data loading and preprocessing for EuroSAT Sentinel-2 vegetation classification.

EuroSAT is a benchmark dataset of 64x64-pixel Sentinel-2 image patches covering
10 land-use / land-cover classes sampled across 34 European countries.

Reference:
    Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
    EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
    Land Cover Classification. IEEE Journal of Selected Topics in Applied
    Earth Observations and Remote Sensing.
    https://doi.org/10.1109/JSTARS.2019.2918242
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# Alphabetical order matches torchvision's folder-sorting convention
CLASS_NAMES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_DISPLAY_NAMES: dict[str, str] = {
    "AnnualCrop":           "Annual Crop",
    "Forest":               "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation",
    "Highway":              "Highway",
    "Industrial":           "Industrial",
    "Pasture":              "Pasture",
    "PermanentCrop":        "Permanent Crop",
    "Residential":          "Residential",
    "River":                "River",
    "SeaLake":              "Sea / Lake",
}

# Vegetation-related classes (the focus of the ecological analysis)
VEGETATION_CLASSES: set[str] = {
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Pasture",
    "PermanentCrop",
}

# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
# Computed from the EuroSAT RGB training split (approximate but representative).
# Using dataset-specific statistics rather than ImageNet stats is generally
# preferable for domain-shifted imagery such as satellite data.
EUROSAT_MEAN: List[float] = [0.3444, 0.3803, 0.4078]
EUROSAT_STD:  List[float] = [0.2025, 0.1363, 0.1148]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(train: bool = True, image_size: int = 64) -> transforms.Compose:
    """
    Return a torchvision transform pipeline for training or evaluation.

    Augmentation choices are tailored to the remote-sensing context:
    - Horizontal/vertical flips are valid: satellite imagery has no canonical
      orientation (unlike natural-scene photography).
    - 90-degree rotation is valid for the same reason.
    - Colour jitter is mild to avoid distorting spectral reflectance
      relationships, which carry the primary classification signal.
    - No perspective or elastic distortions: the patches are already
      geometrically rectified and the spatial structure is meaningful.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])


def get_raw_transform(image_size: int = 64) -> transforms.Compose:
    """Minimal transform that returns a [0, 1] tensor without normalisation.

    Used for visualisation so that images can be displayed without
    reversing the normalisation step.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_eurosat(
    data_dir: str | Path,
    transform=None,
    download: bool = True,
) -> datasets.EuroSAT:
    """Load (and optionally download) the EuroSAT RGB dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return datasets.EuroSAT(
        root=str(data_dir),
        transform=transform,
        download=download,
    )


def make_splits(
    dataset_size: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Generate reproducible random train / validation / test index splits.

    Note on spatial autocorrelation
    --------------------------------
    This is a simple random image-level split.  Because EuroSAT patches are
    sampled from geographically distributed tiles, nearby patches may share
    environmental gradients, sensor conditions, or atmospheric state.
    Random splitting places near-neighbours in both train and test sets,
    which can overestimate generalisation to genuinely unseen regions.
    A spatially blocked or leave-region-out split would provide a more
    conservative estimate for a production landscape-mapping application.
    See README §Validation strategy for further discussion.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)
    n_train = int(train_frac * dataset_size)
    n_val   = int(val_frac   * dataset_size)
    train_idx = indices[:n_train].tolist()
    val_idx   = indices[n_train : n_train + n_val].tolist()
    test_idx  = indices[n_train + n_val :].tolist()
    return train_idx, val_idx, test_idx


def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 64,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[int], List[int], List[int]]:
    """
    Build train / val / test DataLoaders from the EuroSAT RGB dataset.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names,
    train_idx, val_idx, test_idx
    """
    # Probe the dataset once (without transforms) to get metadata.
    base = load_eurosat(data_dir, transform=None, download=download)
    class_names = base.classes

    train_idx, val_idx, test_idx = make_splits(
        len(base), train_frac, val_frac, seed
    )

    # Re-attach per-split transforms via separate dataset instances.
    train_ds = load_eurosat(
        data_dir, transform=get_transforms(train=True,  image_size=image_size), download=False
    )
    eval_ds = load_eurosat(
        data_dir, transform=get_transforms(train=False, image_size=image_size), download=False
    )

    def _make_loader(ds, indices, shuffle):
        return DataLoader(
            Subset(ds, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    train_loader = _make_loader(train_ds, train_idx, shuffle=True)
    val_loader   = _make_loader(eval_ds,  val_idx,   shuffle=False)
    test_loader  = _make_loader(eval_ds,  test_idx,  shuffle=False)

    return train_loader, val_loader, test_loader, class_names, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Baseline feature extraction
# ---------------------------------------------------------------------------

def extract_mean_features(
    data_dir: str | Path,
    split_indices: List[int],
    image_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    download: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-channel mean pixel values as features for the spectral baseline.

    For each image patch this computes the arithmetic mean of each colour
    channel across all spatial positions, yielding a 3-dimensional feature
    vector (mean R, mean G, mean B).

    This deliberately discards all spatial information, representing what a
    classifier can learn from aggregate spectral reflectance alone.  The
    improvement of the CNN over this baseline quantifies how much spatial
    structure contributes to classification accuracy.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_channels)
    y : np.ndarray, shape (n_samples,)
    """
    dataset = load_eurosat(data_dir, transform=get_raw_transform(image_size), download=download)
    loader  = DataLoader(
        Subset(dataset, split_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    X_parts, y_parts = [], []
    for images, labels in loader:
        # images: (B, C, H, W) → mean over H, W → (B, C)
        X_parts.append(images.mean(dim=[2, 3]).numpy())
        y_parts.append(labels.numpy())
    return np.vstack(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# Split persistence
# ---------------------------------------------------------------------------

def save_split_indices(
    train_idx: List[int],
    val_idx:   List[int],
    test_idx:  List[int],
    path: str | Path,
) -> None:
    """Save split indices to JSON so all scripts use identical data partitions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)


def load_split_indices(path: str | Path) -> Tuple[List[int], List[int], List[int]]:
    """Load split indices previously saved by save_split_indices."""
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]

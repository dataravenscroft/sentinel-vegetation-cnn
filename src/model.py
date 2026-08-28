"""
Model definitions for EuroSAT vegetation / land-cover classification.

Two model families are provided:

SmallCNN
    A compact convolutional network trained from scratch on EuroSAT.
    Four convolutional blocks with batch normalisation and max-pooling
    are followed by global average pooling and a two-layer classifier head.
    Designed to be transparent — every architectural choice is visible in
    a few dozen lines of PyTorch — and to train to convergence in under
    an hour on a modern laptop or a free Colab GPU session.

ResNet-18 (optional, secondary experiment)
    ImageNet-pretrained ResNet-18 with a replaced classification head,
    used to quantify the benefit of transfer learning relative to the
    from-scratch SmallCNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Convolutional building block: Conv2d → BatchNorm → ReLU → optional MaxPool.

    Batch normalisation stabilises training across the range of spectral
    values found in satellite imagery and allows higher learning rates.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        pool:         bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,           # bias redundant when followed by BN
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


# ---------------------------------------------------------------------------
# Small CNN
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """
    Compact CNN for 64 x 64 remote-sensing image patches.

    Architecture summary
    --------------------
    Input         64 x 64 x C  (C = 3 for RGB)
    ConvBlock 1    32 filters, 3x3, stride-2 pool  →  32 x 32
    ConvBlock 2    64 filters, 3x3, stride-2 pool  →  16 x 16
    ConvBlock 3   128 filters, 3x3, stride-2 pool  →   8 x  8
    ConvBlock 4   256 filters, 3x3, stride-2 pool  →   4 x  4
    GlobalAvgPool                                  → 256-d vector
    Dropout (p=0.4)
    FC 256 → 128 → ReLU → Dropout (p=0.2) → FC 128 → num_classes

    Global average pooling (rather than flattening) reduces the parameter
    count substantially and provides a degree of spatial invariance,
    which is appropriate for land-cover patches where the class is
    distributed across the whole tile rather than localised to one region.

    Parameter count: ~600 k (much smaller than ResNet-18 at ~11 M).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels,  32,  pool=True),   # 64→32
            ConvBlock(32,           64,  pool=True),   # 32→16
            ConvBlock(64,          128,  pool=True),   # 16→8
            ConvBlock(128,         256,  pool=True),   # 8→4
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-d embedding (after global pool) for visualisation."""
        x = self.features(x)
        return self.global_pool(x).flatten(1)


# ---------------------------------------------------------------------------
# ResNet-18 (transfer learning, secondary experiment)
# ---------------------------------------------------------------------------

def get_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with ImageNet pretraining, classification head replaced for EuroSAT.

    This is provided as a secondary experiment to assess whether ImageNet
    feature representations transfer usefully to Sentinel-2 RGB imagery.
    The first-layer filters are retained without modification; ResNet-18
    expects 3-channel RGB input in [0, 1] normalised with ImageNet statistics,
    which is consistent with the EuroSAT RGB JPEG patches.
    """
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    architecture: str  = "small_cnn",
    num_classes:  int  = 10,
    in_channels:  int  = 3,
    pretrained:   bool = True,
    dropout:      float = 0.4,
) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    architecture : "small_cnn" | "resnet18"
    num_classes  : number of output classes (10 for EuroSAT)
    in_channels  : input channels (3 for RGB; 13 for full Sentinel-2 MS)
    pretrained   : whether to load ImageNet weights (resnet18 only)
    dropout      : dropout rate (small_cnn only)
    """
    if architecture == "small_cnn":
        return SmallCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    elif architecture == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            "Choose 'small_cnn' or 'resnet18'."
        )

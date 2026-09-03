# Sentinel-2 Vegetation Classification with PyTorch CNNs

A CNN-based land-cover classifier built on the [EuroSAT benchmark dataset](https://github.com/phelber/EuroSAT), using Sentinel-2 RGB imagery and PyTorch. This is a direct comparison to an earlier classification project I completed using traditional remote-sensing methods, a decision-tree classifier on expert-labelled Landsat imagery in northeastern Minnesota. I wanted to see what changes and what stays the same when you replace that pipeline with a convolutional network.

> I used Claude Code to generate this codebase. I had a clear idea of what I wanted to build and compare, but I'm not going to pretend I wrote every line from scratch. What I am doing is working through it carefully — understanding each design decision, noting what warrante further exploration/analysis, and modifying it as I go. This repo is a work in progress in that sense.
> EuroSAT is a benchmark dataset, not novel research. The point is the comparison and the analysis, not the accuracy number.

---

## Scientific question

> *How well can a convolutional neural network distinguish vegetation and land-cover classes from Sentinel-2 imagery, and what can its errors tell us about the spectral and spatial similarities among those classes?*

The secondary question:

> *What does the CNN learn that a simple spectral-mean classifier cannot, and where does that additional capacity break down?*

---

## Background

### Sentinel-2

Sentinel-2 is a European Space Agency (ESA) Copernicus mission satellite carrying a MultiSpectral Instrument (MSI). It captures imagery in 13 spectral bands spanning the visible (400–700 nm), near-infrared (NIR, ~800–900 nm), and short-wave infrared (SWIR, ~1400–2400 nm) at spatial resolutions of 10–60 m. The 10 m bands — Blue (B2), Green (B3), Red (B4), and NIR (B8) — are the most commonly used for vegetation analysis.

Key vegetation-relevant spectral features:
- **Chlorophyll absorption** at ~450 nm and ~670 nm (blue and red bands) —  plants look green
- **Green reflectance peak** (~550 nm) — low absorption by chlorophyll
- **Red-edge** (~700–740 nm) — abrupt transition from chlorophyll absorption to NIR plateau; diagnostic vegetation signals, captured by Sentinel-2 bands B5/B6/B7
- **NIR plateau** (~750–900 nm) — high reflectance from leaf cell structure scattering; strongly differentiates healthy vegetation from bare soil or water

### EuroSAT

EuroSAT (Helber et al. 2019) is a land-use / land-cover benchmark comprising **27,000 labelled 64×64-pixel image patches** from Sentinel-2, sampled across 34 European countries. Each patch covers approximately 0.64 km² at 10 m resolution. The dataset has 10 classes:

| Class | Vegetation? | Description |
|---|---|---|
| Annual Crop | Yes | Arable fields with seasonal crops (wheat, maize, sunflower, etc.) |
| Forest | Yes | Continuous tree cover; mixed or broadleaf/conifer stands |
| Herbaceous Vegetation | Yes | Non-woody vegetation: meadows, rough grassland, shrubland |
| Highway | No | Major roads and surrounding built environment |
| Industrial | No | Industrial buildings, warehouses, car parks |
| Pasture | Yes | Managed grassland for livestock grazing |
| Permanent Crop | Yes | Orchards, vineyards, olive groves |
| Residential | No | Urban housing; often includes trees and gardens |
| River | No | Flowing water bodies |
| Sea / Lake | No | Open water: lakes, reservoirs, coastal sea |

The RGB version (used here) corresponds to Sentinel-2 bands B4, B3, B2. A full 13-band multispectral version is available separately — extending to that is a natural next step.

**Reference:** Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.* https://doi.org/10.1109/JSTARS.2019.2918242

---

## Why CNNs add value over spectral baselines

A simple spectral classifier can use the aggregate reflectance in each band to distinguish classes. For EuroSAT RGB, this means each image is reduced to three numbers (mean R, mean G, mean B). This works well for spectrally distinctive classes — water has low, blue-dominated reflectance; bare industrial surfaces are bright and spectrally flat — but breaks down for vegetation types with similar spectral signatures.

Convolutional neural networks can additionally learn **spatial texture features**:
- Forest has a characteristically irregular, high-contrast texture from canopy gaps, individual crown shapes, and within-canopy shadow.
- Permanent crops (orchards, vineyards) often exhibit regular row structure at 10 m resolution.
- Annual crop fields frequently show geometric boundaries.
- Pasture and herbaceous vegetation appear as more spatially uniform, low-texture patches.

By learning these spatial patterns through convolutional filters, the CNN can discriminate classes that are spectrally similar but structurally distinct. The improvement of the CNN over the spectral-mean baseline quantifies how much spatial information contributes to classification accuracy.

---

## Repository structure

```
sentinel-vegetation-cnn/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py          # Dataset loading, splits, transforms, feature extraction
│   ├── model.py         # SmallCNN, ResNet-18, build_model factory
│   ├── train.py         # Training loop, checkpointing, history logging
│   └── evaluate.py      # Metrics, confusion matrix, error analysis plots
├── notebooks/
│   └── exploration_and_error_analysis.ipynb
├── figures/             # Generated plots (committed selectively)
├── results/             # JSON metrics and split indices
└── models/              # Model checkpoints (.gitignored)
```

---

## Model architecture

### SmallCNN (primary model)

A compact convolutional network implemented directly in PyTorch, designed to be transparent and trainable without a GPU in under an hour.

```
Input          64 × 64 × 3 (RGB)
ConvBlock 1    Conv2d(3→32, 3×3) → BN → ReLU → MaxPool(2×2)   →  32 × 32 × 32
ConvBlock 2    Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)  →  16 × 16 × 64
ConvBlock 3    Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2×2) →   8 ×  8 × 128
ConvBlock 4    Conv2d(128→256, 3×3) → BN → ReLU → MaxPool(2×2) →  4 ×  4 × 256
GlobalAvgPool                                                   →        256-d
Dropout (p=0.4)
FC 256 → 128 → ReLU → Dropout (p=0.2) → FC 128 → 10 classes
```

Global average pooling replaces flattening: it reduces the parameter count substantially and provides spatial invariance appropriate for patch-level land-cover classification, where the label is distributed across the full patch rather than localised to a specific region.

Total trainable parameters: ~600k

### ResNet-18 (secondary experiment)

ImageNet-pretrained ResNet-18 with a replaced classification head, used to assess the benefit of transfer learning relative to the from-scratch SmallCNN. Run with `--architecture resnet18`.

### Spectral-mean baseline

For comparison, logistic regression and random forest classifiers are trained on three-dimensional features (mean R, mean G, mean B per patch). This establishes a lower bound representing what can be learned from spectral information alone, with all spatial structure discarded.

---

## Installation and quickstart

```bash
# Clone and set up environment
git clone <your-repo-url> sentinel-vegetation-cnn
cd sentinel-vegetation-cnn
pip install -r requirements.txt

# Train the SmallCNN (downloads EuroSAT automatically ~90 MB)
python -m src.train

# Evaluate and generate all figures
python -m src.evaluate --run_baseline

# Open the analysis notebook
jupyter notebook notebooks/exploration_and_error_analysis.ipynb
```

All paths default to relative directories (`data/`, `results/`, `models/`, `figures/`) within the repository root. Run all commands from the project root directory.

### Training options

```bash
python -m src.train --help

# Key arguments:
#   --architecture   small_cnn | resnet18           (default: small_cnn)
#   --epochs         number of training epochs      (default: 40)
#   --batch_size     mini-batch size                (default: 64)
#   --lr             initial learning rate           (default: 1e-3)
#   --early_stopping patience in epochs             (default: 10)
#   --seed           random seed for reproducibility (default: 42)
```

### Evaluation options

```bash
python -m src.evaluate --checkpoint models/small_cnn_best.pt --run_baseline
```

---

## Training details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Initial LR | 1e-3 |
| LR schedule | Cosine annealing to 1e-6 |
| Weight decay | 1e-4 |
| Loss function | Cross-entropy with label smoothing (ε=0.05) |
| Gradient clipping | max_norm = 5.0 |
| Dropout | 0.4 (after global pool), 0.2 (in FC head) |
| Epochs | up to 40 with early stopping (patience=10) |
| Batch size | 64 |

### Data augmentation

Augmentations are chosen to respect the remote-sensing context:
- **Random horizontal/vertical flip** — valid because satellite imagery has no canonical orientation
- **Random 90° rotation** — valid for the same reason
- **Mild colour jitter** (brightness ±10%, contrast ±10%, saturation ±5%) — conservative to avoid distorting spectral reflectance relationships

No perspective distortion or elastic transforms are applied — the patches are geometrically rectified and spatial structure is meaningful.

### Data splits

70% train / 15% validation / 15% test, generated once with a fixed random seed and saved to `results/split_indices.json` so all scripts use identical partitions.

---

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (mean RGB) | 40.5% | 0.371 |
| Random Forest (mean RGB) | 60.4% | 0.595 |
| SmallCNN (RGB, from scratch) | **95.7%** | **0.956** |
| ResNet-18 (pretrained, optional) | — | — |

The CNN's 35-point accuracy gain over the Random Forest baseline directly quantifies the contribution of spatial texture to classification performance. Per-class F1 ranges from 0.920 (Permanent Crop) to 0.987 (Sea / Lake), with the lowest scores concentrated among ecologically similar vegetation types.

---

## Confusion and error analysis

The most frequent misclassifications involve ecologically similar vegetation types:

**Pasture ↔ Herbaceous Vegetation**
Both classes are dominated by non-woody, low-growing green vegetation with nearly identical RGB spectral signatures. The distinction is ecological management (grazed vs. natural/semi-natural) rather than a spectral or structural property consistently visible at 10 m resolution. This confusion is well-documented in operational vegetation mapping. Time-series phenology and NIR/red-edge data would be required to improve separation.

**Annual Crop ↔ Permanent Crop / Herbaceous Vegetation**
At a single point in the growing season, an annual crop field may be visually indistinguishable from rough grassland. The CNN exploits geometric field boundaries and regular row texture where present, but these cues are not always visible at patch scale.

**Forest**
Forest is among the best-classified vegetation types, reflecting both its distinctive spectral signature (lower red reflectance due to canopy shadow and chlorophyll absorption) and its characteristic high-contrast spatial texture from individual tree crowns and gaps.

**Residential ↔ HerbaceousVegetation**
Residential patches with high tree/garden cover can superficially resemble herbaceous or even forested patches, producing asymmetric confusion in both directions.

The improvement of the CNN over the spectral-mean baseline is largest for classes with characteristic spatial texture (Forest, PermanentCrop, Highway, Industrial) and smallest for spectrally distinctive classes (Sea/Lake, Industrial) where mean reflectance alone is nearly sufficient.

---

## Validation strategy and limitations

### Why random splitting may overestimate performance

The random image-level split used for benchmark evaluation assigns each patch independently to train, val, or test. This is appropriate for comparing methods, but for a **production vegetation mapping application**, it likely **overestimates** generalisation performance:

1. **Spatial autocorrelation.** Nearby patches share soil type, climate, topography, and atmospheric state at acquisition. Patches from the same geographic neighbourhood appear in both training and test sets, violating the independence assumption.

2. **Tile-level consistency.** Patches from the same Sentinel-2 tile are processed with the same calibration and atmospheric correction. Random splitting distributes patches from the same tile across train and test.

3. **Distribution shift.** The vegetation composition of a novel region may differ systematically from the training distribution even within EuroSAT's European coverage.

### Stronger validation approaches for landscape-scale application

- **Spatially blocked cross-validation:** assign patches to geographic blocks (grid cells, watersheds, administrative regions) and evaluate using leave-one-block-out. This prevents spatial neighbours from appearing in both train and test.
- **Leave-region-out:** hold out an entire country, ecoregion, or Sentinel-2 tile as the test set.
- **Temporal validation:** train on one acquisition date, test on another to assess robustness to inter-annual phenological variation.

EuroSAT does not provide patch coordinates, so spatial blocking cannot be directly implemented without geolocating patches from the source data.

### Other limitations

- **RGB only.** The NIR and red-edge bands that carry the strongest vegetation discrimination signal are unavailable in the RGB variant.
- **Single date.** Crop phenology and seasonal greenness patterns, which are powerful temporal discriminators, are not captured.
- **Patch scale.** Real landscape mapping involves edge effects, spatial context, and class transitions beyond a single 64×64 patch.
- **European training domain.** Generalisation outside Europe is untested.

---

## Scaling to landscape mapping

This patch-classification experiment represents one component of a full landscape mapping pipeline. In a production workflow:

1. **Tile-based inference:** a sliding window or grid of patches tiles the landscape; each patch receives a class prediction.
2. **Spatial smoothing / CRF:** neighbouring patch predictions are regularised using conditional random fields or morphological post-processing to remove isolated pixels.
3. **Temporal compositing:** multi-date imagery is used to construct cloud-free composites and phenological features, substantially improving crop/grassland discrimination.
4. **Spatial validation:** accuracy is assessed using geographically independent test regions, not random splits.
5. **Uncertainty quantification:** prediction confidence maps identify areas requiring field validation.

This is also where the comparison to traditional methods becomes most relevant — the earlier Minnesota workflow applied the same general pipeline but with hand-engineered spectral indices and texture metrics in place of learned convolutional features.

---

## Prior work

Earlier in my research career I worked on a classification project using Landsat imagery and an existing forest composition map to assign forest cover classes across a large landscape in northeastern Minnesota. That workflow used spectral indices, spatial filters, and a decision-tree classifier applied to expert-labelled training polygons — the standard traditional remote-sensing toolkit.

This project is the same scientific problem approached differently. Comparing the two is the point.

---

## Reproducibility

The training seed, data splits, and all hyperparameters are set deterministically and logged to `results/`. To reproduce exactly:

```bash
python -m src.train --seed 42
python -m src.evaluate --run_baseline
```

Split indices are saved to `results/split_indices.json` and loaded by the evaluation script, ensuring train and test partitions are identical across runs.

---

## Project context

The codebase was generated with Claude Code as a starting point. I'm working through it — reading the architecture, understanding the parameter choices, checking the evaluation setup, and making changes where something doesn't make sense or could be cleaner. Notes on specific decisions are scattered through the code comments and this README as I go.

The underlying goal is comparison: I've done this class of problem before with traditional remote-sensing tools, and I want a concrete side-by-side view of where CNNs add something and where the older approach held up fine.

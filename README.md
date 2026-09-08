# Sentinel-2 Vegetation Classification with PyTorch CNNs

A CNN-based land-cover classifier built on the [EuroSAT benchmark dataset](https://github.com/phelber/EuroSAT), using Sentinel-2 RGB imagery and PyTorch. This is a comparison to an earlier classification project I completed using traditional remote-sensing methods — engineered spectral and spatial predictors, a decision-tree classifier applied to expert-labelled Landsat imagery in northeastern Minnesota. The question is what changes, and what the tradeoffs are, when you replace hand-engineered features with learned image representations.

> Claude Code was used as a coding assistant to generate the initial codebase. I defined the scientific comparison and analysis, reviewed the implementation, ran the training and evaluation, investigated the errors, and documented where the design does and does not support the conclusions. EuroSAT is a benchmark dataset, not novel research. The 95.7% test accuracy is a benchmark result on a single random split; what the numbers establish and what they do not is discussed in the [evaluation section](#evaluation-design-validity-and-limitations) below.

---

## Result

The SmallCNN trained from scratch on EuroSAT RGB patches reached **95.7% test accuracy and 0.956 macro F1** on a held-out 4,050-patch test partition (one run, seed 42). The mean-RGB baselines scored 60.4% / 0.595 F1 (Random Forest) and 40.5% / 0.371 F1 (Logistic Regression).

The 35.3-percentage-point gap over the Random Forest shows that the complete pixel grid contains highly predictive information that is lost when each patch is reduced to three channel means. However, the comparison changes both the input representation and the model capacity simultaneously — so the gap cannot be attributed to spatial texture alone. Per-class F1 ranges from 0.920 (Permanent Crop) to 0.987 (Sea / Lake); the weakest results are concentrated among vegetation types with overlapping spectral signatures at single-date RGB resolution.

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

## Image representation: full pixel grid vs. spectral summary

The mean-RGB baseline reduces each patch to three numbers — the spatial average of each channel. This discards all information about how pixel values are arranged within the patch. A CNN operating on the full pixel grid retains that arrangement and can learn filters sensitive to spatial structure.

The comparison tests whether the complete image contains useful predictive information beyond what channel means capture — not whether spatial texture is specifically responsible for any improvement. The CNN and the spectral-mean classifiers differ in both input representation and model expressiveness, so observed performance differences reflect both factors.

Spatial signals that may be available to the CNN but not to the mean-RGB baseline:
- Forest: irregular, high-contrast texture from canopy gaps, crown shapes, and within-canopy shadow.
- Permanent crops: regular row structure (orchards, vineyards) often visible at 10 m resolution.
- Annual crop fields: geometric field boundaries and uniform intra-field texture.
- Pasture / Herbaceous Vegetation: spectrally similar and spatially more uniform — the harder cases.

Whether and to what degree the CNN uses these signals would require saliency analysis or controlled ablations (e.g. spatially shuffling pixels to preserve pixel-value distributions while destroying spatial arrangement).

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
- **Random rotation up to ±90°** — valid for the same reason (continuous uniform draw from [−90°, +90°], not discrete 90° steps)
- **Mild colour jitter** (brightness ±10%, contrast ±10%, saturation ±5%, hue ±2%) — conservative to avoid distorting spectral reflectance relationships

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

Per-class F1 ranges from 0.920 (Permanent Crop) to 0.987 (Sea / Lake), with the weakest scores among vegetation classes with similar spectral signatures at single-date RGB resolution. See the [Result](#result) section for interpretation of the baseline gap.

---

## Confusion and error analysis

The most frequent misclassifications involve ecologically similar vegetation types:

**Pasture ↔ Herbaceous Vegetation**
Both classes are dominated by non-woody, low-growing green vegetation with nearly identical RGB spectral signatures. The distinction is ecological management (grazed vs. natural/semi-natural) rather than a spectral or structural property consistently visible at 10 m resolution. This confusion is well-documented in operational vegetation mapping. Time-series phenology and NIR/red-edge data would be required to improve separation.

**Annual Crop ↔ Permanent Crop / Herbaceous Vegetation**
At a single point in the growing season, an annual crop field may be visually indistinguishable from rough grassland. Geometric field boundaries and regular row texture are plausible discriminating cues where visible at patch scale, but what the CNN is actually using is not established from error patterns alone.

**Forest**
Forest is among the best-classified classes (F1: 0.984), consistent with having both a distinctive spectral signature (lower red reflectance from canopy shadow and chlorophyll absorption) and characteristically irregular spatial texture from individual tree crowns and gaps. Whether both factors contribute, and in what proportion, is not established by these results alone.

**Residential ↔ Herbaceous Vegetation**
Residential patches with high tree/garden cover can superficially resemble herbaceous or even forested patches, producing confusion in both directions.

These error patterns motivate hypotheses about spectral similarity, spatial structure, management regimes, and phenology — but the confusion matrix alone does not prove what the CNN learned. Saliency analysis and controlled ablations would be needed to test those hypotheses.

---

## Evaluation design, validity, and limitations

### What this evaluation establishes

- **Fixed partitions:** a single 70 / 15 / 15 train / validation / test split, generated with seed 42 and saved to `results/split_indices.json`. The same partitions are used by all scripts.
- **Checkpoint selection on validation accuracy:** the saved model is the best-performing checkpoint on the validation set, not selected on test performance.
- **Test-set metrics:** accuracy and macro F1 on a 4,050-patch held-out test partition, with per-class precision, recall, and F1.
- **Error analysis:** confusion matrix and example patches for the most frequent misclassification pairs.
- **Logged artefacts:** split indices, per-epoch training history, full hyperparameters (stored in the checkpoint), and test metrics are all saved to `results/`.

### What this evaluation does not establish

- **Geographic transfer.** The test set is drawn from the same random partition as training data. There is no guarantee of geographic independence between train and test patches.
- **Independence from spatial autocorrelation.** Nearby patches share soil type, climate, topography, and atmospheric conditions. Random image-level splitting places geographic neighbours in both train and test, which can inflate measured accuracy relative to a genuinely unseen region.
- **Tile-level independence.** Patches from the same Sentinel-2 tile share calibration and atmospheric correction; random splitting distributes them across train and test.
- **Seed stability.** Only one training run was recorded. Results may vary across random seeds; no uncertainty intervals or calibration metrics are reported.
- **Temporal or ecological distribution shift.** The dataset is single-date and drawn from a curated European domain.
- **Production landscape-mapping performance.** Patch classification does not address tiled inference, mixed pixels, class boundaries, cloud cover, or domain shift to a new landscape.

The 95.7% accuracy should be read as a benchmark result on this dataset and split — not as evidence of geographic generalisation.

### Stronger validation approaches

- **Spatially blocked cross-validation:** assign patches to geographic blocks (grid cells, watersheds, administrative regions) and evaluate using leave-one-block-out.
- **Leave-region-out:** hold out an entire country, ecoregion, or Sentinel-2 tile as the test set.
- **Repeated seeds:** run training multiple times and report mean and variance of test metrics.
- **Temporal validation:** train on one acquisition date, test on another to assess phenological robustness.

EuroSAT does not include patch coordinates, so spatial blocking cannot be applied directly without geolocating patches from the source data.

### Limitations

- **Mean RGB is an intentionally lossy baseline**, not a competitive conventional remote-sensing pipeline. Colour histograms, texture descriptors, and engineered spectral / spatial features would be stronger conventional comparators.
- **The baseline comparison conflates two factors:** input representation (mean vs. full pixel grid) and model capacity (linear / tree vs. CNN). The performance gap cannot be attributed to either factor alone.
- **RGB only.** NIR and red-edge bands carry the strongest vegetation discrimination signal and are absent from this experiment.
- **Single date.** Crop phenology and seasonal greenness — powerful temporal discriminators — are not captured.
- **Patch scale.** Landscape mapping involves edge effects, mixed pixels, class transitions, and spatial context beyond a single 64 × 64 patch.
- **Curated European domain.** EuroSAT covers 34 European countries with human-verified labels; generalisation outside this domain is untested.
- **One training run.** Without repeated seeds or confidence intervals, the reported metrics carry unknown variance.

---

## What I would do next

The main limitation is evaluation design, not model architecture. A different architecture or further accuracy-tuning would not address the core issues.

1. **Spatially blocked validation.** Reconstruct or adopt a georeferenced version of EuroSAT and perform grouped, spatially blocked, or leave-region-out cross-validation to get a less optimistic accuracy estimate.
2. **Stronger conventional baselines.** Add colour histograms, texture descriptors (GLCM, LBP), and engineered spectral / spatial indices. Mean RGB is a useful floor but not a competitive remote-sensing baseline.
3. **Controlled ablations.** Train and test on spatially shuffled patches — pixels permuted randomly within each patch, destroying spatial arrangement while preserving pixel-value distributions. Comparing shuffled vs. unshuffled performance would quantify the contribution of pixel arrangement independently of model capacity differences.
4. **Repeated seeds and calibration.** Run training across multiple random seeds and report mean, standard deviation, and calibration of test metrics.
5. **Spectral bands.** Compare RGB against NIR, red-edge, and the full 13-band Sentinel-2 stack to quantify what is lost by restricting to visible wavelengths.
6. **Multi-date imagery.** Test whether a second acquisition date improves separation of phenologically similar classes (Annual Crop, Pasture, Herbaceous Vegetation).
7. **Tiled landscape inference.** Apply the classifier to a geographically independent landscape using a sliding window, assess accuracy against independent reference data, and characterise errors at patch boundaries and in mixed-cover areas.

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

The random seed, data splits, and hyperparameters are fixed and logged to `results/`. Data partitions are saved to `results/split_indices.json` and loaded by the evaluation script, so train and test sets are consistent across runs. Note that GPU operations can be non-deterministic even with a fixed seed; exact numeric reproducibility is not guaranteed across different hardware or library versions.

```bash
python -m src.train --seed 42
python -m src.evaluate --run_baseline
```

---

## Project context

Claude Code was used to generate the initial codebase. I defined the scientific comparison — what to build, what to measure, and how to interpret it — reviewed the implementation, ran training and evaluation, and worked through the error analysis and limitations. The README and code comments reflect that review process and are updated as I go.

The underlying goal is to have a working CNN pipeline I can compare directly against my earlier traditional-methods project — not to achieve a state-of-the-art benchmark result.

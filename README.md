# 🧠 Brain Tumor Segmentation with U-Net

A PyTorch implementation of a U-Net architecture for automated brain tumor segmentation, trained on the **BRISC 2025** dataset. The model performs pixel-level binary segmentation to identify tumor regions in grayscale MRI scans.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

---

## Overview

This project tackles brain tumor segmentation as a binary semantic segmentation task. Given a grayscale MRI image, the model predicts a pixel-wise mask indicating the presence of tumor tissue. The pipeline covers:

- Custom PyTorch `Dataset` with grayscale loading and mask alignment
- Class imbalance handling via weighted `BCEWithLogitsLoss`
- A fully custom U-Net built from scratch with encoder/decoder blocks
- Multi-GPU training support via `DataParallel`
- Visualization utilities for predictions vs. ground truth

---

## Architecture

The model follows the classic **U-Net** encoder–decoder design with skip connections.

```
Input (1×256×256)
│
├─ EncoderBlock 1 → (16 channels)   ─────────────────────────────────┐ skip
├─ EncoderBlock 2 → (32 channels)   ──────────────────────────────┐  │ skip
├─ EncoderBlock 3 → (64 channels)   ───────────────────────────┐  │  │ skip
├─ EncoderBlock 4 → (128 channels)  ────────────────────────┐  │  │  │ skip
├─ Bottleneck     → (256 channels, no pooling)              │  │  │  │
│                                                           │  │  │  │
├─ DecoderBlock 5 → (128 channels) ←────────────────────────┘  │  │  │
├─ DecoderBlock 4 → (64 channels)  ←───────────────────────────┘  │  │
├─ DecoderBlock 3 → (32 channels)  ←──────────────────────────────┘  │
├─ DecoderBlock 2 → (16 channels)  ←─────────────────────────────────┘
│
└─ Conv 1×1 → Output Logit (1×256×256)
```

Each **EncoderBlock** applies two Conv2d + ReLU layers followed by MaxPool2d, outputting both the downsampled feature map and a skip connection. Each **DecoderBlock** applies a transposed convolution to upsample, crops and concatenates the corresponding skip connection, then passes through another encoder-style conv block. The final 1×1 convolution produces a single-channel logit map.

---

## Dataset

The project uses the [BRISC 2025](https://www.kaggle.com/) segmentation dataset structured as:

```
brisc2025/
└── segmentation_task/
    ├── train/
    │   ├── images/   # .jpg MRI scans
    │   └── masks/    # .png binary tumor masks
    └── test/
        ├── images/
        └── masks/
```

**Preprocessing:**
- Images converted to grayscale
- Resized and center-cropped to **256×256**
- Images normalized to `[0, 1]` via `ToTensor()`
- Masks binarized and normalized to `[0, 1]`
- Samples smaller than 256×256 are skipped

**Class imbalance:** The dataset is heavily imbalanced (background >> tumor pixels). A `pos_weight` is computed from the training set ratio of background to foreground pixels and passed to `BCEWithLogitsLoss`.

---

## Project Structure

```
.
├── brain-tumor-segmentation.ipynb   # Main notebook (data → model → train → evaluate)
├── README.md
└── /kaggle/working/
    └── Segmentation_model.pk        # Saved model (pickle)
```

---

## Requirements

```
torch
torchvision
numpy
Pillow
matplotlib
pandas
```

Install with:

```bash
pip install torch torchvision numpy Pillow matplotlib pandas
```

> **Note:** The notebook is designed to run on Kaggle with GPU acceleration. A CUDA-capable device is strongly recommended for training.

---

## Getting Started

1. Clone or download the notebook.
2. Upload to [Kaggle](https://www.kaggle.com/) or a local Jupyter environment.
3. Attach the **BRISC 2025** dataset at `/kaggle/input/datasets/briscdataset/`.
4. Run all cells in order.

---

## Training

The model is trained with the following configuration:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss function | `BCEWithLogitsLoss` (weighted) |
| Batch size | 16 |
| Epochs | 40 |
| Input size | 256×256 |
| Base channels | 16 |
| Multi-GPU | `DataParallel` (2 GPUs) |

Training progress is printed per batch and averaged loss per epoch is logged and plotted.

---

## Inference

After training, predictions are generated using a configurable sigmoid threshold:

```python
predict(model, image, threshold=0.99)
```

- The model output logit is passed through `Sigmoid`.
- Pixels with probability ≥ `threshold` are classified as tumor.
- The `plot_predictions` utility overlays predicted masks on original MRI images for visual comparison against ground truth.

The trained model is saved as a pickle file:

```python
save_model(model, filename="Segmentation_model.pk")
```

---

## Results

### 🖼️ Training Samples with Ground Truth Masks

The figure below shows a sample of MRI brain scans from the training set with their ground truth tumor masks overlaid in white. Each image is center-cropped to 256×256 and converted to grayscale before being fed to the model.

![Training samples with ground truth masks](https://github.com/user-attachments/assets/09b9f8a0-26f4-4071-8805-60f99c3342b0)

---

### 🔬 Predicted Masks vs. Ground Truth

The figure below shows side-by-side comparisons of actual annotated MRI scans (left) and the model's predicted segmentation masks (right) on the test set. White regions indicate identified tumor areas. Predictions are generated using a sigmoid threshold of `0.99`.

![Predicted vs actual segmentation masks](https://github.com/user-attachments/assets/5439c766-8052-4920-af69-cd888287d01e)

Loss convergence is tracked over 40 epochs and plotted with `plot_losses()`.

---

## Notes

- The `DeconderBlock` class name is a typo in the source code (should be `DecoderBlock`) — preserved here for consistency with the implementation.
- The dataset loader silently skips images smaller than 256×256 by cycling to the next sample; this may cause subtle epoch-level inconsistencies for very small datasets.
- For production use, consider replacing `pickle` model saving with `torch.save(model.state_dict(), ...)` for better portability.

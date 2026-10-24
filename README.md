# Day 04 — PCA via SVD: Image Compression & Spatial Denoising

> **Pure NumPy SVD engine** — no `sklearn.decomposition.PCA` used anywhere.  
> Dataset: **Olivetti Faces** (400 × 64×64 grayscale images, D = 4 096)  
> Offline fallback chain: `fetch_olivetti_faces` → `load_digits` (upsampled) → synthetic structured faces

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Architecture — `ScratchPCA` Class](#architecture--scratchpca-class)
- [Experiments & Results](#experiments--results)
  - [Experiment A — Image Compression](#experiment-a--image-compression)
  - [Experiment B — Spatial Denoising](#experiment-b--spatial-denoising)
- [Visual Dashboard](#visual-dashboard)
- [Dataset Fallback Chain](#dataset-fallback-chain)
- [Key Insights](#key-insights)

---

## Overview

This script implements **Principal Component Analysis (PCA)** entirely from scratch in pure NumPy using **Singular Value Decomposition (SVD)**. It demonstrates two classical computer vision tasks:

| Task | Method | Key Metric |
|------|--------|------------|
| **Image Compression** | Project to top-*k* eigenvectors, reconstruct | Compression Ratio, MSE |
| **Spatial Denoising** | Project noisy image onto clean eigenbasis, reconstruct | PSNR gain (dB), MSE reduction |

The script produces a **dark-themed 2×2 Matplotlib dashboard** (`day04_pca_artifact.png`) and prints a full terminal summary table.

---

## Mathematical Foundation

### SVD Decomposition

Given the mean-centred data matrix **X**_centered ∈ ℝ^(N×D):

```
X_centered = U Σ Vᵀ
```

where:
- **U** ∈ ℝ^(N×K) — left singular vectors (sample scores)
- **Σ** = diag(σ₁, …, σ_K) — singular values, σ₁ ≥ σ₂ ≥ … ≥ 0
- **V** ∈ ℝ^(D×K) — right singular vectors (**principal components / eigenfaces**)

### Eigenvalues & Explained Variance

```
λᵢ = σᵢ² / (N − 1)          # eigenvalue of sample covariance matrix
EVRᵢ = λᵢ / Σλⱼ             # explained variance ratio
```

### Projection & Reconstruction

```
Z_k  = (X − μ) @ V_k         # (N, D) -> (N, k)   encode
X̂_k  = Z_k @ V_kᵀ + μ        # (N, k) -> (N, D)   decode
```

Reconstruction error: **MSE** = ‖X − X̂_k‖²_F / (N·D)

### PSNR (Denoising Quality)

```
PSNR = 10 · log₁₀(MAX² / MSE)    [dB]
```

---

## Project Structure

```
Project4/
├── day04_pca_image_compression_denoising.py   # Main script (all-in-one)
├── day04_pca_artifact.png                      # Generated dashboard (auto-created)
└── README.md                                   # This file
```

---

## Requirements

```txt
numpy
matplotlib
scikit-learn          # ONLY for dataset loading (fetch_olivetti_faces / load_digits)
```

Install with:

```bash
pip install numpy matplotlib scikit-learn
```

The script is compatible with **Python 3.7+** and handles Windows `cp1252` encoding automatically (stdout reconfigured to UTF-8 at startup).

---

## Quick Start

```bash
cd "c:\AI(Projects)\Project4"
python day04_pca_image_compression_denoising.py
```

Expected runtime: **~5–15 seconds** (dominated by SVD on the 360×4096 matrix).  
Output: terminal summary + `day04_pca_artifact.png` in the same directory.

---

## Architecture — `ScratchPCA` Class

```python
pca = ScratchPCA()
pca.fit(X_train)              # runs economy SVD, stores components & EVR
Z_k   = pca.project(X, k)    # (N, D) -> (N, k)
X_hat = pca.reconstruct(Z_k, k)   # (N, k) -> (N, D)
```

| Method / Attribute | Description |
|--------------------|-------------|
| `fit(X)` | Computes mean μ, runs `np.linalg.svd(full_matrices=False)`, stores V rows as principal components |
| `project(X, k)` | Encodes X into k-dimensional latent space |
| `reconstruct(Z_k, k)` | Decodes back to pixel space + adds mean |
| `mean_` | Mean face vector μ ∈ ℝ^D |
| `components_` | Principal components (rows of Vᵀ), shape (K, D) |
| `singular_values_` | Sorted singular values σ₁ ≥ … ≥ σ_K |
| `explained_variance_ratio_` | Per-component EVR array |
| `cumulative_variance_ratio_` | Cumulative EVR (monotone increasing to 1.0) |
| `n_components_for_variance(thr)` | Min k such that cumulative EVR ≥ threshold |
| `compression_ratio(k, D, N)` | `(N·D) / (N·k + k·D + D)` |

---

## Experiments & Results

### Experiment A — Image Compression

Reconstructs a single test face at four values of k:

```
k   Cum. Var %   Comp. Ratio      MSE    Bits Saved
---  ----------  -----------  -------  ----------
  5     54.35%       55.91x   0.0095       98.2%
 20     76.28%       15.82x   0.0054       93.7%
 50     87.57%        6.50x   0.0042       84.6%
150     96.64%        2.19x   0.0028       54.4%
```

> With just **k = 20 components** (0.5% of the 4096-D space), the face is visually recognisable at 93.7% memory reduction.

### Experiment B — Spatial Denoising

Adds Gaussian noise (σ = 0.25) to test images, then projects onto **top-40 clean eigenvectors**:

```
Metric       Noisy Input   PCA Recon    Gain
-----------  -----------  ----------  ------
MSE            0.053855    0.005128   -0.0487
PSNR (dB)        12.69       22.90   +10.21 dB
```

> PCA acts as a **low-pass spatial filter** — noise lives in the high-frequency tail of the spectrum; projecting onto the top-k eigenbasis discards it cleanly.

### Variance Thresholds

| Threshold | Minimum k |
|-----------|-----------|
| 80%       | k = 27    |
| 90%       | k = 65    |
| 95%       | k = 118   |

---

## Visual Dashboard

`day04_pca_artifact.png` — dark-themed 2×2 grid:

| Panel | Content |
|-------|---------|
| **Panel 1** | Cumulative Explained Variance vs. k, with dashed guide lines at 80 / 90 / 95 % |
| **Panel 2** | Mean face (grayscale) + top-5 Eigenfaces (PC1–PC5) coloured with `RdBu_r`, captioned with % variance each explains |
| **Panel 3** | Side-by-side progression: Original (4096-D) → k=5 → k=20 → k=50 → k=150, annotated with MSE and compression ratio |
| **Panel 4** | Denoising comparison: Clean → Noisy (σ=0.25) → PCA Reconstructed (k=40) → Residual heat map \|Noisy − Recon\| |

---

## Dataset Fallback Chain

The script is designed to **run 100% reliably offline**:

```
1. fetch_olivetti_faces()          400 x 64x64  float64 in [0,1]
        |  (network / permission failure)
        v
2. load_digits() upsampled         1797 x 8x8 -> nearest-neighbour resize to 64x64
        |  (sklearn unavailable)
        v
3. Synthetic structured faces      400 x 64x64  Gaussian-blob face templates + noise
```

---

## Key Insights

- **Eigenfaces encode facial structure hierarchically** — PC1 captures global illumination, PC2–PC5 encode symmetry axes, glasses, hair boundaries, and mouth shape.
- **SVD is numerically identical to PCA on the covariance matrix** but avoids explicitly forming the D×D covariance (saving memory when D >> N, and here D=4096 >> N=360).
- **Economy SVD** (`full_matrices=False`) returns only min(N,D) = 360 components — the practical rank of the dataset — making all downstream operations fast.
- **PCA denoising** exploits the spectral gap: face structure lives in the top eigenvectors, Gaussian noise is uniformly spread across all dimensions. Truncating to k=40 removes ~90% of noise energy.

---

*Part of the **ML/CV from Scratch** daily series — Day 04.*

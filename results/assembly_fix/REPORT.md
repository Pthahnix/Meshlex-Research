# Assembly Fix — PCA Inverse Rotation Verification

**Date**: 2026-03-21

## Bug Description

VQ-VAE reconstruction was missing the PCA inverse rotation step.
The normalization does: `centered → PCA rotate → scale` (forward).
The inverse should be: `scale → PCA inverse rotate → translate` (backward).
But the buggy code only did: `scale → translate` (missing rotation).

## Fix

Added `aligned @ Vt` step where `Vt` = `principal_axes` from SVD.
Modified:
- `scripts/encode_sequences.py` — now saves `principal_axes` in sequence NPZ
- `scripts/visualize_mesh_comparison.py` — applies PCA inverse in decode

## Results

| Mesh | CD (Buggy) | CD (Fixed) | Improvement |
|------|-----------|-----------|-------------|
| 001cfadfb920... | 0.019789 | 0.002670 | 86.5% |
| 002b5dcd1a78... | 0.024744 | 0.005475 | 77.9% |
| 0045e8726266... | 0.022935 | 0.002778 | 87.9% |
| 006387f76f68... | 0.001469 | 0.000159 | 89.2% |
| 007941851778... | 0.006090 | 0.000496 | 91.9% |
| **Average** | **0.015006** | **0.002316** | **84.6%** |

## Visualizations

| File | Description |
|------|-------------|
| `compare_XX_*.png` | Original vs Buggy vs Fixed, 2 views per mesh |
| `summary_comparison.png` | Bar chart of CD comparison + improvement |
| `results.json` | Raw numerical results |

# MeshLex Research

**MeshLex: Learning a Topology-aware Patch Vocabulary for Compositional Mesh Generation**

A research project exploring whether 3D triangle meshes possess a finite, reusable "vocabulary" of local topological patterns — analogous to how BPE tokens form a vocabulary for natural language.

## Motivation

All current mesh generation methods serialize meshes into 1D token sequences and feed them to transformers. They differ only in *how* they serialize (BPT, EdgeBreaker, FACE, etc.) and *what* backbone they use (GPT, DiT, Mamba, etc.). But mesh is fundamentally a graph — forcing it into a sequence is like cutting a map into strips and asking a model to reassemble it.

MeshLex takes a different approach: instead of generating meshes face-by-face, we learn a **codebook of ~4096 topology-aware patches** (each covering 20-50 faces) and generate meshes by selecting, deforming, and assembling patches from this codebook. A 4000-face mesh becomes ~130 tokens — an order of magnitude more compact than the state-of-the-art (FACE, ICML 2026: ~400 tokens).

## Core Hypothesis

> Mesh local topology is low-entropy and universal across object categories. A finite codebook of ~4096 topology prototypes, combined with continuous deformation parameters, can reconstruct arbitrary meshes with high fidelity.

## Timeline

- **Day 1 (2026-03-06)**: Project inception, gap analysis, idea generation, experiment design
- **Day 2 (2026-03-07)**: Full codebase implementation (14 tasks), unit tests, initial experiment
- **Day 3 (2026-03-08)**: Diagnosed codebook collapse, fixed SimVQ implementation, Exp1 v2 (A-stage 5cat) training + eval completed — **STRONG GO**. B-stage code implemented (rotation trick + multi-token KV decoder)
- **Day 4 (2026-03-09)**: Exp3 (B-stage 5cat) completed — **STRONG GO** (CD -6.2%). Discovered rotation trick incompatible with SimVQ. LVIS-Wide data prepared (844 categories, 71K patches). Exp2 (A-stage LVIS-Wide) completed — **STRONG GO** (ratio 1.07x, util 67.8%). Key finding: more categories = better generalization

## Current Status

**Phase: 4-experiment matrix (A/B stage × 5cat/LVIS-Wide). 3/4 completed, all STRONG GO.**

| # | Experiment | Status | Result |
|---|-----------|--------|--------|
| 1 | A-stage × 5-Category | **Done** | STRONG GO (ratio 1.14x, util 46%) |
| 2 | A-stage × LVIS-Wide | **Done** | **STRONG GO (ratio 1.07x, util 67.8%)** |
| 3 | B-stage × 5-Category | **Done** | STRONG GO (ratio 1.18x, CD -6.2%, util 99%) |
| 4 | B-stage × LVIS-Wide | Pending | — |

Key findings so far:
- SimVQ collapse fix successful: utilization 0.46% → 99%+ (217x improvement)
- B-stage multi-token KV decoder effective: reconstruction CD reduced 6.2%
- Rotation trick incompatible with SimVQ (causes rapid collapse)
- Cross-category generalization validated: CD ratio 1.07-1.18x (< 1.2 threshold)
- **More categories = better generalization**: LVIS-Wide (844 cat) outperforms 5-cat in all metrics

## Pipeline

```
ShapeNet OBJ → Decimation (pyfqmr) → Normalize [-1,1]
    → METIS Patch Segmentation (~35 faces/patch)
    → PCA-aligned local coordinates
    → Face features (15-dim: vertices + normal + angles)
    → SAGEConv GNN Encoder → 128-dim embedding
    → SimVQ Codebook (K=4096, learnable reparameterization)
    → Cross-attention MLP Decoder → Reconstructed vertices
```

## Repository Structure

```
src/                               # Core modules
├── data_prep.py                   # Mesh loading, decimation, normalization
├── patch_segment.py               # METIS patch segmentation + PCA normalization
├── patch_dataset.py               # NPZ serialization + PyTorch/PyG Dataset
├── model.py                       # PatchEncoder, SimVQCodebook, PatchDecoder, MeshLexVQVAE
├── losses.py                      # Masked Chamfer Distance loss
├── trainer.py                     # Training loop (encoder warmup + K-means init + dead code revival)
└── evaluate.py                    # Evaluation metrics + Go/No-Go decision

scripts/                           # CLI entry points
├── download_shapenet.py           # Download ShapeNet from HuggingFace
├── run_preprocessing.py           # Batch preprocess ShapeNet (with train/test split)
├── train.py                       # Training (supports --resume)
├── init_codebook.py               # K-means codebook initialization
├── evaluate.py                    # Same-cat / cross-cat evaluation
├── visualize.py                   # t-SNE, utilization histogram, training curves
└── validate_task*.py              # Per-task real data validation scripts

tests/                             # 17 unit tests
├── test_data_prep.py              # 2 tests
├── test_patch_segment.py          # 4 tests
├── test_patch_dataset.py          # 3 tests
└── test_model.py                  # 8 tests

results/                           # Validation outputs (committed)
├── task1_3_validation/            # Data prep + patch segmentation
├── task4_validation/              # Dataset serialization
├── task5_7_validation/            # Encoder/Codebook/Decoder
├── task8_10_validation/           # VQ-VAE + Training
├── task12_validation/             # Visualization
├── task13_validation/             # K-means init
├── exp1_v2_collapse_fix/          # Exp1 v2 training reports + model analysis
├── exp1_A_5cat/                   # Exp1 A-stage 5cat eval + report
├── exp2_A_lvis_wide/              # Exp2 A-stage LVIS-Wide eval + report
└── exp3_B_5cat/                   # Exp3 B-stage 5cat eval + report

.context/                          # Research documents (chronological)
├── 00-09_*.md                     # Research evolution documents
├── material/                      # Analysis summaries of key papers
└── paper/                         # [gitignored] 300+ paper markdown files
```

## Key Differentiators

| | MeshMosaic | FreeMesh | FACE | **MeshLex** |
|---|---|---|---|---|
| Approach | Divide-and-conquer | BPE on coordinates | One-face-one-token | **Topology patch codebook** |
| Still per-face generation? | Yes (within each patch) | Yes (merged coordinates) | Yes | **No** |
| Has codebook? | No | Yes (coordinate-level) | No | **Yes (topology-level)** |
| Compression (4K faces) | N/A | ~300 tokens | ~400 tokens | **~130 tokens** |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Run unit tests
python -m pytest tests/ -v

# See RUN_GUIDE.md for full training pipeline
```

## Target Venue

CCF-A conferences: CVPR / NeurIPS / ICCV

## License

Apache-2.0

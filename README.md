# MeshLex Research

**MeshLex: Learning a Topology-aware Patch Vocabulary for Compositional Mesh Generation**

A research project exploring whether 3D triangle meshes possess a finite, reusable "vocabulary" of local topological patterns — analogous to how BPE tokens form a vocabulary for natural language.

## Motivation

All current mesh generation methods serialize meshes into 1D token sequences and feed them to transformers. They differ only in *how* they serialize (BPT, EdgeBreaker, FACE, etc.) and *what* backbone they use (GPT, DiT, Mamba, etc.). But mesh is fundamentally a graph — forcing it into a sequence is like cutting a map into strips and asking a model to reassemble it.

MeshLex takes a different approach: instead of generating meshes face-by-face, we learn a **codebook of ~4096 topology-aware patches** (each covering 20-50 faces) and generate meshes by selecting, deforming, and assembling patches from this codebook. A 4000-face mesh becomes ~130 tokens — an order of magnitude more compact than the state-of-the-art (FACE, ICML 2026: ~400 tokens).

## Core Hypothesis

> Mesh local topology is low-entropy and universal across object categories. A finite codebook of ~4096 topology prototypes, combined with continuous deformation parameters, can reconstruct arbitrary meshes with high fidelity.

## Research Evolution

| Stage | Document | Summary |
|-------|----------|---------|
| 0 | `00_original_prompt.md` | Initial vision: Large Mesh Model (LMM) for unified reconstruction + generation |
| 1 | `01_gap_analysis_lmm.md` | 75+ paper survey, 7 research gaps identified |
| 2 | `02_idea_generation_lmm.md` | 5 candidate ideas → MeshFoundation v2 selected |
| 3 | `03_experiment_design_lmm.md` | Full experiment design for MeshFoundation v2 |
| 4 | `04_pplx_comprehensive_evaluation.md` | Independent review (Gap 88% accuracy, Idea 78/100, Exp 82/100) |
| 5 | `05_cc_pplx_debate.md` | Paradigm shift: from "better serialization" to "should we serialize at all?" → MeshLex |
| 6 | `06_plan_meshlex_validation.md` | Validation experiment plan for MeshLex feasibility |
| 7 | `07_impl_plan_meshlex_validation.md` | 14-Task implementation plan (completed) |

## Current Status

**Phase: Validation experiment code complete. Ready for full-scale training.**

All 14 implementation tasks are complete with 17 unit tests passing. The pipeline is ready for end-to-end training on ShapeNet data. See [`RUN_GUIDE.md`](RUN_GUIDE.md) for the complete operational guide from data preparation to Go/No-Go decision.

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
├── trainer.py                     # Training loop with staged VQ
└── evaluate.py                    # Evaluation metrics + Go/No-Go decision

scripts/                           # CLI entry points
├── run_preprocessing.py           # Batch preprocess ShapeNet
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
└── task13_validation/             # K-means init

.context/                          # Research documents (chronological)
├── 00-07_*.md                     # Research evolution documents
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

Research use only. Not yet published.

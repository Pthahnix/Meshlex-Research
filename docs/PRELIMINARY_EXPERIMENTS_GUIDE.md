# Preliminary Experiments — Remote Execution Guide

**Date**: 2026-03-21
**Branch**: `quick-validation`
**Estimated time**: ~2 hours total (5 min analysis + 1-2 hrs MDLM training)

## Prerequisites

Ensure these are available on the remote machine:
- VQ-VAE checkpoint: `data/checkpoints/rvq_lvis/checkpoint_final.pt`
- Sequence data: `data/sequences/rvq_lvis/` (4934 `*_sequence.npz` files)
- Patch data: `data/patches/lvis_wide/` (for category mapping)

If data is missing, download from HuggingFace:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
# Download checkpoint
hf_hub_download('Pthahnix/MeshLex-Research', 'checkpoints/rvq_lvis/checkpoint_final.pt',
                local_dir='data', repo_type='model')
"
```

If sequences are missing, they need to be re-encoded (requires checkpoint + patches).

## Step 1: Pull latest code

```bash
cd ~/MeshLex-Research
git fetch origin quick-validation
git checkout quick-validation
git pull origin quick-validation
```

## Step 2: Install dependencies

```bash
pip install umap-learn  # For Exp 3
# torch, numpy, scipy, matplotlib should already be available
```

## Step 3: Run Exp 1-4 (analysis, ~5 min)

```bash
PYTHONPATH=. python scripts/run_preliminary_analysis.py \
    --seq_dir data/sequences/rvq_lvis \
    --patch_dir data/patches/lvis_wide \
    --checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
    --output_dir results/preliminary_exp
```

This produces:
- `results/preliminary_exp/exp1_per_category/` — per-group distribution analysis
- `results/preliminary_exp/exp2_spatial/` — spatial correlation
- `results/preliminary_exp/exp3_codebook_viz/` — UMAP visualizations
- `results/preliminary_exp/exp4_rvq_dependency/` — mutual information

## Step 4: Run Exp 5 (MDLM training, ~1-2 hrs on GPU)

```bash
PYTHONPATH=. python scripts/run_mdlm_prototype.py \
    --seq_dir data/sequences/rvq_lvis \
    --output_dir results/preliminary_exp/exp5_mdlm \
    --epochs 100 \
    --batch_size 64
```

This produces:
- `results/preliminary_exp/exp5_mdlm/` — training curve, generated tokens, comparison with AR

## Step 5: Write summary report

After all experiments finish, create `results/preliminary_exp/REPORT.md` summarizing:

1. **Exp 1 verdict**: Is lognormal consistent across groups or a mixture artifact?
2. **Exp 2 verdict**: Do tokens have spatial locality?
3. **Exp 3 observations**: Does codebook have cluster structure?
4. **Exp 4 verdict**: How dependent are RVQ levels? (NMI values)
5. **Exp 5 verdict**: Is MDLM feasible? (ppl, accuracy, KL vs AR baseline)
6. **Updated direction recommendation** (see Decision Framework in spec)

## Step 6: Commit and push

```bash
git add results/preliminary_exp/ scripts/run_preliminary_analysis.py scripts/run_mdlm_prototype.py
git commit -m "research: preliminary experiments (5 exp) — token analysis + MDLM prototype

Exp 1: Per-category distribution analysis
Exp 2: Token spatial correlation
Exp 3: Codebook UMAP visualization
Exp 4: RVQ inter-level mutual information
Exp 5: Toy MDLM masked diffusion prototype

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

git push origin quick-validation
```

## Decision Framework

After reviewing results, update direction based on:

| Exp 1 (per-category) | Exp 5 (MDLM) | Recommended Direction |
|-----------------------|---------------|----------------------|
| Consistent lognormal | MDLM feasible | Both directions, theory-driven priority |
| Consistent lognormal | MDLM not feasible | Theory-driven only |
| Mixture artifact | MDLM feasible | PatchDiffusion only |
| Mixture artifact | MDLM not feasible | Focus on main pipeline, drop sub-directions |

**IMPORTANT**: These are 5% scale preliminary results. Do NOT make final direction decisions.
Record observations and flag what needs full-scale validation.

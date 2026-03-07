# MeshLex Validation Experiment Execution Design

**Date**: 2026-03-07
**Status**: Approved

## Goal

Run the MeshLex validation experiment on real ShapeNet data to reach a Go/No-Go decision on the core hypothesis: mesh local topology forms a universal, finite vocabulary.

## Execution Strategy

Split into two phases. Phase A+B first (data prep + quick validation), then Phase C-G (full training + evaluation) if no issues found.

### Phase A: Data Preparation (~1-2h)

1. **HF Login** — `huggingface-cli login` on the machine
2. **Download 5 category zips** from `ShapeNet/ShapeNetCore` via `huggingface_hub`:
   - Chair (03001627, 1.97GB), Table (04379243, 1.67GB), Airplane (02691156, 3.36GB)
   - Car (02958343, 5.69GB), Lamp (03636649, 745MB)
   - Total: ~13.4GB
3. **Extract OBJ files** — only keep `model_normalized.obj`, discard textures
4. **Preprocess** — use existing `scripts/run_preprocessing.py`:
   - Decimation to 1000 faces (pyfqmr)
   - Normalize to [-1, 1]
   - METIS patch segmentation (~35 faces/patch)
   - NPZ serialization per patch
5. **Train/Test split**:
   - Chair/Table/Airplane: 400 train + 100 test each
   - Car/Lamp: 500 each, all for cross-category test
6. **Validate** — patch count/size distribution stats, visual spot-check

### Phase B: Quick Training Validation (~2-3h)

1. **Encoder-Only** — 20 epochs, vq_start_epoch=999
2. **K-means Init** — codebook initialization from encoder embeddings
3. **Full VQ-VAE** — 10-20 epochs only, verify VQ loss converges + no codebook collapse
4. **Quick Eval** — CD + utilization on small test subset

### Phase C-G: Full Training + Evaluation (if Phase B passes)

- Full VQ-VAE: 200 epochs
- Evaluation: same-cat + cross-cat CD
- Visualization: t-SNE, utilization, training curves
- Go/No-Go decision per matrix in `06_plan_meshlex_validation.md`

## Code Changes Needed

- **New**: `scripts/download_shapenet.py` — download + extract from HF
- **Modify**: `scripts/run_preprocessing.py` — adapt to ShapeNet directory structure
- **Existing code unchanged**: all `src/` modules, train/eval/visualize scripts

## Hardware

- GPU: RTX 4090 24GB
- RAM: 503GB
- Disk: 57GB available
- All sufficient for full-scale experiment

## Data Budget

- 2500 meshes x ~30 patches/mesh = ~75K patches
- Each patch NPZ: ~5KB → ~375MB total patches
- Checkpoints: ~200MB each
- Total disk usage estimate: ~20GB (zips + extracted OBJ + patches + checkpoints)

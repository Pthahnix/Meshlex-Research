# MeshLex v2 — Assembly Fix + Full Dataset Retrain Design

> Date: 2026-03-19
> Status: REVIEWED
> Scope: Fix assembly pipeline bug, build unified Objaverse-LVIS + ShapeNet dataset on HF, retrain all models from scratch

---

## 1. Problem Statement

### 1.1 Assembly Bug (Critical)

VQ-VAE reconstruction comparison shows **shattered, unrecognizable meshes** despite low per-patch reconstruction loss (0.177). Root cause:

**Encoding** (`patch_segment.py:_normalize_patch_coords`):
```
local_vertices = (vertices - centroid) @ Vt.T / scale
```

**Decoding** (`visualize_mesh_comparison.py:decode_training_sequence`):
```python
world_verts = local_verts * scale + centroid   # BUG: missing @ Vt
```

**Correct inverse**:
```python
world_verts = (local_verts * scale) @ Vt + centroid
```

The PCA rotation matrix `Vt` (3x3) is saved in patch NPZ files but:
- **Not saved** in sequence NPZ files
- **Not predicted** by the AR model

Each patch's local coordinates are un-scaled and un-centered but **not un-rotated**, producing randomly oriented fragments.

Evidence from PatchNets (ECCV 2020, 131 citations): per-patch extrinsics **must** include `(center, radius, rotation)` — rotation is mandatory for correct world-space assembly.

### 1.2 Data Starvation (Critical)

| Metric | Current | Target (LVIS+ShapeNet) |
|--------|---------|------------------------|
| Available objects | 46K (LVIS only) | **~97K** (46K LVIS + 51K ShapeNet) |
| Meshes for AR training | 4,674 | ~65,000+ |
| Total tokens | 4.25M | ~60M+ |
| Token/param ratio | 0.21 | ~3.0+ |
| Patches for VQ-VAE | 188K | ~3M+ |
| Categories | 1,156 | ~1,200+ (1,156 LVIS + 55 ShapeNet) |

Only **10%** of available Objaverse-LVIS data was used. Adding ShapeNetCore v2 doubles the dataset.

### 1.3 Storage Constraint

RunPod has 80GB disk. Neither Objaverse cache (~50-100GB for 46K GLBs) nor ShapeNetCore (~30GB zip) fit alongside code + checkpoints. Need a streaming pipeline: download → process → upload to HF → delete local.

---

## 2. Solution Overview

### 2.1 Execution Phases

```
Phase A: Fix assembly rotation (1h, existing data, local only)
    ↓ Go/No-Go: VQ-VAE recon quality validated?
Phase D: Stream-process LVIS (46K) + ShapeNet (51K) → HF Dataset (10-16h CPU)
    ↓ HF Dataset: Pthahnix/MeshLex-Patches
Phase B: PCA + Rotation Tokens pipeline (retrain all on full data)
Phase C: No-PCA baseline pipeline (retrain all on full data)
    ↓
Phase E: Ablation comparison (B vs C)
```

### 2.2 Two Pipeline Variants

| | Pipeline B (PCA + rot tokens) | Pipeline C (No PCA) |
|---|---|---|
| Patch normalization | center + PCA rotate + scale | center + scale only |
| Token format | 11 tokens/patch | 7 tokens/patch |
| Vocab size | 2112 | 1852 |
| Seq len (130 patches) | 1430 | 910 |
| VQ-VAE codebook | Rotation-invariant (compact) | Rotation-variant (needs more capacity) |
| Assembly | `(local * scale) @ R + centroid` | `local * scale + centroid` |
| Rotation repr | Quaternion (4 tokens) | None |

Both pipelines use the **same unified dataset** from HF.

---

## 3. Phase A: Fix Assembly Rotation

**Goal**: Validate that VQ-VAE reconstruction is correct when PCA rotation is properly applied.

**Prerequisite**: Patch NPZ files must still be available on disk (in `data/patches/lvis_wide/`). If deleted, Phase A must be done after Phase D provides the data.

### 3.1 Changes

**`scripts/visualize_mesh_comparison.py` — `decode_training_sequence`**:

```python
def decode_training_sequence(seq_path, vqvae, device, patch_dir=None):
    """Decode a training sequence NPZ to world-space vertices.

    If patch_dir is provided, reads principal_axes from original patch NPZs
    for correct PCA inverse rotation.
    """
    data = np.load(seq_path)
    centroids = data["centroids"]
    scales = data["scales"]
    tokens = data["tokens"]

    # Try to load principal_axes from patch NPZ files
    mesh_id = Path(seq_path).stem.replace("_sequence", "")
    rotations = _load_patch_rotations(mesh_id, patch_dir) if patch_dir else None

    all_world_verts = []
    for i in range(len(centroids)):
        with torch.no_grad():
            tok_indices = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok_indices)
            n_verts = torch.tensor([30], device=device)
            local_verts = vqvae.decoder(z_hat, n_verts)[0, :30].cpu().numpy()

        scale = max(scales[i], 0.01)
        scaled = local_verts * scale

        if rotations is not None and i < len(rotations):
            # Apply inverse PCA rotation: @ Vt
            world_verts = scaled @ rotations[i] + centroids[i]
        else:
            world_verts = scaled + centroids[i]

        all_world_verts.append(world_verts)
    return all_world_verts
```

**Helper function**:
```python
def _load_patch_rotations(mesh_id, patch_dir):
    """Load principal_axes from original patch NPZ files."""
    if patch_dir is None:
        return None
    patch_dir = Path(patch_dir)
    rotations = []
    for npz_path in sorted(patch_dir.rglob(f"{mesh_id}_patch_*.npz")):
        data = np.load(str(npz_path))
        rotations.append(data["principal_axes"])  # (3, 3)
    return rotations if rotations else None
```

### 3.2 Diagnostic Output

- Re-run 8 reconstruction comparisons with corrected transform
- Compute Chamfer Distance: original mesh vertices ↔ reconstructed point cloud
- Side-by-side PNG comparison
- Save to `results/assembly_fix_validation/`

### 3.3 Go/No-Go Gate

| Metric | GO | NO-GO |
|--------|-----|-------|
| Visual similarity | Recognizable shape | Still shattered |
| CD (normalized) | Comparable to v1 (~200 range) | 10x+ worse |

If NO-GO: deeper VQ-VAE problem exists, must diagnose before proceeding.

---

## 4. Phase D: Unified Dataset → HuggingFace

**Goal**: Process **all** Objaverse-LVIS (46K) + ShapeNetCore v2 (51K) objects into patch NPZ files, stored on HuggingFace.

### 4.1 Data Sources

| Source | Objects | Categories | Format | Access |
|--------|---------|------------|--------|--------|
| Objaverse-LVIS | 46,207 | 1,156 | GLB (via `objaverse` Python API) | Open |
| ShapeNetCore v2 | 51,300 | 55 | OBJ (HF gated dataset) | Approved |
| **Total** | **~97,500** | **~1,200** | | |

**Category overlap**: Some ShapeNet categories (chair, table, airplane, car, lamp) overlap with LVIS categories. This is fine — more data per category improves codebook quality. We track `source` field to distinguish them.

### 4.2 HuggingFace Dataset

**Repo**: `Pthahnix/MeshLex-Patches` (new HF dataset)

**Structure on HF**:
```
Pthahnix/MeshLex-Patches/
├── metadata.json              # {mesh_id: {category, source, n_patches, n_faces, n_verts}}
├── splits.json                # {seen_train: [...], seen_test: [...], unseen: [...]}
├── stats.json                 # Aggregate statistics
├── objaverse/                 # Objaverse-LVIS patches
│   ├── batch_000/             # ~500 meshes per batch
│   │   ├── {uid}_patch_000.npz
│   │   └── ...
│   ├── batch_001/
│   └── ...
└── shapenet/                  # ShapeNet patches
    ├── batch_000/
    │   ├── {synsetId}_{modelId}_patch_000.npz
    │   └── ...
    ├── batch_001/
    └── ...
```

**`splits.json` schema**:
```json
{
  "seen_train": ["mesh_id_1", "mesh_id_2", ...],
  "seen_test": ["mesh_id_3", ...],
  "unseen": ["mesh_id_4", ...],
  "unseen_categories": ["category_1", "category_2", ...],
  "seen_categories": ["category_3", ...],
  "split_seed": 42,
  "holdout_count": 100,
  "test_ratio": 0.2
}
```

### 4.3 Patch NPZ Format

Each patch NPZ contains:

```python
np.savez_compressed(
    path,
    # PCA-normalized (for Pipeline B)
    local_vertices=pca_normalized,      # (V, 3) — center + PCA rotate + scale
    principal_axes=Vt,                  # (3, 3) — PCA rotation matrix
    # Non-PCA-normalized (for Pipeline C)
    local_vertices_nopca=nopca_normalized,  # (V, 3) — center + scale only
    # Shared fields
    faces=faces,                        # (F, 3) local face indices
    vertices=world_vertices,            # (V, 3) world-space vertex coords
    centroid=centroid,                  # (3,) patch centroid
    scale=scale,                       # (1,) bounding sphere radius
    # Note: scale is rotation-invariant (||Vt @ x|| == ||x||), so
    # a single scale value serves both PCA and noPCA modes.
    boundary_vertices=boundary,         # (B,) local indices
    global_face_indices=global_fi,      # (F,) into original mesh
)
```

**Note on scale**: Since `Vt` is orthogonal, `max(||centered @ Vt.T||) == max(||centered||)`. A single `scale` field is correct for both normalization modes.

### 4.4 Streaming Processing Pipeline

New script: `scripts/process_full_dataset_streaming.py`

**Two sub-pipelines in one script:**

#### 4.4.1 Objaverse-LVIS Stream

```
1. Load LVIS annotations → get all 46K UIDs with category labels
2. Split into batches of 500 UIDs
3. For each batch:
   a. objaverse.load_objects(uids) → GLB files in ~/.objaverse/ cache
   b. For each GLB:
      - load_and_preprocess_mesh(glb_path, target_faces=1000) → trimesh
      - segment_mesh_to_patches(mesh) → list[MeshPatch]
      - Save dual-normalization NPZ to local temp dir
      - Record metadata entry
      - On failure: log warning, continue to next mesh
   c. Upload batch folder to HF: api.upload_folder(folder_path, path_in_repo=f"objaverse/batch_{N:03d}")
   d. Delete local temp dir
   e. Clear objaverse cache: rm -rf ~/.objaverse/hf-objaverse-v1/glbs/{uid}*
   f. Append to progress.json (for resume)
```

#### 4.4.2 ShapeNet Stream

```
1. Download ShapeNetCore v2 from HF (gated): hf_hub_download to local temp
   - ShapeNetCore.v2.zip is ~30GB → extract in streaming fashion
   - Or: use HF datasets API to access individual files
2. Parse taxonomy.json → synsetId → category name mapping
3. Walk directory structure: {synsetId}/{modelId}/models/model_normalized.obj
4. Same batch processing loop:
   a. Load OBJ → decimate → segment → dual-normalization NPZ
   b. Upload batch to HF
   c. Delete processed files
   d. Update progress
```

**ShapeNet disk strategy**: ShapeNetCore v2 is ~30GB compressed. We cannot extract it all at once. Options:
- **Option A (recommended)**: Use `hf_hub_download` to download individual category folders one at a time (~0.5-2GB each). Process → upload → delete. 55 category iterations.
- **Option B**: Extract zip in streaming mode using Python `zipfile` module, processing files as they're extracted.

### 4.5 Split Strategy

Combined split across both datasets:
- Pool all categories from LVIS (1,156) + ShapeNet (55, deduped with LVIS where overlapping)
- **100 categories held out** as unseen test set (scaled up from 50 given 2x more data)
- Remaining categories: **80/20 mesh-level split** (seen_train / seen_test)
- Overlapping LVIS-ShapeNet categories (e.g., "chair") are treated as the same category
- Split assignments recorded in `splits.json`

### 4.6 Estimated Resources

| Step | Time | Disk (peak) | Notes |
|------|------|-------------|-------|
| Objaverse: download+process per batch | ~5 min / 500 meshes | ~2GB | 16 CPU cores |
| Objaverse: total (46K, ~92 batches) | **6-10h** | <5GB local | Resume-safe |
| ShapeNet: per category | ~10 min / category | ~3GB | 55 categories |
| ShapeNet: total (51K, 55 categories) | **4-6h** | <5GB local | Resume-safe |
| **Combined total** | **10-16h** | <5GB local peak | |
| HF dataset final size | — | ~40-50 GB on HF | ~3M+ patches |

### 4.7 Validation

After processing, verify:
- Objaverse: ≥ 35,000 meshes processed (75% success rate)
- ShapeNet: ≥ 45,000 meshes processed (88% success rate, cleaner data)
- Total: ≥ 75,000 meshes, ≥ 2.5M patches
- Category distribution is reasonable
- Sample 20 random patches (10 per source): load NPZ, verify shapes are valid
- Verify both `local_vertices` and `local_vertices_nopca` are present and different

---

## 5. Phase B: PCA + Rotation Tokens Pipeline

**Prerequisite**: Phase D dataset on HF.

### 5.1 Rotation Representation: Quaternion (4 tokens)

**Why not Euler angles**: Euler angles (ZYX convention) suffer from gimbal lock when the middle angle approaches ±90°, causing large jumps in two angles for small rotation changes. PCA matrices from mesh patches can easily produce ±90° orientations (e.g., vertical surfaces), making gimbal lock a real risk.

**Quaternion advantages**:
- No singularities (except double-cover: q and -q represent same rotation → fix by canonical form w≥0)
- Smooth and uniform parameterization
- 4 values in [-1, 1] → straightforward quantization

**Encoding**:
```python
from scipy.spatial.transform import Rotation as R

def quantize_rotation(Vt, n_bins=64):
    """PCA rotation matrix → 4 quantized quaternion tokens."""
    quat = R.from_matrix(Vt).as_quat()  # [x, y, z, w]
    # Canonical form: ensure w >= 0
    if quat[3] < 0:
        quat = -quat
    # Quantize each component from [-1, 1] to [0, n_bins)
    bins = ((quat + 1.0) / 2.0 * n_bins).astype(int).clip(0, n_bins - 1)
    return bins  # (4,) int array

def dequantize_rotation(bins, n_bins=64):
    """4 quantized tokens → rotation matrix."""
    quat = (bins.astype(float) + 0.5) / n_bins * 2.0 - 1.0  # [-1, 1]
    # Re-normalize to unit quaternion
    quat = quat / np.linalg.norm(quat)
    return R.from_quat(quat).as_matrix()  # (3, 3)
```

### 5.2 Token Format

New (11 tokens/patch):
```
(pos_x, pos_y, pos_z, scale, rot_qx, rot_qy, rot_qz, rot_qw, cb_L1, cb_L2, cb_L3)
```

**Token layout**:
| Field | Range | Bins |
|-------|-------|------|
| pos_x | [0, 255] | 256 |
| pos_y | [256, 511] | 256 |
| pos_z | [512, 767] | 256 |
| scale | [768, 831] | 64 |
| rot_qx | [832, 895] | 64 |
| rot_qy | [896, 959] | 64 |
| rot_qz | [960, 1023] | 64 |
| rot_qw | [1024, 1087] | 64 |
| cb_L1 | [1088, 2111] | 1024 |
| cb_L2 | [1088, 2111] | 1024 |
| cb_L3 | [1088, 2111] | 1024 |

**Vocab size**: 3×256 + 64 + 4×64 + 1024 = 768 + 64 + 256 + 1024 = **2112**

**Seq len**: 130 patches × 11 = **1430 tokens**

### 5.3 Encode/Decode Pseudocode (11-token format)

**Encode** (in `patches_to_token_sequence()`):
```python
def patches_to_token_sequence_rot(centroids, scales, rotations, tokens,
                                   n_pos_bins=256, n_scale_bins=64, n_rot_bins=64):
    """Build 11-token-per-patch flat sequence with quaternion rotation.

    Args:
        centroids: (N, 3) patch centroids
        scales: (N,) patch scales
        rotations: (N, 3, 3) PCA rotation matrices (Vt)
        tokens: (N, 3) RVQ codebook indices per patch
    Returns:
        flat_sequence: (N*11,) int array of token IDs
    """
    off_y = n_pos_bins                                    # 256
    off_z = 2 * n_pos_bins                                # 512
    off_scale = 3 * n_pos_bins                            # 768
    off_rot = 3 * n_pos_bins + n_scale_bins               # 832
    off_code = off_rot + 4 * n_rot_bins                   # 1088

    # Normalize positions to [0,1] per axis (per-mesh min/max)
    pos_min = centroids.min(axis=0)
    pos_max = centroids.max(axis=0)
    pos_range = np.maximum(pos_max - pos_min, 1e-8)
    pos_norm = (centroids - pos_min) / pos_range          # (N, 3) in [0, 1]

    # Normalize scales to [0,1]
    s_min, s_max = scales.min(), scales.max()
    s_range = max(s_max - s_min, 1e-8)
    s_norm = (scales - s_min) / s_range                   # (N,) in [0, 1]

    sequence = []
    for i in range(len(centroids)):
        px = int(pos_norm[i, 0] * (n_pos_bins - 1))
        py = int(pos_norm[i, 1] * (n_pos_bins - 1)) + off_y
        pz = int(pos_norm[i, 2] * (n_pos_bins - 1)) + off_z
        sc = int(s_norm[i] * (n_scale_bins - 1)) + off_scale
        # Quaternion rotation
        rot_bins = quantize_rotation(rotations[i], n_rot_bins)  # (4,) in [0, n_rot_bins)
        qx = rot_bins[0] + off_rot
        qy = rot_bins[1] + off_rot + n_rot_bins
        qz = rot_bins[2] + off_rot + 2 * n_rot_bins
        qw = rot_bins[3] + off_rot + 3 * n_rot_bins
        # Codebook tokens
        c1 = tokens[i, 0] + off_code
        c2 = tokens[i, 1] + off_code
        c3 = tokens[i, 2] + off_code
        sequence.extend([px, py, pz, sc, qx, qy, qz, qw, c1, c2, c3])
    return np.array(sequence, dtype=np.int64)
```

**Data flow**: `encode_sequences.py` reads `principal_axes` (3×3) from patch NPZ → saves raw quaternions `(N, 4)` in sequence NPZ → `patches_to_token_sequence_rot()` receives the rotation matrices and quantizes inline during token sequence construction.

**Decode** (11-token → world vertices):

```python
def decode_sequence_to_patches(sequence, vqvae, device,
                                n_pos_bins=256, n_scale_bins=64, n_rot_bins=64):
    tokens_per_patch = 11
    off_y = n_pos_bins
    off_z = 2 * n_pos_bins
    off_scale = 3 * n_pos_bins
    off_rot = 3 * n_pos_bins + n_scale_bins          # 832
    off_code = 3 * n_pos_bins + n_scale_bins + 4 * n_rot_bins  # 1088

    n_patches = len(sequence) // tokens_per_patch
    all_world_verts = []

    for i in range(n_patches):
        b = i * tokens_per_patch
        # Positions: dequantize from [0,1]
        pos_x = int(sequence[b + 0]) / 255.0
        pos_y = (int(sequence[b + 1]) - off_y) / 255.0
        pos_z = (int(sequence[b + 2]) - off_z) / 255.0
        # Scale
        scale_tok = int(sequence[b + 3]) - off_scale
        scale = max(scale_tok / 63.0, 0.01)
        # Rotation quaternion
        rot_bins = np.array([
            int(sequence[b + 4]) - off_rot,
            int(sequence[b + 5]) - off_rot - n_rot_bins,
            int(sequence[b + 6]) - off_rot - 2 * n_rot_bins,
            int(sequence[b + 7]) - off_rot - 3 * n_rot_bins,
        ])
        R_mat = dequantize_rotation(rot_bins, n_rot_bins)
        # Codebook tokens
        tok1 = int(sequence[b + 8]) - off_code
        tok2 = int(sequence[b + 9]) - off_code
        tok3 = int(sequence[b + 10]) - off_code

        # Decode vertices through RVQ
        with torch.no_grad():
            tok_indices = torch.tensor([[tok1, tok2, tok3]], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok_indices)
            n_verts = torch.tensor([30], device=device)
            local_verts = vqvae.decoder(z_hat, n_verts)[0, :30].cpu().numpy()

        # Transform to world space
        pos = np.array([pos_x, pos_y, pos_z])
        world_verts = (local_verts * scale) @ R_mat + pos
        all_world_verts.append(world_verts)

    return all_world_verts
```

### 5.4 Changes to Source Files

| File | Change |
|------|--------|
| `src/patch_sequence.py` | Add `quantize_rotation()`, `dequantize_rotation()`. Update `patches_to_token_sequence()` to accept rotations param and emit 11-token format. Add `compute_vocab_size()` mode for rotation. |
| `src/patch_dataset.py` | `MeshSequenceDataset.__getitem__()`: load `rotations` from sequence NPZ, pass to token encoder. Update `max_seq_len` default to 1500. |
| `scripts/encode_sequences.py` | Read `principal_axes` from patch NPZ files. Convert to quaternion. Save `rotations` array (N, 4) in sequence NPZ. |
| `scripts/visualize_mesh_comparison.py` | `decode_sequence_to_patches()`: parse 11-token format with quaternion decode. `decode_training_sequence()`: load rotations from sequence NPZ. |
| `scripts/train_ar.py` | Update `max_seq_len=1500`, `vocab_size=2112`. |
| `src/ar_model.py` | No changes (PatchGPT is vocab-size agnostic). |
| `src/model_rvq.py` | No changes. |
| `src/patch_segment.py` | No changes (PCA normalization stays). |

### 5.5 Training Pipeline

1. **Download patches from HF** → local `data/patches/full/`
   - Use `hf_hub_download(local_dir=...)` to download directly, avoiding HF cache duplication
   - Only need PCA-related fields: `local_vertices`, `principal_axes`, `faces`, `centroid`, `scale`
2. **Train RVQ VQ-VAE** on full ~3M patches (PCA-normalized `local_vertices`)
   - batch_size=512 (up from 256, fits in 24GB VRAM)
   - 100-200 epochs depending on convergence
3. **Encode sequences**: patches → RVQ tokens + centroids + scales + **quaternion rotations**
4. **Train AR**: 11-token format, vocab=2112, max_seq_len=1500
   - ~65K sequences × 1430 tokens = **93M tokens**
   - Token/param ratio: 93M / 20.4M = **4.56** (22x improvement over current)
   - 200-300 epochs
5. **Generate + Evaluate**: full pipeline with quaternion-aware decode

### 5.6 Resource Estimates (Corrected)

| Step | GPU Time | Notes |
|------|----------|-------|
| Download from HF | 2-4h | ~20GB, direct to local_dir |
| VQ-VAE training (3M patches, bs=512, 100ep) | **40-60h** | Bottleneck; monitor convergence |
| Sequence encoding | 4-6h | |
| AR training (65K seq, 200ep) | **40-60h** | Corrected: 65K/4674 ≈ 14x more data per epoch |
| **Total Pipeline B** | **~90-130h** | |

**AR training time correction**: Current AR trains 4674 sequences in ~32s/epoch. With 65K sequences: ~450s/epoch. 200 epochs × 450s = 25h. With gradient accumulation overhead: ~30-40h. Previous estimate of 30h was roughly correct for 35K sequences but underestimated for 65K. Revised to 40-60h.

---

## 6. Phase C: No-PCA Baseline Pipeline

**Prerequisite**: Phase D dataset on HF (uses `local_vertices_nopca` field).

### 6.1 Changes to Source Files

| File | Change |
|------|--------|
| `src/patch_dataset.py` | `PatchGraphDataset.__getitem__()`: add config flag to load `local_vertices_nopca` instead of `local_vertices`. |
| Token format | Same 7 tokens/patch (no rotation needed): `(pos_x, pos_y, pos_z, scale, cb_L1, cb_L2, cb_L3)` |
| Assembly | `world = local * scale + centroid` (current code, already correct) |
| `scripts/train_ar.py` | vocab=1852, max_seq_len=1024 (same as current) |

### 6.2 Training Pipeline

Same as Phase B but:
- VQ-VAE trained on `local_vertices_nopca` features (no PCA rotation)
- 7 tokens/patch, vocab=1852, max_seq_len=1024
- Shorter sequences → faster AR training

### 6.3 Resource Estimates

| Step | GPU Time |
|------|----------|
| VQ-VAE training (3M patches, bs=512, 100ep) | 40-60h |
| Sequence encoding | 4-6h |
| AR training (65K seq, 200ep, shorter seqs) | 25-40h |
| **Total Pipeline C** | **~70-110h** |

---

## 7. Phase E: Ablation Comparison

### 7.1 Metrics

| Metric | Description |
|--------|-------------|
| **Recon CD** | Chamfer Distance: original mesh ↔ VQ-VAE reconstruction |
| **Codebook utilization** | % of codebook entries used (per RVQ level) |
| **CD ratio** | Cross-cat CD / same-cat CD (generalization) |
| **AR loss / perplexity** | Final training metrics |
| **Generation CD diversity** | Pairwise CD among generated meshes |
| **Visual quality** | Side-by-side rendered comparisons |
| **FID-like metric** | Distribution distance: generated vs real point clouds |

### 7.2 Comparison Table (to fill)

| Metric | Pipeline B (PCA+quat) | Pipeline C (no PCA) |
|--------|----------------------|---------------------|
| VQ-VAE recon CD | | |
| Codebook utilization | | |
| CD ratio (generalization) | | |
| AR final loss | | |
| AR perplexity | | |
| Generation visual quality | | |
| Tokens per mesh | 1430 | 910 |
| Total vocab | 2112 | 1852 |

### 7.3 Paper-Worthy Findings

This ablation directly addresses a core design question: **Is PCA alignment worth the extra rotation tokens?**

- If B >> C: PCA alignment is essential, rotation-invariant codebook is more efficient
- If B ≈ C: Simpler no-PCA pipeline preferred (Occam's razor)
- If B << C: Something wrong with rotation token learning

Additional cross-dataset finding: **Does ShapeNet + Objaverse generalize better than either alone?** Can evaluate by comparing codebook utilization on held-out ShapeNet categories when trained on LVIS+ShapeNet vs LVIS-only.

---

## 8. Training Time Mitigation

With ~3M patches (16x current), VQ-VAE training is the bottleneck.

**Mitigations (must implement)**:
1. **Increase batch size**: current=256, target=512 or 1024 (RTX 4090, 24GB VRAM, ~2M param model)
2. **Early stopping**: Monitor validation loss every 10 epochs. Stop if no improvement for 20 epochs.
3. **Checkpoint to HF every 20 epochs**: Prevents loss from pod reset.
4. **Learning rate scheduling**: Cosine with warmup already in place; may increase initial LR with larger batch.

**Projected epoch times** (3M patches):
- bs=512: ~5860 iter/epoch × ~1.4s/iter ≈ 8200s/epoch ≈ 2.3h/epoch → 100ep ≈ 230h
- bs=1024: ~2930 iter/epoch × ~2.8s/iter ≈ 8200s/epoch ≈ 2.3h/epoch → 100ep ≈ 230h
- (Larger batch = fewer iters but slower per iter; wall time similar)

**Revised mitigation**: Full 3M patches at any batch size takes ~230h for 100 epochs — too long.

**Alternative**: Use **subset training** for VQ-VAE:
- The VQ-VAE codebook (1024 entries per level) may saturate well before seeing all 3M patches
- Train on a random 500K-patch subset (5x current data) as a pragmatic middle ground
- If codebook utilization reaches 100% early, more data won't help the codebook — it helps the encoder/decoder
- **Decision**: Start with 500K patches, evaluate utilization at epoch 50. If < 95%, increase data.

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Phase A shows VQ-VAE recon is bad even with correct rotation | HIGH | Diagnose encoder/decoder before proceeding |
| Processing fails for >50% objects | MEDIUM | Still 40K+ meshes even at 50% rate |
| VQ-VAE training too slow on full data | HIGH | Subset training (500K patches); increase batch size |
| HF upload bandwidth limits | LOW | Batch upload; resume-safe |
| RunPod pod reset during training | HIGH | `--resume` + HF checkpoint every 20 epochs |
| ShapeNet download requires special handling | LOW | Use HF gated dataset API; per-category streaming |
| Quaternion quantization error | LOW | 64 bins per component → ~1.8° resolution, acceptable |

---

## 10. Implementation Order

| # | Phase | What | Time | Dependency |
|---|-------|------|------|------------|
| 1 | A | Fix assembly rotation, validate VQ-VAE | 1h | None |
| 2 | D | Stream-process LVIS+ShapeNet → HF dataset | 10-16h CPU | Phase A GO |
| 3 | B | PCA + quaternion rotation: train all | 90-130h GPU | Phase D |
| 4 | C | No-PCA baseline: train all | 70-110h GPU | Phase D |
| 5 | E | Ablation comparison | 4h | Phase B + C |

Phase B and C can run on **separate pods** in parallel.

---

## 11. Deliverables

1. **HF Dataset**: `Pthahnix/MeshLex-Patches` — unified LVIS + ShapeNet processed patches
2. **HF Checkpoints** in `Pthahnix/MeshLex-Research`:
   - `checkpoints/rvq_full_pca/` — Phase B VQ-VAE
   - `checkpoints/ar_full_pca_quat/` — Phase B AR (quaternion rotation)
   - `checkpoints/rvq_full_nopca/` — Phase C VQ-VAE
   - `checkpoints/ar_full_nopca/` — Phase C AR
3. **Results**: `results/ablation_full/` — comparison dashboard + report
4. **Assembly validation**: `results/assembly_fix_validation/` — Phase A output

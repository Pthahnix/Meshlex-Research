# Assembly Fix + Full Dataset Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the PCA rotation assembly bug, build a unified LVIS+ShapeNet dataset on HuggingFace, and retrain all models (PCA+quaternion and no-PCA pipelines) from scratch on the full data.

**Architecture:** Five phases — (A) validate the rotation fix on existing data, (D) stream-process 97K meshes to HF, (B) PCA+quaternion 11-token pipeline retrain, (C) no-PCA 7-token baseline retrain, (E) ablation comparison. Phases B and C share the same HF dataset but use different NPZ fields (`local_vertices` vs `local_vertices_nopca`).

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, trimesh, pymetis, pyfqmr, Open3D, scipy, objaverse, huggingface_hub, numpy

**Spec:** `docs/superpowers/specs/2026-03-19-assembly-fix-full-retrain-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/rotation.py` | Quaternion encode/decode utilities (quantize_rotation, dequantize_rotation) |
| `scripts/process_full_dataset_streaming.py` | Stream-process LVIS+ShapeNet → HF dataset |
| `scripts/validate_assembly_fix.py` | Phase A: validate rotation fix with CD metrics |
| `tests/test_rotation.py` | Unit tests for quaternion utilities |
| `tests/test_streaming_pipeline.py` | Unit tests for streaming pipeline helpers |

### Modified Files
| File | Changes |
|------|---------|
| `src/patch_sequence.py` | Add `patches_to_token_sequence_rot()`, update `compute_vocab_size()` with rotation mode |
| `src/patch_dataset.py` | Add `use_nopca` flag to `PatchGraphDataset`, add rotation support to `MeshSequenceDataset` |
| `src/patch_segment.py` | Add `_normalize_patch_coords_nopca()`, update `segment_mesh_to_patches()` to return both |
| `scripts/encode_sequences.py` | Save `principal_axes` (as quaternion) in sequence NPZ |
| `scripts/visualize_mesh_comparison.py` | Fix decode with rotation, support 11-token format |
| `scripts/train_ar.py` | Accept `--rotation` flag for 11-token mode |
| `tests/test_patch_sequence.py` | Add tests for rotation token sequence |

---

## Phase A: Fix Assembly Rotation (Tasks 1-3)

### Task 1: Quaternion Rotation Utilities

**Files:**
- Create: `src/rotation.py`
- Create: `tests/test_rotation.py`

- [ ] **Step 1: Write failing tests for quaternion encode/decode**

```python
# tests/test_rotation.py
"""Tests for src.rotation — quaternion encode/decode utilities."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


def test_quantize_rotation_output_shape():
    """quantize_rotation returns 4 ints in [0, n_bins)."""
    from src.rotation import quantize_rotation
    Vt = np.eye(3)
    bins = quantize_rotation(Vt, n_bins=64)
    assert bins.shape == (4,)
    assert bins.dtype == np.int64 or bins.dtype == np.intp
    assert (bins >= 0).all()
    assert (bins < 64).all()


def test_dequantize_rotation_returns_valid_matrix():
    """dequantize_rotation returns a valid 3x3 rotation matrix."""
    from src.rotation import dequantize_rotation
    bins = np.array([32, 32, 0, 63])  # arbitrary
    R_mat = dequantize_rotation(bins, n_bins=64)
    assert R_mat.shape == (3, 3)
    # Check orthogonality: R @ R.T ≈ I
    np.testing.assert_allclose(R_mat @ R_mat.T, np.eye(3), atol=0.1)


def test_roundtrip_identity():
    """Identity rotation survives quantize → dequantize."""
    from src.rotation import quantize_rotation, dequantize_rotation
    Vt = np.eye(3)
    bins = quantize_rotation(Vt, n_bins=64)
    R_rec = dequantize_rotation(bins, n_bins=64)
    np.testing.assert_allclose(R_rec, np.eye(3), atol=0.05)


def test_roundtrip_random_rotation():
    """Random rotation survives roundtrip within ~2° error."""
    from src.rotation import quantize_rotation, dequantize_rotation
    rng = np.random.default_rng(42)
    for _ in range(20):
        Vt = R.random(random_state=rng).as_matrix()
        bins = quantize_rotation(Vt, n_bins=64)
        R_rec = dequantize_rotation(bins, n_bins=64)
        # Angular error
        angle = R.from_matrix(R_rec @ Vt.T).magnitude()
        assert angle < np.radians(5), f"Angular error {np.degrees(angle):.1f}° > 5°"


def test_canonical_form_w_positive():
    """Quaternion canonical form ensures w >= 0."""
    from src.rotation import quantize_rotation
    # A rotation that produces negative w
    Vt = R.from_euler('z', 180, degrees=True).as_matrix()
    bins = quantize_rotation(Vt, n_bins=64)
    # The w bin should be >= n_bins/2 (since w >= 0 maps to [0.5, 1] → [32, 63])
    assert bins[3] >= 32, f"w bin {bins[3]} suggests w < 0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rotation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rotation'`

- [ ] **Step 3: Implement rotation utilities**

```python
# src/rotation.py
"""Quaternion rotation encode/decode for MeshLex patch assembly."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def quantize_rotation(Vt: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """PCA rotation matrix (3x3) → 4 quantized quaternion bins.

    Uses canonical form (w >= 0) to avoid double-cover ambiguity.

    Args:
        Vt: (3, 3) rotation matrix from SVD.
        n_bins: quantization bins per quaternion component.

    Returns:
        (4,) int array with values in [0, n_bins), order [qx, qy, qz, qw].
    """
    quat = R.from_matrix(Vt).as_quat()  # [x, y, z, w]
    # Canonical form: ensure w >= 0
    if quat[3] < 0:
        quat = -quat
    # Quantize from [-1, 1] to [0, n_bins)
    bins = ((quat + 1.0) / 2.0 * n_bins).astype(int).clip(0, n_bins - 1)
    return bins


def dequantize_rotation(bins: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """4 quantized bins → rotation matrix (3x3).

    Args:
        bins: (4,) int array in [0, n_bins), order [qx, qy, qz, qw].
        n_bins: quantization bins per component.

    Returns:
        (3, 3) rotation matrix.
    """
    quat = (bins.astype(float) + 0.5) / n_bins * 2.0 - 1.0  # [-1, 1]
    # Re-normalize to unit quaternion
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.eye(3)
    quat = quat / norm
    return R.from_quat(quat).as_matrix()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rotation.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rotation.py tests/test_rotation.py
git commit -m "feat: add quaternion rotation encode/decode utilities"
git push
```

---

### Task 2: Fix Assembly Decode in Visualization Script

**Files:**
- Modify: `scripts/visualize_mesh_comparison.py:166-184` (decode_training_sequence)
- Modify: `scripts/visualize_mesh_comparison.py:129-163` (decode_sequence_to_patches)

- [ ] **Step 1: Fix `decode_training_sequence` to apply PCA rotation inverse**

In `scripts/visualize_mesh_comparison.py`, replace the `decode_training_sequence` function (lines 166-184):

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

    # Load principal_axes from patch NPZ files
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
            world_verts = scaled @ rotations[i] + centroids[i]
        else:
            world_verts = scaled + centroids[i]

        all_world_verts.append(world_verts)
    return all_world_verts
```

- [ ] **Step 2: Add `_load_patch_rotations` helper**

Add this function before `decode_training_sequence` in the same file:

```python
def _load_patch_rotations(mesh_id, patch_dir):
    """Load principal_axes from original patch NPZ files."""
    if patch_dir is None:
        return None
    patch_dir = Path(patch_dir)
    rotations = []
    for npz_path in sorted(patch_dir.rglob(f"{mesh_id}_patch_*.npz")):
        data = np.load(str(npz_path))
        if "principal_axes" in data:
            rotations.append(data["principal_axes"])  # (3, 3)
    return rotations if rotations else None
```

- [ ] **Step 3: Update `main()` to pass `--patch_dir` argument**

Add CLI argument and pass it through:

```python
# In argparse section, add:
parser.add_argument("--patch_dir", default=None,
                    help="Directory with patch NPZ files (for PCA rotation fix)")

# In do_reconstruction_comparison call, pass patch_dir:
do_reconstruction_comparison(
    args.mesh_dir, args.seq_dir, vqvae, device,
    args.output_dir, n_samples=args.n_recon, patch_dir=args.patch_dir)
```

Update `do_reconstruction_comparison` signature to accept and forward `patch_dir`:

```python
def do_reconstruction_comparison(mesh_dir, seq_dir, vqvae, device, output_dir,
                                  n_samples=5, patch_dir=None):
    # ... existing code ...
    # Change the decode call:
    all_world_verts = decode_training_sequence(seq_path, vqvae, device, patch_dir=patch_dir)
```

- [ ] **Step 4: Verify the fix runs without errors**

Run (on existing data):
```bash
python scripts/visualize_mesh_comparison.py \
    --vqvae_checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
    --ar_checkpoint data/checkpoints/ar_v2/checkpoint_final.pt \
    --patch_dir data/patches/lvis_wide/seen_train \
    --output_dir results/assembly_fix_validation \
    --n_recon 5 --n_gen 0
```
Expected: Runs without error, produces PNG files in `results/assembly_fix_validation/reconstruction/`

- [ ] **Step 5: Commit**

```bash
git add scripts/visualize_mesh_comparison.py
git commit -m "fix: apply PCA rotation inverse in assembly decode"
git push
```

---

### Task 3: Phase A Validation Script with Go/No-Go Gate

**Files:**
- Create: `scripts/validate_assembly_fix.py`

- [ ] **Step 1: Write validation script**

```python
# scripts/validate_assembly_fix.py
"""Phase A: Validate VQ-VAE reconstruction with corrected PCA rotation.

Computes Chamfer Distance between original mesh vertices and
VQ-VAE reconstructed point cloud (with rotation fix applied).

Go/No-Go gate:
  GO: Reconstructed shapes are visually recognizable, CD in reasonable range
  NO-GO: Still shattered or CD is 10x+ worse than expected
"""
import argparse
import json
import sys
import numpy as np
import torch
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_rvq import MeshLexRVQVAE


def chamfer_distance_np(pts_a, pts_b):
    """Compute Chamfer Distance between two point clouds (numpy)."""
    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return (dist_a.mean() + dist_b.mean()) / 2.0


def decode_with_rotation(seq_path, patch_dir, vqvae, device):
    """Decode sequence with PCA rotation fix."""
    data = np.load(seq_path)
    centroids = data["centroids"]
    scales = data["scales"]
    tokens = data["tokens"]

    mesh_id = Path(seq_path).stem.replace("_sequence", "")
    rotations = []
    for npz_path in sorted(Path(patch_dir).rglob(f"{mesh_id}_patch_*.npz")):
        d = np.load(str(npz_path))
        if "principal_axes" in d:
            rotations.append(d["principal_axes"])

    all_verts = []
    for i in range(len(centroids)):
        with torch.no_grad():
            tok = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok)
            n_v = torch.tensor([30], device=device)
            local = vqvae.decoder(z_hat, n_v)[0, :30].cpu().numpy()

        scale = max(scales[i], 0.01)
        scaled = local * scale
        if i < len(rotations):
            world = scaled @ rotations[i] + centroids[i]
        else:
            world = scaled + centroids[i]
        all_verts.append(world)

    return np.concatenate(all_verts, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_checkpoint", required=True)
    parser.add_argument("--seq_dir", default="data/sequences/rvq_lvis")
    parser.add_argument("--patch_dir", required=True)
    parser.add_argument("--mesh_dir", default="data/meshes/lvis_wide")
    parser.add_argument("--output_dir", default="results/assembly_fix_validation")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    vqvae = MeshLexRVQVAE().to(device)
    vqvae.load_state_dict(ckpt["model_state_dict"], strict=False)
    vqvae.eval()

    seq_dir = Path(args.seq_dir)
    mesh_dir = Path(args.mesh_dir)
    seq_files = sorted(seq_dir.glob("*_sequence.npz"))
    obj_files = {obj.stem: obj for obj in mesh_dir.rglob("*.obj")}

    results = []
    for sf in seq_files[:args.n_samples]:
        mesh_id = sf.stem.replace("_sequence", "")
        if mesh_id not in obj_files:
            continue
        orig = trimesh.load(str(obj_files[mesh_id]), force='mesh')
        recon_pts = decode_with_rotation(sf, args.patch_dir, vqvae, device)
        cd = chamfer_distance_np(np.array(orig.vertices), recon_pts)
        results.append({"mesh_id": mesh_id, "cd": float(cd)})
        print(f"  {mesh_id}: CD = {cd:.6f}")

    cds = [r["cd"] for r in results]
    summary = {"n_samples": len(results), "mean_cd": float(np.mean(cds)),
               "std_cd": float(np.std(cds)), "results": results}
    with open(out / "validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean CD: {summary['mean_cd']:.6f}")
    # Note: CD scale depends on mesh normalization (unit cube [-1,1]).
    # v1 results had CD ~200 on unnormalized meshes.
    # On normalized meshes, CD < 0.1 is good, < 0.5 is acceptable.
    if summary["mean_cd"] < 0.5:
        print("RESULT: GO")
    else:
        print("RESULT: NEEDS INVESTIGATION — check CD scale vs mesh normalization")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validation**

```bash
python scripts/validate_assembly_fix.py \
    --vqvae_checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
    --patch_dir data/patches/lvis_wide/seen_train \
    --n_samples 10
```
Expected: CD values printed, GO result

- [ ] **Step 3: Evaluate Go/No-Go — if NO-GO, stop and diagnose**

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_assembly_fix.py results/assembly_fix_validation/
git commit -m "feat: Phase A validation — assembly fix Go/No-Go gate"
git push
```

---

## Phase D: Unified Dataset → HuggingFace (Tasks 4-6)

### Task 4: Dual-Normalization in Patch Segmentation

**Files:**
- Modify: `src/patch_segment.py:38-57` (_normalize_patch_coords)
- Modify: `src/patch_segment.py:149-184` (segment_mesh_to_patches, MeshPatch)
- Modify: `tests/test_patch_segment.py`

- [ ] **Step 1: Write failing test for no-PCA normalization**

Add to `tests/test_patch_segment.py`:

```python
def test_normalize_nopca_no_rotation():
    """No-PCA normalization should center and scale but NOT rotate."""
    from src.patch_segment import _normalize_patch_coords_nopca
    verts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    normalized, centroid, scale = _normalize_patch_coords_nopca(verts)
    # Centroid should be mean
    np.testing.assert_allclose(centroid, [4, 5, 6])
    # Normalized should be centered and scaled, but NOT rotated
    centered = verts - centroid
    expected = centered / scale
    np.testing.assert_allclose(normalized, expected, atol=1e-6)


def test_dual_normalization_same_scale():
    """PCA and no-PCA normalization should produce the same scale value."""
    from src.patch_segment import _normalize_patch_coords, _normalize_patch_coords_nopca
    rng = np.random.default_rng(42)
    verts = rng.standard_normal((20, 3)).astype(np.float32)
    pca_norm, _, _, pca_scale = _normalize_patch_coords(verts)
    nopca_norm, _, nopca_scale = _normalize_patch_coords_nopca(verts)
    # Scales must be equal (rotation-invariant)
    np.testing.assert_allclose(pca_scale, nopca_scale, rtol=1e-5)


def test_mesh_patch_has_nopca_field():
    """MeshPatch should have local_vertices_nopca field."""
    from src.patch_segment import MeshPatch
    import dataclasses
    fields = {f.name for f in dataclasses.fields(MeshPatch)}
    assert "local_vertices_nopca" in fields
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patch_segment.py::test_normalize_nopca_no_rotation -v`
Expected: FAIL — `ImportError: cannot import name '_normalize_patch_coords_nopca'`

- [ ] **Step 3: Implement dual normalization**

Add to `src/patch_segment.py` after `_normalize_patch_coords`:

```python
def _normalize_patch_coords_nopca(vertices: np.ndarray):
    """Center and scale patch vertices WITHOUT PCA rotation."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-8:
        scale = 1.0
    normalized = centered / scale
    return normalized, centroid, scale
```

Update `MeshPatch` dataclass — add field with default value (to avoid breaking existing callers):

```python
local_vertices_nopca: np.ndarray = None  # (V, 3) centered + unit-scaled, NO PCA rotation
```

Update `segment_mesh_to_patches` — in the patch building loop (after line 173), add:

```python
local_verts_nopca, _, _ = _normalize_patch_coords_nopca(vertices)
```

And pass it to `MeshPatch`:

```python
patches.append(MeshPatch(
    ...,
    local_vertices=local_verts,
    local_vertices_nopca=local_verts_nopca,
))
```

- [ ] **Step 4: Run all patch_segment tests**

Run: `python -m pytest tests/test_patch_segment.py -v`
Expected: All tests PASS (including new ones)

- [ ] **Step 5: Update `process_and_save_patches` to save nopca field**

In `src/patch_dataset.py:76-106`, add to the `np.savez_compressed` call:

```python
local_vertices_nopca=patch.local_vertices_nopca,
```

- [ ] **Step 6: Commit**

```bash
git add src/patch_segment.py src/patch_dataset.py tests/test_patch_segment.py
git commit -m "feat: dual normalization — PCA + no-PCA in patch segmentation"
git push
```

---

### Task 5: Streaming Dataset Pipeline Script

**Files:**
- Create: `scripts/process_full_dataset_streaming.py`
- Create: `tests/test_streaming_pipeline.py`

This is a large script. Build it incrementally: helpers first, then Objaverse stream, then ShapeNet stream.

- [ ] **Step 1: Write test for batch processing helper**

```python
# tests/test_streaming_pipeline.py
"""Tests for streaming pipeline helpers."""
import numpy as np
import pytest
import tempfile
from pathlib import Path


def test_process_single_mesh_produces_npz(tmp_path):
    """process_single_mesh should produce patch NPZ files with dual normalization."""
    import trimesh
    mesh = trimesh.creation.box()
    mesh_path = tmp_path / "test.obj"
    mesh.export(str(mesh_path))

    from scripts.process_full_dataset_streaming import process_single_mesh
    result = process_single_mesh(str(mesh_path), "test_box", str(tmp_path / "out"))

    assert result is not None
    assert result["mesh_id"] == "test_box"
    assert result["n_patches"] > 0

    npz_files = list((tmp_path / "out").glob("*.npz"))
    assert len(npz_files) > 0
    data = np.load(str(npz_files[0]))
    assert "local_vertices" in data
    assert "local_vertices_nopca" in data
    assert "principal_axes" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_streaming_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement core processing function**

Create `scripts/process_full_dataset_streaming.py` with:
- `process_single_mesh(mesh_path, mesh_id, output_dir, target_faces=1000)` — loads mesh via `load_and_preprocess_mesh`, segments via `process_and_save_patches`, returns metadata dict or None
- `upload_batch_to_hf(local_dir, hf_repo, path_in_repo)` — uses `HfApi.upload_folder()`
- `load_progress(path)` / `save_progress(path, data)` — JSON resume support

See spec section 4.2 for full pseudocode. Key imports:
```python
from src.data_prep import load_and_preprocess_mesh
from src.patch_dataset import process_and_save_patches
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_streaming_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit core**

```bash
git add scripts/process_full_dataset_streaming.py tests/test_streaming_pipeline.py
git commit -m "feat: streaming pipeline — core mesh processing + helpers"
git push
```

- [ ] **Step 6: Implement Objaverse streaming loop**

Add `stream_objaverse(hf_repo, batch_size=500, progress_path)` function:
1. Load LVIS annotations via `objaverse.load_lvis_annotations()`
2. Collect all UIDs with categories (no `max_per_cat` limit — use ALL)
3. Split into batches of `batch_size`
4. For each batch: download → process → upload to HF `objaverse/batch_NNN/` → delete local
5. Clear objaverse cache after each batch
6. Resume support via progress JSON

See spec section 4.2 for full pseudocode.

- [ ] **Step 7: Commit Objaverse stream**

```bash
git add scripts/process_full_dataset_streaming.py
git commit -m "feat: streaming pipeline — Objaverse-LVIS batch processing"
git push
```

- [ ] **Step 8: Implement ShapeNet streaming loop**

Add `stream_shapenet(hf_repo, batch_size=500, progress_path)` function:
1. Download taxonomy.json from `ShapeNet/ShapeNetCore` HF dataset
2. List all synset directories and model IDs
3. For each batch: download OBJ via `hf_hub_download` → process → upload to HF `shapenet/batch_NNN/` → delete local
4. Resume support via progress JSON

Key: ShapeNet is a gated HF dataset — user must have accepted terms. Use `hf_hub_download(repo_id="ShapeNet/ShapeNetCore", filename=..., repo_type="dataset")`.

- [ ] **Step 9: Add main() with CLI and split generation**

Add `generate_splits(metadata, holdout_count=100, test_ratio=0.2, seed=42)` and `main()`:
- CLI args: `--source {objaverse,shapenet,both}`, `--hf_repo`, `--batch_size`
- After all processing: generate splits, upload `metadata.json`, `splits.json`, `stats.json` to HF root
- Split logic: hold out 100 categories as unseen, 80/20 mesh-level split for seen

- [ ] **Step 10: Commit full streaming pipeline**

```bash
git add scripts/process_full_dataset_streaming.py
git commit -m "feat: streaming pipeline — ShapeNet + splits + main CLI"
git push
```

---

### Task 6: Execute Streaming Pipeline (RunPod)

**Prerequisites:** Tasks 4-5 complete, RunPod pod running, HF token configured.

- [ ] **Step 1: Create HF dataset repo**

```bash
pip install huggingface_hub
huggingface-cli repo create MeshLex-Patches --type dataset
```

- [ ] **Step 2: Run Objaverse streaming**

```bash
python scripts/process_full_dataset_streaming.py \
    --source objaverse --hf_repo Pthahnix/MeshLex-Patches --batch_size 500
```
Expected: ~46K meshes processed in ~92 batches, ~8-12h. Monitor disk usage with `df -h /`.

- [ ] **Step 3: Run ShapeNet streaming**

```bash
python scripts/process_full_dataset_streaming.py \
    --source shapenet --hf_repo Pthahnix/MeshLex-Patches --batch_size 500
```
Expected: ~51K meshes processed in ~102 batches, ~8-12h.

- [ ] **Step 4: Verify HF dataset**

```python
from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("Pthahnix/MeshLex-Patches")
print(f"Files: {len(list(api.list_repo_tree('Pthahnix/MeshLex-Patches', repo_type='dataset')))}")
```

- [ ] **Step 5: Commit progress files**

```bash
git add data/objaverse_progress.json data/shapenet_progress.json
git commit -m "data: streaming pipeline complete — LVIS + ShapeNet on HF"
git push
```

---

## Phase B: PCA + Quaternion Pipeline Retrain (Tasks 7-10)

### Task 7: 11-Token Sequence Encoding with Rotation

**Files:**
- Modify: `src/patch_sequence.py:81-167`
- Modify: `scripts/encode_sequences.py:87-114`
- Modify: `tests/test_patch_sequence.py`

- [ ] **Step 1: Write failing test for rotation token sequence**

Add to `tests/test_patch_sequence.py`:

```python
def test_patches_to_sequence_rvq_rot():
    """RVQ+rotation mode: 10 patches -> 10*11 = 110 tokens."""
    M = 10
    centroids = torch.randn(M, 3)
    scales = torch.rand(M) + 0.1
    codebook_tokens = torch.randint(0, 1024, (M, 3))
    rotations = torch.eye(3).unsqueeze(0).expand(M, -1, -1)  # identity rotations

    from src.patch_sequence import patches_to_token_sequence_rot
    seq = patches_to_token_sequence_rot(
        centroids, scales, rotations, codebook_tokens,
        n_pos_bins=256, n_scale_bins=64, n_rot_bins=64,
    )
    assert seq.shape == (M * 11,)
    assert (seq >= 0).all()
    # Max token should be < vocab_size (2112)
    assert seq.max() < 2112


def test_compute_vocab_size_rot():
    """Vocab size with rotation: 3*256 + 64 + 4*64 + 1024 = 2112."""
    from src.patch_sequence import compute_vocab_size_rot
    assert compute_vocab_size_rot() == 2112
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patch_sequence.py::test_patches_to_sequence_rvq_rot -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement rotation token sequence**

Add to `src/patch_sequence.py`:

```python
from src.rotation import quantize_rotation


def patches_to_token_sequence_rot(
    centroids, scales, rotations, codebook_tokens,
    n_pos_bins=256, n_scale_bins=64, n_rot_bins=64,
):
    """Convert patch data into 11-token-per-patch flat sequence with quaternion rotation.

    Token layout per patch:
        pos_x  : [0, n_pos_bins)
        pos_y  : [n_pos_bins, 2*n_pos_bins)
        pos_z  : [2*n_pos_bins, 3*n_pos_bins)
        scale  : [3*n_pos_bins, 3*n_pos_bins + n_scale_bins)
        rot_qx : [off_rot, off_rot + n_rot_bins)
        rot_qy : [off_rot + n_rot_bins, off_rot + 2*n_rot_bins)
        rot_qz : [off_rot + 2*n_rot_bins, off_rot + 3*n_rot_bins)
        rot_qw : [off_rot + 3*n_rot_bins, off_rot + 4*n_rot_bins)
        cb_L1  : [off_code, off_code + K)
        cb_L2  : [off_code, off_code + K)
        cb_L3  : [off_code, off_code + K)

    Args:
        centroids: (M, 3) float
        scales: (M,) float
        rotations: (M, 3, 3) rotation matrices (PCA Vt)
        codebook_tokens: (M, 3) RVQ indices
    """
    import numpy as np

    # Convert to numpy if needed
    if hasattr(centroids, 'numpy'):
        centroids = centroids.numpy()
    if hasattr(scales, 'numpy'):
        scales = scales.numpy()
    if hasattr(rotations, 'numpy'):
        rotations = rotations.numpy()
    if hasattr(codebook_tokens, 'numpy'):
        codebook_tokens = codebook_tokens.numpy()

    M = centroids.shape[0]
    off_y = n_pos_bins
    off_z = 2 * n_pos_bins
    off_scale = 3 * n_pos_bins
    off_rot = off_scale + n_scale_bins
    off_code = off_rot + 4 * n_rot_bins

    # Quantize positions
    pos_min = centroids.min(axis=0)
    pos_range = np.maximum(centroids.max(axis=0) - pos_min, 1e-8)
    pos_norm = (centroids - pos_min) / pos_range

    # Quantize scales
    s_min, s_max = scales.min(), scales.max()
    s_range = max(s_max - s_min, 1e-8)
    s_norm = (scales - s_min) / s_range

    # Morton sort (reuse existing function in same module)
    morton = morton_code_3d(torch.tensor(centroids, dtype=torch.float32), n_pos_bins)
    order = morton.argsort().numpy()

    sequence = []
    for idx in order:
        px = int(pos_norm[idx, 0] * (n_pos_bins - 1))
        py = int(pos_norm[idx, 1] * (n_pos_bins - 1)) + off_y
        pz = int(pos_norm[idx, 2] * (n_pos_bins - 1)) + off_z
        sc = int(s_norm[idx] * (n_scale_bins - 1)) + off_scale

        rot_bins = quantize_rotation(rotations[idx], n_rot_bins)
        qx = int(rot_bins[0]) + off_rot
        qy = int(rot_bins[1]) + off_rot + n_rot_bins
        qz = int(rot_bins[2]) + off_rot + 2 * n_rot_bins
        qw = int(rot_bins[3]) + off_rot + 3 * n_rot_bins

        c1 = int(codebook_tokens[idx, 0]) + off_code
        c2 = int(codebook_tokens[idx, 1]) + off_code
        c3 = int(codebook_tokens[idx, 2]) + off_code

        sequence.extend([px, py, pz, sc, qx, qy, qz, qw, c1, c2, c3])

    return torch.tensor(sequence, dtype=torch.int64)


def compute_vocab_size_rot(n_pos_bins=256, n_scale_bins=64, n_rot_bins=64, codebook_K=1024):
    """Total vocabulary size for 11-token rotation format."""
    return 3 * n_pos_bins + n_scale_bins + 4 * n_rot_bins + codebook_K
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patch_sequence.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/patch_sequence.py tests/test_patch_sequence.py
git commit -m "feat: 11-token rotation sequence encoding"
git push
```

---

### Task 8: Update Encode Sequences to Save Rotation

**Files:**
- Modify: `scripts/encode_sequences.py:87-114`

- [ ] **Step 1: Update encode_sequences.py to save principal_axes**

In the mesh loop (line 98-114), add loading of `principal_axes` from patch NPZs and save as quaternion:

```python
# After loading centroids and scales, also load rotations:
rotations = []
for pf in sorted(patch_files):
    data = np.load(str(pf))
    if "principal_axes" in data:
        rotations.append(data["principal_axes"])  # (3, 3)
    else:
        rotations.append(np.eye(3))

mesh_rotations = np.array(rotations, dtype=np.float32)  # (N, 3, 3)

# Update savez call:
np.savez(out_path,
         centroids=mesh_centroids,
         scales=mesh_scales,
         tokens=mesh_tokens,
         rotations=mesh_rotations)  # NEW: (N, 3, 3) PCA rotation matrices
```

- [ ] **Step 2: Verify by running on existing data**

```bash
python scripts/encode_sequences.py \
    --patch_dirs data/patches/lvis_wide/seen_train \
    --checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
    --output_dir data/sequences/rvq_lvis_rot \
    --mode rvq
```
Expected: Sequence NPZs now contain `rotations` key

- [ ] **Step 3: Verify NPZ contents**

```python
import numpy as np
d = np.load("data/sequences/rvq_lvis_rot/<first_file>.npz")
print(list(d.keys()))  # Should include 'rotations'
print(d["rotations"].shape)  # Should be (N, 3, 3)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/encode_sequences.py
git commit -m "feat: encode_sequences saves rotation matrices in sequence NPZ"
git push
```

---

### Task 9: Update MeshSequenceDataset for Rotation Mode

**Files:**
- Modify: `src/patch_dataset.py:202-242` (MeshSequenceDataset)

- [ ] **Step 1: Add rotation support to MeshSequenceDataset**

Update `MeshSequenceDataset.__init__` to accept `use_rotation=False`:

```python
class MeshSequenceDataset(Dataset):
    def __init__(self, sequence_dir, mode="rvq", max_seq_len=1024, use_rotation=False):
        self.sequence_dir = Path(sequence_dir)
        self.files = sorted(self.sequence_dir.glob("*_sequence.npz"))
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.use_rotation = use_rotation

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        centroids = data["centroids"]
        scales = data["scales"]
        tokens = data["tokens"]

        if self.use_rotation and "rotations" in data:
            from src.patch_sequence import patches_to_token_sequence_rot
            rotations = data["rotations"]
            seq = patches_to_token_sequence_rot(
                centroids, scales, rotations, tokens)
        else:
            from src.patch_sequence import patches_to_token_sequence
            seq = patches_to_token_sequence(centroids, scales, tokens, mode=self.mode)

        seq_len = min(len(seq), self.max_seq_len + 1)
        seq = seq[:seq_len]
        input_ids = np.zeros(self.max_seq_len, dtype=np.int64)
        target_ids = np.full(self.max_seq_len, -100, dtype=np.int64)
        input_ids[:seq_len - 1] = seq[:-1]
        target_ids[:seq_len - 1] = seq[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
```

- [ ] **Step 2: Write test for rotation mode dataset**

Add to `tests/test_patch_dataset.py` (or create if needed):

```python
def test_mesh_sequence_dataset_rotation_mode(tmp_path):
    """MeshSequenceDataset with use_rotation=True produces 11-token sequences."""
    import numpy as np
    M = 5  # patches
    np.savez(tmp_path / "test_sequence.npz",
             centroids=np.random.randn(M, 3).astype(np.float32),
             scales=np.random.rand(M).astype(np.float32) + 0.1,
             tokens=np.random.randint(0, 1024, (M, 3)),
             rotations=np.tile(np.eye(3), (M, 1, 1)).astype(np.float32))

    from src.patch_dataset import MeshSequenceDataset
    ds = MeshSequenceDataset(str(tmp_path), mode="rvq", max_seq_len=1430, use_rotation=True)
    input_ids, target_ids = ds[0]
    # 5 patches × 11 tokens = 55 tokens, so first 54 should be non-zero in input
    assert input_ids.shape == (1430,)
    assert (input_ids[:54] != 0).any()
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_patch_dataset.py::test_mesh_sequence_dataset_rotation_mode -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/patch_dataset.py
git commit -m "feat: MeshSequenceDataset supports rotation mode"
git push
```

---

### Task 10: Update AR Training Script for Rotation Mode

**Files:**
- Modify: `scripts/train_ar.py`

- [ ] **Step 1: Add `--rotation` flag to train_ar.py**

```python
# In argparse section, add:
parser.add_argument("--rotation", action="store_true",
                    help="Use 11-token rotation format (PCA pipeline)")
```

Update dataset and vocab size:

```python
# Replace dataset creation:
dataset = MeshSequenceDataset(
    args.sequence_dir, mode=args.mode,
    max_seq_len=args.max_seq_len, use_rotation=args.rotation)

# Replace vocab_size:
if args.rotation:
    from src.patch_sequence import compute_vocab_size_rot
    vocab_size = compute_vocab_size_rot(codebook_K=args.codebook_size)
else:
    vocab_size = compute_vocab_size(codebook_K=args.codebook_size)
```

Also update `max_seq_len` default: when `--rotation`, 130 patches × 11 = 1430 tokens, so need `--max_seq_len 1430`.

- [ ] **Step 2: Also save `rotation: True` in checkpoint config**

When `--rotation` is used, add to the config dict saved in checkpoints:

```python
config["rotation"] = True
config["tokens_per_patch"] = 11
```

This allows generation scripts to auto-detect the token format from the checkpoint.

- [ ] **Step 3: Update generate_v2_pipeline.py for 11-token decode**

The generation script needs to decode 11-token sequences (with quaternion rotation) when using PCA pipeline checkpoints. Add rotation-aware decode:

```python
# In generate_v2_pipeline.py, update the decode function:
def decode_sequence_to_patches_rot(sequence, vqvae, device,
                                    n_pos_bins=256, n_scale_bins=64, n_rot_bins=64):
    """Decode 11-token-per-patch sequence with quaternion rotation."""
    from src.rotation import dequantize_rotation
    tokens_per_patch = 11
    off_y = n_pos_bins
    off_z = 2 * n_pos_bins
    off_scale = 3 * n_pos_bins
    off_rot = off_scale + n_scale_bins
    off_code = off_rot + 4 * n_rot_bins

    n_patches = len(sequence) // tokens_per_patch
    all_world_verts = []

    for i in range(n_patches):
        base = i * tokens_per_patch
        pos_x = int(sequence[base + 0]) / 255.0
        pos_y = (int(sequence[base + 1]) - off_y) / 255.0
        pos_z = (int(sequence[base + 2]) - off_z) / 255.0
        scale_tok = int(sequence[base + 3]) - off_scale
        scale = max(scale_tok / 63.0, 0.01)

        # Quaternion rotation
        rot_bins = np.array([
            int(sequence[base + 4]) - off_rot,
            int(sequence[base + 5]) - off_rot - n_rot_bins,
            int(sequence[base + 6]) - off_rot - 2 * n_rot_bins,
            int(sequence[base + 7]) - off_rot - 3 * n_rot_bins,
        ])
        Vt = dequantize_rotation(rot_bins, n_rot_bins)

        # Codebook tokens
        tok1 = int(sequence[base + 8]) - off_code
        tok2 = int(sequence[base + 9]) - off_code
        tok3 = int(sequence[base + 10]) - off_code

        with torch.no_grad():
            tok_indices = torch.tensor([[tok1, tok2, tok3]], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok_indices)
            n_verts = torch.tensor([30], device=device)
            local_verts = vqvae.decoder(z_hat, n_verts)[0, :30].cpu().numpy()

        pos = np.array([pos_x, pos_y, pos_z])
        world_verts = (local_verts * scale) @ Vt + pos
        all_world_verts.append(world_verts)

    return all_world_verts
```

Add `--rotation` flag to the generation script CLI and use the appropriate decode function based on checkpoint config.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ar.py scripts/generate_v2_pipeline.py
git commit -m "feat: train_ar + generate_v2_pipeline support rotation mode"
git push
```

---

### Task 11: Phase B Full Retrain (RunPod)

**Prerequisites:** Tasks 7-10 complete, HF dataset available, RunPod pod running.

- [ ] **Step 1: Download processed patches from HF**

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Pthahnix/MeshLex-Patches',
    repo_type='dataset',
    local_dir='data/patches_full',
    allow_patterns=['metadata.json', 'splits.json'],
)
"
```

- [ ] **Step 2: Download training split patches**

```python
# Download only seen_train patches (use splits.json to filter)
import json
from huggingface_hub import hf_hub_download

with open("data/patches_full/splits.json") as f:
    splits = json.load(f)

# Download batch-by-batch based on which meshes are in seen_train
# (Implementation depends on HF repo structure)
```

- [ ] **Step 3: Train RVQ VQ-VAE on full data (PCA mode)**

```bash
python scripts/train_rvq.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint_dir data/checkpoints/rvq_full_pca \
    --epochs 100 --batch_size 1024
```
Expected: ~20-40h depending on subset strategy. Monitor with `nvidia-smi`.

- [ ] **Step 4: Upload VQ-VAE checkpoint to HF**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='data/checkpoints/rvq_full_pca/checkpoint_final.pt',
    path_in_repo='checkpoints/rvq_full_pca/checkpoint_final.pt',
    repo_id='Pthahnix/MeshLex-Research',
    repo_type='model',
)
print('✅ VQ-VAE PCA checkpoint uploaded')
"
```

- [ ] **Step 5: Encode sequences with rotation**

```bash
python scripts/encode_sequences.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --output_dir data/sequences/rvq_full_pca \
    --mode rvq
```

- [ ] **Step 6: Train AR model (11-token rotation format)**

```bash
python scripts/train_ar.py \
    --sequence_dir data/sequences/rvq_full_pca \
    --checkpoint_dir data/checkpoints/ar_full_pca \
    --rotation --max_seq_len 1430 \
    --d_model 512 --n_heads 8 --n_layers 8 \
    --batch_size 4 --grad_accum_steps 8 \
    --epochs 200 --warmup_epochs 10
```
Expected: ~40-60h. Loss should reach ~1.5 or lower.

- [ ] **Step 7: Upload AR checkpoint to HF**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='data/checkpoints/ar_full_pca/checkpoint_final.pt',
    path_in_repo='checkpoints/ar_full_pca/checkpoint_final.pt',
    repo_id='Pthahnix/MeshLex-Research',
    repo_type='model',
)
print('✅ AR PCA checkpoint uploaded')
"
```

- [ ] **Step 8: Generate and evaluate (PCA pipeline)**

```bash
python scripts/generate_v2_pipeline.py \
    --vqvae_checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --ar_checkpoint data/checkpoints/ar_full_pca/checkpoint_final.pt \
    --output_dir results/generation_pca \
    --n_meshes 10 --temperatures 0.7 0.8 0.9 1.0
```

- [ ] **Step 9: Commit results**

```bash
git add results/generation_pca/
git commit -m "results: Phase B — PCA+quaternion pipeline generation"
git push
```

---

## Phase C: No-PCA Baseline Retrain (Tasks 12-14)

### Task 12: No-PCA Dataset Support

**Files:**
- Modify: `src/patch_dataset.py` (PatchGraphDataset)

- [ ] **Step 1: Add `use_nopca` flag to PatchGraphDataset**

In `PatchGraphDataset.__init__`, add `use_nopca=False` parameter. When True, load `local_vertices_nopca` instead of `local_vertices` from patch NPZs:

```python
class PatchGraphDataset(Dataset):
    def __init__(self, patch_dirs, max_verts=30, use_nopca=False):
        self.use_nopca = use_nopca
        # ... existing init code ...

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        if self.use_nopca and "local_vertices_nopca" in data:
            verts = data["local_vertices_nopca"]
        else:
            verts = data["local_vertices"]
        # ... rest of existing code ...
```

- [ ] **Step 2: Commit**

```bash
git add src/patch_dataset.py
git commit -m "feat: PatchGraphDataset supports no-PCA mode"
git push
```

---

### Task 13: No-PCA Training Scripts + Encode Support

**Files:**
- Modify: `scripts/train_rvq.py`
- Modify: `scripts/encode_sequences.py`

- [ ] **Step 1: Add `--nopca` flag to train_rvq.py**

```python
parser.add_argument("--nopca", action="store_true",
                    help="Train on non-PCA-normalized vertices")
```

Pass to dataset:
```python
dataset = PatchGraphDataset(args.patch_dirs, use_nopca=args.nopca)
```

- [ ] **Step 2: Add `--nopca` flag to encode_sequences.py**

The encode script uses `PatchGraphDataset` internally to load patch vertices for VQ-VAE encoding. When running the no-PCA pipeline, it must load `local_vertices_nopca` instead of `local_vertices`:

```python
# In argparse:
parser.add_argument("--nopca", action="store_true",
                    help="Use non-PCA-normalized vertices for encoding")

# When creating the dataset or loading patches:
# In the mesh loop, when loading patch NPZ:
if args.nopca and "local_vertices_nopca" in data:
    verts = data["local_vertices_nopca"]
else:
    verts = data["local_vertices"]
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_rvq.py scripts/encode_sequences.py
git commit -m "feat: train_rvq and encode_sequences support --nopca flag"
git push
```

---

### Task 14: Phase C Full Retrain (RunPod)

**Prerequisites:** Task 12-13 complete, HF dataset available.

- [ ] **Step 1: Train RVQ VQ-VAE (no-PCA mode)**

```bash
python scripts/train_rvq.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint_dir data/checkpoints/rvq_full_nopca \
    --nopca --epochs 100 --batch_size 1024
```

- [ ] **Step 2: Upload VQ-VAE checkpoint**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='data/checkpoints/rvq_full_nopca/checkpoint_final.pt',
    path_in_repo='checkpoints/rvq_full_nopca/checkpoint_final.pt',
    repo_id='Pthahnix/MeshLex-Research',
    repo_type='model',
)
print('✅ VQ-VAE no-PCA checkpoint uploaded')
"
```

- [ ] **Step 3: Encode sequences (no rotation — standard 7-token format, nopca vertices)**

```bash
python scripts/encode_sequences.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_nopca/checkpoint_final.pt \
    --output_dir data/sequences/rvq_full_nopca \
    --mode rvq --nopca
```

- [ ] **Step 4: Train AR model (7-token, no rotation)**

```bash
python scripts/train_ar.py \
    --sequence_dir data/sequences/rvq_full_nopca \
    --checkpoint_dir data/checkpoints/ar_full_nopca \
    --max_seq_len 1024 \
    --d_model 512 --n_heads 8 --n_layers 8 \
    --batch_size 4 --grad_accum_steps 8 \
    --epochs 200 --warmup_epochs 10
```
Expected: ~30-40h. Vocab size = 1852 (no rotation tokens).

- [ ] **Step 5: Upload AR checkpoint**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='data/checkpoints/ar_full_nopca/checkpoint_final.pt',
    path_in_repo='checkpoints/ar_full_nopca/checkpoint_final.pt',
    repo_id='Pthahnix/MeshLex-Research',
    repo_type='model',
)
print('✅ AR no-PCA checkpoint uploaded')
"
```

- [ ] **Step 6: Generate and evaluate (no-PCA pipeline)**

```bash
python scripts/generate_v2_pipeline.py \
    --vqvae_checkpoint data/checkpoints/rvq_full_nopca/checkpoint_final.pt \
    --ar_checkpoint data/checkpoints/ar_full_nopca/checkpoint_final.pt \
    --output_dir results/generation_nopca \
    --n_meshes 10 --temperatures 0.7 0.8 0.9 1.0
```

- [ ] **Step 7: Commit results**

```bash
git add results/generation_nopca/
git commit -m "results: Phase C — no-PCA baseline generation"
git push
```

---

## Phase E: Ablation Comparison (Tasks 15-16)

### Task 15: Ablation Comparison Script

**Files:**
- Create: `scripts/ablation_comparison.py`

- [ ] **Step 1: Write ablation comparison script**

```python
# scripts/ablation_comparison.py
"""Phase E: Compare PCA+quaternion vs no-PCA pipelines.

Metrics:
  - Chamfer Distance (reconstruction quality)
  - Codebook utilization
  - Token diversity (unique tokens in generated sequences)
  - Spatial coherence (pairwise CD between generated meshes)
  - Visual comparison (side-by-side renders)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_eval_results(eval_dir):
    """Load evaluation_results.json from a generation directory."""
    path = Path(eval_dir) / "evaluation_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compare_pipelines(pca_dir, nopca_dir, output_dir):
    """Generate comparison report and visualizations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pca_results = load_eval_results(pca_dir)
    nopca_results = load_eval_results(nopca_dir)

    if not pca_results or not nopca_results:
        print("ERROR: Missing evaluation results. Run evaluate_generation.py first.")
        return

    # Compare per-temperature metrics
    comparison = {}
    for temp in ["0.7", "0.8", "0.9", "1.0"]:
        pca = pca_results.get("per_temperature", {}).get(temp, {})
        nopca = nopca_results.get("per_temperature", {}).get(temp, {})
        comparison[temp] = {
            "pca_cd": pca.get("mean_pairwise_cd", 0),
            "nopca_cd": nopca.get("mean_pairwise_cd", 0),
            "pca_spread": pca.get("mean_spatial_spread", 0),
            "nopca_spread": nopca.get("mean_spatial_spread", 0),
        }

    # Save comparison
    with open(out / "ablation_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Generate comparison chart
    temps = list(comparison.keys())
    pca_cds = [comparison[t]["pca_cd"] for t in temps]
    nopca_cds = [comparison[t]["nopca_cd"] for t in temps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(temps))
    w = 0.35
    axes[0].bar(x - w/2, pca_cds, w, label="PCA+Quaternion", color="steelblue")
    axes[0].bar(x + w/2, nopca_cds, w, label="No-PCA", color="coral")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Mean Pairwise CD")
    axes[0].set_title("Generation Diversity (Chamfer Distance)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"T={t}" for t in temps])
    axes[0].legend()

    pca_spreads = [comparison[t]["pca_spread"] for t in temps]
    nopca_spreads = [comparison[t]["nopca_spread"] for t in temps]
    axes[1].bar(x - w/2, pca_spreads, w, label="PCA+Quaternion", color="steelblue")
    axes[1].bar(x + w/2, nopca_spreads, w, label="No-PCA", color="coral")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Spatial Spread")
    axes[1].set_title("Spatial Coherence")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"T={t}" for t in temps])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out / "ablation_comparison.png", dpi=150)
    plt.close()

    print(f"Ablation comparison saved to {out}")
    print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_dir", default="results/generation_pca")
    parser.add_argument("--nopca_dir", default="results/generation_nopca")
    parser.add_argument("--output_dir", default="results/ablation")
    args = parser.parse_args()
    compare_pipelines(args.pca_dir, args.nopca_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/ablation_comparison.py
git commit -m "feat: ablation comparison script — PCA vs no-PCA"
git push
```

---

### Task 16: Run Ablation and Final Report

- [ ] **Step 1: Run evaluation on both pipelines**

```bash
python scripts/evaluate_generation.py \
    --gen_dir results/generation_pca --output_dir results/generation_pca

python scripts/evaluate_generation.py \
    --gen_dir results/generation_nopca --output_dir results/generation_nopca
```

- [ ] **Step 2: Run ablation comparison**

```bash
python scripts/ablation_comparison.py \
    --pca_dir results/generation_pca \
    --nopca_dir results/generation_nopca \
    --output_dir results/ablation
```

- [ ] **Step 3: Run reconstruction comparison with rotation fix**

```bash
python scripts/visualize_mesh_comparison.py \
    --vqvae_checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --ar_checkpoint data/checkpoints/ar_full_pca/checkpoint_final.pt \
    --patch_dir data/patches_full/seen_train \
    --output_dir results/mesh_comparison_v3 \
    --n_recon 10 --n_gen 10
```

- [ ] **Step 4: Commit all results**

```bash
git add results/ablation/ results/mesh_comparison_v3/
git commit -m "results: Phase E — ablation comparison PCA vs no-PCA"
git push
```

- [ ] **Step 5: Update CLAUDE.md with new status**

Update the Current Status section to reflect completed phases and new checkpoint locations.

```bash
git add CLAUDE.md
git commit -m "docs: update status — assembly fix + full retrain complete"
git push
```

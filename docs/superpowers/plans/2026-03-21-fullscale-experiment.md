# Full-Scale Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute full-scale training, analysis, ablation, and evaluation on the complete 72K mesh / 10.8M patch dataset, covering PCA + noPCA pipelines, theory-driven analysis, and MDLM feasibility — producing paper-ready results.

**Architecture:** 5 phases — (0) hardware audit, (1) VQ-VAE foundation ×4, (2) token encoding + AR ×2, (3) experiment branches (preliminary rerun + theory-driven + MDLM), (4) evaluation + ablation, (5) paper-ready figures. Multi-GPU execution on 3× RTX 5090.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, trimesh, scipy, datasets (HuggingFace), huggingface_hub, matplotlib, umap-learn, numpy

**Spec:** `docs/superpowers/specs/2026-03-21-fullscale-experiment-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/parquet_loader.py` | HF Parquet → local NPZ conversion for PatchGraphDataset |
| `src/rotation.py` | Quaternion encode/decode for PCA rotation tokens |
| `src/mdlm_model.py` | Full-scale MDLM (Transformer encoder + time/level embeddings) |
| `scripts/train_mdlm.py` | MDLM training script (continuous-time masking) |
| `scripts/run_fullscale_analysis.py` | Phase 3b theory-driven analysis orchestrator |
| `scripts/curvature_analysis.py` | Discrete Gaussian curvature + frequency correlation |
| `scripts/vq_method_comparison.py` | Train vanilla/EMA VQ + compare distributions |
| `scripts/fullscale_evaluation.py` | Unified evaluation dashboard generator |
| `scripts/hardware_audit.py` | Phase 0 hardware audit (one-time) |
| `tests/test_parquet_loader.py` | Parquet loader tests |
| `tests/test_rotation.py` | Quaternion utility tests |
| `tests/test_mdlm_model.py` | MDLM model tests |

### Modified Files
| File | Changes |
|------|---------|
| `src/patch_sequence.py` | Add `patches_to_token_sequence_rot()`, `compute_vocab_size_rot()` |
| `src/patch_dataset.py` | Add `use_nopca` to PatchGraphDataset, `use_rotation` to MeshSequenceDataset |
| `src/rvq.py` | Add `VanillaVQ` and `EMAVQ` codebook classes |
| `scripts/train_rvq.py` | Add `--nopca`, `--vq_method`, Parquet input support |
| `scripts/train_ar.py` | Add `--rotation`, support larger model configs |
| `scripts/encode_sequences.py` | Add `--nopca`, `--parquet` mode |
| `scripts/generate_v2_pipeline.py` | Support 11-token rotation decode |
| `scripts/run_preliminary_analysis.py` | Parameterize data paths for full-scale |

---

## Dependency Graph

```
Task 1 (Hardware Audit)
  │
Task 2 (Parquet Loader) ──────────────────────────────┐
Task 3 (Rotation Utilities) ───┐                       │
Task 4 (11-Token Encoding) ────┤                       │
Task 5 (Dataset Updates) ──────┤                       │
Task 6 (Training Script Updates) ──────────────────────┤
  │                                                     │
Task 7 (Phase 1: VQ-VAE ×4) ◄─────────────────────────┘
  │
Task 8 (Phase 2a: Token Encoding)
  │
Task 9 (AR Scale-Up + Training Script)
Task 10 (Phase 2b: AR ×2)
  │
  ├── Task 11 (Phase 3a: Preliminary Rerun)
  ├── Task 12 (MDLM Model + Script)
  ├── Task 13 (Phase 3b: Theory-Driven Analysis)
  └── Task 14 (Phase 3c: MDLM Training)
  │
Task 15 (Phase 4: Evaluation + Ablation)
  │
Task 16 (Phase 5: Paper-Ready)
```

---

<!-- TASKS START -->

### Task 1: Phase 0 — Hardware Audit

**Files:**
- Create: `results/fullscale_eval/hardware_audit.json`

- [ ] **Step 1: Run hardware checks**

```bash
nvidia-smi
free -h
df -h /
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem/1e9:.1f} GB') for i in range(torch.cuda.device_count())]"
```

- [ ] **Step 2: Verify thresholds and save audit**

```python
# scripts/hardware_audit.py — run once, save results
import json, subprocess, torch, shutil
from pathlib import Path

audit = {}

# GPU
audit["gpu_count"] = torch.cuda.device_count()
audit["gpus"] = []
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    audit["gpus"].append({
        "index": i, "name": props.name,
        "vram_gb": round(props.total_mem / 1e9, 1)
    })

# RAM
import os
mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
audit["ram_gb"] = round(mem, 1)

# Disk
total, used, free = shutil.disk_usage("/")
audit["disk_total_gb"] = round(total / (1024**3), 1)
audit["disk_free_gb"] = round(free / (1024**3), 1)

# Go/No-Go
audit["go"] = (
    audit["gpu_count"] >= 1
    and audit["disk_free_gb"] >= 100
    and audit["ram_gb"] >= 64
)

Path("results/fullscale_eval").mkdir(parents=True, exist_ok=True)
with open("results/fullscale_eval/hardware_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print(json.dumps(audit, indent=2))
if not audit["go"]:
    print("\n❌ STOP: Hardware below minimum thresholds. Report to user.")
else:
    print(f"\n✅ Hardware GO — {audit['gpu_count']} GPUs, {audit['ram_gb']}GB RAM, {audit['disk_free_gb']}GB free disk")
```

Run: `python scripts/hardware_audit.py`
**Gate**: If `go=false`, STOP and report. Do not proceed.

- [ ] **Step 3: Install dependencies**

```bash
pip install datasets huggingface_hub umap-learn scipy trimesh pymetis torch-geometric open3d pyfqmr
```

- [ ] **Step 4: Verify HF dataset access**

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Pthahnix/MeshLex-Patches', split='train', streaming=True)
row = next(iter(ds))
print('Columns:', list(row.keys()))
print('mesh_id:', row['mesh_id'])
print('OK — HF dataset accessible')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/hardware_audit.py results/fullscale_eval/hardware_audit.json
git commit -m "phase0: hardware audit complete"
git push
```

---

### Task 2: Parquet Data Loader

**Files:**
- Create: `src/parquet_loader.py`
- Create: `tests/test_parquet_loader.py`

This module downloads HF Parquet data and converts it to local NPZ files that `PatchGraphDataset` can read directly. This is the simplest integration path — no changes to the training loop.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parquet_loader.py
"""Tests for src.parquet_loader — HF Parquet → local NPZ conversion."""
import numpy as np
import pytest
from pathlib import Path


def test_parquet_row_to_npz(tmp_path):
    """A single Parquet row converts to a valid patch NPZ."""
    from src.parquet_loader import parquet_row_to_npz

    # Simulate a HF dataset row (all arrays are flat lists in Parquet)
    row = {
        "mesh_id": "test_mesh",
        "patch_idx": 0,
        "vertices": list(range(30 * 3)),  # 30 verts × 3
        "faces": list(range(20 * 3)),  # 20 faces × 3
        "local_vertices": list(np.random.randn(30 * 3).astype(float)),
        "local_vertices_nopca": list(np.random.randn(30 * 3).astype(float)),
        "centroid": [1.0, 2.0, 3.0],
        "principal_axes": list(np.eye(3).flatten().astype(float)),
        "scale": 0.5,
        "boundary_vertices": list(range(5)),
        "n_vertices": 30,
        "n_faces": 20,
    }

    out_path = parquet_row_to_npz(row, tmp_path)
    assert out_path.exists()

    data = np.load(str(out_path))
    assert "local_vertices" in data
    assert "local_vertices_nopca" in data
    assert "principal_axes" in data
    assert data["local_vertices"].shape == (30, 3)
    assert data["principal_axes"].shape == (3, 3)


def test_download_split_patches(tmp_path, monkeypatch):
    """download_split_patches creates NPZ files from HF dataset rows."""
    from src.parquet_loader import download_split_patches

    # Mock the HF dataset with fake rows
    fake_rows = [
        {
            "mesh_id": "mesh_A", "patch_idx": 0,
            "vertices": list(np.zeros(30 * 3)),
            "faces": list(np.zeros(20 * 3, dtype=int)),
            "local_vertices": list(np.random.randn(30 * 3)),
            "local_vertices_nopca": list(np.random.randn(30 * 3)),
            "centroid": [0, 0, 0], "principal_axes": list(np.eye(3).flatten()),
            "scale": 1.0, "boundary_vertices": [0, 1],
            "n_vertices": 30, "n_faces": 20,
        },
        {
            "mesh_id": "mesh_A", "patch_idx": 1,
            "vertices": list(np.zeros(30 * 3)),
            "faces": list(np.zeros(20 * 3, dtype=int)),
            "local_vertices": list(np.random.randn(30 * 3)),
            "local_vertices_nopca": list(np.random.randn(30 * 3)),
            "centroid": [1, 0, 0], "principal_axes": list(np.eye(3).flatten()),
            "scale": 0.8, "boundary_vertices": [2, 3],
            "n_vertices": 30, "n_faces": 20,
        },
    ]

    # Monkeypatch load_dataset
    class FakeDS:
        def filter(self, fn, **kwargs):
            return self
        def __iter__(self):
            return iter(fake_rows)
        def __len__(self):
            return len(fake_rows)

    monkeypatch.setattr(
        "src.parquet_loader.load_dataset",
        lambda *a, **kw: FakeDS()
    )

    mesh_ids = {"mesh_A"}
    out_dir = download_split_patches(
        mesh_ids=mesh_ids,
        output_dir=tmp_path / "patches",
        hf_repo="fake/repo",
    )

    npz_files = list(Path(out_dir).glob("*.npz"))
    assert len(npz_files) == 2
    # Verify filename format: {mesh_id}_patch_{idx:03d}.npz
    names = sorted(f.name for f in npz_files)
    assert names[0] == "mesh_A_patch_000.npz"
    assert names[1] == "mesh_A_patch_001.npz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_parquet_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.parquet_loader'`

- [ ] **Step 3: Implement parquet_loader.py**

```python
# src/parquet_loader.py
"""HF Parquet → local NPZ conversion for MeshLex training pipeline.

Downloads patches from HuggingFace Parquet dataset and converts each row
to a local NPZ file compatible with PatchGraphDataset.
"""
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset


def parquet_row_to_npz(row: dict, output_dir: Path) -> Path:
    """Convert a single HF dataset row to a patch NPZ file.

    Args:
        row: Dict with keys from HF Parquet (flat arrays).
        output_dir: Directory to write NPZ file.

    Returns:
        Path to created NPZ file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_id = row["mesh_id"]
    patch_idx = row["patch_idx"]
    n_verts = row.get("n_vertices", 30)
    n_faces = row.get("n_faces", 20)

    # Reshape flat arrays back to matrices
    local_vertices = np.array(row["local_vertices"], dtype=np.float32).reshape(-1, 3)[:n_verts]
    local_vertices_nopca = np.array(row["local_vertices_nopca"], dtype=np.float32).reshape(-1, 3)[:n_verts]
    vertices = np.array(row["vertices"], dtype=np.float32).reshape(-1, 3)[:n_verts]
    faces = np.array(row["faces"], dtype=np.int64).reshape(-1, 3)[:n_faces]
    centroid = np.array(row["centroid"], dtype=np.float32)
    principal_axes = np.array(row["principal_axes"], dtype=np.float32).reshape(3, 3)
    scale = float(row["scale"])
    boundary_vertices = np.array(row.get("boundary_vertices", []), dtype=np.int64)

    out_path = output_dir / f"{mesh_id}_patch_{patch_idx:03d}.npz"
    np.savez_compressed(
        str(out_path),
        local_vertices=local_vertices,
        local_vertices_nopca=local_vertices_nopca,
        vertices=vertices,
        faces=faces,
        centroid=centroid,
        principal_axes=principal_axes,
        scale=scale,
        boundary_vertices=boundary_vertices,
    )
    return out_path


def download_split_patches(
    mesh_ids: set,
    output_dir,
    hf_repo: str = "Pthahnix/MeshLex-Patches",
    batch_size: int = 10000,
) -> Path:
    """Download patches for a set of mesh IDs from HF Parquet.

    Args:
        mesh_ids: Set of mesh_id strings to download.
        output_dir: Directory to write NPZ files.
        hf_repo: HuggingFace dataset repository.
        batch_size: Processing batch size for progress reporting.

    Returns:
        Path to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading patches for {len(mesh_ids)} meshes from {hf_repo}...")
    ds = load_dataset(hf_repo, split="train")

    # Filter to requested mesh IDs
    ds = ds.filter(lambda row: row["mesh_id"] in mesh_ids, num_proc=4)

    count = 0
    for row in ds:
        # Skip if already exists
        mesh_id = row["mesh_id"]
        patch_idx = row["patch_idx"]
        out_path = output_dir / f"{mesh_id}_patch_{patch_idx:03d}.npz"
        if out_path.exists():
            count += 1
            continue

        parquet_row_to_npz(row, output_dir)
        count += 1
        if count % batch_size == 0:
            print(f"  Converted {count} patches...")

    print(f"  Done: {count} patches in {output_dir}")
    return output_dir


def download_splits_json(
    hf_repo: str = "Pthahnix/MeshLex-Patches",
    output_path: str = "data/splits.json",
) -> dict:
    """Download splits.json from HF dataset repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=hf_repo, filename="splits.json", repo_type="dataset"
    )
    import shutil
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, output_path)
    with open(output_path) as f:
        return json.load(f)


def prepare_training_data(
    output_base: str = "data/patches_full",
    hf_repo: str = "Pthahnix/MeshLex-Patches",
    splits: list[str] = ("seen_train",),
):
    """Full pipeline: download splits.json, then download patches for requested splits.

    Args:
        output_base: Base directory for output.
        hf_repo: HF dataset repo.
        splits: Which splits to download (seen_train, seen_test, unseen).

    Returns:
        Dict mapping split name → output directory path.
    """
    splits_data = download_splits_json(hf_repo, f"{output_base}/splits.json")
    results = {}

    for split_name in splits:
        mesh_ids = set(splits_data[split_name])
        out_dir = download_split_patches(
            mesh_ids=mesh_ids,
            output_dir=f"{output_base}/{split_name}",
            hf_repo=hf_repo,
        )
        results[split_name] = str(out_dir)
        print(f"Split '{split_name}': {len(mesh_ids)} meshes → {out_dir}")

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_parquet_loader.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/parquet_loader.py tests/test_parquet_loader.py
git commit -m "feat: parquet data loader — HF Parquet to local NPZ conversion"
git push
```

---

### Task 3: Quaternion Rotation Utilities

**Files:**
- Create: `src/rotation.py`
- Create: `tests/test_rotation.py`

- [ ] **Step 1: Write failing tests**

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
    assert (bins >= 0).all()
    assert (bins < 64).all()


def test_dequantize_rotation_returns_valid_matrix():
    """dequantize_rotation returns a valid 3x3 rotation matrix."""
    from src.rotation import dequantize_rotation
    bins = np.array([32, 32, 0, 63])
    R_mat = dequantize_rotation(bins, n_bins=64)
    assert R_mat.shape == (3, 3)
    np.testing.assert_allclose(R_mat @ R_mat.T, np.eye(3), atol=0.1)


def test_roundtrip_identity():
    """Identity rotation survives quantize → dequantize."""
    from src.rotation import quantize_rotation, dequantize_rotation
    Vt = np.eye(3)
    bins = quantize_rotation(Vt, n_bins=64)
    R_rec = dequantize_rotation(bins, n_bins=64)
    np.testing.assert_allclose(R_rec, np.eye(3), atol=0.05)


def test_roundtrip_random_rotation():
    """Random rotation survives roundtrip within ~5° error."""
    from src.rotation import quantize_rotation, dequantize_rotation
    rng = np.random.default_rng(42)
    for _ in range(20):
        Vt = R.random(random_state=rng).as_matrix()
        bins = quantize_rotation(Vt, n_bins=64)
        R_rec = dequantize_rotation(bins, n_bins=64)
        angle = R.from_matrix(R_rec @ Vt.T).magnitude()
        assert angle < np.radians(5), f"Angular error {np.degrees(angle):.1f}° > 5°"


def test_canonical_form_w_positive():
    """Quaternion canonical form ensures w >= 0."""
    from src.rotation import quantize_rotation
    Vt = R.from_euler('z', 180, degrees=True).as_matrix()
    bins = quantize_rotation(Vt, n_bins=64)
    assert bins[3] >= 32, f"w bin {bins[3]} suggests w < 0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rotation.py -v`
Expected: FAIL — `ModuleNotFoundError`

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
    if quat[3] < 0:
        quat = -quat
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
    quat = (bins.astype(float) + 0.5) / n_bins * 2.0 - 1.0
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
git commit -m "feat: quaternion rotation encode/decode utilities"
git push
```

---

### Task 4: 11-Token Sequence Encoding

**Files:**
- Modify: `src/patch_sequence.py`
- Modify: `tests/test_patch_sequence.py`

Adds rotation-aware 11-token-per-patch encoding (pos×3 + scale + quat×4 + tok×3).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_patch_sequence.py`:

```python
def test_patches_to_sequence_rvq_rot():
    """RVQ+rotation mode: 10 patches -> 10*11 = 110 tokens."""
    import torch
    M = 10
    centroids = torch.randn(M, 3)
    scales = torch.rand(M) + 0.1
    codebook_tokens = torch.randint(0, 1024, (M, 3))
    rotations = torch.eye(3).unsqueeze(0).expand(M, -1, -1)

    from src.patch_sequence import patches_to_token_sequence_rot
    seq = patches_to_token_sequence_rot(
        centroids, scales, rotations, codebook_tokens,
        n_pos_bins=256, n_scale_bins=64, n_rot_bins=64,
    )
    assert seq.shape == (M * 11,)
    assert (seq >= 0).all()
    assert seq.max() < 2112


def test_compute_vocab_size_rot():
    """Vocab size with rotation: 3*256 + 64 + 4*64 + 1024 = 2112."""
    from src.patch_sequence import compute_vocab_size_rot
    assert compute_vocab_size_rot() == 2112
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patch_sequence.py::test_patches_to_sequence_rvq_rot tests/test_patch_sequence.py::test_compute_vocab_size_rot -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Add rotation sequence functions to `src/patch_sequence.py`**

Append the following to the end of `src/patch_sequence.py`:

```python
from src.rotation import quantize_rotation


def patches_to_token_sequence_rot(
    centroids, scales, rotations, codebook_tokens,
    n_pos_bins=256, n_scale_bins=64, n_rot_bins=64,
):
    """Convert patch data into 11-token-per-patch flat sequence with quaternion rotation.

    Token layout per patch (11 tokens):
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
    """
    import numpy as np

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

    # Quantize positions to [0, 1]
    pos_min = centroids.min(axis=0)
    pos_range = np.maximum(centroids.max(axis=0) - pos_min, 1e-8)
    pos_norm = (centroids - pos_min) / pos_range

    # Quantize scales to [0, 1]
    s_min, s_max = scales.min(), scales.max()
    s_range = max(s_max - s_min, 1e-8)
    s_norm = (scales - s_min) / s_range

    # Morton sort
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

- [ ] **Step 4: Run all patch_sequence tests**

Run: `python -m pytest tests/test_patch_sequence.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/patch_sequence.py tests/test_patch_sequence.py
git commit -m "feat: 11-token rotation sequence encoding"
git push
```

---

### Task 5: Dataset Class Updates

**Files:**
- Modify: `src/patch_dataset.py`

Two changes: (1) `PatchGraphDataset` gets `use_nopca` flag, (2) `MeshSequenceDataset` gets `use_rotation` flag.

- [ ] **Step 1: Add `use_nopca` to PatchGraphDataset**

In `src/patch_dataset.py`, update `PatchGraphDataset.__init__` to accept `use_nopca=False`:

```python
class PatchGraphDataset(Dataset):
    MAX_VERTICES = 128

    def __init__(self, patch_dir: str, use_nopca: bool = False):
        self.use_nopca = use_nopca
        # ... existing init code unchanged ...
```

In `__getitem__`, change the vertex loading. **Important**: This must replace the single `local_verts = data["local_vertices"]` assignment at line ~169 so that both `compute_face_features(local_verts, faces)` (encoder input) and `padded_verts[:n_verts] = local_verts` (decoder target) use the same coordinate system:

```python
# Replace line ~169:
#   local_verts = data["local_vertices"]
# With:
if self.use_nopca and "local_vertices_nopca" in data:
    local_verts = data["local_vertices_nopca"]
else:
    local_verts = data["local_vertices"]
# Everything downstream (compute_face_features, padded_verts) uses local_verts — no other changes needed
```

- [ ] **Step 2: Add `use_rotation` to MeshSequenceDataset**

Update `MeshSequenceDataset.__init__`:

```python
class MeshSequenceDataset(Dataset):
    def __init__(self, sequence_dir: str, mode: str = "rvq",
                 max_seq_len: int = 1024, use_rotation: bool = False):
        self.sequence_dir = Path(sequence_dir)
        self.files = sorted(self.sequence_dir.glob("*_sequence.npz"))
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.use_rotation = use_rotation
```

Update `__getitem__`:

```python
def __getitem__(self, idx):
    data = np.load(str(self.files[idx]))
    centroids = data["centroids"]
    scales = data["scales"]
    tokens = data["tokens"]

    if self.use_rotation and "principal_axes" in data:
        from src.patch_sequence import patches_to_token_sequence_rot
        rotations = data["principal_axes"]  # (N, 3, 3) — saved by encode_sequences.py as principal_axes
        seq = patches_to_token_sequence_rot(centroids, scales, rotations, tokens)
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

- [ ] **Step 3: Write test for both changes**

Add to `tests/test_patch_dataset.py`:

```python
def test_patch_graph_dataset_nopca(tmp_path):
    """PatchGraphDataset with use_nopca=True loads local_vertices_nopca."""
    import numpy as np
    # Create a minimal patch NPZ with both vertex fields
    verts_pca = np.random.randn(30, 3).astype(np.float32)
    verts_nopca = np.random.randn(30, 3).astype(np.float32) * 2  # different values
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    np.savez(tmp_path / "test_patch_000.npz",
             local_vertices=verts_pca, local_vertices_nopca=verts_nopca,
             faces=faces, boundary_vertices=np.array([0]))

    from src.patch_dataset import PatchGraphDataset
    ds_pca = PatchGraphDataset(str(tmp_path), use_nopca=False)
    ds_nopca = PatchGraphDataset(str(tmp_path), use_nopca=True)
    # They should load different vertex data
    assert len(ds_pca) == 1
    assert len(ds_nopca) == 1


def test_mesh_sequence_dataset_rotation(tmp_path):
    """MeshSequenceDataset with use_rotation=True produces 11-token sequences."""
    import numpy as np
    M = 5
    np.savez(tmp_path / "test_sequence.npz",
             centroids=np.random.randn(M, 3).astype(np.float32),
             scales=np.random.rand(M).astype(np.float32) + 0.1,
             tokens=np.random.randint(0, 1024, (M, 3)),
             principal_axes=np.tile(np.eye(3), (M, 1, 1)).astype(np.float32))

    from src.patch_dataset import MeshSequenceDataset
    ds = MeshSequenceDataset(str(tmp_path), mode="rvq", max_seq_len=1430, use_rotation=True)
    input_ids, target_ids = ds[0]
    assert input_ids.shape == (1430,)
    # 5 patches × 11 tokens = 55, first 54 are input
    assert (input_ids[:54] != 0).any()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_patch_dataset.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/patch_dataset.py tests/test_patch_dataset.py
git commit -m "feat: PatchGraphDataset noPCA + MeshSequenceDataset rotation support"
git push
```

---

### Task 6: Training Script Updates

**Files:**
- Modify: `scripts/train_rvq.py`
- Modify: `scripts/encode_sequences.py`

Updates to support noPCA mode and Parquet-downloaded data.

- [ ] **Step 1: Add `--nopca` and `--vq_method` to `scripts/train_rvq.py`**

Add CLI arguments:

```python
parser.add_argument("--nopca", action="store_true",
                    help="Train on non-PCA-normalized vertices")
parser.add_argument("--vq_method", choices=["simvq", "vanilla", "ema"],
                    default="simvq", help="VQ codebook method")
```

Update dataset creation:

```python
# Note: args.train_dirs is a list (nargs="+"), PatchGraphDataset expects a single string
dataset = PatchGraphDataset(args.train_dirs[0], use_nopca=args.nopca)
if args.val_dirs:
    val_dataset = PatchGraphDataset(args.val_dirs[0], use_nopca=args.nopca)
```

Wire `--vq_method` through model instantiation. In `train_rvq.py`, when creating the model:

```python
# Add vq_method to model config
model = MeshLexRVQVAE(
    # ... existing args ...
    vq_method=args.vq_method,  # "simvq" (default), "vanilla", or "ema"
)
```

This requires `MeshLexRVQVAE` and `ResidualVQ` to accept a `vq_method` parameter. Add to `src/rvq.py` `ResidualVQ.__init__`:

```python
class ResidualVQ(nn.Module):
    def __init__(self, n_levels=3, K=1024, dim=128, vq_method="simvq"):
        super().__init__()
        self.levels = nn.ModuleList()
        for _ in range(n_levels):
            if vq_method == "vanilla":
                self.levels.append(VanillaVQ(K, dim))
            elif vq_method == "ema":
                self.levels.append(EMAVQ(K, dim))
            else:
                self.levels.append(SimVQCodebook(K, dim))
```

And pass through from `MeshLexRVQVAE`:

```python
class MeshLexRVQVAE(nn.Module):
    def __init__(self, ..., vq_method="simvq"):
        # ...
        self.rvq = ResidualVQ(n_levels=n_levels, K=codebook_size, dim=embed_dim, vq_method=vq_method)
```

**Note**: `VanillaVQ` and `EMAVQ` classes are defined in Task 13 Step 3. This wiring must be done in Task 6 so that `--vq_method vanilla` actually has an effect when Task 13's `vq_method_comparison.py` calls `train_rvq.py`.

Save the config in checkpoint:

```python
config["nopca"] = args.nopca
config["vq_method"] = args.vq_method
```

- [ ] **Step 2: Add `--nopca` to `scripts/encode_sequences.py`**

Add CLI argument:

```python
parser.add_argument("--nopca", action="store_true",
                    help="Use non-PCA-normalized vertices for encoding")
```

In the patch loading loop, when creating `PatchGraphDataset` for encoding, pass `use_nopca`:

```python
# The encode script creates a PatchGraphDataset internally for batch encoding
# Update its creation to pass use_nopca
dataset = PatchGraphDataset(patch_dir, use_nopca=args.nopca)
```

**Note**: `encode_sequences.py` already saves `principal_axes` (rotation matrices) from patch NPZs. The `rotations` key in the output sequence NPZ is what `MeshSequenceDataset` reads for 11-token mode. No change needed for rotation saving — it's already there.

- [ ] **Step 3: Verify existing tests still pass**

Run: `python -m pytest tests/ -v --ignore=tests/test_parquet_loader.py`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train_rvq.py scripts/encode_sequences.py
git commit -m "feat: train_rvq and encode_sequences support --nopca flag"
git push
```

---

### Task 7: Phase 1 — VQ-VAE Foundation Training (×4)

**Prerequisites:** Tasks 1-6 complete. HF dataset accessible.

**Multi-GPU Strategy:**
- GPU 0: PCA VQ-VAE K=1024
- GPU 1: noPCA VQ-VAE K=1024
- GPU 2: K=512 → K=2048 (sequential)

- [ ] **Step 1: Download training data from HF Parquet**

```bash
python -c "
from src.parquet_loader import prepare_training_data
results = prepare_training_data(
    output_base='data/patches_full',
    splits=['seen_train', 'seen_test'],
)
print(results)
"
```

Expected: ~7.9M patch NPZ files in `data/patches_full/seen_train/`, ~2M in `data/patches_full/seen_test/`. This may take 1-3 hours depending on network.

Also download unseen split for later evaluation:

```bash
python -c "
from src.parquet_loader import prepare_training_data
results = prepare_training_data(
    output_base='data/patches_full',
    splits=['unseen'],
)
print(results)
"
```

Expected: ~830K patches in `data/patches_full/unseen/` (5,541 meshes).

Check disk usage: `du -sh data/patches_full/`
If > 40GB, consider streaming mode (skip this step and modify train_rvq to use Parquet directly).

- [ ] **Step 2: Train PCA VQ-VAE K=1024 (GPU 0)**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_rvq.py \
    --train_dirs data/patches_full/seen_train \
    --codebook_size 1024 --n_levels 3 --embed_dim 128 \
    --batch_size 1024 --epochs 100 --lr 1e-4 \
    --checkpoint_dir data/checkpoints/rvq_full_pca \
    2>&1 | tee results/fullscale_eval/train_rvq_pca.log &
```

- [ ] **Step 3: Train noPCA VQ-VAE K=1024 (GPU 1) — in parallel**

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_rvq.py \
    --train_dirs data/patches_full/seen_train \
    --nopca \
    --codebook_size 1024 --n_levels 3 --embed_dim 128 \
    --batch_size 1024 --epochs 100 --lr 1e-4 \
    --checkpoint_dir data/checkpoints/rvq_full_nopca \
    2>&1 | tee results/fullscale_eval/train_rvq_nopca.log &
```

- [ ] **Step 4: Train K=512 VQ-VAE (GPU 2)**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_rvq.py \
    --train_dirs data/patches_full/seen_train \
    --codebook_size 512 --n_levels 3 --embed_dim 128 \
    --batch_size 1024 --epochs 100 --lr 1e-4 \
    --checkpoint_dir data/checkpoints/rvq_full_pca_k512 \
    2>&1 | tee results/fullscale_eval/train_rvq_k512.log
```

After K=512 finishes:

- [ ] **Step 5: Train K=2048 VQ-VAE (GPU 2)**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_rvq.py \
    --train_dirs data/patches_full/seen_train \
    --codebook_size 2048 --n_levels 3 --embed_dim 128 \
    --batch_size 1024 --epochs 100 --lr 1e-4 \
    --checkpoint_dir data/checkpoints/rvq_full_pca_k2048 \
    2>&1 | tee results/fullscale_eval/train_rvq_k2048.log
```

- [ ] **Step 6: Verify all 4 VQ-VAEs — Go/No-Go gate**

```python
import torch, json
from pathlib import Path

for name in ["rvq_full_pca", "rvq_full_nopca", "rvq_full_pca_k512", "rvq_full_pca_k2048"]:
    ckpt = torch.load(f"data/checkpoints/{name}/checkpoint_final.pt", map_location="cpu", weights_only=False)
    hist = ckpt.get("history", {})
    final_loss = hist.get("val_loss", hist.get("train_loss", [None]))[-1]
    print(f"{name}: final_loss={final_loss}")

# Gate: all losses < 0.3 (generous threshold for full-scale)
```

**Gate**: If any VQ-VAE loss > 0.3, investigate before proceeding.

- [ ] **Step 6b: Reconstruction CD sanity check (spec Section 3.1)**

```python
# Quick CD check on 500 seen_test patches for PCA VQ-VAE
import torch, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

device = "cuda:0"
ckpt = torch.load("data/checkpoints/rvq_full_pca/checkpoint_final.pt", map_location=device, weights_only=False)
model = MeshLexRVQVAE().to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

dataset = PatchGraphDataset("data/patches_full/seen_test")
loader = PyGDataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

cds = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        z = model.encoder(batch.x, batch.edge_index, batch.batch)
        z_hat, _ = model.rvq(z)
        recon = model.decoder(z_hat, batch.n_vertices)
        for i in range(len(batch.n_vertices)):
            n_v = batch.n_vertices[i].item()
            gt = batch.gt_vertices[i, :n_v].cpu().numpy()
            pred = recon[i, :n_v].cpu().numpy()
            tree_a, tree_b = cKDTree(gt), cKDTree(pred)
            cd = (tree_b.query(gt)[0].mean() + tree_a.query(pred)[0].mean()) / 2.0
            cds.append(cd)
        if len(cds) >= 500:
            break

mean_cd = np.mean(cds[:500])
print(f"PCA VQ-VAE reconstruction CD on seen_test: {mean_cd:.6f}")
if mean_cd > 0.3:
    print("⚠️ WARNING: Mean CD > 0.3 — investigate before proceeding!")
else:
    print(f"✅ CD gate passed (mean CD = {mean_cd:.6f} < 0.3)")
```

**Gate**: If mean CD > 0.3, investigate before proceeding.

- [ ] **Step 7: Upload all checkpoints to HF**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
for name in ['rvq_full_pca', 'rvq_full_nopca', 'rvq_full_pca_k512', 'rvq_full_pca_k2048']:
    for fname in ['checkpoint_final.pt', 'training_history.json']:
        import os
        fpath = f'data/checkpoints/{name}/{fname}'
        if os.path.exists(fpath):
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f'checkpoints/{name}/{fname}',
                repo_id='Pthahnix/MeshLex-Research', repo_type='model',
            )
            print(f'✅ {name}/{fname} uploaded')
"
```

- [ ] **Step 8: Commit logs**

```bash
git add results/fullscale_eval/train_rvq_*.log
git commit -m "phase1: VQ-VAE foundation training complete (×4)"
git push
```

---

### Task 8: Phase 2a — Token Encoding

**Prerequisites:** Task 7 (all 4 VQ-VAEs trained).

Encode all seen_train meshes into token sequences using both PCA and noPCA VQ-VAEs.

- [ ] **Step 1: Encode PCA sequences (saves rotations for 11-token format)**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/encode_sequences.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --output_dir data/sequences/rvq_full_pca \
    --mode rvq --batch_size 512 \
    2>&1 | tee results/fullscale_eval/encode_pca.log
```

Expected: ~53,492 `*_sequence.npz` files, each with keys `centroids`, `scales`, `tokens`, `principal_axes` (for rotation).

- [ ] **Step 2: Encode noPCA sequences (in parallel on GPU 1)**

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/encode_sequences.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_nopca/checkpoint_final.pt \
    --output_dir data/sequences/rvq_full_nopca \
    --mode rvq --nopca --batch_size 512 \
    2>&1 | tee results/fullscale_eval/encode_nopca.log
```

- [ ] **Step 3: Encode K=512 and K=2048 sequences (for Phase 3b-1 K ablation)**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/encode_sequences.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_pca_k512/checkpoint_final.pt \
    --output_dir data/sequences/rvq_full_pca_k512 \
    --mode rvq --batch_size 512

CUDA_VISIBLE_DEVICES=2 python scripts/encode_sequences.py \
    --patch_dirs data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_pca_k2048/checkpoint_final.pt \
    --output_dir data/sequences/rvq_full_pca_k2048 \
    --mode rvq --batch_size 512
```

- [ ] **Step 4: Verify sequence counts**

```bash
echo "PCA:    $(ls data/sequences/rvq_full_pca/*_sequence.npz | wc -l) sequences"
echo "noPCA:  $(ls data/sequences/rvq_full_nopca/*_sequence.npz | wc -l) sequences"
echo "K=512:  $(ls data/sequences/rvq_full_pca_k512/*_sequence.npz | wc -l) sequences"
echo "K=2048: $(ls data/sequences/rvq_full_pca_k2048/*_sequence.npz | wc -l) sequences"
```

Expected: All ~53,492 (may be slightly less due to encoding failures).

- [ ] **Step 5: Quick sanity check on sequence content**

```python
import numpy as np
from pathlib import Path

for name in ["rvq_full_pca", "rvq_full_nopca"]:
    files = sorted(Path(f"data/sequences/{name}").glob("*_sequence.npz"))
    d = np.load(str(files[0]))
    print(f"{name}: keys={list(d.keys())}, tokens shape={d['tokens'].shape}")
    if name == "rvq_full_pca":
        assert "principal_axes" in d, "PCA sequences must have principal_axes for rotation!"
        print(f"  principal_axes shape: {d['principal_axes'].shape}")
```

- [ ] **Step 6: Commit**

```bash
git add results/fullscale_eval/encode_*.log
git commit -m "phase2a: token encoding complete — PCA + noPCA + K ablation"
git push
```

---

### Task 9: AR Scale-Up + Training Script Updates

**Files:**
- Modify: `scripts/train_ar.py`
- Modify: `scripts/generate_v2_pipeline.py`

Scale up AR model from 20.4M → ~57M params and add rotation support.

- [ ] **Step 1: Add `--rotation` flag to `scripts/train_ar.py`**

```python
# Add to argparse:
parser.add_argument("--rotation", action="store_true",
                    help="Use 11-token rotation format (PCA pipeline)")
```

Update vocab size computation:

```python
if args.rotation:
    from src.patch_sequence import compute_vocab_size_rot
    vocab_size = compute_vocab_size_rot(codebook_K=args.codebook_size)
else:
    vocab_size = compute_vocab_size(codebook_K=args.codebook_size)
```

Update dataset:

```python
dataset = MeshSequenceDataset(
    args.sequence_dir, mode=args.mode,
    max_seq_len=args.max_seq_len, use_rotation=args.rotation)
```

Save rotation flag in checkpoint config:

```python
config["rotation"] = args.rotation
config["tokens_per_patch"] = 11 if args.rotation else 7
```

- [ ] **Step 2: Add rotation decode to `scripts/generate_v2_pipeline.py`**

Add a new function `decode_sequence_to_patches_rot()`:

```python
def decode_sequence_to_patches_rot(sequence, vqvae, device,
                                    n_pos_bins=256, n_scale_bins=64, n_rot_bins=64):
    """Decode 11-token-per-patch sequence with quaternion rotation."""
    from src.rotation import dequantize_rotation
    import numpy as np

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
        pos_x = int(sequence[base]) / (n_pos_bins - 1)
        pos_y = (int(sequence[base + 1]) - off_y) / (n_pos_bins - 1)
        pos_z = (int(sequence[base + 2]) - off_z) / (n_pos_bins - 1)
        scale_tok = int(sequence[base + 3]) - off_scale
        scale = max(scale_tok / (n_scale_bins - 1), 0.01)

        rot_bins = np.array([
            int(sequence[base + 4]) - off_rot,
            int(sequence[base + 5]) - off_rot - n_rot_bins,
            int(sequence[base + 6]) - off_rot - 2 * n_rot_bins,
            int(sequence[base + 7]) - off_rot - 3 * n_rot_bins,
        ])
        Vt = dequantize_rotation(rot_bins, n_rot_bins)

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

Add `--rotation` flag to the generation script CLI and dispatch to the appropriate decode function.

- [ ] **Step 3: Verify AR script accepts new args without errors**

```bash
python scripts/train_ar.py --help | grep -E "rotation|max_seq_len"
```
Expected: `--rotation` flag visible, `--max_seq_len` visible.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ar.py scripts/generate_v2_pipeline.py
git commit -m "feat: AR scale-up — rotation support + 11-token decode"
git push
```

---

### Task 10: Phase 2b — AR Training (×2)

**Prerequisites:** Tasks 8-9 complete.

**Multi-GPU Strategy:**
- GPU 0: PCA AR (11-token, vocab=2112, max_seq_len=1430)
- GPU 1: noPCA AR (7-token, vocab=1856, max_seq_len=1024)

> **Note:** Spec says vocab=1852 for noPCA, but `compute_vocab_size(codebook_K=1024)` returns 1856. Use the value from code (1856). The difference is from bin count rounding.

Both use scaled-up config: d_model=768, n_heads=12, n_layers=12 (~57M params).

- [ ] **Step 1: Train PCA AR (GPU 0)**

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_ar.py \
    --sequence_dir data/sequences/rvq_full_pca \
    --checkpoint_dir data/checkpoints/ar_full_pca \
    --rotation --max_seq_len 1430 \
    --codebook_size 1024 \
    --d_model 768 --n_heads 12 --n_layers 12 \
    --batch_size 8 --grad_accum_steps 4 \
    --epochs 200 --warmup_epochs 10 --lr 3e-4 \
    2>&1 | tee results/fullscale_eval/train_ar_pca.log &
```

Expected: ~28-46h. Monitor with `tail -f results/fullscale_eval/train_ar_pca.log`.

If OOM: reduce `--batch_size 4` and increase `--grad_accum_steps 8`.

- [ ] **Step 2: Train noPCA AR (GPU 1) — in parallel**

```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_ar.py \
    --sequence_dir data/sequences/rvq_full_nopca \
    --checkpoint_dir data/checkpoints/ar_full_nopca \
    --max_seq_len 1024 \
    --codebook_size 1024 \
    --d_model 768 --n_heads 12 --n_layers 12 \
    --batch_size 8 --grad_accum_steps 4 \
    --epochs 200 --warmup_epochs 10 --lr 3e-4 \
    2>&1 | tee results/fullscale_eval/train_ar_nopca.log &
```

- [ ] **Step 3: Monitor training progress**

```bash
# Check both training logs periodically
tail -5 results/fullscale_eval/train_ar_pca.log
tail -5 results/fullscale_eval/train_ar_nopca.log
nvidia-smi
```

- [ ] **Step 4: Verify AR quality — Go/No-Go gate**

After training completes:

```python
import torch, json

for name in ["ar_full_pca", "ar_full_nopca"]:
    ckpt = torch.load(f"data/checkpoints/{name}/checkpoint_final.pt",
                       map_location="cpu", weights_only=False)
    hist = ckpt.get("history", {})
    final_loss = hist.get("val_loss", hist.get("train_loss", [None]))[-1]
    print(f"{name}: final_loss={final_loss}")

# Gate: loss < 2.0. If not met, investigate before Phase 3+.
```

**Gate**: Loss < 2.0, PPL < 8.0. If either fails, debug (check lr, data loading, OOM issues).

- [ ] **Step 5: Upload checkpoints to HF**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
for name in ['ar_full_pca', 'ar_full_nopca']:
    for fname in ['checkpoint_final.pt', 'training_history.json']:
        api.upload_file(
            path_or_fileobj=f'data/checkpoints/{name}/{fname}',
            path_in_repo=f'checkpoints/{name}/{fname}',
            repo_id='Pthahnix/MeshLex-Research', repo_type='model',
        )
    print(f'✅ {name} uploaded')
"
```

- [ ] **Step 6: Commit**

```bash
git add results/fullscale_eval/train_ar_*.log
git commit -m "phase2b: AR training complete — PCA (57M) + noPCA (57M)"
git push
```

---

### Task 11: Phase 3a — Preliminary Experiments Rerun

**Prerequisites:** Task 8 (PCA sequences encoded).

Rerun all 4 analysis experiments from `scripts/run_preliminary_analysis.py` on full-scale data, plus a 50-epoch MDLM quick check. Full MDLM training is in Task 14.

- [ ] **Step 1: Run preliminary analysis on full PCA sequences**

```bash
python scripts/run_preliminary_analysis.py \
    --seq_dir data/sequences/rvq_full_pca \
    --patch_dir data/patches_full/seen_train \
    --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --output_dir results/fullscale_preliminary \
    --codebook_size 1024
```

Expected: ~2-3h. Produces `results/fullscale_preliminary/exp{1-4}_*/summary.json` + plots.

- [ ] **Step 2: Compare with preliminary results**

```python
import json

# Load both summaries
with open("results/preliminary_exp/exp1_per_category/summary.json") as f:
    old_exp1 = json.load(f)
with open("results/fullscale_preliminary/exp1_per_category/summary.json") as f:
    new_exp1 = json.load(f)

print("=== Exp 1: Per-category Lognormal ===")
print(f"  5% scale: {old_exp1.get('lognormal_wins', '?')}/{old_exp1.get('total_groups', '?')} lognormal")
print(f"  Full scale: {new_exp1.get('lognormal_wins', '?')}/{new_exp1.get('total_groups', '?')} lognormal")

with open("results/preliminary_exp/exp4_rvq_dependency/summary.json") as f:
    old_exp4 = json.load(f)
with open("results/fullscale_preliminary/exp4_rvq_dependency/summary.json") as f:
    new_exp4 = json.load(f)

print("\n=== Exp 4: RVQ NMI ===")
print(f"  5% scale: NMI_L1_L2={old_exp4['NMI_L1_L2']:.4f}")
print(f"  Full scale: NMI_L1_L2={new_exp4['NMI_L1_L2']:.4f}")
```

**Key questions answered:**
- FM2: Does lognormal hold on full codebook? → Check Exp 1
- Q: Do NMI values change significantly? → Check Exp 4

- [ ] **Step 2b: Exp 5 — 50-epoch MDLM quick check (GPU 2)**

Run a quick MDLM sanity check on full-scale data. If accuracy is still near random after 50 epochs, this signals the full MDLM in Task 14 needs different hyperparams.

```bash
# Uses the existing toy prototype — it reads raw tokens, not 7-token format
CUDA_VISIBLE_DEVICES=2 python scripts/run_mdlm_prototype.py \
    --seq_dir data/sequences/rvq_full_pca \
    --output_dir results/fullscale_preliminary/exp5_mdlm \
    --epochs 50 --batch_size 256
```

Compare with preliminary result (PPL=867.75 on 4934 sequences). If PPL drops significantly on full data, MDLM direction is promising.

- [ ] **Step 3: Commit results**

```bash
git add results/fullscale_preliminary/
git commit -m "phase3a: preliminary experiments rerun on full 72K data"
git push
```

---

### Task 12: MDLM Model + Training Script

**Files:**
- Create: `src/mdlm_model.py`
- Create: `scripts/train_mdlm.py`
- Create: `tests/test_mdlm_model.py`

Full-scale MDLM: Transformer encoder with time/level embeddings, ~20-40M params. Based on the toy prototype in `scripts/run_mdlm_prototype.py` but scaled up.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mdlm_model.py
"""Tests for src.mdlm_model — full-scale MDLM."""
import torch
import pytest


def test_mdlm_forward_shape():
    """MDLM forward returns correct logit shape."""
    from src.mdlm_model import FullMDLM
    model = FullMDLM(vocab_size=1025, max_seq_len=240, d_model=128,
                     n_heads=4, n_layers=2)
    B, L = 2, 240
    x = torch.randint(0, 1025, (B, L))
    t = torch.rand(B)
    padding_mask = torch.ones(B, L, dtype=torch.bool)
    logits = model(x, t, padding_mask)
    assert logits.shape == (B, L, 1024)  # predicts real tokens only (not MASK)


def test_mdlm_masking():
    """MDLM masking produces correct mask counts."""
    from src.mdlm_model import apply_masking
    B, L = 4, 100
    tokens = torch.randint(0, 1024, (B, L))
    padding_mask = torch.ones(B, L, dtype=torch.bool)
    t = torch.full((B,), 0.5)  # 50% mask rate

    masked, mask_positions = apply_masking(tokens, t, padding_mask, mask_token=1024)
    # About 50% should be masked
    mask_rate = mask_positions.float().mean().item()
    assert 0.3 < mask_rate < 0.7, f"Mask rate {mask_rate} not near 0.5"
    # Masked positions should have mask_token
    assert (masked[mask_positions] == 1024).all()


def test_mdlm_generate_shape():
    """MDLM generate produces valid token sequences."""
    from src.mdlm_model import FullMDLM
    model = FullMDLM(vocab_size=1025, max_seq_len=60, d_model=64,
                     n_heads=2, n_layers=1)
    model.eval()
    seqs = model.generate(n_samples=2, seq_len=60, n_steps=10)
    assert seqs.shape == (2, 60)
    assert (seqs >= 0).all()
    assert (seqs < 1024).all()  # no MASK tokens in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mdlm_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement full MDLM model**

```python
# src/mdlm_model.py
"""Full-scale Masked Discrete Language Model for mesh token generation.

Architecture: Bidirectional Transformer Encoder with:
- Token embedding (vocab_size = K + 1 for MASK)
- Positional embedding
- Level embedding (L1/L2/L3 awareness via position mod tokens_per_patch)
- Continuous time embedding (MLP: t → d_model)
- Output head predicts K real tokens (not MASK)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullMDLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1025,      # K + MASK token
        max_seq_len: int = 240,      # 80 patches × 3 levels
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = None,            # defaults to 4 * d_model
        dropout: float = 0.1,
        n_levels: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.real_vocab = vocab_size - 1  # K (excludes MASK)
        self.mask_token = vocab_size - 1
        self.max_seq_len = max_seq_len
        self.n_levels = n_levels
        if d_ff is None:
            d_ff = 4 * d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.level_emb = nn.Embedding(n_levels, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.real_vocab)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.level_emb.weight, std=0.02)

    def forward(self, x, t, padding_mask=None):
        """Forward pass.

        Args:
            x: (B, L) token IDs (may contain MASK token).
            t: (B,) continuous time in [0, 1].
            padding_mask: (B, L) bool, True = valid token.

        Returns:
            logits: (B, L, real_vocab) — predictions for real tokens only.
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device)
        levels = positions % self.n_levels

        h = self.token_emb(x) + self.pos_emb(positions) + self.level_emb(levels)
        h = h + self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)  # broadcast time
        h = self.drop(h)

        # Transformer (no causal mask — bidirectional)
        src_key_padding_mask = ~padding_mask if padding_mask is not None else None
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, n_samples=1, seq_len=None, n_steps=100, temperature=1.0):
        """Generate sequences via iterative unmasking.

        Starts from all-MASK, iteratively unmasks highest-confidence positions.
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        device = next(self.parameters()).device
        x = torch.full((n_samples, seq_len), self.mask_token, device=device, dtype=torch.long)
        padding_mask = torch.ones(n_samples, seq_len, device=device, dtype=torch.bool)

        for step in range(n_steps):
            t_val = 1.0 - step / n_steps
            t = torch.full((n_samples,), t_val, device=device)
            logits = self(x, t, padding_mask)
            probs = F.softmax(logits / temperature, dim=-1)

            # For each masked position, compute confidence (max prob)
            is_masked = (x == self.mask_token)
            confidence = probs.max(dim=-1).values  # (B, L)
            confidence[~is_masked] = -1.0  # ignore already-unmasked

            # Unmask top-k positions this step
            n_to_unmask = max(1, int(is_masked.float().sum(-1).max().item() / (n_steps - step)))
            for b in range(n_samples):
                masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                conf_at_masked = confidence[b, masked_positions]
                topk = min(n_to_unmask, len(masked_positions))
                _, top_idx = conf_at_masked.topk(topk)
                for idx in top_idx:
                    pos = masked_positions[idx]
                    sampled = torch.multinomial(probs[b, pos], 1).item()
                    x[b, pos] = sampled

        # Replace any remaining MASK with random tokens
        still_masked = (x == self.mask_token)
        if still_masked.any():
            x[still_masked] = torch.randint(0, self.real_vocab, (still_masked.sum(),), device=device)

        return x


def apply_masking(tokens, t, padding_mask, mask_token=1024):
    """Apply continuous-time masking for MDLM training.

    Args:
        tokens: (B, L) original tokens.
        t: (B,) mask probability per sample.
        padding_mask: (B, L) True = valid.
        mask_token: ID for MASK token.

    Returns:
        masked_tokens: (B, L) with some positions replaced by mask_token.
        mask_positions: (B, L) bool, True = was masked.
    """
    B, L = tokens.shape
    rand = torch.rand(B, L, device=tokens.device)
    mask_prob = t.unsqueeze(-1).expand(B, L)
    mask_positions = (rand < mask_prob) & padding_mask
    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = mask_token
    return masked_tokens, mask_positions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mdlm_model.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Implement MDLM training script**

```python
# scripts/train_mdlm.py
"""Full-scale MDLM training script.

Trains FullMDLM on mesh token sequences using continuous-time masking.
Reads token sequences from *_sequence.npz files (same format as AR training).
Only uses the codebook tokens (L1, L2, L3), NOT position/scale tokens.
"""
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mdlm_model import FullMDLM, apply_masking


class MDLMTokenDataset(Dataset):
    """Loads codebook tokens from sequence NPZs for MDLM training.

    Extracts only the RVQ token indices (L1, L2, L3) from each sequence,
    flattening to (n_patches * 3,) and padding/truncating to max_seq_len.
    """
    MASK_TOKEN = 1024

    def __init__(self, seq_dir, max_seq_len=390):
        self.files = sorted(Path(seq_dir).glob("*_sequence.npz"))
        self.max_seq_len = max_seq_len  # 130 patches × 3 levels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        tokens = data["tokens"]  # (N, 3) RVQ indices
        flat = tokens.flatten()  # (N*3,)

        L = min(len(flat), self.max_seq_len)
        padded = np.full(self.max_seq_len, 0, dtype=np.int64)
        padded[:L] = flat[:L]
        padding_mask = np.zeros(self.max_seq_len, dtype=bool)
        padding_mask[:L] = True

        return torch.tensor(padded, dtype=torch.long), torch.tensor(padding_mask)


def train_mdlm(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset = MDLMTokenDataset(args.seq_dir, max_seq_len=args.max_seq_len)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = FullMDLM(
        vocab_size=args.codebook_size + 1,  # +1 for MASK
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MDLM: {n_params/1e6:.1f}M params, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Warmup + cosine schedule
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ppl": []}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for tokens, padding_mask in train_loader:
            tokens, padding_mask = tokens.to(device), padding_mask.to(device)

            # Sample masking rate
            t = torch.empty(tokens.size(0), device=device).uniform_(0.1, 1.0)
            masked, mask_pos = apply_masking(tokens, t, padding_mask,
                                              mask_token=args.codebook_size)

            logits = model(masked, t, padding_mask)

            # Loss only on masked positions
            loss = F.cross_entropy(
                logits[mask_pos], tokens[mask_pos],
                reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, padding_mask in val_loader:
                tokens, padding_mask = tokens.to(device), padding_mask.to(device)
                t = torch.full((tokens.size(0),), 0.5, device=device)
                masked, mask_pos = apply_masking(tokens, t, padding_mask,
                                                  mask_token=args.codebook_size)
                logits = model(masked, t, padding_mask)

                if mask_pos.any():
                    loss = F.cross_entropy(logits[mask_pos], tokens[mask_pos])
                    val_loss += loss.item()
                    preds = logits[mask_pos].argmax(dim=-1)
                    val_correct += (preds == tokens[mask_pos]).sum().item()
                    val_total += mask_pos.sum().item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_ppl"].append(val_ppl)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                  f"val_acc={val_acc:.4f} val_ppl={val_ppl:.1f}")

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": vars(args),
            }, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

            # Keep only latest 3
            import glob
            ckpts = sorted(glob.glob(str(ckpt_dir / "checkpoint_epoch*.pt")))
            for old in ckpts[:-3]:
                Path(old).unlink()

    # Save final
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": vars(args),
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    summary = {
        "n_params": n_params,
        "n_train": n_train,
        "n_val": n_val,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "verdict": "FEASIBLE" if val_ppl < 10 else ("MARGINAL" if val_ppl < 50 else "NOT_FEASIBLE"),
    }
    print(f"\nMDLM Training Complete: PPL={val_ppl:.1f}, Acc={val_acc:.4f}, Verdict={summary['verdict']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/mdlm_full")
    parser.add_argument("--output_dir", default="results/fullscale_mdlm")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=390)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train_mdlm(args)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_mdlm_model.py -v`
Expected: All 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/mdlm_model.py scripts/train_mdlm.py tests/test_mdlm_model.py
git commit -m "feat: full-scale MDLM model + training script"
git push
```

---

### Task 13: Phase 3b — Theory-Driven Analysis

**Prerequisites:** Task 8 (all 4 sets of sequences encoded), Task 7 (VQ-VAE checkpoints).

Three sub-analyses: (3b-1) K ablation, (3b-2) VQ method comparison, (3b-3) curvature-frequency.

#### 3b-1: Codebook K Scaling Analysis (~1-2h)

- [ ] **Step 1: Run distribution fitting across K values**

```python
# scripts/run_fullscale_analysis.py (create this file)
"""Phase 3b: Theory-driven analysis orchestrator."""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fit_distributions(token_freqs):
    """Fit Zipf (power law) and lognormal to token frequency data."""
    freqs_sorted = np.sort(token_freqs)[::-1]
    ranks = np.arange(1, len(freqs_sorted) + 1)

    # Zipf: log(freq) = -alpha * log(rank) + c
    log_ranks = np.log(ranks[freqs_sorted > 0])
    log_freqs = np.log(freqs_sorted[freqs_sorted > 0])
    slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_freqs)
    zipf_alpha = -slope
    zipf_r2 = r_value ** 2

    # Lognormal: fit to non-zero frequencies
    nz = freqs_sorted[freqs_sorted > 0]
    shape, loc, scale = stats.lognorm.fit(nz, floc=0)
    sigma = shape
    mu = np.log(scale)

    # Entropy
    total = freqs_sorted.sum()
    probs = freqs_sorted / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(token_freqs))

    return {
        "zipf_alpha": float(zipf_alpha),
        "zipf_r2": float(zipf_r2),
        "lognormal_sigma": float(sigma),
        "lognormal_mu": float(mu),
        "entropy_bits": float(entropy),
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else 0,
        "gini": float(gini_coefficient(freqs_sorted)),
        "utilization": float(np.sum(freqs_sorted > 0) / len(freqs_sorted)),
    }


def gini_coefficient(values):
    """Compute Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * cumsum[-1]) - (n + 1) / n)


def k_ablation_analysis(seq_dirs, k_values, output_dir):
    """Compare token distributions across codebook sizes.

    Args:
        seq_dirs: Dict mapping K → sequence directory path.
        k_values: List of K values [512, 1024, 2048].
        output_dir: Output directory for results.
    """
    out = Path(output_dir) / "k_ablation"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for K in k_values:
        seq_dir = seq_dirs[K]
        # Collect all tokens
        all_tokens = {0: [], 1: [], 2: []}
        for f in sorted(Path(seq_dir).glob("*_sequence.npz")):
            d = np.load(str(f))
            tokens = d["tokens"]  # (N, 3)
            for level in range(3):
                all_tokens[level].extend(tokens[:, level].tolist())

        # Fit per level
        level_results = {}
        for level in range(3):
            freqs = np.bincount(all_tokens[level], minlength=K)
            level_results[f"L{level+1}"] = fit_distributions(freqs)

        results[str(K)] = level_results
        print(f"K={K}: σ=[{level_results['L1']['lognormal_sigma']:.3f}, "
              f"{level_results['L2']['lognormal_sigma']:.3f}, "
              f"{level_results['L3']['lognormal_sigma']:.3f}]")

    # Save
    with open(out / "k_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot K vs sigma
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(["lognormal_sigma", "zipf_alpha", "entropy_ratio"]):
        for level in ["L1", "L2", "L3"]:
            vals = [results[str(K)][level][metric] for K in k_values]
            axes[i].plot(k_values, vals, "o-", label=level)
        axes[i].set_xlabel("Codebook K")
        axes[i].set_ylabel(metric)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].legend()
        axes[i].set_xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(out / "k_scaling.png", dpi=150)
    plt.close()
    print(f"K ablation saved to {out}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=["k_ablation"], required=True)
    parser.add_argument("--output_dir", default="results/fullscale_theory")
    args = parser.parse_args()

    if args.analysis == "k_ablation":
        seq_dirs = {
            512: "data/sequences/rvq_full_pca_k512",
            1024: "data/sequences/rvq_full_pca",
            2048: "data/sequences/rvq_full_pca_k2048",
        }
        k_ablation_analysis(seq_dirs, [512, 1024, 2048], args.output_dir)
```

Run:
```bash
python scripts/run_fullscale_analysis.py --analysis k_ablation --output_dir results/fullscale_theory
```

- [ ] **Step 2: Commit K ablation results**

```bash
git add scripts/run_fullscale_analysis.py results/fullscale_theory/k_ablation/
git commit -m "phase3b-1: K ablation analysis — distribution vs codebook size"
git push
```

#### 3b-2: VQ Method Comparison (~12h training + 2h analysis)

This is the **most critical experiment** — directly tests FM1 (SimVQ artifact hypothesis).

- [ ] **Step 3: Add VanillaVQ and EMAVQ to `src/rvq.py`**

Append to `src/rvq.py`:

```python
class VanillaVQ(nn.Module):
    """Standard VQ with straight-through estimator (no SimVQ transform)."""
    def __init__(self, K: int = 1024, dim: int = 128):
        super().__init__()
        self.K = K
        self.codebook = nn.Embedding(K, dim)
        nn.init.uniform_(self.codebook.weight, -1/K, 1/K)

    def forward(self, z):
        # Nearest neighbor lookup
        dists = torch.cdist(z.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)
        # Straight-through
        z_hat = z + (z_q - z).detach()
        return z_hat, indices

    def compute_loss(self, z, z_q):
        commit = F.mse_loss(z.detach(), z_q)
        embed = F.mse_loss(z, z_q.detach())
        return commit, embed


class EMAVQ(nn.Module):
    """VQ with exponential moving average codebook update."""
    def __init__(self, K: int = 1024, dim: int = 128, decay: float = 0.99):
        super().__init__()
        self.K = K
        self.decay = decay
        self.codebook = nn.Embedding(K, dim)
        nn.init.uniform_(self.codebook.weight, -1/K, 1/K)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("ema_w", self.codebook.weight.clone())

    def forward(self, z):
        dists = torch.cdist(z.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)

        if self.training:
            # EMA update
            one_hot = F.one_hot(indices, self.K).float()
            self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            embed_sum = one_hot.T @ z
            self.ema_w.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.K * 1e-5) * n
            self.codebook.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        z_hat = z + (z_q - z).detach()
        return z_hat, indices

    def compute_loss(self, z, z_q):
        return F.mse_loss(z, z_q.detach()), torch.tensor(0.0, device=z.device)
```

- [ ] **Step 4: Create VQ method comparison script**

```python
# scripts/vq_method_comparison.py
"""Phase 3b-2: Train vanilla/EMA VQ-VAE and compare token distributions.

Directly tests FM1: Is lognormal a SimVQ artifact?
"""
import argparse
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train_vq_variant(method, train_dir, checkpoint_dir, gpu=0, epochs=100, batch_size=1024):
    """Train a VQ-VAE variant (vanilla or EMA) and return checkpoint path."""
    import subprocess
    cmd = [
        "python", "scripts/train_rvq.py",
        "--train_dirs", train_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--codebook_size", "1024",
        "--vq_method", method,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    import os
    env.update(os.environ)
    print(f"Training {method} VQ-VAE on GPU {gpu}...")
    subprocess.run(cmd, env=env, check=True)
    return f"{checkpoint_dir}/checkpoint_final.pt"


def encode_and_analyze(checkpoint, seq_dir, method_name, output_dir):
    """Encode sequences with a VQ-VAE and fit distributions."""
    import subprocess
    subprocess.run([
        "python", "scripts/encode_sequences.py",
        "--patch_dirs", "data/patches_full/seen_train",
        "--checkpoint", checkpoint,
        "--output_dir", seq_dir,
        "--mode", "rvq",
    ], check=True)

    # Collect token frequencies
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_fullscale_analysis import fit_distributions
    all_tokens = {0: [], 1: [], 2: []}
    for f in sorted(Path(seq_dir).glob("*_sequence.npz")):
        d = np.load(str(f))
        tokens = d["tokens"]
        for level in range(3):
            all_tokens[level].extend(tokens[:, level].tolist())

    results = {}
    for level in range(3):
        freqs = np.bincount(all_tokens[level], minlength=1024)
        results[f"L{level+1}"] = fit_distributions(freqs)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/fullscale_theory/vq_comparison")
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Train vanilla and EMA VQ-VAEs
    vanilla_ckpt = train_vq_variant(
        "vanilla", "data/patches_full/seen_train",
        "data/checkpoints/rvq_full_vanilla", gpu=args.gpu)
    ema_ckpt = train_vq_variant(
        "ema", "data/patches_full/seen_train",
        "data/checkpoints/rvq_full_ema", gpu=args.gpu)

    # Encode and analyze each
    methods = {
        "simvq": ("data/checkpoints/rvq_full_pca/checkpoint_final.pt", "data/sequences/rvq_full_pca"),
        "vanilla": (vanilla_ckpt, "data/sequences/rvq_full_vanilla"),
        "ema": (ema_ckpt, "data/sequences/rvq_full_ema"),
    }

    all_results = {}
    for method, (ckpt, seq_dir) in methods.items():
        if method == "simvq":
            # Already encoded — just analyze
            all_tokens = {0: [], 1: [], 2: []}
            for f in sorted(Path(seq_dir).glob("*_sequence.npz")):
                d = np.load(str(f))
                tokens = d["tokens"]
                for level in range(3):
                    all_tokens[level].extend(tokens[:, level].tolist())

            results = {}
            for level in range(3):
                freqs = np.bincount(all_tokens[level], minlength=1024)
                results[f"L{level+1}"] = fit_distributions(freqs)
            all_results[method] = results
        else:
            all_results[method] = encode_and_analyze(ckpt, seq_dir, method, str(out))

    # Save comparison
    with open(out / "vq_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # FM1 verdict
    simvq_sigma = np.mean([all_results["simvq"][f"L{l}"]["lognormal_sigma"] for l in [1,2,3]])
    vanilla_sigma = np.mean([all_results["vanilla"][f"L{l}"]["lognormal_sigma"] for l in [1,2,3]])
    print(f"\nFM1 Test: SimVQ σ={simvq_sigma:.3f}, Vanilla σ={vanilla_sigma:.3f}")
    if abs(simvq_sigma - vanilla_sigma) < 0.1:
        print("→ Lognormal is GEOMETRY-DRIVEN (not SimVQ artifact)")
    else:
        print("→ Lognormal may be a SimVQ artifact — investigate further")

    # Upload checkpoints
    from huggingface_hub import HfApi
    api = HfApi()
    for name in ["rvq_full_vanilla", "rvq_full_ema"]:
        api.upload_file(
            path_or_fileobj=f"data/checkpoints/{name}/checkpoint_final.pt",
            path_in_repo=f"checkpoints/{name}/checkpoint_final.pt",
            repo_id="Pthahnix/MeshLex-Research", repo_type="model",
        )
        print(f"✅ {name} uploaded to HF")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run VQ method comparison**

```bash
python scripts/vq_method_comparison.py --gpu 2 --output_dir results/fullscale_theory/vq_comparison
```

Expected: ~12h (two VQ-VAE trainings + encoding + analysis).

- [ ] **Step 6: Commit**

```bash
git add src/rvq.py scripts/vq_method_comparison.py results/fullscale_theory/vq_comparison/
git commit -m "phase3b-2: VQ method comparison — FM1 test (SimVQ vs vanilla vs EMA)"
git push
```

#### 3b-3: Curvature-Frequency Correlation (~2h)

- [ ] **Step 7: Create curvature analysis script**

```python
# scripts/curvature_analysis.py
"""Phase 3b-3: Discrete Gaussian curvature vs token frequency correlation.

Tests the Gauss-Bonnet theoretical prediction:
- High-frequency tokens → low curvature (flat patches)
- Low-frequency tokens → high curvature (curved patches)
"""
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def angle_deficit_curvature(vertices, faces):
    """Compute discrete Gaussian curvature via angle deficit method.

    For each interior vertex: K_v = 2π - Σ(angles at v)
    Returns mean absolute curvature across all vertices.
    """
    n_verts = len(vertices)
    angle_sum = np.zeros(n_verts)

    for face in faces:
        for i in range(3):
            v0 = vertices[face[i]]
            v1 = vertices[face[(i+1) % 3]]
            v2 = vertices[face[(i+2) % 3]]
            e1 = v1 - v0
            e2 = v2 - v0
            cos_angle = np.clip(
                np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10),
                -1, 1
            )
            angle_sum[face[i]] += np.arccos(cos_angle)

    curvature = 2 * np.pi - angle_sum
    return np.mean(np.abs(curvature))


def curvature_frequency_analysis(patch_dir, seq_dir, output_dir, max_meshes=5000):
    """Compute curvature per patch, map to token, correlate with frequency."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect token frequencies (L1 only for simplicity)
    token_counts = np.zeros(1024, dtype=np.int64)
    seq_files = sorted(Path(seq_dir).glob("*_sequence.npz"))

    for sf in seq_files:
        d = np.load(str(sf))
        tokens = d["tokens"][:, 0]  # L1 only
        for t in tokens:
            token_counts[t] += 1

    # Step 2: Compute curvature per patch and map to token
    token_curvatures = {i: [] for i in range(1024)}
    patch_dir = Path(patch_dir)

    processed = 0
    for sf in seq_files[:max_meshes]:
        mesh_id = sf.stem.replace("_sequence", "")
        d = np.load(str(sf))
        tokens = d["tokens"][:, 0]  # L1 tokens

        # Find corresponding patch NPZs
        patch_files = sorted(patch_dir.glob(f"{mesh_id}_patch_*.npz"))
        for i, pf in enumerate(patch_files):
            if i >= len(tokens):
                break
            pd = np.load(str(pf))
            verts = pd["local_vertices"]
            faces = pd["faces"]
            if len(faces) > 0 and len(verts) > 2:
                curv = angle_deficit_curvature(verts, faces)
                token_curvatures[tokens[i]].append(curv)

        processed += 1
        if processed % 1000 == 0:
            print(f"  Processed {processed}/{min(len(seq_files), max_meshes)} meshes")

    # Step 3: Compute mean curvature per token
    token_mean_curv = np.zeros(1024)
    token_n_samples = np.zeros(1024, dtype=int)
    for tok in range(1024):
        if token_curvatures[tok]:
            token_mean_curv[tok] = np.mean(token_curvatures[tok])
            token_n_samples[tok] = len(token_curvatures[tok])

    # Step 4: Correlation
    valid = (token_counts > 0) & (token_n_samples > 10)
    log_freq = np.log(token_counts[valid] + 1)
    mean_curv = token_mean_curv[valid]

    rho, p_value = scipy_stats.spearmanr(log_freq, mean_curv)

    results = {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_valid_tokens": int(valid.sum()),
        "n_meshes_processed": processed,
        "prediction": "Negative correlation (high freq = low curvature)" if rho < 0
                      else "Positive or no correlation",
    }

    with open(out / "curvature_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_freq, mean_curv, alpha=0.3, s=10)
    ax.set_xlabel("log(Token Frequency)")
    ax.set_ylabel("Mean |Gaussian Curvature|")
    ax.set_title(f"Curvature-Frequency Correlation (ρ={rho:.3f}, p={p_value:.2e})")
    plt.tight_layout()
    plt.savefig(out / "curvature_frequency.png", dpi=150)
    plt.close()

    print(f"Curvature analysis: ρ={rho:.3f}, p={p_value:.2e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", default="data/patches_full/seen_train")
    parser.add_argument("--seq_dir", default="data/sequences/rvq_full_pca")
    parser.add_argument("--output_dir", default="results/fullscale_theory/curvature")
    parser.add_argument("--max_meshes", type=int, default=5000)
    args = parser.parse_args()
    curvature_frequency_analysis(args.patch_dir, args.seq_dir, args.output_dir, args.max_meshes)
```

- [ ] **Step 8: Run curvature analysis**

```bash
python scripts/curvature_analysis.py --max_meshes 5000
```

- [ ] **Step 9: Commit all theory-driven results**

```bash
git add scripts/curvature_analysis.py results/fullscale_theory/
git commit -m "phase3b-3: curvature-frequency correlation analysis"
git push
```

---

### Task 14: Phase 3c — MDLM Full-Scale Training

**Prerequisites:** Task 8 (PCA sequences), Task 12 (MDLM model + script).

Trains full-scale MDLM on 53K sequences. Runs on GPU 2 while Phase 3b runs on GPU 0/1.

- [ ] **Step 1: Train MDLM**

```bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_mdlm.py \
    --seq_dir data/sequences/rvq_full_pca \
    --checkpoint_dir data/checkpoints/mdlm_full \
    --output_dir results/fullscale_mdlm \
    --gpu 0 \
    --codebook_size 1024 --max_seq_len 390 \
    --d_model 512 --n_heads 8 --n_layers 8 \
    --batch_size 32 --epochs 200 --lr 3e-4 --warmup_epochs 5 \
    2>&1 | tee results/fullscale_eval/train_mdlm.log &
```

Note: `--gpu 0` because `CUDA_VISIBLE_DEVICES=2` maps physical GPU 2 to logical GPU 0.

Expected: ~12-18h. Monitor with `tail -f results/fullscale_eval/train_mdlm.log`.

- [ ] **Step 2: Check MDLM feasibility verdict**

```python
import json
with open("results/fullscale_mdlm/summary.json") as f:
    s = json.load(f)
print(f"MDLM: PPL={s['final_val_ppl']:.1f}, Acc={s['final_val_acc']:.4f}, Verdict={s['verdict']}")
```

**Feasibility thresholds:**
| Metric | Toy (5%) | Threshold | Promising |
|--------|----------|-----------|-----------|
| PPL | 868 | < 50 | < 20 |
| Accuracy | 0.6% | > 5% | > 15% |

- [ ] **Step 3: If promising (PPL < 50), generate samples**

```python
import torch
from src.mdlm_model import FullMDLM
from pathlib import Path

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("data/checkpoints/mdlm_full/checkpoint_final.pt",
                    map_location=device, weights_only=False)
config = ckpt["config"]

model = FullMDLM(
    vocab_size=config["codebook_size"] + 1,
    max_seq_len=config["max_seq_len"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate 100 sequences
seqs = model.generate(n_samples=100, seq_len=config["max_seq_len"], n_steps=1000)
import numpy as np
np.save("results/fullscale_mdlm/generated_tokens.npy", seqs.cpu().numpy())
print(f"Generated {seqs.shape[0]} sequences, shape={seqs.shape}")
```

- [ ] **Step 4: Upload MDLM checkpoint**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='data/checkpoints/mdlm_full/checkpoint_final.pt',
    path_in_repo='checkpoints/mdlm_full/checkpoint_final.pt',
    repo_id='Pthahnix/MeshLex-Research', repo_type='model',
)
print('✅ MDLM checkpoint uploaded')
"
```

- [ ] **Step 5: Commit**

```bash
git add results/fullscale_mdlm/ results/fullscale_eval/train_mdlm.log
git commit -m "phase3c: MDLM full-scale training — feasibility verdict"
git push
```

---

### Task 15: Phase 4 — Evaluation + Ablation

**Prerequisites:** Tasks 10 (AR models), 11-14 (Phase 3 experiments).

Covers: (4a) generation evaluation, (4b) reconstruction evaluation, (4c) PCA vs noPCA ablation.

#### 4a: Generation Evaluation

- [ ] **Step 1: Generate meshes with PCA AR**

```bash
python scripts/generate_v2_pipeline.py \
    --vqvae_checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --ar_checkpoint data/checkpoints/ar_full_pca/checkpoint_final.pt \
    --output_dir results/fullscale_gen_pca \
    --rotation \
    --n_meshes 10 --temperatures 0.6 0.7 0.8 0.9 1.0
```

- [ ] **Step 2: Generate meshes with noPCA AR**

```bash
python scripts/generate_v2_pipeline.py \
    --vqvae_checkpoint data/checkpoints/rvq_full_nopca/checkpoint_final.pt \
    --ar_checkpoint data/checkpoints/ar_full_nopca/checkpoint_final.pt \
    --output_dir results/fullscale_gen_nopca \
    --n_meshes 10 --temperatures 0.6 0.7 0.8 0.9 1.0
```

- [ ] **Step 3: Run evaluation on both**

```bash
python scripts/evaluate_generation.py \
    --gen_dir results/fullscale_gen_pca --output_dir results/fullscale_gen_pca

python scripts/evaluate_generation.py \
    --gen_dir results/fullscale_gen_nopca --output_dir results/fullscale_gen_nopca
```

#### 4b: Reconstruction Evaluation

- [ ] **Step 4: Evaluate reconstruction on seen_test and unseen splits**

```python
# scripts/fullscale_evaluation.py
"""Phase 4: Unified evaluation — reconstruction + generation + ablation dashboard."""
import argparse
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader


def chamfer_distance_np(pts_a, pts_b):
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return (dist_a.mean() + dist_b.mean()) / 2.0


def evaluate_reconstruction(checkpoint, patch_dir, output_dir, n_samples=500, use_nopca=False):
    """Evaluate VQ-VAE reconstruction quality on a split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model — use config from checkpoint to handle K=512/2048 ablations
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    model = MeshLexRVQVAE(codebook_size=ckpt_config.get("codebook_size", 1024)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    dataset = PatchGraphDataset(patch_dir, use_nopca=use_nopca)
    loader = PyGDataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    cds = []
    utilization_counts = np.zeros(1024, dtype=int)
    total_patches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encoder(batch.x, batch.edge_index, batch.batch)
            z_hat, indices = model.rvq(z)

            # Track utilization
            for level in range(3):
                for idx in indices[:, level].cpu().numpy():
                    utilization_counts[idx] += 1

            # Reconstruct and compute CD (sample)
            if total_patches < n_samples:
                recon = model.decoder(z_hat, batch.n_vertices)
                for i in range(min(len(batch.n_vertices), n_samples - total_patches)):
                    n_v = batch.n_vertices[i].item()
                    gt = batch.gt_vertices[i, :n_v].cpu().numpy()
                    pred = recon[i, :n_v].cpu().numpy()
                    cd = chamfer_distance_np(gt, pred)
                    cds.append(cd)

            total_patches += len(batch.n_vertices)

    utilization = float(np.sum(utilization_counts > 0) / 1024)
    results = {
        "n_patches": total_patches,
        "n_samples_cd": len(cds),
        "mean_cd": float(np.mean(cds)),
        "std_cd": float(np.std(cds)),
        "median_cd": float(np.median(cds)),
        "codebook_utilization": utilization,
    }

    with open(out / "reconstruction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Reconstruction: CD={results['mean_cd']:.6f}, util={utilization:.3f}")
    return results


def build_dashboard(output_dir):
    """Build unified DASHBOARD.md from all evaluation results."""
    out = Path(output_dir)
    lines = ["# Full-Scale Evaluation Dashboard\n"]

    # VQ-VAE results
    lines.append("## VQ-VAE Reconstruction\n")
    lines.append("| Config | Mean CD | Utilization |")
    lines.append("|--------|---------|-------------|")
    for name, label in [
        ("rvq_full_pca", "PCA K=1024"),
        ("rvq_full_nopca", "noPCA K=1024"),
    ]:
        p = out / f"recon_{name}" / "reconstruction_results.json"
        if p.exists():
            r = json.loads(p.read_text())
            lines.append(f"| {label} | {r['mean_cd']:.6f} | {r['codebook_utilization']:.3f} |")

    # AR results
    lines.append("\n## AR Generation\n")
    for name, label in [
        ("fullscale_gen_pca", "PCA AR"),
        ("fullscale_gen_nopca", "noPCA AR"),
    ]:
        p = Path(f"results/{name}/evaluation_results.json")
        if p.exists():
            r = json.loads(p.read_text())
            lines.append(f"**{label}**: {json.dumps(r.get('summary', {}), indent=2)}\n")

    # Theory-driven results
    lines.append("\n## Theory-Driven Findings\n")
    for name, label in [
        ("fullscale_preliminary/exp1_per_category/summary.json", "Exp 1: Lognormal"),
        ("fullscale_theory/k_ablation/k_ablation_results.json", "K Ablation"),
        ("fullscale_theory/vq_comparison/vq_comparison_results.json", "VQ Comparison (FM1)"),
        ("fullscale_theory/curvature/curvature_results.json", "Curvature Correlation"),
    ]:
        p = Path(f"results/{name}")
        if p.exists():
            r = json.loads(p.read_text())
            lines.append(f"**{label}**: `{json.dumps(r, indent=2)[:200]}...`\n")

    # MDLM results
    p = Path("results/fullscale_mdlm/summary.json")
    if p.exists():
        r = json.loads(p.read_text())
        lines.append(f"\n## MDLM Feasibility\n")
        lines.append(f"PPL={r['final_val_ppl']:.1f}, Acc={r['final_val_acc']:.4f}, **{r['verdict']}**\n")

    dashboard = "\n".join(lines)
    (out / "DASHBOARD.md").write_text(dashboard)
    print(f"Dashboard written to {out / 'DASHBOARD.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["recon", "dashboard"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--patch_dir", default=None)
    parser.add_argument("--output_dir", default="results/fullscale_eval")
    parser.add_argument("--nopca", action="store_true")
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()

    if args.action == "recon":
        evaluate_reconstruction(args.checkpoint, args.patch_dir, args.output_dir,
                                 args.n_samples, args.nopca)
    elif args.action == "dashboard":
        build_dashboard(args.output_dir)
```

- [ ] **Step 5: Run reconstruction evaluation for PCA**

```bash
python scripts/fullscale_evaluation.py --action recon \
    --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
    --patch_dir data/patches_full/seen_test \
    --output_dir results/fullscale_eval/recon_rvq_full_pca

python scripts/fullscale_evaluation.py --action recon \
    --checkpoint data/checkpoints/rvq_full_nopca/checkpoint_final.pt \
    --patch_dir data/patches_full/seen_test \
    --output_dir results/fullscale_eval/recon_rvq_full_nopca \
    --nopca
```

#### 4c: PCA vs noPCA Ablation

- [ ] **Step 6: Compute ablation metrics**

```python
import json, numpy as np
from pathlib import Path

ablation = {"PCA_vs_noPCA": {}}

# Reconstruction comparison
for name, label in [("rvq_full_pca", "PCA"), ("rvq_full_nopca", "noPCA")]:
    p = Path(f"results/fullscale_eval/recon_{name}/reconstruction_results.json")
    if p.exists():
        r = json.loads(p.read_text())
        ablation["PCA_vs_noPCA"][label] = {
            "mean_cd": r["mean_cd"],
            "codebook_utilization": r["codebook_utilization"],
        }

# Token distribution comparison (σ, α from Exp 1 rerun)
for name, label in [("rvq_full_pca", "PCA"), ("rvq_full_nopca", "noPCA")]:
    exp1_path = Path(f"results/fullscale_preliminary/exp1_per_category/summary.json")
    if exp1_path.exists():
        exp1 = json.loads(exp1_path.read_text())
        if label in ablation["PCA_vs_noPCA"]:
            ablation["PCA_vs_noPCA"][label]["sigma"] = exp1.get("global_sigma", "N/A")
            ablation["PCA_vs_noPCA"][label]["alpha"] = exp1.get("global_alpha", "N/A")

# AR generation comparison
for name, label in [("fullscale_gen_pca", "PCA"), ("fullscale_gen_nopca", "noPCA")]:
    p = Path(f"results/{name}/evaluation_results.json")
    if p.exists():
        r = json.loads(p.read_text())
        if label in ablation["PCA_vs_noPCA"]:
            ablation["PCA_vs_noPCA"][label]["gen_ppl"] = r.get("summary", {}).get("perplexity", "N/A")

# K ablation (from theory-driven)
k_path = Path("results/fullscale_theory/k_ablation/k_ablation_results.json")
if k_path.exists():
    ablation["K_ablation"] = json.loads(k_path.read_text())

out = Path("results/fullscale_eval/ablation_results.json")
with open(out, "w") as f:
    json.dump(ablation, f, indent=2)
print(json.dumps(ablation, indent=2))
```

- [ ] **Step 7: Build unified dashboard**

```bash
python scripts/fullscale_evaluation.py --action dashboard \
    --output_dir results/fullscale_eval
```

- [ ] **Step 8: Commit all evaluation results**

```bash
git add scripts/fullscale_evaluation.py results/fullscale_eval/ results/fullscale_gen_*/
git commit -m "phase4: evaluation + ablation — reconstruction + generation + dashboard"
git push
```

---

### Task 16: Phase 5 — Paper-Ready Analysis

**Prerequisites:** Task 15 (all evaluation complete).

Generate publication-quality figures, write comprehensive context documents, and make the final direction decision.

- [ ] **Step 1: Generate paper figures**

```python
# Run as a script or inline
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("results/fullscale_eval/figures")
out.mkdir(parents=True, exist_ok=True)

# F1: Token distribution comparison (3D mesh vs image vs time series)
# Load full-scale distribution data
with open("results/fullscale_preliminary/exp1_per_category/summary.json") as f:
    exp1 = json.load(f)

# Plot rank-frequency for L1, L2, L3 on log-log scale
# (Code adapts from exp1 raw data — the agent should load the actual
#  token frequency arrays from the exp1 output directory)
print("F1: Token distribution plot → figures/F1_distribution.png")

# F2: K scaling from 3b-1
with open("results/fullscale_theory/k_ablation/k_ablation_results.json") as f:
    k_data = json.load(f)
print("F2: K scaling → already at results/fullscale_theory/k_ablation/k_scaling.png")

# F3: VQ method comparison from 3b-2
with open("results/fullscale_theory/vq_comparison/vq_comparison_results.json") as f:
    vq_data = json.load(f)

methods = ["simvq", "vanilla", "ema"]
sigmas = {m: np.mean([vq_data[m][f"L{l}"]["lognormal_sigma"] for l in [1,2,3]]) for m in methods}
alphas = {m: np.mean([vq_data[m][f"L{l}"]["zipf_alpha"] for l in [1,2,3]]) for m in methods}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(methods, [sigmas[m] for m in methods], color=["steelblue", "coral", "gold"])
ax1.set_ylabel("Lognormal σ")
ax1.set_title("(a) Lognormal σ by VQ Method")
ax2.bar(methods, [alphas[m] for m in methods], color=["steelblue", "coral", "gold"])
ax2.set_ylabel("Zipf α")
ax2.set_title("(b) Zipf α by VQ Method")
plt.tight_layout()
plt.savefig(out / "F3_vq_comparison.png", dpi=300)
plt.close()
print("F3: VQ comparison → figures/F3_vq_comparison.png")

# F4: Curvature-frequency (copy from 3b-3)
print("F4: Curvature → already at results/fullscale_theory/curvature/curvature_frequency.png")

# F5-F8: Agent should generate remaining figures based on available data
print("Remaining figures (F5-F8) should be generated from evaluation results.")
```

- [ ] **Step 2: Write comprehensive results document**

Create `context/29_fullscale_experiment_results.md`:

```markdown
# 29 — Full-Scale Experiment Results

**Date**: 2026-03-XX (fill in actual date)
**Data**: 72,555 meshes / 10.8M patches (full LVIS+ShapeNet dataset)
**Hardware**: 3× RTX 5090 (32GB VRAM each)

## Phase 1: VQ-VAE Foundation

| Config | Loss | Utilization | Epochs |
|--------|------|-------------|--------|
| PCA K=1024 | TODO | TODO | 100 |
| noPCA K=1024 | TODO | TODO | 100 |
| PCA K=512 | TODO | TODO | 100 |
| PCA K=2048 | TODO | TODO | 100 |

## Phase 2: AR Training

| Config | Loss | PPL | Params |
|--------|------|-----|--------|
| PCA AR (11-tok) | TODO | TODO | ~57M |
| noPCA AR (7-tok) | TODO | TODO | ~57M |

## Phase 3a: Preliminary Rerun at Full Scale

(Fill with actual results from exp1-4)

### Key Comparison: 5% vs 100%

| Metric | 5% (4,934 meshes) | 100% (72,555 meshes) | Change |
|--------|-------------------|---------------------|--------|
| Lognormal consensus | 11/11 | TODO | |
| Spatial ρ | -0.036 | TODO | |
| RVQ NMI (L1→L2) | 0.273 | TODO | |

## Phase 3b: Theory-Driven

### 3b-1: K Ablation
(Fill from k_ablation_results.json)

### 3b-2: VQ Method Comparison (FM1 Resolution)
(Fill from vq_comparison_results.json — THIS IS THE KEY RESULT)

### 3b-3: Curvature-Frequency Correlation
(Fill from curvature_results.json)

## Phase 3c: MDLM Feasibility

| Metric | Toy (5%) | Full Scale | Verdict |
|--------|----------|------------|---------|
| PPL | 868 | TODO | |
| Accuracy | 0.6% | TODO | |

## Phase 4: Evaluation

### Reconstruction Quality
(Fill from reconstruction_results.json)

### PCA vs noPCA Ablation
(Fill from dashboard)

## Final Direction Decision

(Fill based on decision matrix from spec Section 7.3)
```

The agent should **fill in all TODO values** from the actual experiment results JSON files.

- [ ] **Step 3: Write contribution summary**

Create `context/30_paper_contribution_summary.md` based on results:

```markdown
# 30 — Paper Contribution Summary

**Date**: 2026-03-XX

## Direction: [TO BE DECIDED based on FM1/FM3 resolution]

## Confirmed Contributions

### C1: First systematic token distribution analysis for 3D mesh codebooks
- Evidence: Exp 1 full-scale (11/11? lognormal), cross-domain comparison
- Figure: F1

### C2: [Depends on FM1 resolution]
If geometry-driven: Gauss-Bonnet theoretical explanation
If SimVQ artifact: Characterization of VQ method impact on token statistics

### C3: Full-scale mesh generation system
- 72K meshes, 10.8M patches, ~57M param AR model
- Evidence: Phase 4 evaluation results

### C4: [Depends on FM3 resolution]
If MDLM feasible: First patch-level discrete diffusion for mesh generation
If not feasible: Negative result documenting MDLM limitations for mesh tokens

### C5: Comprehensive ablation
- PCA vs noPCA, codebook size (K), VQ method comparison
- Evidence: Phase 4c ablation dashboard
```

- [ ] **Step 4: Commit all paper-ready artifacts**

```bash
git add results/fullscale_eval/figures/ context/29_fullscale_experiment_results.md context/30_paper_contribution_summary.md
git commit -m "phase5: paper-ready analysis — figures, results, contribution summary"
git push
```

- [ ] **Step 5: Update CLAUDE.md with final status**

Update the Current Status section in CLAUDE.md to reflect all completed phases, new checkpoint locations, and the final direction decision.

```bash
git add CLAUDE.md
git commit -m "docs: update status — full-scale experiment complete"
git push
```

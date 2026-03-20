# Daft Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream-process ~46K Objaverse-LVIS + ~51K ShapeNetCore v2 meshes into dual-normalization patch Parquet data on HuggingFace dataset `Pthahnix/MeshLex-Patches` via the Daft dataframe engine.

**Architecture:** Each streaming script processes meshes in batches (500 for Objaverse, per-synset for ShapeNet). Per batch: download 3D files → trimesh preprocessing → METIS patch segmentation → flatten arrays to Python lists → `daft.from_pydict()` → cast to float32/int32 → `df.write_huggingface()`. Resume-safe via `progress.json`. Post-processing generates train/test/unseen splits.

**Tech Stack:** Python 3.10+, daft[huggingface], trimesh, pymetis, pyfqmr, objaverse, huggingface_hub, numpy

**Spec:** `docs/superpowers/specs/2026-03-20-daft-dataset-pipeline-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/daft_utils.py` | `patches_to_daft_rows()`, `build_patch_dataframe()`, `get_hf_io_config()`, `make_empty_rows()`, `accumulate_rows()`, `PATCH_COLUMN_TYPES` |
| `src/stream_utils.py` | `ProgressTracker`, `MetadataCollector`, `batch_uids()`, `SHAPENET_SYNSET_MAP` |
| `scripts/stream_objaverse_daft.py` | Objaverse-LVIS streaming → Daft → HF |
| `scripts/stream_shapenet_daft.py` | ShapeNetCore v2 streaming → Daft → HF |
| `scripts/generate_splits_daft.py` | Read metadata JSON → generate splits → upload to HF |
| `scripts/validate_dataset_daft.py` | Read HF Parquet via Daft → validate thresholds |
| `scripts/run_dataset_pipeline.sh` | Overnight tmux orchestrator |
| `tests/test_daft_utils.py` | Unit tests for daft_utils |
| `tests/test_stream_utils.py` | Unit tests for stream helpers |
| `tests/test_generate_splits.py` | Unit tests for split logic |

### Modified Files
| File | Changes |
|------|---------|
| `src/patch_segment.py:8-23` | Add `local_vertices_nopca` field to `MeshPatch` dataclass |
| `src/patch_segment.py:172-184` | Compute no-PCA coords after `_normalize_patch_coords` call |
| `requirements.txt` | Add `daft[huggingface]>=0.5.0`, `objaverse`, `huggingface_hub` |

---

## Task 1: Update Dependencies + Add `local_vertices_nopca` to MeshPatch

**Files:**
- Modify: `requirements.txt`
- Modify: `src/patch_segment.py:8-23`
- Modify: `src/patch_segment.py:172-184`
- Test: `tests/test_patch_segment.py`

- [ ] **Step 1: Update requirements.txt**

Add these lines to `requirements.txt`:

```
daft[huggingface]>=0.5.0
objaverse
huggingface_hub
```

- [ ] **Step 2: Install new dependencies**

Run: `pip install "daft[huggingface]>=0.5.0" objaverse huggingface_hub`
Expected: Installs without errors.

Verify: `python -c "import daft; print(daft.__version__)"`
Expected: Prints version >= 0.5.0

- [ ] **Step 3: Write failing test for nopca field**

Add to `tests/test_patch_segment.py`:

```python
def test_patch_has_local_vertices_nopca():
    """Each patch should have a local_vertices_nopca field (center+scale, no PCA)."""
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    for p in patches:
        assert hasattr(p, "local_vertices_nopca"), "MeshPatch missing local_vertices_nopca"
        assert p.local_vertices_nopca is not None
        assert p.local_vertices_nopca.shape == p.local_vertices.shape
        norms_nopca = np.linalg.norm(p.local_vertices_nopca, axis=1)
        assert norms_nopca.max() <= 1.05, f"nopca not normalized: max norm {norms_nopca.max()}"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_patch_segment.py::test_patch_has_local_vertices_nopca -v`
Expected: FAIL with `AttributeError: 'MeshPatch' object has no attribute 'local_vertices_nopca'`

- [ ] **Step 5: Add field to MeshPatch and compute it**

In `src/patch_segment.py`, update the `MeshPatch` dataclass (line ~8-23):

```python
@dataclass
class MeshPatch:
    # Topology (local indices)
    faces: np.ndarray              # (F, 3) local vertex indices
    vertices: np.ndarray           # (V, 3) world-space vertex coords
    global_face_indices: np.ndarray  # (F,) indices into the original mesh
    boundary_vertices: list[int]   # local indices of boundary verts

    # Geometry (for reconstruction)
    centroid: np.ndarray           # (3,)
    principal_axes: np.ndarray     # (3, 3) PCA rotation
    scale: float                   # bounding sphere radius

    # Normalized local coordinates
    local_vertices: np.ndarray     # (V, 3) centered + PCA-aligned + unit-scaled
    local_vertices_nopca: np.ndarray = None  # (V, 3) centered + unit-scaled (no PCA)
```

In `segment_mesh_to_patches`, after the `_normalize_patch_coords` call (line ~173), add:

```python
        # Normalize (PCA)
        local_verts, centroid, axes, scale = _normalize_patch_coords(vertices)

        # No-PCA normalization: center + scale only
        centered = vertices - centroid
        local_verts_nopca = centered / scale if scale > 1e-8 else centered

        patches.append(MeshPatch(
            faces=local_faces,
            vertices=vertices,
            global_face_indices=face_indices,
            boundary_vertices=sorted(boundary_local),
            centroid=centroid,
            principal_axes=axes,
            scale=scale,
            local_vertices=local_verts,
            local_vertices_nopca=local_verts_nopca,
        ))
```

- [ ] **Step 6: Run all patch_segment tests**

Run: `pytest tests/test_patch_segment.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/patch_segment.py tests/test_patch_segment.py
git commit -m "feat: add local_vertices_nopca to MeshPatch + daft dependencies"
git push
```

---

## Task 2: Stream Processing Helpers

**Files:**
- Create: `src/stream_utils.py`
- Create: `tests/test_stream_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stream_utils.py
"""Tests for streaming pipeline helpers."""
import json
import pytest


def test_progress_tracker_save_load(tmp_path):
    """ProgressTracker should persist completed batches across restarts."""
    from src.stream_utils import ProgressTracker
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    assert not tracker.is_done("batch_000")
    tracker.mark_done("batch_000", {"meshes": 450, "patches": 12000})
    tracker.save()
    tracker2 = ProgressTracker(str(tmp_path / "progress.json"))
    assert tracker2.is_done("batch_000")
    assert not tracker2.is_done("batch_001")


def test_metadata_collector_accumulate(tmp_path):
    """MetadataCollector should accumulate entries and save to JSON."""
    from src.stream_utils import MetadataCollector
    collector = MetadataCollector(str(tmp_path / "metadata.json"))
    collector.add("mesh_001", {"category": "chair", "source": "objaverse"})
    collector.add("mesh_002", {"category": "table", "source": "shapenet"})
    collector.save()
    with open(tmp_path / "metadata.json") as f:
        data = json.load(f)
    assert "mesh_001" in data
    assert data["mesh_002"]["source"] == "shapenet"


def test_metadata_collector_resume(tmp_path):
    """MetadataCollector should load existing entries on init."""
    from src.stream_utils import MetadataCollector
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"existing": {"category": "lamp"}}, f)
    collector = MetadataCollector(str(tmp_path / "metadata.json"))
    assert "existing" in collector.data
    collector.add("new_mesh", {"category": "car"})
    assert len(collector.data) == 2


def test_shapenet_synset_to_category():
    """Should map synset IDs to human-readable category names."""
    from src.stream_utils import SHAPENET_SYNSET_MAP
    assert SHAPENET_SYNSET_MAP["03001627"] == "chair"
    assert SHAPENET_SYNSET_MAP["02691156"] == "airplane"
    assert len(SHAPENET_SYNSET_MAP) == 55


def test_batch_uids():
    """batch_uids should split a list into chunks of given size."""
    from src.stream_utils import batch_uids
    uids = list(range(1250))
    batches = list(batch_uids(uids, batch_size=500))
    assert len(batches) == 3
    assert len(batches[0]) == 500
    assert len(batches[2]) == 250
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stream_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement stream_utils**

```python
# src/stream_utils.py
"""Helpers for streaming dataset processing pipeline."""
import json
from pathlib import Path
from typing import Iterator


class ProgressTracker:
    """Track completed batches for resume-safe processing."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.completed: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.completed = json.load(f)

    def is_done(self, batch_id: str) -> bool:
        return batch_id in self.completed

    def mark_done(self, batch_id: str, stats: dict):
        self.completed[batch_id] = stats

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.completed, f, indent=2)


class MetadataCollector:
    """Accumulate per-mesh metadata and persist to JSON."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def add(self, mesh_id: str, entry: dict):
        self.data[mesh_id] = entry

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)


def batch_uids(uids: list, batch_size: int = 500) -> Iterator[list]:
    """Yield successive chunks of uids."""
    for i in range(0, len(uids), batch_size):
        yield uids[i:i + batch_size]


# ShapeNetCore v2: 55 synset IDs → human-readable category names
SHAPENET_SYNSET_MAP = {
    "02691156": "airplane", "02747177": "trash_bin", "02773838": "bag",
    "02801938": "basket", "02808440": "bathtub", "02818832": "bed",
    "02828884": "bench", "02834778": "bicycle", "02843684": "birdhouse",
    "02858304": "boat", "02871439": "bookshelf", "02876657": "bottle",
    "02880940": "bowl", "02924116": "bus", "02933112": "cabinet",
    "02942699": "camera", "02946921": "can", "02954340": "cap",
    "02958343": "car", "02992529": "cellphone", "03001627": "chair",
    "03046257": "clock", "03085013": "keyboard", "03207941": "dishwasher",
    "03211117": "display", "03261776": "earphone", "03325088": "faucet",
    "03337140": "file_cabinet", "03467517": "guitar", "03513137": "helmet",
    "03593526": "jar", "03624134": "knife", "03636649": "lamp",
    "03642806": "laptop", "03691459": "loudspeaker", "03710193": "mailbox",
    "03759954": "microphone", "03761084": "microwave", "03790512": "motorbike",
    "03797390": "mug", "03928116": "piano", "03938244": "pillow",
    "03948459": "pistol", "03991062": "pot", "04004475": "printer",
    "04074963": "remote", "04090263": "rifle", "04099429": "rocket",
    "04225987": "skateboard", "04256520": "sofa", "04330267": "stove",
    "04379243": "table", "04401088": "telephone", "04460130": "tower",
    "04468005": "train", "04530566": "watercraft", "04554684": "washer",
}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_stream_utils.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/stream_utils.py tests/test_stream_utils.py
git commit -m "feat: streaming pipeline helpers (progress tracker, metadata, synset map)"
git push
```

---

## Task 3: Daft Utilities (Row Conversion + Schema)

**Files:**
- Create: `src/daft_utils.py`
- Create: `tests/test_daft_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_daft_utils.py
"""Tests for Daft DataFrame utilities."""
import numpy as np
import pytest


def test_make_empty_rows_has_all_columns():
    """make_empty_rows should return dict with all 15 expected columns."""
    from src.daft_utils import make_empty_rows
    rows = make_empty_rows()
    expected = [
        "mesh_id", "patch_idx", "category", "source",
        "n_faces", "n_verts", "faces", "vertices",
        "local_vertices", "local_vertices_nopca",
        "centroid", "principal_axes", "scale",
        "boundary_vertices", "global_face_indices",
    ]
    assert list(rows.keys()) == expected
    assert all(isinstance(v, list) and len(v) == 0 for v in rows.values())


def test_accumulate_rows_merges():
    """accumulate_rows should extend target lists with source lists."""
    from src.daft_utils import make_empty_rows, accumulate_rows
    target = make_empty_rows()
    source = make_empty_rows()
    source["mesh_id"].append("m1")
    source["patch_idx"].append(0)
    source["category"].append("chair")
    source["source"].append("objaverse")
    source["n_faces"].append(30)
    source["n_verts"].append(50)
    source["faces"].append([0, 1, 2])
    source["vertices"].append([0.1, 0.2, 0.3])
    source["local_vertices"].append([0.1, 0.2, 0.3])
    source["local_vertices_nopca"].append([0.1, 0.2, 0.3])
    source["centroid"].append([0.0, 0.0, 0.0])
    source["principal_axes"].append([1.0] * 9)
    source["scale"].append(1.0)
    source["boundary_vertices"].append([0, 1])
    source["global_face_indices"].append([5, 6, 7])
    accumulate_rows(target, source)
    assert len(target["mesh_id"]) == 1
    assert target["mesh_id"][0] == "m1"


def test_patches_to_daft_rows_flattens_arrays():
    """patches_to_daft_rows should flatten all ndarray columns to Python lists."""
    from src.daft_utils import patches_to_daft_rows
    from src.patch_segment import MeshPatch

    patch = MeshPatch(
        faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
        global_face_indices=np.array([10, 11], dtype=np.int32),
        boundary_vertices=[0, 3],
        centroid=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        principal_axes=np.eye(3, dtype=np.float32),
        scale=1.0,
        local_vertices=np.array([[-.5, -.5, 0], [.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0]], dtype=np.float32),
        local_vertices_nopca=np.array([[-.5, -.5, 0], [.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0]], dtype=np.float32),
    )
    rows = patches_to_daft_rows([patch], "test_mesh", "chair", "objaverse")

    assert len(rows["mesh_id"]) == 1
    assert rows["mesh_id"][0] == "test_mesh"
    assert rows["n_faces"][0] == 2
    assert rows["n_verts"][0] == 4
    # faces: (2,3) flattened → 6 ints
    assert isinstance(rows["faces"][0], list)
    assert len(rows["faces"][0]) == 6
    # vertices: (4,3) flattened → 12 floats
    assert len(rows["vertices"][0]) == 12
    # centroid: 3 floats
    assert len(rows["centroid"][0]) == 3
    # principal_axes: (3,3) flattened → 9 floats
    assert len(rows["principal_axes"][0]) == 9
    # scale: scalar float
    assert isinstance(rows["scale"][0], float)


def test_build_patch_dataframe_schema():
    """build_patch_dataframe should produce a Daft DataFrame with correct schema."""
    import daft
    from src.daft_utils import patches_to_daft_rows, build_patch_dataframe
    from src.patch_segment import MeshPatch

    patch = MeshPatch(
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        global_face_indices=np.array([0], dtype=np.int32),
        boundary_vertices=[0, 1, 2],
        centroid=np.zeros(3, dtype=np.float32),
        principal_axes=np.eye(3, dtype=np.float32),
        scale=1.0,
        local_vertices=np.zeros((3, 3), dtype=np.float32),
        local_vertices_nopca=np.zeros((3, 3), dtype=np.float32),
    )
    rows = patches_to_daft_rows([patch], "m1", "chair", "objaverse")
    df = build_patch_dataframe(rows)

    schema = df.schema()
    # Check float columns are float32, not float64
    assert schema["scale"].dtype == daft.DataType.float32()
    # Check list columns exist
    assert "vertices" in schema.column_names()
    assert "faces" in schema.column_names()
    # Verify row count
    assert df.count_rows() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_daft_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.daft_utils'`

- [ ] **Step 3: Implement daft_utils**

```python
# src/daft_utils.py
"""Daft DataFrame utilities for MeshLex patch data."""
import os

import daft
import numpy as np

from src.patch_segment import MeshPatch


# Column name → target Daft DataType for float32/int32 enforcement
PATCH_COLUMN_TYPES = {
    "patch_idx": daft.DataType.int32(),
    "n_faces": daft.DataType.int32(),
    "n_verts": daft.DataType.int32(),
    "faces": daft.DataType.list(daft.DataType.int32()),
    "vertices": daft.DataType.list(daft.DataType.float32()),
    "local_vertices": daft.DataType.list(daft.DataType.float32()),
    "local_vertices_nopca": daft.DataType.list(daft.DataType.float32()),
    "centroid": daft.DataType.list(daft.DataType.float32()),
    "principal_axes": daft.DataType.list(daft.DataType.float32()),
    "scale": daft.DataType.float32(),
    "boundary_vertices": daft.DataType.list(daft.DataType.int32()),
    "global_face_indices": daft.DataType.list(daft.DataType.int32()),
}

_ALL_COLUMNS = [
    "mesh_id", "patch_idx", "category", "source",
    "n_faces", "n_verts", "faces", "vertices",
    "local_vertices", "local_vertices_nopca",
    "centroid", "principal_axes", "scale",
    "boundary_vertices", "global_face_indices",
]


def make_empty_rows() -> dict[str, list]:
    """Create empty row accumulator with all expected columns."""
    return {col: [] for col in _ALL_COLUMNS}


def accumulate_rows(target: dict, source: dict):
    """Merge source rows into target (in-place)."""
    for key in target:
        target[key].extend(source[key])


def patches_to_daft_rows(
    patches: list[MeshPatch],
    mesh_id: str,
    category: str,
    source: str,
) -> dict[str, list]:
    """Convert one mesh's patches to column-oriented dict for Daft.

    Arrays are flattened to Python lists for Parquet-native list(float32/int32).
    """
    rows = make_empty_rows()
    for i, p in enumerate(patches):
        rows["mesh_id"].append(mesh_id)
        rows["patch_idx"].append(i)
        rows["category"].append(category)
        rows["source"].append(source)
        rows["n_faces"].append(p.faces.shape[0])
        rows["n_verts"].append(p.local_vertices.shape[0])
        rows["faces"].append(p.faces.astype(np.int32).flatten().tolist())
        rows["vertices"].append(p.vertices.astype(np.float32).flatten().tolist())
        rows["local_vertices"].append(p.local_vertices.astype(np.float32).flatten().tolist())
        rows["local_vertices_nopca"].append(p.local_vertices_nopca.astype(np.float32).flatten().tolist())
        rows["centroid"].append(p.centroid.astype(np.float32).tolist())
        rows["principal_axes"].append(p.principal_axes.astype(np.float32).flatten().tolist())
        rows["scale"].append(float(p.scale))
        rows["boundary_vertices"].append(np.array(p.boundary_vertices, dtype=np.int32).tolist())
        rows["global_face_indices"].append(p.global_face_indices.astype(np.int32).tolist())
    return rows


def build_patch_dataframe(rows: dict[str, list]) -> daft.DataFrame:
    """Build a Daft DataFrame from accumulated rows, casting to float32/int32.

    Daft infers float64 from Python lists; we cast columns to enforce float32/int32.
    """
    df = daft.from_pydict(rows)
    # Cast columns that need explicit type enforcement
    for col_name, target_type in PATCH_COLUMN_TYPES.items():
        df = df.with_column(col_name, daft.col(col_name).cast(target_type))
    return df


def get_hf_io_config():
    """Return Daft IOConfig for HuggingFace writes."""
    from daft.io import IOConfig, HuggingFaceConfig
    return IOConfig(hf=HuggingFaceConfig(
        token=os.environ.get("HF_TOKEN"),
    ))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_daft_utils.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/daft_utils.py tests/test_daft_utils.py
git commit -m "feat: Daft utilities (row conversion, schema casting, HF config)"
git push
```

---

## Task 4: Objaverse-LVIS Streaming Script

**Files:**
- Create: `scripts/stream_objaverse_daft.py`

- [ ] **Step 1: Write the script**

```python
# scripts/stream_objaverse_daft.py
"""Stream-process Objaverse-LVIS objects → Daft → HF Parquet.

Usage:
    python scripts/stream_objaverse_daft.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --batch_size 500 \
        --work_dir /tmp/meshlex_objaverse
"""
import argparse
import gc
import logging
import shutil
import time
from pathlib import Path

import objaverse

from src.data_prep import load_and_preprocess_mesh
from src.patch_segment import segment_mesh_to_patches
from src.daft_utils import (
    patches_to_daft_rows, build_patch_dataframe,
    get_hf_io_config, make_empty_rows, accumulate_rows,
)
from src.stream_utils import ProgressTracker, MetadataCollector, batch_uids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def process_batch(
    batch_idx: int,
    uids: list[str],
    uid_to_cat: dict[str, str],
    work_dir: Path,
    hf_repo: str,
    progress: ProgressTracker,
    metadata: MetadataCollector,
    io_config,
    download_processes: int = 8,
    target_faces: int = 1000,
):
    batch_id = f"batch_{batch_idx:03d}"
    if progress.is_done(batch_id):
        log.info(f"Skipping {batch_id} (already done)")
        return

    # Download GLBs
    log.info(f"[{batch_id}] Downloading {len(uids)} objects...")
    t0 = time.time()
    objects = objaverse.load_objects(uids=uids, download_processes=download_processes)
    log.info(f"[{batch_id}] Downloaded in {time.time()-t0:.0f}s")

    accumulated = make_empty_rows()
    n_ok, n_fail, n_patches_total = 0, 0, 0

    for uid in uids:
        glb_path = objects.get(uid)
        if glb_path is None:
            n_fail += 1
            continue
        try:
            mesh = load_and_preprocess_mesh(glb_path, target_faces=target_faces)
            if mesh is None:
                n_fail += 1
                continue
            patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
            if len(patches) == 0:
                n_fail += 1
                continue

            category = uid_to_cat.get(uid, "unknown")
            rows = patches_to_daft_rows(patches, uid, category, "objaverse")
            accumulate_rows(accumulated, rows)

            metadata.add(uid, {
                "category": category, "source": "objaverse",
                "n_patches": len(patches),
                "n_faces": int(mesh.faces.shape[0]),
                "n_verts": int(mesh.vertices.shape[0]),
            })
            n_ok += 1
            n_patches_total += len(patches)
        except Exception as e:
            log.warning(f"[{batch_id}] Failed {uid}: {e}")
            n_fail += 1

    log.info(f"[{batch_id}] Processed: {n_ok} ok, {n_fail} fail, {n_patches_total} patches")

    # Write to HF via Daft
    if n_ok > 0:
        log.info(f"[{batch_id}] Writing {n_patches_total} patches to HF...")
        df = build_patch_dataframe(accumulated)
        df.write_huggingface(hf_repo, io_config=io_config)

    # Cleanup objaverse caches
    objaverse_cache = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"
    if objaverse_cache.exists():
        for uid in uids:
            uid_file = objaverse_cache / uid[:2] / f"{uid}.glb"
            if uid_file.exists():
                try:
                    uid_file.unlink()
                except Exception:
                    pass
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for cache_dir in hf_cache.glob("models--allenai--objaverse*"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    progress.mark_done(batch_id, {
        "meshes_ok": n_ok, "meshes_fail": n_fail, "patches": n_patches_total,
    })
    progress.save()
    metadata.save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--download_processes", type=int, default=8)
    parser.add_argument("--work_dir", default="/tmp/meshlex_objaverse")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Max batches to process (-1=all, for dry-run use 1 or 2)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    io_config = get_hf_io_config()

    progress = ProgressTracker(str(work_dir / "progress.json"))
    metadata = MetadataCollector(str(work_dir / "metadata.json"))

    log.info("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    uid_to_cat = {}
    all_uids = []
    for cat_name, uids in sorted(lvis.items()):
        for uid in uids:
            if uid not in uid_to_cat:
                uid_to_cat[uid] = cat_name
                all_uids.append(uid)
    log.info(f"Total UIDs: {len(all_uids)}")

    batches = list(batch_uids(all_uids, batch_size=args.batch_size))
    if args.max_batches > 0:
        batches = batches[:args.max_batches]
    log.info(f"Processing {len(batches)} batches")

    for i, batch in enumerate(batches):
        process_batch(i, batch, uid_to_cat, work_dir, args.hf_repo,
                      progress, metadata, io_config,
                      args.download_processes, args.target_faces)

    # Upload metadata JSON
    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=str(work_dir / "metadata.json"),
        path_in_repo="metadata_objaverse.json",
        repo_id=args.hf_repo, repo_type="dataset",
    )
    log.info("Objaverse streaming complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses**

Run: `python -c "import ast; ast.parse(open('scripts/stream_objaverse_daft.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Dry-run with 2 batches of 3 on RunPod**

Run:
```bash
python scripts/stream_objaverse_daft.py \
    --hf_repo Pthahnix/MeshLex-Patches \
    --batch_size 3 --max_batches 2 \
    --work_dir /tmp/meshlex_test 2>&1 | tail -20
```
Expected: Downloads 6 GLBs, processes them, writes 2 batches of Parquet to HF.
Verify: `cat /tmp/meshlex_test/progress.json` shows `batch_000` and `batch_001` done.

- [ ] **Step 4: Commit**

```bash
git add scripts/stream_objaverse_daft.py
git commit -m "feat: Objaverse-LVIS streaming pipeline (Daft → HF Parquet)"
git push
```

---

## Task 5: ShapeNetCore v2 Streaming Script

**Files:**
- Create: `scripts/stream_shapenet_daft.py`

- [ ] **Step 1: Write the script**

```python
# scripts/stream_shapenet_daft.py
"""Stream-process ShapeNetCore v2 → Daft → HF Parquet, one synset at a time.

Usage:
    python scripts/stream_shapenet_daft.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --work_dir /tmp/meshlex_shapenet
"""
import argparse
import gc
import logging
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.data_prep import load_and_preprocess_mesh
from src.patch_segment import segment_mesh_to_patches
from src.daft_utils import (
    patches_to_daft_rows, build_patch_dataframe,
    get_hf_io_config, make_empty_rows, accumulate_rows,
)
from src.stream_utils import (
    ProgressTracker, MetadataCollector, SHAPENET_SYNSET_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SHAPENET_HF_REPO = "ShapeNet/ShapeNetCore"


def process_category(
    synset_id: str, cat_name: str,
    work_dir: Path, hf_repo: str,
    progress: ProgressTracker, metadata: MetadataCollector,
    io_config, target_faces: int = 1000, sub_batch_size: int = 500,
):
    cat_key = f"shapenet_{synset_id}"
    if progress.is_done(cat_key):
        log.info(f"Skipping {cat_name} ({synset_id}) — already done")
        return

    log.info(f"[{cat_name}] Downloading category {synset_id}...")
    t0 = time.time()
    local_dir = work_dir / "shapenet_raw"
    try:
        snapshot_download(
            repo_id=SHAPENET_HF_REPO, repo_type="dataset",
            allow_patterns=f"{synset_id}/**/model_normalized.obj",
            local_dir=str(local_dir),
        )
    except Exception as e:
        log.error(f"[{cat_name}] Download failed: {e}")
        progress.mark_done(cat_key, {"error": str(e)})
        progress.save()
        return
    log.info(f"[{cat_name}] Downloaded in {time.time()-t0:.0f}s")

    synset_dir = local_dir / synset_id
    obj_files = sorted(synset_dir.rglob("model_normalized.obj")) if synset_dir.exists() else []
    log.info(f"[{cat_name}] Found {len(obj_files)} models")

    accumulated = make_empty_rows()
    n_ok, n_fail, n_patches_total, mesh_count = 0, 0, 0, 0

    for obj_file in obj_files:
        model_id = obj_file.parent.parent.name
        mesh_id = f"{synset_id}_{model_id}"
        try:
            mesh = load_and_preprocess_mesh(str(obj_file), target_faces=target_faces)
            if mesh is None:
                n_fail += 1
                continue
            patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
            if len(patches) == 0:
                n_fail += 1
                continue

            rows = patches_to_daft_rows(patches, mesh_id, cat_name, "shapenet")
            accumulate_rows(accumulated, rows)
            metadata.add(mesh_id, {
                "category": cat_name, "source": "shapenet",
                "synset_id": synset_id, "n_patches": len(patches),
                "n_faces": int(mesh.faces.shape[0]),
                "n_verts": int(mesh.vertices.shape[0]),
            })
            n_ok += 1
            n_patches_total += len(patches)
            mesh_count += 1
        except Exception as e:
            log.warning(f"[{cat_name}] Failed {mesh_id}: {e}")
            n_fail += 1

        # Write sub-batch when full
        if mesh_count >= sub_batch_size:
            log.info(f"[{cat_name}] Writing sub-batch ({len(accumulated['mesh_id'])} patches)...")
            df = build_patch_dataframe(accumulated)
            df.write_huggingface(hf_repo, io_config=io_config)
            accumulated = make_empty_rows()
            mesh_count = 0
            metadata.save()

    # Write remaining
    if len(accumulated["mesh_id"]) > 0:
        df = build_patch_dataframe(accumulated)
        df.write_huggingface(hf_repo, io_config=io_config)

    # Cleanup
    shutil.rmtree(local_dir / synset_id, ignore_errors=True)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for cache_dir in hf_cache.glob("datasets--ShapeNet--ShapeNetCore*"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    log.info(f"[{cat_name}] Done: {n_ok} ok, {n_fail} fail, {n_patches_total} patches")
    progress.mark_done(cat_key, {
        "meshes_ok": n_ok, "meshes_fail": n_fail, "patches": n_patches_total,
    })
    progress.save()
    metadata.save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--work_dir", default="/tmp/meshlex_shapenet")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--sub_batch_size", type=int, default=500)
    parser.add_argument("--only_synset", type=str, default=None,
                        help="Process only this synset ID (for dry-run)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    io_config = get_hf_io_config()
    progress = ProgressTracker(str(work_dir / "progress.json"))
    metadata = MetadataCollector(str(work_dir / "metadata.json"))

    synsets = SHAPENET_SYNSET_MAP
    if args.only_synset:
        synsets = {args.only_synset: SHAPENET_SYNSET_MAP[args.only_synset]}

    log.info(f"Processing {len(synsets)} ShapeNet categories")
    for synset_id, cat_name in sorted(synsets.items()):
        process_category(synset_id, cat_name, work_dir, args.hf_repo,
                         progress, metadata, io_config,
                         args.target_faces, args.sub_batch_size)

    HfApi().upload_file(
        path_or_fileobj=str(work_dir / "metadata.json"),
        path_in_repo="metadata_shapenet.json",
        repo_id=args.hf_repo, repo_type="dataset",
    )
    log.info("ShapeNet streaming complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses**

Run: `python -c "import ast; ast.parse(open('scripts/stream_shapenet_daft.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Dry-run with rocket category (04099429, ~85 models)**

Run:
```bash
python scripts/stream_shapenet_daft.py \
    --hf_repo Pthahnix/MeshLex-Patches \
    --only_synset 04099429 \
    --work_dir /tmp/meshlex_shapenet_test 2>&1 | tail -20
```
Expected: Downloads ~85 OBJ files, processes them, writes Parquet to HF.

- [ ] **Step 4: Commit**

```bash
git add scripts/stream_shapenet_daft.py
git commit -m "feat: ShapeNetCore v2 streaming pipeline (Daft → HF Parquet)"
git push
```

---

## Task 6: Generate Splits + Stats

**Files:**
- Create: `scripts/generate_splits_daft.py`
- Create: `tests/test_generate_splits.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generate_splits.py
"""Tests for split generation logic."""
import pytest


def _make_fake_metadata():
    meta = {}
    for cat_idx in range(10):
        cat_name = f"cat_{cat_idx:02d}"
        for mesh_idx in range(20):
            mesh_id = f"{cat_name}_mesh_{mesh_idx:03d}"
            meta[mesh_id] = {
                "category": cat_name,
                "source": "objaverse" if cat_idx < 5 else "shapenet",
                "n_patches": 30, "n_faces": 1000, "n_verts": 500,
            }
    return meta


def test_generate_splits_all_meshes_assigned():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    all_ids = set(splits["seen_train"] + splits["seen_test"] + splits["unseen"])
    assert all_ids == set(meta.keys())


def test_generate_splits_unseen_categories_excluded():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    unseen_cats = set(splits["unseen_categories"])
    for mesh_id in splits["seen_train"] + splits["seen_test"]:
        assert meta[mesh_id]["category"] not in unseen_cats


def test_generate_splits_test_ratio():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    n_seen = len(splits["seen_train"]) + len(splits["seen_test"])
    ratio = len(splits["seen_test"]) / n_seen
    assert 0.15 < ratio < 0.25


def test_generate_splits_holdout_count():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    assert len(splits["unseen_categories"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement generate_splits_daft.py**

```python
# scripts/generate_splits_daft.py
"""Generate train/test/unseen splits from metadata + verify via Daft.

Usage:
    python scripts/generate_splits_daft.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --holdout_count 100 --test_ratio 0.2 --seed 42
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def generate_splits(metadata, holdout_count=100, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    cat_to_meshes = {}
    for mesh_id, info in metadata.items():
        cat_to_meshes.setdefault(info["category"], []).append(mesh_id)

    all_cats = sorted(cat_to_meshes.keys())
    actual_holdout = min(holdout_count, len(all_cats) // 2)
    perm = rng.permutation(len(all_cats))
    unseen_cats = [all_cats[i] for i in perm[:actual_holdout]]
    seen_cats = [all_cats[i] for i in perm[actual_holdout:]]

    unseen = [m for c in unseen_cats for m in cat_to_meshes[c]]
    seen_meshes = [m for c in seen_cats for m in cat_to_meshes[c]]
    rng.shuffle(seen_meshes)
    n_test = int(len(seen_meshes) * test_ratio)

    return {
        "seen_train": sorted(seen_meshes[n_test:]),
        "seen_test": sorted(seen_meshes[:n_test]),
        "unseen": sorted(unseen),
        "unseen_categories": sorted(unseen_cats),
        "seen_categories": sorted(seen_cats),
        "split_seed": seed, "holdout_count": actual_holdout, "test_ratio": test_ratio,
    }


def compute_stats(metadata, splits):
    total_patches = sum(m["n_patches"] for m in metadata.values())
    return {
        "total_meshes": len(metadata),
        "total_patches": total_patches,
        "avg_patches_per_mesh": round(total_patches / max(len(metadata), 1), 1),
        "source_counts": dict(Counter(m["source"] for m in metadata.values())),
        "n_categories": len(set(m["category"] for m in metadata.values())),
        "split_sizes": {k: len(splits[k]) for k in ["seen_train", "seen_test", "unseen"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--holdout_count", type=int, default=100)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="/tmp/meshlex_splits")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    hf_api = HfApi()

    # Download metadata JSONs
    log.info("Downloading metadata from HF...")
    meta_all = {}
    for fname in ["metadata_objaverse.json", "metadata_shapenet.json"]:
        try:
            path = hf_hub_download(repo_id=args.hf_repo, filename=fname,
                                   repo_type="dataset", local_dir=str(work_dir))
            with open(path) as f:
                meta_all.update(json.load(f))
        except Exception as e:
            log.warning(f"Could not download {fname}: {e}")

    log.info(f"Total meshes in metadata: {len(meta_all)}")

    # Verify row count via Daft (lazy, no full download)
    import daft
    from src.daft_utils import get_hf_io_config
    io_config = get_hf_io_config()
    try:
        df = daft.read_parquet(
            f"hf://datasets/{args.hf_repo}/**/*.parquet", io_config=io_config,
        )
        parquet_rows = df.count_rows()
        log.info(f"Parquet row count (patches): {parquet_rows}")
    except Exception as e:
        log.warning(f"Could not read Parquet from HF: {e}")

    splits = generate_splits(meta_all, args.holdout_count, args.test_ratio, args.seed)
    stats = compute_stats(meta_all, splits)
    log.info(f"Splits: {stats['split_sizes']}")

    for name, data in [("metadata.json", meta_all), ("splits.json", splits), ("stats.json", stats)]:
        with open(work_dir / name, "w") as f:
            json.dump(data, f, indent=2)

    for name in ["metadata.json", "splits.json", "stats.json"]:
        hf_api.upload_file(path_or_fileobj=str(work_dir / name),
                           path_in_repo=name, repo_id=args.hf_repo, repo_type="dataset")

    log.info("Splits generation complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_generate_splits.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_splits_daft.py tests/test_generate_splits.py
git commit -m "feat: generate train/test/unseen splits + Daft row count verification"
git push
```

---

## Task 7: Validation Script

**Files:**
- Create: `scripts/validate_dataset_daft.py`

- [ ] **Step 1: Write the script**

```python
# scripts/validate_dataset_daft.py
"""Validate HF dataset meets spec thresholds via Daft.

Usage:
    python scripts/validate_dataset_daft.py --hf_repo Pthahnix/MeshLex-Patches
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import daft
from huggingface_hub import hf_hub_download

from src.daft_utils import get_hf_io_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--work_dir", default="/tmp/meshlex_validate")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    io_config = get_hf_io_config()

    # Download metadata + splits
    for fname in ["metadata.json", "splits.json", "stats.json"]:
        hf_hub_download(repo_id=args.hf_repo, filename=fname,
                        repo_type="dataset", local_dir=str(work_dir))
    with open(work_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(work_dir / "splits.json") as f:
        splits = json.load(f)

    checks = []

    # Metadata-based checks
    n_obj = sum(1 for m in metadata.values() if m["source"] == "objaverse")
    n_sn = sum(1 for m in metadata.values() if m["source"] == "shapenet")
    n_total = len(metadata)
    n_patches = sum(m["n_patches"] for m in metadata.values())
    n_cats = len(set(m["category"] for m in metadata.values()))

    checks.append(("Objaverse meshes >= 35,000", n_obj, n_obj >= 35000))
    checks.append(("ShapeNet meshes >= 45,000", n_sn, n_sn >= 45000))
    checks.append(("Total meshes >= 75,000", n_total, n_total >= 75000))
    checks.append(("Total patches >= 2,500,000", n_patches, n_patches >= 2_500_000))
    checks.append(("Categories >= 500", n_cats, n_cats >= 500))

    all_split_ids = set(splits["seen_train"] + splits["seen_test"] + splits["unseen"])
    checks.append(("All meshes in splits", len(all_split_ids), all_split_ids == set(metadata.keys())))

    # Daft-based Parquet validation
    log.info("Reading Parquet from HF via Daft...")
    try:
        df = daft.read_parquet(
            f"hf://datasets/{args.hf_repo}/**/*.parquet", io_config=io_config,
        )
        parquet_rows = df.count_rows()
        checks.append(("Parquet rows == metadata patches", parquet_rows,
                        parquet_rows == n_patches))

        # Sample 10 rows and check columns
        sample = df.limit(10).collect()
        schema = df.schema()
        expected_cols = [
            "mesh_id", "patch_idx", "category", "source",
            "n_faces", "n_verts", "faces", "vertices",
            "local_vertices", "local_vertices_nopca",
            "centroid", "principal_axes", "scale",
            "boundary_vertices", "global_face_indices",
        ]
        missing = [c for c in expected_cols if c not in schema.column_names()]
        checks.append(("All columns present", f"missing={missing}", len(missing) == 0))
        checks.append(("Sample rows fetched", sample.count_rows(), sample.count_rows() == 10))
    except Exception as e:
        checks.append(("Daft Parquet read", str(e), False))

    # Print report
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    all_pass = True
    for name, value, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")
    print("=" * 60)
    print("RESULT:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses**

Run: `python -c "import ast; ast.parse(open('scripts/validate_dataset_daft.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_dataset_daft.py
git commit -m "feat: dataset validation script (Daft Parquet + metadata thresholds)"
git push
```

---

## Task 8: Overnight Pipeline Orchestrator

**Files:**
- Create: `scripts/run_dataset_pipeline.sh`

- [ ] **Step 1: Write the orchestrator**

```bash
#!/bin/bash
# scripts/run_dataset_pipeline.sh
# Overnight dataset pipeline on RunPod.
#
# Usage (in tmux):
#   tmux new -s dataset
#   bash scripts/run_dataset_pipeline.sh 2>&1 | tee /tmp/dataset_pipeline.log

set -e

HF_REPO="Pthahnix/MeshLex-Patches"
WORK_BASE="/tmp/meshlex"

echo "=========================================="
echo "MeshLex Daft Dataset Pipeline — $(date)"
echo "=========================================="

echo ""
echo "[Phase 1/4] Objaverse-LVIS streaming..."
python scripts/stream_objaverse_daft.py \
    --hf_repo "$HF_REPO" \
    --batch_size 500 --download_processes 8 \
    --work_dir "${WORK_BASE}/objaverse" --target_faces 1000

echo ""
echo "[Phase 2/4] ShapeNetCore v2 streaming..."
python scripts/stream_shapenet_daft.py \
    --hf_repo "$HF_REPO" \
    --work_dir "${WORK_BASE}/shapenet" --target_faces 1000

echo ""
echo "[Phase 3/4] Generating splits..."
python scripts/generate_splits_daft.py \
    --hf_repo "$HF_REPO" \
    --holdout_count 100 --test_ratio 0.2 --seed 42 \
    --work_dir "${WORK_BASE}/splits"

echo ""
echo "[Phase 4/4] Validating dataset..."
python scripts/validate_dataset_daft.py \
    --hf_repo "$HF_REPO" --work_dir "${WORK_BASE}/validate"

echo ""
echo "=========================================="
echo "Pipeline complete — $(date)"
echo "=========================================="
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/run_dataset_pipeline.sh
git add scripts/run_dataset_pipeline.sh
git commit -m "feat: overnight Daft dataset pipeline orchestrator"
git push
```

---

## Execution Summary

| Task | What | Time |
|------|------|------|
| 1 | Dependencies + MeshPatch nopca | 5 min |
| 2 | Stream processing helpers | 10 min |
| 3 | Daft utilities (row conversion + schema) | 10 min |
| 4 | Objaverse streaming script | 15 min (code) + 6-10h (run) |
| 5 | ShapeNet streaming script | 15 min (code) + 4-6h (run) |
| 6 | Generate splits + stats | 10 min |
| 7 | Validation script | 5 min |
| 8 | Overnight orchestrator | 5 min |

**Total coding:** ~75 min
**Total overnight run:** ~10-16h

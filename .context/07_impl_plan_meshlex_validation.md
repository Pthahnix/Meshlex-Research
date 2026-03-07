# MeshLex Validation Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the core MeshLex hypothesis — that mesh local topology forms a universal, finite vocabulary — through a 2-3 day feasibility experiment on ShapeNet.

**Architecture:** METIS-based patch segmentation → SAGEConv GNN encoder → SimVQ codebook (K=4096) → Set-based MLP decoder. Train on Chair+Table+Airplane, test cross-category on Car+Lamp. Go/No-Go based on cross-category CD ratio.

**Tech Stack:** Python 3.11, PyTorch 2.4.1+cu124, torch-geometric (SAGEConv), trimesh, pymeshlab, pymetis, Open3D, matplotlib, scikit-learn, RTX 4090 24GB.

**Reference:** See `.context/06_plan_meshlex_validation.md` for full research plan and Go/No-Go decision matrix.

---

## Task 1: Environment Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Install core dependencies**

```bash
pip install trimesh pymeshlab pymetis numpy scipy scikit-learn tqdm matplotlib
```

**Step 2: Install PyTorch Geometric (must match CUDA 12.4)**

```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

**Step 3: Install visualization dependencies**

```bash
pip install open3d pyvista umap-learn
```

**Step 4: Create requirements.txt**

```
torch>=2.4.0
torch-geometric>=2.5
trimesh>=4.0
pymeshlab
pymetis
numpy
scipy
scikit-learn
tqdm
matplotlib
open3d
pyvista
umap-learn
```

**Step 5: Create project directory structure**

```bash
mkdir -p src tests data/meshes data/patches data/checkpoints results/plots results/meshes
touch src/__init__.py tests/__init__.py
```

**Step 6: Verify installation**

```python
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import torch_geometric; print(f'PyG {torch_geometric.__version__}')
import trimesh; print(f'trimesh {trimesh.__version__}')
import pymeshlab; print('pymeshlab OK')
import pymetis; print('pymetis OK')
"
```

Expected: All imports succeed, CUDA available.

**Step 7: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "feat: project scaffold and dependencies for MeshLex validation"
```

---

## Task 2: ShapeNet Data Download & Preprocessing Pipeline

**Files:**
- Create: `src/data_prep.py`
- Create: `tests/test_data_prep.py`

**Step 1: Write the failing test for mesh loading and preprocessing**

```python
# tests/test_data_prep.py
import pytest
import numpy as np
import trimesh

from src.data_prep import load_and_preprocess_mesh


def test_preprocess_creates_valid_mesh(tmp_path):
    """A preprocessed mesh should be watertight, within face count range, and normalized."""
    # Create a simple test mesh (cube = 12 faces)
    mesh = trimesh.creation.box()
    path_in = tmp_path / "cube.obj"
    mesh.export(str(path_in))

    result = load_and_preprocess_mesh(str(path_in), target_faces=12)

    assert result is not None
    assert isinstance(result, trimesh.Trimesh)
    # Normalized to [-1, 1]
    assert result.vertices.max() <= 1.01
    assert result.vertices.min() >= -1.01
    # Centered near origin
    centroid = result.vertices.mean(axis=0)
    assert np.allclose(centroid, 0, atol=0.1)


def test_preprocess_rejects_degenerate_mesh(tmp_path):
    """Meshes with too few faces after decimation should return None."""
    # A single triangle — too small
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    path_in = tmp_path / "tiny.obj"
    mesh.export(str(path_in))

    result = load_and_preprocess_mesh(str(path_in), target_faces=800)
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_prep'`

**Step 3: Implement `src/data_prep.py`**

```python
# src/data_prep.py
"""Mesh loading, watertight repair, decimation, and normalization."""
import numpy as np
import trimesh
import pymeshlab


def load_and_preprocess_mesh(
    path: str,
    target_faces: int = 1000,
    min_faces: int = 200,
) -> trimesh.Trimesh | None:
    """Load mesh, decimate to target_faces, normalize to unit cube centered at origin.

    Returns None if mesh is degenerate or below min_faces after processing.
    """
    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception:
        return None

    if mesh.faces.shape[0] < min_faces:
        return None

    # Decimation via PyMeshLab if needed
    if mesh.faces.shape[0] > target_faces * 1.2:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
        m = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())

    if mesh.faces.shape[0] < min_faces:
        return None

    # Normalize: center at origin, scale to [-1, 1]
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    scale = np.abs(mesh.vertices).max()
    if scale > 1e-8:
        mesh.vertices /= scale

    return mesh


def preprocess_shapenet_category(
    category_dir: str,
    output_dir: str,
    target_faces: int = 1000,
    max_meshes: int = 500,
) -> list[dict]:
    """Process all OBJ files in a ShapeNet category directory.

    Returns list of metadata dicts: {id, path, n_faces, n_vertices}.
    """
    import os, json
    from pathlib import Path

    cat_path = Path(category_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    obj_files = sorted(cat_path.rglob("*.obj"))[:max_meshes]

    for obj_file in obj_files:
        mesh_id = obj_file.parent.name  # ShapeNet: category/model_id/model.obj
        mesh = load_and_preprocess_mesh(str(obj_file), target_faces=target_faces)
        if mesh is None:
            continue

        out_file = out_path / f"{mesh_id}.obj"
        mesh.export(str(out_file))
        results.append({
            "id": mesh_id,
            "path": str(out_file),
            "n_faces": mesh.faces.shape[0],
            "n_vertices": mesh.vertices.shape[0],
        })

    return results
```

**Step 4: Run tests**

Run: `pytest tests/test_data_prep.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "feat: mesh preprocessing pipeline (load, decimate, normalize)"
```

---

## Task 3: METIS Patch Segmentation

**Files:**
- Create: `src/patch_segment.py`
- Create: `tests/test_patch_segment.py`

**Step 1: Write the failing test**

```python
# tests/test_patch_segment.py
import pytest
import numpy as np
import trimesh

from src.patch_segment import segment_mesh_to_patches, MeshPatch


def _make_sphere(n_faces=200):
    """Create a UV sphere mesh for testing."""
    mesh = trimesh.creation.icosphere(subdivisions=3)  # ~1280 faces
    return mesh


def test_segment_returns_patches():
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    assert len(patches) > 0
    assert all(isinstance(p, MeshPatch) for p in patches)


def test_patch_face_count_in_range():
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    for p in patches:
        assert p.faces.shape[0] >= 10, f"Patch too small: {p.faces.shape[0]} faces"
        assert p.faces.shape[0] <= 80, f"Patch too large: {p.faces.shape[0]} faces"


def test_all_faces_covered():
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    # Every original face should appear in exactly one patch
    all_global_faces = set()
    for p in patches:
        all_global_faces.update(p.global_face_indices.tolist())
    assert len(all_global_faces) == mesh.faces.shape[0]


def test_patch_local_vertices_normalized():
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    for p in patches:
        # Local vertices should be roughly within unit sphere
        norms = np.linalg.norm(p.local_vertices, axis=1)
        assert norms.max() <= 1.05, f"local_vertices not normalized: max norm {norms.max()}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_patch_segment.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `src/patch_segment.py`**

```python
# src/patch_segment.py
"""METIS-based mesh patch segmentation with PCA normalization."""
from dataclasses import dataclass
import numpy as np
import trimesh
import pymetis


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


def _build_face_adjacency(mesh: trimesh.Trimesh):
    """Build adjacency list and weights from face adjacency."""
    n_faces = mesh.faces.shape[0]
    adj_list: list[list[int]] = [[] for _ in range(n_faces)]
    face_adj = mesh.face_adjacency  # (E, 2)
    normals = mesh.face_normals

    for f1, f2 in face_adj:
        adj_list[f1].append(f2)
        adj_list[f2].append(f1)

    return adj_list


def _normalize_patch_coords(vertices: np.ndarray):
    """PCA-align and normalize patch vertices to unit sphere."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # PCA alignment
    if centered.shape[0] >= 3:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        aligned = centered @ Vt.T
    else:
        Vt = np.eye(3)
        aligned = centered

    # Scale to unit sphere
    scale = np.max(np.linalg.norm(aligned, axis=1))
    if scale < 1e-8:
        scale = 1.0
    normalized = aligned / scale

    return normalized, centroid, Vt, scale


def segment_mesh_to_patches(
    mesh: trimesh.Trimesh,
    target_patch_faces: int = 35,
    min_patch_faces: int = 15,
    max_patch_faces: int = 60,
) -> list[MeshPatch]:
    """Segment mesh into patches using METIS graph partitioning.

    Each patch covers ~target_patch_faces faces. Small patches are merged
    into neighbors, large patches are bisected.
    """
    n_faces = mesh.faces.shape[0]
    k = max(2, round(n_faces / target_patch_faces))

    adj_list = _build_face_adjacency(mesh)

    # METIS partitioning
    _, partition = pymetis.part_graph(k, adjacency=adj_list)
    partition = np.array(partition)

    # Group faces by partition
    patch_face_groups: dict[int, list[int]] = {}
    for face_idx, part_id in enumerate(partition):
        patch_face_groups.setdefault(part_id, []).append(face_idx)

    # Post-process: merge small patches into largest neighbor
    final_groups = {}
    for part_id, face_indices in patch_face_groups.items():
        if len(face_indices) < min_patch_faces:
            # Find neighbor partition with most shared edges
            neighbor_counts: dict[int, int] = {}
            for fi in face_indices:
                for nf in adj_list[fi]:
                    np_id = partition[nf]
                    if np_id != part_id:
                        neighbor_counts[np_id] = neighbor_counts.get(np_id, 0) + 1
            if neighbor_counts:
                best_neighbor = max(neighbor_counts, key=neighbor_counts.get)
                patch_face_groups.setdefault(best_neighbor, []).extend(face_indices)
                continue
        final_groups[part_id] = face_indices

    # Post-process: bisect large patches
    result_groups = []
    for part_id, face_indices in final_groups.items():
        if len(face_indices) > max_patch_faces:
            # Simple bisection via METIS on subgraph
            sub_adj = _build_subgraph_adj(face_indices, adj_list)
            if len(sub_adj) >= 2:
                try:
                    _, sub_part = pymetis.part_graph(2, adjacency=sub_adj)
                    g0 = [face_indices[i] for i, p in enumerate(sub_part) if p == 0]
                    g1 = [face_indices[i] for i, p in enumerate(sub_part) if p == 1]
                    if g0:
                        result_groups.append(g0)
                    if g1:
                        result_groups.append(g1)
                    continue
                except Exception:
                    pass
            result_groups.append(face_indices)
        else:
            result_groups.append(face_indices)

    # Build MeshPatch objects
    patches = []
    for face_indices in result_groups:
        face_indices = np.array(face_indices)
        patch_faces_global = mesh.faces[face_indices]  # (F, 3) global vert indices

        # Extract unique vertices and remap
        unique_verts = np.unique(patch_faces_global.flatten())
        vert_map = {g: l for l, g in enumerate(unique_verts)}
        local_faces = np.vectorize(vert_map.get)(patch_faces_global)
        vertices = mesh.vertices[unique_verts]

        # Find boundary vertices (vertices on edges shared with faces outside this patch)
        face_set = set(face_indices.tolist())
        boundary_local = set()
        for fi in face_indices:
            for nf in adj_list[fi]:
                if nf not in face_set:
                    shared = set(mesh.faces[fi]) & set(mesh.faces[nf])
                    for v in shared:
                        if v in vert_map:
                            boundary_local.add(vert_map[v])

        # Normalize
        local_verts, centroid, axes, scale = _normalize_patch_coords(vertices)

        patches.append(MeshPatch(
            faces=local_faces,
            vertices=vertices,
            global_face_indices=face_indices,
            boundary_vertices=sorted(boundary_local),
            centroid=centroid,
            principal_axes=axes,
            scale=scale,
            local_vertices=local_verts,
        ))

    return patches


def _build_subgraph_adj(face_indices: list[int], full_adj: list[list[int]]):
    """Build adjacency list for a subgraph of faces."""
    idx_set = set(face_indices)
    local_map = {g: l for l, g in enumerate(face_indices)}
    sub_adj = [[] for _ in range(len(face_indices))]
    for g_idx in face_indices:
        l_idx = local_map[g_idx]
        for neighbor in full_adj[g_idx]:
            if neighbor in idx_set:
                sub_adj[l_idx].append(local_map[neighbor])
    return sub_adj
```

**Step 4: Run tests**

Run: `pytest tests/test_patch_segment.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/patch_segment.py tests/test_patch_segment.py
git commit -m "feat: METIS patch segmentation with PCA normalization"
```

---

## Task 4: Batch Patch Processing & Dataset Serialization

**Files:**
- Create: `src/patch_dataset.py`
- Create: `tests/test_patch_dataset.py`
- Create: `scripts/run_preprocessing.py`

**Step 1: Write the failing test**

```python
# tests/test_patch_dataset.py
import pytest
import numpy as np
import trimesh
import json
from pathlib import Path

from src.patch_dataset import process_and_save_patches, PatchDataset


def test_process_and_save(tmp_path):
    """Process a mesh and save patches as .npz files."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh_path = tmp_path / "meshes"
    mesh_path.mkdir()
    obj_path = mesh_path / "test_sphere.obj"
    mesh.export(str(obj_path))

    patch_dir = tmp_path / "patches"
    meta = process_and_save_patches(
        mesh_path=str(obj_path),
        mesh_id="test_sphere",
        output_dir=str(patch_dir),
    )

    assert meta["n_patches"] > 0
    npz_files = list(patch_dir.glob("test_sphere_patch_*.npz"))
    assert len(npz_files) == meta["n_patches"]

    # Verify npz contents
    data = np.load(str(npz_files[0]))
    assert "faces" in data
    assert "local_vertices" in data
    assert "centroid" in data
    assert "principal_axes" in data
    assert "scale" in data


def test_patch_dataset_loads(tmp_path):
    """PatchDataset should load .npz files and return torch tensors."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    obj_path = tmp_path / "sphere.obj"
    mesh.export(str(obj_path))

    patch_dir = tmp_path / "patches"
    process_and_save_patches(str(obj_path), "sphere", str(patch_dir))

    ds = PatchDataset(str(patch_dir))
    assert len(ds) > 0

    sample = ds[0]
    assert "face_features" in sample   # (F, 15) input features
    assert "edge_index" in sample      # (2, E) face adjacency
    assert "local_vertices" in sample  # (V, 3) target
    assert "n_vertices" in sample      # int
    assert "n_faces" in sample         # int
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_patch_dataset.py -v`
Expected: FAIL

**Step 3: Implement `src/patch_dataset.py`**

```python
# src/patch_dataset.py
"""Patch serialization to .npz and PyTorch Dataset for training."""
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.patch_segment import segment_mesh_to_patches


def compute_face_features(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute 15-dim face features: 9 vertex coords + 3 normal + 3 edge angles.

    Args:
        vertices: (V, 3) local normalized vertex coordinates
        faces: (F, 3) face vertex indices

    Returns:
        (F, 15) feature array
    """
    n_faces = faces.shape[0]
    features = np.zeros((n_faces, 15), dtype=np.float32)

    for i, (v0, v1, v2) in enumerate(faces):
        p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]

        # 9 vertex coordinates (flattened)
        features[i, :3] = p0
        features[i, 3:6] = p1
        features[i, 6:9] = p2

        # Face normal
        e1, e2 = p1 - p0, p2 - p0
        normal = np.cross(e1, e2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal /= norm_len
        features[i, 9:12] = normal

        # Edge angles (interior angles at each vertex)
        edges = [p1 - p0, p2 - p1, p0 - p2]
        for j in range(3):
            a, b = -edges[j - 1], edges[j]
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            features[i, 12 + j] = np.arccos(np.clip(cos_angle, -1, 1))

    return features


def build_face_edge_index(faces: np.ndarray) -> np.ndarray:
    """Build face adjacency edge_index (2, E) from face array.

    Two faces are adjacent if they share exactly 2 vertices (an edge).
    """
    n_faces = faces.shape[0]
    edge_map: dict[tuple, list[int]] = {}

    for fi in range(n_faces):
        verts = sorted(faces[fi])
        for i in range(3):
            for j in range(i + 1, 3):
                edge_key = (verts[i], verts[j])
                edge_map.setdefault(edge_key, []).append(fi)

    src, dst = [], []
    for edge_key, face_list in edge_map.items():
        if len(face_list) == 2:
            f0, f1 = face_list
            src.extend([f0, f1])
            dst.extend([f1, f0])

    return np.array([src, dst], dtype=np.int64)


def process_and_save_patches(
    mesh_path: str,
    mesh_id: str,
    output_dir: str,
    target_patch_faces: int = 35,
) -> dict:
    """Segment a mesh and save each patch as .npz."""
    mesh = trimesh.load(mesh_path, force="mesh")
    patches = segment_mesh_to_patches(mesh, target_patch_faces=target_patch_faces)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, patch in enumerate(patches):
        np.savez_compressed(
            str(out / f"{mesh_id}_patch_{i:03d}.npz"),
            faces=patch.faces,
            vertices=patch.vertices,
            local_vertices=patch.local_vertices,
            centroid=patch.centroid,
            principal_axes=patch.principal_axes,
            scale=np.array([patch.scale]),
            boundary_vertices=np.array(patch.boundary_vertices),
            global_face_indices=patch.global_face_indices,
        )

    return {
        "mesh_id": mesh_id,
        "n_patches": len(patches),
        "face_counts": [p.faces.shape[0] for p in patches],
    }


class PatchDataset(Dataset):
    """PyTorch Dataset that loads .npz patch files."""

    MAX_FACES = 80
    MAX_VERTICES = 60

    def __init__(self, patch_dir: str):
        self.patch_dir = Path(patch_dir)
        self.files = sorted(self.patch_dir.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        faces = data["faces"]
        local_verts = data["local_vertices"]

        # Face features (F, 15)
        face_feats = compute_face_features(local_verts, faces)

        # Face adjacency edge_index (2, E)
        edge_index = build_face_edge_index(faces)

        n_faces = faces.shape[0]
        n_verts = local_verts.shape[0]

        # Pad to fixed size for batching
        padded_feats = np.zeros((self.MAX_FACES, 15), dtype=np.float32)
        padded_feats[:n_faces] = face_feats

        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return {
            "face_features": torch.tensor(padded_feats),
            "edge_index": torch.tensor(edge_index),
            "local_vertices": torch.tensor(padded_verts),
            "n_faces": n_faces,
            "n_vertices": n_verts,
        }
```

**Step 4: Run tests**

Run: `pytest tests/test_patch_dataset.py -v`
Expected: PASS

**Step 5: Write batch preprocessing script**

```python
# scripts/run_preprocessing.py
"""Batch preprocess ShapeNet meshes and segment into patches."""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.data_prep import load_and_preprocess_mesh
from src.patch_dataset import process_and_save_patches

SHAPENET_CATEGORIES = {
    "chair":    "03001627",
    "table":    "04379243",
    "airplane": "02691156",
    "car":      "02958343",
    "lamp":     "03636649",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--max_per_category", type=int, default=500)
    args = parser.parse_args()

    shapenet = Path(args.shapenet_root)
    output = Path(args.output_root)
    metadata = []

    for cat_name, cat_id in SHAPENET_CATEGORIES.items():
        cat_dir = shapenet / cat_id
        if not cat_dir.exists():
            print(f"Skipping {cat_name}: {cat_dir} not found")
            continue

        mesh_out = output / "meshes" / cat_name
        patch_out = output / "patches" / cat_name
        mesh_out.mkdir(parents=True, exist_ok=True)

        obj_files = sorted(cat_dir.rglob("*.obj"))[:args.max_per_category]
        print(f"\n[{cat_name}] Processing {len(obj_files)} meshes...")

        for obj_file in tqdm(obj_files, desc=cat_name):
            mesh_id = obj_file.parent.name
            mesh = load_and_preprocess_mesh(str(obj_file), target_faces=args.target_faces)
            if mesh is None:
                continue

            # Save preprocessed mesh
            mesh_file = mesh_out / f"{mesh_id}.obj"
            mesh.export(str(mesh_file))

            # Segment and save patches
            meta = process_and_save_patches(
                str(mesh_file), mesh_id, str(patch_out),
            )
            meta["category"] = cat_name
            metadata.append(meta)

    # Save metadata
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    total_patches = sum(m["n_patches"] for m in metadata)
    print(f"\nDone: {len(metadata)} meshes → {total_patches} patches")
    print(f"Metadata saved to {output / 'metadata.json'}")


if __name__ == "__main__":
    main()
```

**Step 6: Commit**

```bash
git add src/patch_dataset.py tests/test_patch_dataset.py scripts/run_preprocessing.py
git commit -m "feat: patch dataset serialization and batch preprocessing script"
```

---

## Task 5: GNN Encoder (SAGEConv)

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing test for encoder**

```python
# tests/test_model.py
import pytest
import torch
from src.model import PatchEncoder


def test_encoder_output_shape():
    """Encoder should produce (B, embed_dim) from batched graph."""
    from torch_geometric.data import Data, Batch

    batch_size = 4
    graphs = []
    for _ in range(batch_size):
        n_faces = 30
        x = torch.randn(n_faces, 15)
        # Random connected graph
        src = torch.randint(0, n_faces, (n_faces * 2,))
        dst = torch.randint(0, n_faces, (n_faces * 2,))
        edge_index = torch.stack([src, dst])
        graphs.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(graphs)
    encoder = PatchEncoder(in_dim=15, hidden_dim=256, out_dim=128)
    out = encoder(batch.x, batch.edge_index, batch.batch)

    assert out.shape == (batch_size, 128)


def test_encoder_deterministic():
    """Same input should give same output."""
    from torch_geometric.data import Data, Batch

    torch.manual_seed(42)
    x = torch.randn(20, 15)
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]])
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])

    encoder = PatchEncoder()
    encoder.eval()
    with torch.no_grad():
        out1 = encoder(batch.x, batch.edge_index, batch.batch)
        out2 = encoder(batch.x, batch.edge_index, batch.batch)
    assert torch.allclose(out1, out2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_encoder_output_shape -v`
Expected: FAIL

**Step 3: Implement PatchEncoder**

```python
# src/model.py
"""MeshLex model components: GNN Encoder, SimVQ Codebook, Patch Decoder."""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class PatchEncoder(nn.Module):
    """4-layer SAGEConv encoder: face features → patch embedding.

    Input: per-face features (F_total, 15) across all patches in batch
    Output: per-patch embedding (B, out_dim)
    """

    def __init__(self, in_dim: int = 15, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(out_dim)

        self.act = nn.GELU()

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: (N_total, in_dim) face features for all patches in batch
            edge_index: (2, E_total) face adjacency edges
            batch: (N_total,) batch assignment vector
        Returns:
            (B, out_dim) patch embeddings
        """
        x = self.act(self.norm1(self.conv1(x, edge_index)))
        x = self.act(self.norm2(self.conv2(x, edge_index)))
        x = self.act(self.norm3(self.conv3(x, edge_index)))
        x = self.act(self.norm4(self.conv4(x, edge_index)))

        # Global mean pooling per patch
        return global_mean_pool(x, batch)  # (B, out_dim)
```

**Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: SAGEConv GNN patch encoder"
```

---

## Task 6: SimVQ Codebook

**Files:**
- Modify: `src/model.py` — add `SimVQCodebook`
- Modify: `tests/test_model.py` — add codebook tests

**Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
from src.model import SimVQCodebook


def test_codebook_output_shape():
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(8, 128)
    quantized, indices = codebook(z)
    assert quantized.shape == (8, 128)
    assert indices.shape == (8,)
    assert indices.min() >= 0
    assert indices.max() < 64


def test_codebook_straight_through_gradient():
    """Gradients should flow through quantization via straight-through."""
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(4, 128, requires_grad=True)
    quantized, _ = codebook(z)
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_codebook_utilization():
    """With diverse inputs, utilization should be non-trivial."""
    codebook = SimVQCodebook(K=32, dim=16)
    z = torch.randn(256, 16)
    _, indices = codebook(z)
    unique_codes = indices.unique().numel()
    # At least 25% utilization with random inputs
    assert unique_codes >= 8, f"Only {unique_codes}/32 codes used"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_codebook_output_shape -v`
Expected: FAIL — `ImportError`

**Step 3: Implement SimVQCodebook in `src/model.py`**

Append to `src/model.py`:

```python
class SimVQCodebook(nn.Module):
    """SimVQ codebook with learnable linear reparameterization.

    Reference: SimVQ (ICCV 2025) — linear transform prevents codebook collapse.
    """

    def __init__(self, K: int = 4096, dim: int = 128):
        super().__init__()
        self.K = K
        self.dim = dim
        self.codebook = nn.Embedding(K, dim)
        self.linear = nn.Linear(dim, dim, bias=False)

        # Initialize codebook from uniform sphere
        nn.init.normal_(self.codebook.weight, std=0.02)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, dim) encoder output
        Returns:
            quantized: (B, dim) quantized embedding (straight-through)
            indices: (B,) codebook indices
        """
        z_proj = self.linear(z)  # SimVQ reparameterization

        # L2 distance to codebook entries
        distances = torch.cdist(
            z_proj.unsqueeze(0),
            self.codebook.weight.unsqueeze(0),
        ).squeeze(0)  # (B, K)

        indices = distances.argmin(dim=-1)  # (B,)
        quantized = self.codebook(indices)  # (B, dim)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        return quantized_st, indices

    def compute_loss(self, z: torch.Tensor, quantized_st: torch.Tensor, indices: torch.Tensor):
        """Compute commitment + embedding losses.

        Returns:
            commit_loss: ||z - sg(quantized)||²
            embed_loss: ||sg(z) - quantized||²
        """
        quantized = self.codebook(indices)
        commit_loss = torch.mean((z - quantized.detach()) ** 2)
        embed_loss = torch.mean((z.detach() - quantized) ** 2)
        return commit_loss, embed_loss

    @torch.no_grad()
    def get_utilization(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in given indices."""
        return indices.unique().numel() / self.K
```

**Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: SimVQ codebook with straight-through and utilization tracking"
```

---

## Task 7: Patch Decoder

**Files:**
- Modify: `src/model.py` — add `PatchDecoder`
- Modify: `tests/test_model.py` — add decoder tests

**Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
from src.model import PatchDecoder


def test_decoder_output_shape():
    decoder = PatchDecoder(embed_dim=128, max_vertices=60)
    z = torch.randn(4, 128)
    n_vertices = torch.tensor([20, 30, 25, 40])
    out = decoder(z, n_vertices)
    assert out.shape == (4, 60, 3)


def test_decoder_masked_output():
    """Vertices beyond n_vertices should be zero (masked)."""
    decoder = PatchDecoder(embed_dim=128, max_vertices=60)
    z = torch.randn(2, 128)
    n_vertices = torch.tensor([10, 15])
    out = decoder(z, n_vertices)
    # Padded positions should be zero
    assert torch.allclose(out[0, 10:], torch.zeros(50, 3))
    assert torch.allclose(out[1, 15:], torch.zeros(45, 3))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_decoder_output_shape -v`
Expected: FAIL

**Step 3: Implement PatchDecoder in `src/model.py`**

Append to `src/model.py`:

```python
class PatchDecoder(nn.Module):
    """MLP decoder: codebook embedding → per-vertex coordinates.

    Uses learnable vertex query positions + cross-attention from codebook embedding,
    with masking for variable vertex counts.
    """

    def __init__(self, embed_dim: int = 128, max_vertices: int = 60):
        super().__init__()
        self.max_vertices = max_vertices

        # Learnable positional queries for each vertex slot
        self.vertex_queries = nn.Parameter(torch.randn(max_vertices, embed_dim) * 0.02)

        # Cross-attention: vertex queries attend to patch embedding
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # MLP to decode xyz
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, z: torch.Tensor, n_vertices: torch.Tensor):
        """
        Args:
            z: (B, embed_dim) codebook embedding
            n_vertices: (B,) actual vertex count per patch
        Returns:
            (B, max_vertices, 3) predicted vertex coordinates (masked beyond n_vertices)
        """
        B = z.shape[0]

        # Expand queries for batch: (B, max_V, D)
        queries = self.vertex_queries.unsqueeze(0).expand(B, -1, -1)

        # Key/Value = patch embedding repeated: (B, 1, D)
        kv = z.unsqueeze(1)

        # Cross-attention
        attn_out, _ = self.cross_attn(queries, kv, kv)
        attn_out = self.norm(attn_out + queries)  # residual

        # Decode to xyz
        coords = self.mlp(attn_out)  # (B, max_V, 3)

        # Mask: zero out positions beyond actual vertex count
        mask = torch.arange(self.max_vertices, device=z.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        coords = coords * mask.unsqueeze(-1).float()

        return coords
```

**Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: cross-attention patch decoder with vertex masking"
```

---

## Task 8: Full VQ-VAE Assembly + Chamfer Distance Loss

**Files:**
- Modify: `src/model.py` — add `MeshLexVQVAE`
- Create: `src/losses.py`
- Modify: `tests/test_model.py` — add end-to-end test

**Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
from src.model import MeshLexVQVAE


def test_vqvae_forward():
    """Full forward pass: graph input → reconstructed vertices + losses."""
    from torch_geometric.data import Data, Batch

    model = MeshLexVQVAE(codebook_size=64, embed_dim=128)
    graphs = []
    n_verts_list = []
    for _ in range(4):
        nf = 30
        nv = 20
        x = torch.randn(nf, 15)
        ei = torch.stack([torch.randint(0, nf, (60,)), torch.randint(0, nf, (60,))])
        graphs.append(Data(x=x, edge_index=ei))
        n_verts_list.append(nv)

    batch = Batch.from_data_list(graphs)
    n_vertices = torch.tensor(n_verts_list)
    gt_vertices = torch.randn(4, 60, 3)

    result = model(batch.x, batch.edge_index, batch.batch, n_vertices, gt_vertices)

    assert "recon_vertices" in result
    assert "total_loss" in result
    assert "recon_loss" in result
    assert "commit_loss" in result
    assert "indices" in result
    assert result["recon_vertices"].shape == (4, 60, 3)
    assert result["total_loss"].requires_grad
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_vqvae_forward -v`
Expected: FAIL

**Step 3: Implement `src/losses.py`**

```python
# src/losses.py
"""Loss functions for MeshLex."""
import torch


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked Chamfer Distance between predicted and GT vertices.

    Args:
        pred: (B, max_V, 3) predicted vertex coordinates
        gt: (B, max_V, 3) ground truth vertex coordinates
        mask: (B, max_V) boolean mask — True for valid vertices

    Returns:
        Scalar mean Chamfer Distance across batch.
    """
    B = pred.shape[0]
    total_cd = 0.0

    for b in range(B):
        m = mask[b]  # (max_V,)
        p = pred[b][m]  # (Np, 3)
        g = gt[b][m]    # (Ng, 3)

        if p.shape[0] == 0 or g.shape[0] == 0:
            continue

        # pred → gt: for each predicted point, find nearest GT
        dist_p2g = torch.cdist(p, g)  # (Np, Ng)
        min_p2g = dist_p2g.min(dim=1).values  # (Np,)

        # gt → pred: for each GT point, find nearest predicted
        min_g2p = dist_p2g.min(dim=0).values  # (Ng,)

        cd = min_p2g.mean() + min_g2p.mean()
        total_cd += cd

    return total_cd / B
```

**Step 4: Implement `MeshLexVQVAE` in `src/model.py`**

Append to `src/model.py`:

```python
from src.losses import chamfer_distance


class MeshLexVQVAE(nn.Module):
    """Full MeshLex VQ-VAE: Encoder → SimVQ → Decoder.

    Combines PatchEncoder, SimVQCodebook, PatchDecoder into end-to-end model.
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        codebook_size: int = 4096,
        max_vertices: int = 60,
        lambda_commit: float = 0.25,
        lambda_embed: float = 1.0,
    ):
        super().__init__()
        self.encoder = PatchEncoder(in_dim, hidden_dim, embed_dim)
        self.codebook = SimVQCodebook(codebook_size, embed_dim)
        self.decoder = PatchDecoder(embed_dim, max_vertices)
        self.max_vertices = max_vertices
        self.lambda_commit = lambda_commit
        self.lambda_embed = lambda_embed

    def forward(self, x, edge_index, batch, n_vertices, gt_vertices):
        """
        Args:
            x: (N_total, in_dim) face features
            edge_index: (2, E_total) face adjacency
            batch: (N_total,) batch vector
            n_vertices: (B,) actual vertex count per patch
            gt_vertices: (B, max_V, 3) ground truth local vertices (padded)
        Returns:
            dict with recon_vertices, total_loss, recon_loss, commit_loss, embed_loss, indices
        """
        # Encode
        z = self.encoder(x, edge_index, batch)  # (B, embed_dim)

        # Quantize
        z_q, indices = self.codebook(z)  # (B, embed_dim), (B,)

        # Decode
        recon = self.decoder(z_q, n_vertices)  # (B, max_V, 3)

        # Losses
        mask = torch.arange(self.max_vertices, device=x.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        recon_loss = chamfer_distance(recon, gt_vertices, mask)
        commit_loss, embed_loss = self.codebook.compute_loss(z, z_q, indices)

        total_loss = recon_loss + self.lambda_commit * commit_loss + self.lambda_embed * embed_loss

        return {
            "recon_vertices": recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "embed_loss": embed_loss,
            "indices": indices,
            "z": z,
        }

    def encode_only(self, x, edge_index, batch):
        """Encode patches without decoding (for codebook init and eval)."""
        return self.encoder(x, edge_index, batch)
```

**Step 5: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/model.py src/losses.py tests/test_model.py
git commit -m "feat: full MeshLex VQ-VAE with Chamfer Distance loss"
```

---

## Task 9: PyG-compatible DataLoader + Collate Function

**Files:**
- Modify: `src/patch_dataset.py` — add `PatchGraphDataset` (returns `torch_geometric.data.Data`)
- Modify: `tests/test_patch_dataset.py` — add DataLoader test

**Step 1: Write the failing test**

Add to `tests/test_patch_dataset.py`:

```python
from src.patch_dataset import PatchGraphDataset


def test_graph_dataset_and_loader(tmp_path):
    """PatchGraphDataset should work with PyG DataLoader."""
    from torch_geometric.loader import DataLoader

    mesh = trimesh.creation.icosphere(subdivisions=3)
    obj_path = tmp_path / "sphere.obj"
    mesh.export(str(obj_path))

    patch_dir = tmp_path / "patches"
    process_and_save_patches(str(obj_path), "sphere", str(patch_dir))

    ds = PatchGraphDataset(str(patch_dir))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert hasattr(batch, "x")           # (N_total, 15) face features
    assert hasattr(batch, "edge_index")  # (2, E_total)
    assert hasattr(batch, "batch")       # (N_total,) batch vector
    assert hasattr(batch, "gt_vertices") # (B, max_V, 3)
    assert hasattr(batch, "n_vertices")  # (B,)
    assert batch.gt_vertices.shape[0] == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_patch_dataset.py::test_graph_dataset_and_loader -v`
Expected: FAIL

**Step 3: Implement `PatchGraphDataset` in `src/patch_dataset.py`**

Add to `src/patch_dataset.py`:

```python
from torch_geometric.data import Data as PyGData


class PatchGraphDataset(Dataset):
    """PyTorch Geometric compatible dataset. Returns Data objects
    with graph structure for SAGEConv + padded vertex targets for decoder.
    """

    MAX_VERTICES = 60

    def __init__(self, patch_dir: str):
        self.patch_dir = Path(patch_dir)
        self.files = sorted(self.patch_dir.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        faces = data["faces"]
        local_verts = data["local_vertices"].astype(np.float32)

        # Face features (F, 15)
        face_feats = compute_face_features(local_verts, faces)

        # Face adjacency graph
        edge_index = build_face_edge_index(faces)

        n_verts = local_verts.shape[0]
        n_faces = faces.shape[0]

        # Pad vertices to max size
        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return PyGData(
            x=torch.tensor(face_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            gt_vertices=torch.tensor(padded_verts, dtype=torch.float32),
            n_vertices=torch.tensor(n_verts, dtype=torch.long),
            n_faces=torch.tensor(n_faces, dtype=torch.long),
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_patch_dataset.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/patch_dataset.py tests/test_patch_dataset.py
git commit -m "feat: PyG-compatible patch graph dataset for DataLoader"
```

---

## Task 10: Training Script

**Files:**
- Create: `scripts/train.py`
- Create: `src/trainer.py`

**Step 1: Implement `src/trainer.py`**

```python
# src/trainer.py
"""Training loop for MeshLex VQ-VAE."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import time


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 200,
        checkpoint_dir: str = "data/checkpoints",
        device: str = "cuda",
        vq_start_epoch: int = 20,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.vq_start_epoch = vq_start_epoch
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            if val_dataset else None
        )

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.history = []

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_recon = 0
        n_batches = 0
        all_indices = []

        for batch in self.train_loader:
            batch = batch.to(self.device)
            gt_verts = batch.gt_vertices  # (B, max_V, 3)
            n_verts = batch.n_vertices    # (B,)

            result = self.model(
                batch.x, batch.edge_index, batch.batch, n_verts, gt_verts,
            )

            # Before vq_start_epoch: zero out VQ losses (train encoder+decoder only)
            if epoch < self.vq_start_epoch:
                loss = result["recon_loss"]
            else:
                loss = result["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += result["recon_loss"].item()
            all_indices.append(result["indices"].detach().cpu())
            n_batches += 1

        self.scheduler.step()

        # Codebook utilization
        all_idx = torch.cat(all_indices)
        utilization = all_idx.unique().numel() / self.model.codebook.K

        return {
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "codebook_utilization": utilization,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self, loader=None):
        """Evaluate on validation or test set."""
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return {}

        self.model.eval()
        total_recon = 0
        n_batches = 0
        all_indices = []

        for batch in loader:
            batch = batch.to(self.device)
            result = self.model(
                batch.x, batch.edge_index, batch.batch,
                batch.n_vertices, batch.gt_vertices,
            )
            total_recon += result["recon_loss"].item()
            all_indices.append(result["indices"].cpu())
            n_batches += 1

        all_idx = torch.cat(all_indices)
        return {
            "val_recon_loss": total_recon / max(n_batches, 1),
            "val_utilization": all_idx.unique().numel() / self.model.codebook.K,
        }

    def train(self):
        """Full training loop."""
        for epoch in range(self.epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "time_sec": elapsed}
            self.history.append(metrics)

            # Print progress
            util = train_metrics["codebook_utilization"]
            print(
                f"Epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
                f"recon {train_metrics['recon_loss']:.4f} | "
                f"util {util:.1%} | {elapsed:.1f}s"
            )

            # Checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)

                # Early warning: codebook collapse
                if epoch >= self.vq_start_epoch and util < 0.30:
                    print(f"WARNING: Codebook utilization {util:.1%} < 30% at epoch {epoch}")

        # Final checkpoint
        self.save_checkpoint(self.epochs - 1, tag="final")
        self.save_history()

    def save_checkpoint(self, epoch, tag=None):
        name = f"checkpoint_epoch{epoch:03d}.pt" if tag is None else f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, self.checkpoint_dir / name)

    def save_history(self):
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
```

**Step 2: Implement `scripts/train.py`**

```python
# scripts/train.py
"""Train MeshLex VQ-VAE on preprocessed patches."""
import argparse
import torch

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True,
                        help="Patch directories for training (e.g., data/patches/chair data/patches/table)")
    parser.add_argument("--val_dirs", nargs="+", default=None,
                        help="Patch directories for validation")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vq_start_epoch", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load datasets
    from torch.utils.data import ConcatDataset
    train_datasets = [PatchGraphDataset(d) for d in args.train_dirs]
    train_dataset = ConcatDataset(train_datasets)
    print(f"Training patches: {len(train_dataset)}")

    val_dataset = None
    if args.val_dirs:
        val_datasets = [PatchGraphDataset(d) for d in args.val_dirs]
        val_dataset = ConcatDataset(val_datasets)
        print(f"Validation patches: {len(val_dataset)}")

    # Create model
    model = MeshLexVQVAE(
        codebook_size=args.codebook_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        vq_start_epoch=args.vq_start_epoch,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/trainer.py scripts/train.py
git commit -m "feat: training loop with staged VQ introduction and codebook monitoring"
```

---

## Task 11: Evaluation Script (Metrics 1-3)

**Files:**
- Create: `src/evaluate.py`
- Create: `scripts/evaluate.py`

**Step 1: Implement `src/evaluate.py`**

```python
# src/evaluate.py
"""Evaluation metrics for MeshLex validation experiment."""
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from collections import Counter


@torch.no_grad()
def evaluate_reconstruction(model, dataset, device="cuda", batch_size=256):
    """Compute reconstruction CD and codebook utilization on a dataset.

    Returns:
        dict with keys:
          - mean_cd: mean Chamfer Distance (× 10³)
          - std_cd: std of per-patch CD
          - utilization: fraction of codebook used
          - code_histogram: Counter of code usage
          - per_patch_cd: list of per-patch CD values
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_cds = []
    all_indices = []

    for batch in loader:
        batch = batch.to(device)
        result = model(
            batch.x, batch.edge_index, batch.batch,
            batch.n_vertices, batch.gt_vertices,
        )

        # Per-sample CD (approximate via batch average)
        all_cds.append(result["recon_loss"].item())
        all_indices.append(result["indices"].cpu())

    all_idx = torch.cat(all_indices)
    code_counts = Counter(all_idx.numpy().tolist())

    utilization = len(code_counts) / model.codebook.K
    mean_cd = np.mean(all_cds) * 1000  # × 10³

    return {
        "mean_cd": mean_cd,
        "std_cd": np.std(all_cds) * 1000,
        "utilization": utilization,
        "code_histogram": code_counts,
        "n_unique_codes": len(code_counts),
        "total_codes": model.codebook.K,
    }


def compute_go_nogo(same_cat_cd: float, cross_cat_cd: float):
    """Apply Go/No-Go decision matrix.

    Args:
        same_cat_cd: mean CD on same-category test set
        cross_cat_cd: mean CD on cross-category test set

    Returns:
        dict with ratio, decision, next_step
    """
    if same_cat_cd < 1e-10:
        return {"ratio": float("inf"), "decision": "ERROR", "next_step": "CD is zero — check data"}

    ratio = cross_cat_cd / same_cat_cd

    if ratio < 1.2:
        decision = "STRONG GO"
        next_step = "Proceed to full MeshLex experiment design"
    elif ratio < 2.0:
        decision = "WEAK GO"
        next_step = "Adjust story to 'transferable vocabulary', continue"
    elif ratio < 3.0:
        decision = "HOLD"
        next_step = "Analyze failure, consider category-adaptive codebook"
    else:
        decision = "NO-GO"
        next_step = "Core hypothesis falsified. Pivot direction."

    return {
        "ratio": ratio,
        "decision": decision,
        "next_step": next_step,
        "same_cat_cd": same_cat_cd,
        "cross_cat_cd": cross_cat_cd,
    }
```

**Step 2: Implement `scripts/evaluate.py`**

```python
# scripts/evaluate.py
"""Run full evaluation: same-category CD, cross-category CD, Go/No-Go."""
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset
from src.evaluate import evaluate_reconstruction, compute_go_nogo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--same_cat_dirs", nargs="+", required=True,
                        help="Test patches from training categories (chair/table/airplane)")
    parser.add_argument("--cross_cat_dirs", nargs="+", required=True,
                        help="Test patches from held-out categories (car/lamp)")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MeshLexVQVAE(codebook_size=args.codebook_size, embed_dim=args.embed_dim)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print("Model loaded.")

    # Same-category evaluation
    same_ds = ConcatDataset([PatchGraphDataset(d) for d in args.same_cat_dirs])
    print(f"Same-category test patches: {len(same_ds)}")
    same_results = evaluate_reconstruction(model, same_ds, device)
    print(f"Same-cat CD: {same_results['mean_cd']:.4f} (×10³)")
    print(f"Codebook utilization: {same_results['utilization']:.1%}")

    # Cross-category evaluation
    cross_ds = ConcatDataset([PatchGraphDataset(d) for d in args.cross_cat_dirs])
    print(f"\nCross-category test patches: {len(cross_ds)}")
    cross_results = evaluate_reconstruction(model, cross_ds, device)
    print(f"Cross-cat CD: {cross_results['mean_cd']:.4f} (×10³)")

    # Go/No-Go
    decision = compute_go_nogo(same_results["mean_cd"], cross_results["mean_cd"])
    print(f"\n{'='*50}")
    print(f"CD Ratio (cross/same): {decision['ratio']:.2f}×")
    print(f"Decision: {decision['decision']}")
    print(f"Next step: {decision['next_step']}")
    print(f"{'='*50}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "same_category": {k: v for k, v in same_results.items() if k != "code_histogram"},
        "cross_category": {k: v for k, v in cross_results.items() if k != "code_histogram"},
        "go_nogo": decision,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/evaluate.py scripts/evaluate.py
git commit -m "feat: evaluation metrics and Go/No-Go decision script"
```

---

## Task 12: Visualization (Metric 4)

**Files:**
- Create: `scripts/visualize.py`

**Step 1: Implement visualization script**

```python
# scripts/visualize.py
"""Codebook visualization: t-SNE/UMAP, utilization histogram, top-K prototypes."""
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset


def plot_utilization_histogram(code_counts: Counter, K: int, save_path: str):
    """Plot histogram of code usage frequencies."""
    counts = np.zeros(K)
    for code_id, freq in code_counts.items():
        counts[code_id] = freq

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sorted frequency bar chart
    sorted_counts = np.sort(counts)[::-1]
    axes[0].bar(range(K), sorted_counts, width=1.0, color="steelblue")
    axes[0].set_xlabel("Code rank")
    axes[0].set_ylabel("Usage frequency")
    axes[0].set_title(f"Codebook Usage (K={K}, active={np.sum(counts > 0)}/{K})")

    # Right: cumulative coverage
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
    axes[1].plot(range(K), cumulative, color="coral")
    axes[1].axhline(y=0.9, color="gray", linestyle="--", label="90% coverage")
    idx_90 = np.searchsorted(cumulative, 0.9)
    axes[1].axvline(x=idx_90, color="gray", linestyle="--")
    axes[1].set_xlabel("Top-N codes")
    axes[1].set_ylabel("Cumulative coverage")
    axes[1].set_title(f"Top-{idx_90} codes cover 90% of patches")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_codebook_tsne(model, save_path: str):
    """t-SNE of codebook embeddings, colored by cluster."""
    from sklearn.manifold import TSNE

    embeddings = model.codebook.codebook.weight.detach().cpu().numpy()  # (K, dim)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.6, c="steelblue")
    plt.title(f"Codebook t-SNE (K={embeddings.shape[0]})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(history_path: str, save_path: str):
    """Plot training loss, recon loss, and utilization over epochs."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    recon = [h["recon_loss"] for h in history]
    util = [h["codebook_utilization"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, loss, label="Total loss")
    axes[0].plot(epochs, recon, label="Recon loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, util, color="coral")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", label="50% threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Utilization")
    axes[1].set_title("Codebook Utilization")
    axes[1].legend()

    axes[2].plot(epochs, recon, color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Recon CD")
    axes[2].set_title("Reconstruction Chamfer Distance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history", type=str, default="data/checkpoints/training_history.json")
    parser.add_argument("--patch_dirs", nargs="+", default=None,
                        help="Patch dirs for utilization histogram")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MeshLexVQVAE(codebook_size=args.codebook_size, embed_dim=args.embed_dim)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # 1. Training curves
    if Path(args.history).exists():
        plot_training_curves(args.history, str(out / "training_curves.png"))

    # 2. Codebook t-SNE
    plot_codebook_tsne(model, str(out / "codebook_tsne.png"))

    # 3. Utilization histogram (if patch dirs provided)
    if args.patch_dirs:
        ds = ConcatDataset([PatchGraphDataset(d) for d in args.patch_dirs])
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        all_indices = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                result = model(
                    batch.x, batch.edge_index, batch.batch,
                    batch.n_vertices, batch.gt_vertices,
                )
                all_indices.append(result["indices"].cpu())

        all_idx = torch.cat(all_indices)
        code_counts = Counter(all_idx.numpy().tolist())
        plot_utilization_histogram(code_counts, model.codebook.K, str(out / "utilization_histogram.png"))

    print("All visualizations complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/visualize.py
git commit -m "feat: codebook visualization (t-SNE, utilization histogram, training curves)"
```

---

## Task 13: K-means Codebook Initialization

**Files:**
- Create: `scripts/init_codebook.py`

Per the 06 plan, codebook should be initialized with K-means on encoder outputs (VQGAN-LC strategy: improves utilization from 11.2% to 99.4%).

**Step 1: Implement initialization script**

```python
# scripts/init_codebook.py
"""Initialize SimVQ codebook with K-means on encoder embeddings."""
import argparse
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pre-VQ checkpoint (from first 20 epochs encoder-only)")
    parser.add_argument("--patch_dirs", nargs="+", required=True)
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--output", type=str, required=True,
                        help="Output checkpoint with initialized codebook")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MeshLexVQVAE(codebook_size=args.codebook_size, embed_dim=args.embed_dim)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Collect encoder embeddings
    ds = ConcatDataset([PatchGraphDataset(d) for d in args.patch_dirs])
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)

    embeddings = []
    print("Collecting encoder embeddings...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode_only(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu().numpy())

    all_embeddings = np.concatenate(embeddings, axis=0)
    print(f"Collected {all_embeddings.shape[0]} embeddings of dim {all_embeddings.shape[1]}")

    # K-means clustering
    print(f"Running K-means (K={args.codebook_size})...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.codebook_size,
        batch_size=4096,
        max_iter=100,
        random_state=42,
    )
    kmeans.fit(all_embeddings)
    centers = kmeans.cluster_centers_  # (K, dim)
    print(f"K-means done. Inertia: {kmeans.inertia_:.4f}")

    # Initialize codebook
    with torch.no_grad():
        model.codebook.codebook.weight.copy_(torch.tensor(centers, dtype=torch.float32))

    # Save
    ckpt["model_state_dict"] = model.state_dict()
    torch.save(ckpt, args.output)
    print(f"Codebook-initialized checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/init_codebook.py
git commit -m "feat: K-means codebook initialization script"
```

---

## Task 14: End-to-End Run Guide

This is the operational runbook. No code to write — this documents the exact sequence of commands.

### Phase A: Data Preparation (~3h)

```bash
# 1. Download ShapeNet Core v2 (assumes access via shapenet.org or mirror)
#    Place at data/ShapeNetCore.v2/
#    Required categories:
#    - 03001627 (chair)
#    - 04379243 (table)
#    - 02691156 (airplane)
#    - 02958343 (car)
#    - 03636649 (lamp)

# 2. Preprocess: decimate + normalize + segment into patches
python scripts/run_preprocessing.py \
    --shapenet_root data/ShapeNetCore.v2 \
    --output_root data \
    --target_faces 1000 \
    --max_per_category 500

# 3. Verify: check patch statistics
python -c "
import json
meta = json.load(open('data/metadata.json'))
cats = {}
for m in meta:
    c = m['category']
    cats.setdefault(c, []).append(m['n_patches'])
for c, patches in sorted(cats.items()):
    import numpy as np
    p = np.array(patches)
    print(f'{c}: {len(patches)} meshes, {p.sum()} patches, median {np.median(p):.0f} patches/mesh')
"
```

### Phase B: Train Encoder-Only (epochs 0-19, ~2h)

```bash
# Split: train on chair+table+airplane, validate on held-out 20%
# (In practice, split patches by mesh_id to avoid data leakage)

python scripts/train.py \
    --train_dirs data/patches/chair data/patches/table data/patches/airplane \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 999 \
    --checkpoint_dir data/checkpoints
```

### Phase C: Initialize Codebook with K-means

```bash
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/checkpoint_epoch019.pt \
    --patch_dirs data/patches/chair data/patches/table data/patches/airplane \
    --codebook_size 4096 \
    --output data/checkpoints/checkpoint_kmeans_init.pt
```

### Phase D: Full VQ-VAE Training (epochs 20-200, ~6h)

```bash
python scripts/train.py \
    --train_dirs data/patches/chair data/patches/table data/patches/airplane \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints
# NOTE: Load from kmeans-initialized checkpoint (modify train.py to accept --resume)
```

### Phase E: Evaluate (~30min)

```bash
# Same-category test (chair+table+airplane held-out patches)
# Cross-category test (car+lamp — never seen during training)

python scripts/evaluate.py \
    --checkpoint data/checkpoints/checkpoint_final.pt \
    --same_cat_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
    --cross_cat_dirs data/patches/car data/patches/lamp \
    --output results/eval_results.json
```

### Phase F: Visualize

```bash
python scripts/visualize.py \
    --checkpoint data/checkpoints/checkpoint_final.pt \
    --history data/checkpoints/training_history.json \
    --patch_dirs data/patches/chair data/patches/table data/patches/airplane \
    --output_dir results/plots
```

### Phase G: Go/No-Go Decision

Read `results/eval_results.json` and apply the decision matrix:

| Cross/Same CD Ratio | Utilization | Decision | Action |
|---------------------|-------------|----------|--------|
| < 1.2× | > 50% | **STRONG GO** | Full MeshLex paper |
| 1.2×–2.0× | > 50% | **WEAK GO** | Adjust story, continue |
| < 2.0× | 30%–50% | **CONDITIONAL GO** | Increase K or try RQ-VAE |
| 2.0×–3.0× | any | **HOLD** | Analyze failure |
| > 3.0× | any | **NO-GO** | Pivot direction |

**Step: Commit final run guide**

```bash
git add .context/07_impl_plan_meshlex_validation.md
git commit -m "docs: complete implementation plan for MeshLex validation"
```

---

## Summary: Task Dependency Graph

```
Task 1: Environment Setup
    │
    ▼
Task 2: Data Preprocessing Pipeline
    │
    ▼
Task 3: METIS Patch Segmentation
    │
    ▼
Task 4: Batch Processing & Dataset
    │
    ├──────────────────┐
    ▼                  ▼
Task 5: GNN Encoder   Task 9: PyG DataLoader
    │                  │
    ▼                  │
Task 6: SimVQ Codebook │
    │                  │
    ▼                  │
Task 7: Patch Decoder  │
    │                  │
    ▼                  │
Task 8: Full VQ-VAE ◄──┘
    │
    ├──────────────┐
    ▼              ▼
Task 10: Train   Task 13: K-means Init
    │              │
    ▼              │
Task 11: Evaluate ◄┘
    │
    ▼
Task 12: Visualize
    │
    ▼
Task 14: Run Guide (this section)
```

Total estimated implementation time: **8-12 hours of coding**, then **12-24 hours of GPU training**.


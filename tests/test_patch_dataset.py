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

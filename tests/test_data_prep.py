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

    result = load_and_preprocess_mesh(str(path_in), target_faces=12, min_faces=4)

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
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    path_in = tmp_path / "tiny.obj"
    mesh.export(str(path_in))

    result = load_and_preprocess_mesh(str(path_in), target_faces=800)
    assert result is None

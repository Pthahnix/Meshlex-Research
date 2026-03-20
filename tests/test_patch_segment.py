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

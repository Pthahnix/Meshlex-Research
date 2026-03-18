import pytest
import numpy as np
import torch

from src.stitching import infer_adjacency, StitchingMLP, merge_boundary_vertices


def test_infer_adjacency():
    """Patches 0 and 1 are close (adjacent), patch 2 is far (not adjacent)."""
    bv0 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    bv1 = np.array([[0.02, 0.0, 0.0], [0.12, 0.0, 0.0]])  # close to patch 0
    bv2 = np.array([[10.0, 10.0, 10.0], [10.1, 10.0, 10.0]])  # far away

    adj = infer_adjacency([bv0, bv1, bv2], threshold=0.05)

    assert (0, 1) in adj
    assert (0, 2) not in adj
    assert (1, 2) not in adj


def test_stitching_mlp_shape():
    """StitchingMLP with input (5, 9) should produce output (5, 3)."""
    mlp = StitchingMLP(input_dim=9, hidden_dim=256)
    x = torch.randn(5, 9)
    out = mlp(x)
    assert out.shape == (5, 3)


def test_merge_boundary_vertices():
    """Merged vertex count should be less than the sum of individual counts."""
    # Two small meshes sharing some boundary vertices
    verts_a = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ])
    faces_a = np.array([[0, 1, 2]])

    # Mesh B shares edge (1,0)-(0.5,1.0) approximately
    verts_b = np.array([
        [1.0, 0.001, 0.0],   # close to verts_a[1]
        [0.5, 1.001, 0.0],   # close to verts_a[2]
        [1.5, 0.5, 0.0],     # new vertex
    ])
    faces_b = np.array([[0, 1, 2]])

    merged_verts, merged_faces = merge_boundary_vertices(
        verts_a, faces_a, verts_b, faces_b, threshold=0.05
    )

    total_before = len(verts_a) + len(verts_b)
    assert len(merged_verts) < total_before
    assert len(merged_faces) == len(faces_a) + len(faces_b)

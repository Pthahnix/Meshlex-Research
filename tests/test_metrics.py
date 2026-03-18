import pytest
import numpy as np

from src.metrics import normal_consistency, f_score, count_non_manifold_edges


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tetrahedron():
    """Regular tetrahedron: 4 vertices, 4 faces, fully watertight."""
    verts = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ])
    return verts, faces


# ── Normal Consistency Tests ─────────────────────────────────────────────────

def test_normal_consistency_identical():
    """NC of a mesh with itself should be > 0.99."""
    verts, faces = _make_tetrahedron()
    nc = normal_consistency(verts, faces, verts, faces)
    assert nc > 0.99


def test_normal_consistency_range():
    """NC should always be in [0, 1]."""
    verts, faces = _make_tetrahedron()
    # Perturb vertices slightly
    perturbed = verts + np.random.default_rng(42).normal(0, 0.1, verts.shape)
    nc = normal_consistency(perturbed, faces, verts, faces)
    assert 0.0 <= nc <= 1.0


# ── F-score Tests ────────────────────────────────────────────────────────────

def test_f_score_identical():
    """F-score of identical point sets should be > 0.99."""
    pts = np.random.default_rng(0).uniform(-1, 1, (200, 3))
    fs = f_score(pts, pts, threshold=0.01)
    assert fs > 0.99


def test_f_score_distant():
    """F-score of completely separated point sets should be < 0.01."""
    pts_a = np.random.default_rng(0).uniform(0, 1, (200, 3))
    pts_b = np.random.default_rng(1).uniform(100, 101, (200, 3))
    fs = f_score(pts_a, pts_b, threshold=0.01)
    assert fs < 0.01


# ── Non-manifold Edge Tests ─────────────────────────────────────────────────

def test_non_manifold_watertight():
    """Tetrahedron is watertight: 0 non-manifold edges, 6 total edges."""
    _, faces = _make_tetrahedron()
    n_nm, n_total = count_non_manifold_edges(faces)
    assert n_nm == 0
    assert n_total == 6


def test_non_manifold_open():
    """Single triangle: all 3 edges are boundary (shared by only 1 face)."""
    faces = np.array([[0, 1, 2]])
    n_nm, n_total = count_non_manifold_edges(faces)
    assert n_nm == 3
    assert n_total == 3

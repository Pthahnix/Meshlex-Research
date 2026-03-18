import numpy as np
import pytest
from src.discretize import (
    build_icosphere_bins,
    discretize_normal,
    discretize_area,
    discretize_dihedral,
    discretize_face_features,
    compute_discretization_mi,
)


def test_icosphere_bins_count():
    """Icosphere bins should have the requested number of directions."""
    bins = build_icosphere_bins(n_bins=64)
    assert bins.shape == (64, 3)
    # All should be unit vectors
    norms = np.linalg.norm(bins, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_discretize_normal_range():
    """Normal discretization should return indices in [0, n_bins)."""
    bins = build_icosphere_bins(64)
    normals = np.random.randn(100, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    indices = discretize_normal(normals, bins)
    assert indices.shape == (100,)
    assert indices.min() >= 0
    assert indices.max() < 64


def test_discretize_area_range():
    """Area discretization should return indices in [0, n_bins)."""
    areas = np.abs(np.random.randn(100)) * 0.1
    indices = discretize_area(areas, n_bins=8)
    assert indices.min() >= 0
    assert indices.max() < 8


def test_discretize_dihedral_range():
    """Dihedral discretization should return indices in [0, n_bins)."""
    angles = np.random.uniform(0, np.pi, 50)
    indices = discretize_dihedral(angles, n_bins=16)
    assert indices.min() >= 0
    assert indices.max() < 16


def test_discretize_face_features_combined():
    """Combined label = normal_bin * n_area + area_bin."""
    normals = np.random.randn(10, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = np.abs(np.random.randn(10)) * 0.1
    labels = discretize_face_features(normals, areas, n_normal=64, n_area=8)
    assert labels.shape == (10,)
    assert labels.max() < 64 * 8


def test_mi_positive():
    """MI between discrete labels and continuous features should be non-negative."""
    np.random.seed(42)
    labels = np.random.randint(0, 10, 200)
    features = np.random.randn(200, 3)
    mi = compute_discretization_mi(labels, features)
    assert mi >= 0.0

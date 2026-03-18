"""Face feature discretization for Graph BPE.

Converts continuous face features (normals, areas, dihedral angles)
into discrete labels for BPE bigram matching.
"""
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def build_icosphere_bins(n_bins: int = 64) -> np.ndarray:
    """Build approximately uniform directions on the unit sphere.

    Uses Fibonacci lattice for near-uniform distribution.

    Returns:
        (n_bins, 3) unit vectors on the sphere.
    """
    indices = np.arange(n_bins, dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2

    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_bins)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    bins = np.stack([x, y, z], axis=1).astype(np.float32)
    bins /= np.linalg.norm(bins, axis=1, keepdims=True)
    return bins


def discretize_normal(normals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Assign each normal to nearest bin direction.

    Args:
        normals: (N, 3) unit normal vectors
        bins: (B, 3) bin directions from build_icosphere_bins

    Returns:
        (N,) integer bin indices
    """
    # Use absolute dot product (normals and -normals are equivalent for faces)
    dots = np.abs(normals @ bins.T)  # (N, B)
    return dots.argmax(axis=1)


def discretize_area(areas: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize face areas into log-scale bins.

    Args:
        areas: (N,) positive face areas
        n_bins: number of bins

    Returns:
        (N,) integer bin indices in [0, n_bins)
    """
    log_areas = np.log1p(areas)
    lo, hi = log_areas.min(), log_areas.max()
    if hi - lo < 1e-10:
        return np.zeros(len(areas), dtype=np.int64)
    normalized = (log_areas - lo) / (hi - lo)
    indices = np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)
    return indices


def discretize_dihedral(angles: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Discretize dihedral angles into uniform angular bins.

    Args:
        angles: (E,) angles in radians [0, pi]
        n_bins: number of bins

    Returns:
        (E,) integer bin indices in [0, n_bins)
    """
    normalized = angles / np.pi  # [0, 1]
    indices = np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)
    return indices


def discretize_face_features(
    normals: np.ndarray,
    areas: np.ndarray,
    n_normal: int = 64,
    n_area: int = 8,
) -> np.ndarray:
    """Combine normal and area discretization into single face (node) label.

    Note: Dihedral angles are edge-level features, discretized separately
    as edge labels in the dual graph (see dual_graph.py). The spec's
    "combined alphabet 64x8x16=8192" counts node labels (64*8=512) and
    edge labels (16) separately -- the 8192 is the bigram space
    (node_label x edge_label x node_label), not the node label space.

    Args:
        normals: (N, 3) face normal vectors
        areas: (N,) face areas

    Returns:
        (N,) combined label indices in [0, n_normal * n_area)
    """
    bins = build_icosphere_bins(n_normal)
    normal_idx = discretize_normal(normals, bins)
    area_idx = discretize_area(areas, n_area)
    return normal_idx * n_area + area_idx


def compute_discretization_mi(
    labels: np.ndarray,
    continuous_features: np.ndarray,
    n_feature_bins: int = 20,
) -> float:
    """Compute mutual information between discrete labels and continuous features.

    Discretizes continuous features into bins, then computes MI.

    Args:
        labels: (N,) discrete labels
        continuous_features: (N, D) continuous feature matrix
        n_feature_bins: bins for discretizing continuous features

    Returns:
        Average MI across feature dimensions.
    """
    mi_total = 0.0
    n_dims = continuous_features.shape[1]

    for d in range(n_dims):
        col = continuous_features[:, d].reshape(-1, 1)
        kbd = KBinsDiscretizer(n_bins=n_feature_bins, encode="ordinal", strategy="quantile")
        binned = kbd.fit_transform(col).ravel().astype(int)
        mi_total += mutual_info_score(labels, binned)

    return mi_total / n_dims

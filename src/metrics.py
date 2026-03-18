"""Extended evaluation metrics for MeshLex v2: normal consistency, F-score, manifold checks."""
import numpy as np
from scipy.spatial import cKDTree


def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute face normals and centroids.

    Args:
        verts: (V, 3) vertex positions.
        faces: (F, 3) face vertex indices.

    Returns:
        normals: (F, 3) unit face normals.
        centroids: (F, 3) face centroids.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    normals = normals / norms

    centroids = (v0 + v1 + v2) / 3.0
    return normals, centroids


def normal_consistency(
    pred_verts: np.ndarray,
    pred_faces: np.ndarray,
    gt_verts: np.ndarray,
    gt_faces: np.ndarray,
) -> float:
    """Normal Consistency (NC) between predicted and ground-truth meshes.

    For each predicted face, finds the nearest GT face (by centroid distance)
    and computes the absolute dot product of their normals. NC is the mean
    of these dot products, in [0, 1].

    Args:
        pred_verts: (V_p, 3) predicted vertices.
        pred_faces: (F_p, 3) predicted faces.
        gt_verts: (V_g, 3) ground-truth vertices.
        gt_faces: (F_g, 3) ground-truth faces.

    Returns:
        NC score in [0, 1].
    """
    pred_normals, pred_centroids = _compute_face_normals(pred_verts, pred_faces)
    gt_normals, gt_centroids = _compute_face_normals(gt_verts, gt_faces)

    tree = cKDTree(gt_centroids)
    _, indices = tree.query(pred_centroids, k=1)

    # Absolute dot product between matched normals
    dots = np.abs(np.sum(pred_normals * gt_normals[indices], axis=1))
    return float(np.mean(dots))


def f_score(
    pred_verts: np.ndarray,
    gt_verts: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """F-score between predicted and ground-truth point sets.

    Precision: fraction of predicted points within `threshold` of any GT point.
    Recall: fraction of GT points within `threshold` of any predicted point.
    F = 2 * precision * recall / (precision + recall).

    Args:
        pred_verts: (N, 3) predicted points.
        gt_verts: (M, 3) ground-truth points.
        threshold: distance threshold.

    Returns:
        F-score in [0, 1].
    """
    tree_gt = cKDTree(gt_verts)
    tree_pred = cKDTree(pred_verts)

    # Precision: pred → GT
    dists_p2g, _ = tree_gt.query(pred_verts, k=1)
    precision = float(np.mean(dists_p2g <= threshold))

    # Recall: GT → pred
    dists_g2p, _ = tree_pred.query(gt_verts, k=1)
    recall = float(np.mean(dists_g2p <= threshold))

    if precision + recall < 1e-10:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def count_non_manifold_edges(faces: np.ndarray) -> tuple[int, int]:
    """Count non-manifold edges in a triangle mesh.

    A manifold edge is shared by exactly 2 faces. Boundary edges (1 face)
    and non-manifold edges (>2 faces) are both counted as non-manifold.

    Args:
        faces: (F, 3) face vertex indices.

    Returns:
        (n_non_manifold, n_total_edges): count of non-manifold edges and total unique edges.
    """
    edge_counts: dict[tuple[int, int], int] = {}

    for face in faces:
        for k in range(3):
            v0 = int(face[k])
            v1 = int(face[(k + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    n_total = len(edge_counts)
    n_non_manifold = sum(1 for c in edge_counts.values() if c != 2)

    return n_non_manifold, n_total

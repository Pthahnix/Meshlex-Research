"""Stitching module for MeshLex v2: adjacency inference, boundary MLP, and vertex merging."""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


def infer_adjacency(boundary_vertices: list[np.ndarray], threshold: float = 0.05) -> set[tuple[int, int]]:
    """Infer patch adjacency from boundary vertices using nearest-neighbor distance.

    For each pair of patches, checks whether any boundary vertex from one patch
    is within `threshold` distance of any boundary vertex from the other patch.

    Args:
        boundary_vertices: list of (N_i, 3) arrays, one per patch.
        threshold: maximum distance to consider two patches adjacent.

    Returns:
        Set of (i, j) tuples with i < j for adjacent patch pairs.
    """
    n = len(boundary_vertices)
    adjacency = set()

    # Build a KD-tree for each patch's boundary vertices
    trees = []
    for bv in boundary_vertices:
        if len(bv) == 0:
            trees.append(None)
        else:
            trees.append(cKDTree(bv))

    for i in range(n):
        if trees[i] is None:
            continue
        for j in range(i + 1, n):
            if trees[j] is None:
                continue
            # Query: for each vertex in patch i, find nearest in patch j
            dists, _ = trees[j].query(boundary_vertices[i], k=1)
            if np.min(dists) <= threshold:
                adjacency.add((i, j))

    return adjacency


class StitchingMLP(nn.Module):
    """3-layer MLP that predicts merged vertex positions from boundary features.

    Input: (N, 9) — concatenation of [pos_a (3), pos_b (3), delta (3)]
    Output: (N, 3) — predicted merged vertex position.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 9) boundary vertex feature pairs.
        Returns:
            (N, 3) predicted merged positions.
        """
        return self.net(x)


def merge_boundary_vertices(
    verts_a: np.ndarray,
    faces_a: np.ndarray,
    verts_b: np.ndarray,
    faces_b: np.ndarray,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two meshes by fusing nearby boundary vertices.

    For each vertex in mesh B that is within `threshold` of a vertex in mesh A,
    the mesh B vertex is replaced by the mesh A vertex (nearest-neighbor matching).

    Args:
        verts_a: (V_a, 3) vertices of mesh A.
        faces_a: (F_a, 3) face indices of mesh A.
        verts_b: (V_b, 3) vertices of mesh B.
        faces_b: (F_b, 3) face indices of mesh B.
        threshold: max distance for vertex merging.

    Returns:
        (merged_verts, merged_faces) with shared vertices deduplicated.
    """
    tree_a = cKDTree(verts_a)
    dists, indices = tree_a.query(verts_b, k=1)

    n_a = len(verts_a)

    # Build remapping for mesh B vertices
    # Start by assigning each B vertex a new index after all A vertices
    remap_b = np.arange(n_a, n_a + len(verts_b))

    # For B vertices close to an A vertex, remap to the A vertex index
    merge_mask = dists <= threshold
    remap_b[merge_mask] = indices[merge_mask]

    # Collect unique kept B vertices (those not merged into A)
    kept_b_mask = ~merge_mask
    kept_b_indices = np.where(kept_b_mask)[0]

    # Reassign compact indices for kept B vertices
    compact_offset = n_a
    for idx in kept_b_indices:
        remap_b[idx] = compact_offset
        compact_offset += 1

    # Build merged vertex array
    kept_b_verts = verts_b[kept_b_mask]
    merged_verts = np.concatenate([verts_a, kept_b_verts], axis=0) if len(kept_b_verts) > 0 else verts_a.copy()

    # Remap faces
    remapped_faces_b = remap_b[faces_b]
    merged_faces = np.concatenate([faces_a, remapped_faces_b], axis=0)

    return merged_verts, merged_faces

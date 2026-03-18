"""Dual graph construction from triangle mesh.

Each face becomes a node. Two nodes are connected if the corresponding
faces share an edge. Nodes carry discretized face labels; edges carry
discretized dihedral angle labels.
"""
from dataclasses import dataclass, field
import numpy as np
import trimesh

from src.discretize import (
    build_icosphere_bins,
    discretize_normal,
    discretize_area,
    discretize_dihedral,
)


@dataclass
class DualGraph:
    """Labeled dual graph of a triangle mesh."""
    n_nodes: int
    node_labels: np.ndarray      # (N,) int -- combined normal+area label per face
    edge_src: np.ndarray          # (E,) int -- source face index
    edge_dst: np.ndarray          # (E,) int -- dest face index
    edge_labels: np.ndarray       # (E,) int -- dihedral angle bin
    # Optional: keep continuous features for analysis
    face_normals: np.ndarray = field(default=None)  # (N, 3)
    face_areas: np.ndarray = field(default=None)     # (N,)


def build_labeled_dual_graph(
    mesh: trimesh.Trimesh,
    n_normal_bins: int = 64,
    n_area_bins: int = 8,
    n_dihedral_bins: int = 16,
) -> DualGraph:
    """Build labeled dual graph from a triangle mesh.

    Args:
        mesh: Triangle mesh (trimesh.Trimesh)
        n_normal_bins: Number of normal direction bins (icosphere)
        n_area_bins: Number of face area bins (log-scale)
        n_dihedral_bins: Number of dihedral angle bins (uniform 0-pi)

    Returns:
        DualGraph with node/edge labels
    """
    n_faces = len(mesh.faces)

    # Face normals and areas
    face_normals = mesh.face_normals.copy()  # (N, 3)
    face_areas = mesh.area_faces.copy()       # (N,)

    # Discretize node labels: combined normal + area
    ico_bins = build_icosphere_bins(n_normal_bins)
    normal_idx = discretize_normal(face_normals, ico_bins)
    area_idx = discretize_area(face_areas, n_area_bins)
    node_labels = normal_idx * n_area_bins + area_idx

    # Build edges from face adjacency
    face_adj = mesh.face_adjacency            # (E_undirected, 2)
    dihedral_angles = mesh.face_adjacency_angles  # (E_undirected,) radians

    # Make bidirectional
    src = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
    dst = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
    angles = np.concatenate([dihedral_angles, dihedral_angles])

    edge_labels = discretize_dihedral(angles, n_dihedral_bins)

    return DualGraph(
        n_nodes=n_faces,
        node_labels=node_labels,
        edge_src=src,
        edge_dst=dst,
        edge_labels=edge_labels,
        face_normals=face_normals,
        face_areas=face_areas,
    )

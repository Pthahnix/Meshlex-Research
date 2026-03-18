import numpy as np
import trimesh
import pytest
from src.dual_graph import build_labeled_dual_graph, DualGraph


def _make_simple_mesh():
    """4-triangle fan mesh (shared center vertex)."""
    vertices = np.array([
        [0, 0, 0],   # center
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_dual_graph_node_count():
    """Dual graph should have one node per face."""
    mesh = _make_simple_mesh()
    dg = build_labeled_dual_graph(mesh, n_normal_bins=32, n_area_bins=4, n_dihedral_bins=8)
    assert dg.n_nodes == 4


def test_dual_graph_edges_symmetric():
    """Each edge (u, v) should have a corresponding (v, u)."""
    mesh = _make_simple_mesh()
    dg = build_labeled_dual_graph(mesh, n_normal_bins=32, n_area_bins=4, n_dihedral_bins=8)
    edge_set = set()
    for u, v in zip(dg.edge_src, dg.edge_dst):
        edge_set.add((u, v))
    for u, v in zip(dg.edge_src, dg.edge_dst):
        assert (v, u) in edge_set, f"Missing reverse edge ({v}, {u})"


def test_dual_graph_labels_valid():
    """Node and edge labels should be non-negative integers."""
    mesh = _make_simple_mesh()
    dg = build_labeled_dual_graph(mesh, n_normal_bins=32, n_area_bins=4, n_dihedral_bins=8)
    assert all(l >= 0 for l in dg.node_labels)
    assert all(l >= 0 for l in dg.edge_labels)
    assert len(dg.node_labels) == dg.n_nodes
    assert len(dg.edge_labels) == len(dg.edge_src)

"""Patch serialization to .npz and PyTorch Dataset for training."""
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from pathlib import Path

from torch_geometric.data import Data as _PyGData
from src.patch_segment import segment_mesh_to_patches


def compute_face_features(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute 15-dim face features: 9 vertex coords + 3 normal + 3 edge angles.

    Args:
        vertices: (V, 3) local normalized vertex coordinates
        faces: (F, 3) face vertex indices

    Returns:
        (F, 15) feature array
    """
    n_faces = faces.shape[0]
    features = np.zeros((n_faces, 15), dtype=np.float32)

    for i, (v0, v1, v2) in enumerate(faces):
        p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]

        # 9 vertex coordinates (flattened)
        features[i, :3] = p0
        features[i, 3:6] = p1
        features[i, 6:9] = p2

        # Face normal
        e1, e2 = p1 - p0, p2 - p0
        normal = np.cross(e1, e2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal /= norm_len
        features[i, 9:12] = normal

        # Edge angles (interior angles at each vertex)
        edges = [p1 - p0, p2 - p1, p0 - p2]
        for j in range(3):
            a, b = -edges[j - 1], edges[j]
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            features[i, 12 + j] = np.arccos(np.clip(cos_angle, -1, 1))

    return features


def build_face_edge_index(faces: np.ndarray) -> np.ndarray:
    """Build face adjacency edge_index (2, E) from face array.

    Two faces are adjacent if they share exactly 2 vertices (an edge).
    """
    n_faces = faces.shape[0]
    edge_map: dict[tuple, list[int]] = {}

    for fi in range(n_faces):
        verts = sorted(faces[fi])
        for i in range(3):
            for j in range(i + 1, 3):
                edge_key = (verts[i], verts[j])
                edge_map.setdefault(edge_key, []).append(fi)

    src, dst = [], []
    for edge_key, face_list in edge_map.items():
        if len(face_list) == 2:
            f0, f1 = face_list
            src.extend([f0, f1])
            dst.extend([f1, f0])

    return np.array([src, dst], dtype=np.int64)


def process_and_save_patches(
    mesh_path: str,
    mesh_id: str,
    output_dir: str,
    target_patch_faces: int = 35,
) -> dict:
    """Segment a mesh and save each patch as .npz."""
    mesh = trimesh.load(mesh_path, force="mesh")
    patches = segment_mesh_to_patches(mesh, target_patch_faces=target_patch_faces)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, patch in enumerate(patches):
        np.savez_compressed(
            str(out / f"{mesh_id}_patch_{i:03d}.npz"),
            faces=patch.faces,
            vertices=patch.vertices,
            local_vertices=patch.local_vertices,
            centroid=patch.centroid,
            principal_axes=patch.principal_axes,
            scale=np.array([patch.scale]),
            boundary_vertices=np.array(patch.boundary_vertices),
            global_face_indices=patch.global_face_indices,
        )

    return {
        "mesh_id": mesh_id,
        "n_patches": len(patches),
        "face_counts": [p.faces.shape[0] for p in patches],
    }


class PatchDataset(Dataset):
    """PyTorch Dataset that loads .npz patch files."""

    MAX_FACES = 80
    MAX_VERTICES = 128

    def __init__(self, patch_dir: str):
        self.patch_dir = Path(patch_dir)
        self.files = sorted(self.patch_dir.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        faces = data["faces"]
        local_verts = data["local_vertices"]

        # Face features (F, 15)
        face_feats = compute_face_features(local_verts, faces)

        # Face adjacency edge_index (2, E)
        edge_index = build_face_edge_index(faces)

        n_faces = faces.shape[0]
        n_verts = local_verts.shape[0]

        # Pad to fixed size for batching
        padded_feats = np.zeros((self.MAX_FACES, 15), dtype=np.float32)
        padded_feats[:n_faces] = face_feats

        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return {
            "face_features": torch.tensor(padded_feats),
            "edge_index": torch.tensor(edge_index),
            "local_vertices": torch.tensor(padded_verts),
            "n_faces": n_faces,
            "n_vertices": n_verts,
        }


class PatchGraphDataset(Dataset):
    """PyTorch Geometric compatible dataset. Returns Data objects
    with graph structure for SAGEConv + padded vertex targets for decoder.
    """

    MAX_VERTICES = 128

    def __init__(self, patch_dir: str):
        self.patch_dir = Path(patch_dir)
        self.files = sorted(self.patch_dir.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        faces = data["faces"]
        local_verts = data["local_vertices"].astype(np.float32)

        # Face features (F, 15)
        face_feats = compute_face_features(local_verts, faces)

        # Face adjacency graph
        edge_index = build_face_edge_index(faces)

        n_verts = local_verts.shape[0]
        n_faces = faces.shape[0]

        # Pad vertices to max size
        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return PatchData(
            x=torch.tensor(face_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            gt_vertices=torch.tensor(padded_verts, dtype=torch.float32),
            n_vertices=torch.tensor(n_verts, dtype=torch.long),
            n_faces=torch.tensor(n_faces, dtype=torch.long),
        )


class PatchData(_PyGData):
    """PyG Data subclass that stacks gt_vertices instead of concatenating."""

    def __cat_dim__(self, key, value, *args, **kw):
        if key in ("gt_vertices",):
            return None  # stack instead of cat
        return super().__cat_dim__(key, value, *args, **kw)

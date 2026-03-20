"""METIS-based mesh patch segmentation with PCA normalization."""
from dataclasses import dataclass
import numpy as np
import trimesh
import pymetis


@dataclass
class MeshPatch:
    # Topology (local indices)
    faces: np.ndarray              # (F, 3) local vertex indices
    vertices: np.ndarray           # (V, 3) world-space vertex coords
    global_face_indices: np.ndarray  # (F,) indices into the original mesh
    boundary_vertices: list[int]   # local indices of boundary verts

    # Geometry (for reconstruction)
    centroid: np.ndarray           # (3,)
    principal_axes: np.ndarray     # (3, 3) PCA rotation
    scale: float                   # bounding sphere radius

    # Normalized local coordinates
    local_vertices: np.ndarray     # (V, 3) centered + PCA-aligned + unit-scaled
    local_vertices_nopca: np.ndarray = None  # (V, 3) centered + unit-scaled (no PCA)


def _build_face_adjacency(mesh: trimesh.Trimesh):
    """Build adjacency list from face adjacency."""
    n_faces = mesh.faces.shape[0]
    adj_list: list[list[int]] = [[] for _ in range(n_faces)]
    face_adj = mesh.face_adjacency  # (E, 2)

    for f1, f2 in face_adj:
        adj_list[f1].append(f2)
        adj_list[f2].append(f1)

    return adj_list


def _normalize_patch_coords(vertices: np.ndarray):
    """PCA-align and normalize patch vertices to unit sphere."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # PCA alignment
    if centered.shape[0] >= 3:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        aligned = centered @ Vt.T
    else:
        Vt = np.eye(3)
        aligned = centered

    # Scale to unit sphere
    scale = np.max(np.linalg.norm(aligned, axis=1))
    if scale < 1e-8:
        scale = 1.0
    normalized = aligned / scale

    return normalized, centroid, Vt, scale


def _build_subgraph_adj(face_indices: list[int], full_adj: list[list[int]]):
    """Build adjacency list for a subgraph of faces."""
    idx_set = set(face_indices)
    local_map = {g: l for l, g in enumerate(face_indices)}
    sub_adj = [[] for _ in range(len(face_indices))]
    for g_idx in face_indices:
        l_idx = local_map[g_idx]
        for neighbor in full_adj[g_idx]:
            if neighbor in idx_set:
                sub_adj[l_idx].append(local_map[neighbor])
    return sub_adj


def segment_mesh_to_patches(
    mesh: trimesh.Trimesh,
    target_patch_faces: int = 35,
    min_patch_faces: int = 15,
    max_patch_faces: int = 60,
) -> list[MeshPatch]:
    """Segment mesh into patches using METIS graph partitioning.

    Each patch covers ~target_patch_faces faces. Small patches are merged
    into neighbors, large patches are bisected.
    """
    n_faces = mesh.faces.shape[0]
    k = max(2, round(n_faces / target_patch_faces))

    adj_list = _build_face_adjacency(mesh)

    # METIS partitioning
    _, partition = pymetis.part_graph(k, adjacency=adj_list)
    partition = np.array(partition)

    # Group faces by partition
    patch_face_groups: dict[int, list[int]] = {}
    for face_idx, part_id in enumerate(partition):
        patch_face_groups.setdefault(part_id, []).append(face_idx)

    # Post-process: merge small patches into largest neighbor
    merged_groups: dict[int, list[int]] = {}
    merge_targets: dict[int, int] = {}  # small_id -> target_id

    for part_id, face_indices in patch_face_groups.items():
        if len(face_indices) < min_patch_faces:
            # Find neighbor partition with most shared edges
            neighbor_counts: dict[int, int] = {}
            for fi in face_indices:
                for nf in adj_list[fi]:
                    np_id = int(partition[nf])
                    if np_id != part_id:
                        neighbor_counts[np_id] = neighbor_counts.get(np_id, 0) + 1
            if neighbor_counts:
                best_neighbor = max(neighbor_counts, key=neighbor_counts.get)
                merge_targets[part_id] = best_neighbor
                continue
        merged_groups[part_id] = face_indices

    # Apply merges
    for small_id, target_id in merge_targets.items():
        # Follow merge chain
        final_target = target_id
        while final_target in merge_targets:
            final_target = merge_targets[final_target]
        if final_target in merged_groups:
            merged_groups[final_target].extend(patch_face_groups[small_id])
        else:
            merged_groups[final_target] = patch_face_groups[small_id]

    # Post-process: bisect large patches
    result_groups = []
    for part_id, face_indices in merged_groups.items():
        if len(face_indices) > max_patch_faces:
            sub_adj = _build_subgraph_adj(face_indices, adj_list)
            if len(sub_adj) >= 2:
                try:
                    _, sub_part = pymetis.part_graph(2, adjacency=sub_adj)
                    g0 = [face_indices[i] for i, p in enumerate(sub_part) if p == 0]
                    g1 = [face_indices[i] for i, p in enumerate(sub_part) if p == 1]
                    if g0:
                        result_groups.append(g0)
                    if g1:
                        result_groups.append(g1)
                    continue
                except Exception:
                    pass
            result_groups.append(face_indices)
        else:
            result_groups.append(face_indices)

    # Build MeshPatch objects
    patches = []
    for face_indices in result_groups:
        face_indices = np.array(face_indices)
        patch_faces_global = mesh.faces[face_indices]  # (F, 3) global vert indices

        # Extract unique vertices and remap
        unique_verts = np.unique(patch_faces_global.flatten())
        vert_map = {int(g): l for l, g in enumerate(unique_verts)}
        local_faces = np.vectorize(vert_map.get)(patch_faces_global)
        vertices = mesh.vertices[unique_verts]

        # Find boundary vertices (vertices on edges shared with faces outside this patch)
        face_set = set(face_indices.tolist())
        boundary_local = set()
        for fi in face_indices:
            for nf in adj_list[int(fi)]:
                if nf not in face_set:
                    shared = set(mesh.faces[int(fi)].tolist()) & set(mesh.faces[nf].tolist())
                    for v in shared:
                        if v in vert_map:
                            boundary_local.add(vert_map[v])

        # Normalize
        local_verts, centroid, axes, scale = _normalize_patch_coords(vertices)

        # No-PCA normalization: center + scale only
        centered = vertices - centroid
        local_verts_nopca = centered / scale if scale > 1e-8 else centered

        patches.append(MeshPatch(
            faces=local_faces,
            vertices=vertices,
            global_face_indices=face_indices,
            boundary_vertices=sorted(boundary_local),
            centroid=centroid,
            principal_axes=axes,
            scale=scale,
            local_vertices=local_verts,
            local_vertices_nopca=local_verts_nopca,
        ))

    return patches

"""Mesh loading, decimation (pyfqmr), and normalization."""
import numpy as np
import trimesh
import pyfqmr


def load_and_preprocess_mesh(
    path: str,
    target_faces: int = 1000,
    min_faces: int = 200,
) -> trimesh.Trimesh | None:
    """Load mesh, decimate to target_faces, normalize to unit cube centered at origin.

    Returns None if mesh is degenerate or below min_faces after processing.
    """
    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception:
        return None

    if mesh.faces.shape[0] < min_faces:
        return None

    # Decimation via pyfqmr if needed
    if mesh.faces.shape[0] > target_faces * 1.2:
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int64),
        )
        simplifier.simplify_mesh(
            target_count=target_faces, aggressiveness=7, preserve_border=True,
        )
        new_verts, new_faces, _ = simplifier.getMesh()
        mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)

    if mesh.faces.shape[0] < min_faces:
        return None

    # Normalize: center at origin, scale to [-1, 1]
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    scale = np.abs(mesh.vertices).max()
    if scale > 1e-8:
        mesh.vertices /= scale

    return mesh


def preprocess_shapenet_category(
    category_dir: str,
    output_dir: str,
    target_faces: int = 1000,
    max_meshes: int = 500,
) -> list[dict]:
    """Process all OBJ files in a ShapeNet category directory.

    Returns list of metadata dicts: {id, path, n_faces, n_vertices}.
    """
    from pathlib import Path

    cat_path = Path(category_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    obj_files = sorted(cat_path.rglob("*.obj"))[:max_meshes]

    for obj_file in obj_files:
        mesh_id = obj_file.parent.name  # ShapeNet: category/model_id/model.obj
        mesh = load_and_preprocess_mesh(str(obj_file), target_faces=target_faces)
        if mesh is None:
            continue

        out_file = out_path / f"{mesh_id}.obj"
        mesh.export(str(out_file))
        results.append({
            "id": mesh_id,
            "path": str(out_file),
            "n_faces": mesh.faces.shape[0],
            "n_vertices": mesh.vertices.shape[0],
        })

    return results

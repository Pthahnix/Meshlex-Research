"""Daft DataFrame utilities for MeshLex patch data."""
import os

import daft
import numpy as np

from src.patch_segment import MeshPatch


# Column name → target Daft DataType for float32/int32 enforcement
PATCH_COLUMN_TYPES = {
    "patch_idx": daft.DataType.int32(),
    "n_faces": daft.DataType.int32(),
    "n_verts": daft.DataType.int32(),
    "faces": daft.DataType.list(daft.DataType.int32()),
    "vertices": daft.DataType.list(daft.DataType.float32()),
    "local_vertices": daft.DataType.list(daft.DataType.float32()),
    "local_vertices_nopca": daft.DataType.list(daft.DataType.float32()),
    "centroid": daft.DataType.list(daft.DataType.float32()),
    "principal_axes": daft.DataType.list(daft.DataType.float32()),
    "scale": daft.DataType.float32(),
    "boundary_vertices": daft.DataType.list(daft.DataType.int32()),
    "global_face_indices": daft.DataType.list(daft.DataType.int32()),
}

_ALL_COLUMNS = [
    "mesh_id", "patch_idx", "category", "source",
    "n_faces", "n_verts", "faces", "vertices",
    "local_vertices", "local_vertices_nopca",
    "centroid", "principal_axes", "scale",
    "boundary_vertices", "global_face_indices",
]


def make_empty_rows() -> dict[str, list]:
    """Create empty row accumulator with all expected columns."""
    return {col: [] for col in _ALL_COLUMNS}


def accumulate_rows(target: dict, source: dict):
    """Merge source rows into target (in-place)."""
    for key in target:
        target[key].extend(source[key])


def patches_to_daft_rows(
    patches: list[MeshPatch],
    mesh_id: str,
    category: str,
    source: str,
) -> dict[str, list]:
    """Convert one mesh's patches to column-oriented dict for Daft.

    Arrays are flattened to Python lists for Parquet-native list(float32/int32).
    """
    rows = make_empty_rows()
    for i, p in enumerate(patches):
        rows["mesh_id"].append(mesh_id)
        rows["patch_idx"].append(i)
        rows["category"].append(category)
        rows["source"].append(source)
        rows["n_faces"].append(p.faces.shape[0])
        rows["n_verts"].append(p.local_vertices.shape[0])
        rows["faces"].append(p.faces.astype(np.int32).flatten().tolist())
        rows["vertices"].append(p.vertices.astype(np.float32).flatten().tolist())
        rows["local_vertices"].append(p.local_vertices.astype(np.float32).flatten().tolist())
        rows["local_vertices_nopca"].append(p.local_vertices_nopca.astype(np.float32).flatten().tolist())
        rows["centroid"].append(p.centroid.astype(np.float32).tolist())
        rows["principal_axes"].append(p.principal_axes.astype(np.float32).flatten().tolist())
        rows["scale"].append(float(p.scale))
        rows["boundary_vertices"].append(np.array(p.boundary_vertices, dtype=np.int32).tolist())
        rows["global_face_indices"].append(p.global_face_indices.astype(np.int32).tolist())
    return rows


def build_patch_dataframe(rows: dict[str, list]) -> daft.DataFrame:
    """Build a Daft DataFrame from accumulated rows, casting to float32/int32.

    Daft infers float64 from Python lists; we cast columns to enforce float32/int32.
    """
    df = daft.from_pydict(rows)
    for col_name, target_type in PATCH_COLUMN_TYPES.items():
        df = df.with_column(col_name, daft.col(col_name).cast(target_type))
    return df


def get_hf_io_config():
    """Return Daft IOConfig for HuggingFace writes."""
    from daft.io import IOConfig, HuggingFaceConfig
    return IOConfig(hf=HuggingFaceConfig(
        token=os.environ.get("HF_TOKEN"),
    ))

"""Tests for Daft DataFrame utilities."""
import numpy as np
import pytest


def test_make_empty_rows_has_all_columns():
    """make_empty_rows should return dict with all 15 expected columns."""
    from src.daft_utils import make_empty_rows
    rows = make_empty_rows()
    expected = [
        "mesh_id", "patch_idx", "category", "source",
        "n_faces", "n_verts", "faces", "vertices",
        "local_vertices", "local_vertices_nopca",
        "centroid", "principal_axes", "scale",
        "boundary_vertices", "global_face_indices",
    ]
    assert list(rows.keys()) == expected
    assert all(isinstance(v, list) and len(v) == 0 for v in rows.values())


def test_accumulate_rows_merges():
    """accumulate_rows should extend target lists with source lists."""
    from src.daft_utils import make_empty_rows, accumulate_rows
    target = make_empty_rows()
    source = make_empty_rows()
    source["mesh_id"].append("m1")
    source["patch_idx"].append(0)
    source["category"].append("chair")
    source["source"].append("objaverse")
    source["n_faces"].append(30)
    source["n_verts"].append(50)
    source["faces"].append([0, 1, 2])
    source["vertices"].append([0.1, 0.2, 0.3])
    source["local_vertices"].append([0.1, 0.2, 0.3])
    source["local_vertices_nopca"].append([0.1, 0.2, 0.3])
    source["centroid"].append([0.0, 0.0, 0.0])
    source["principal_axes"].append([1.0] * 9)
    source["scale"].append(1.0)
    source["boundary_vertices"].append([0, 1])
    source["global_face_indices"].append([5, 6, 7])
    accumulate_rows(target, source)
    assert len(target["mesh_id"]) == 1
    assert target["mesh_id"][0] == "m1"


def test_patches_to_daft_rows_flattens_arrays():
    """patches_to_daft_rows should flatten all ndarray columns to Python lists."""
    from src.daft_utils import patches_to_daft_rows
    from src.patch_segment import MeshPatch

    patch = MeshPatch(
        faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
        global_face_indices=np.array([10, 11], dtype=np.int32),
        boundary_vertices=[0, 3],
        centroid=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        principal_axes=np.eye(3, dtype=np.float32),
        scale=1.0,
        local_vertices=np.array([[-.5, -.5, 0], [.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0]], dtype=np.float32),
        local_vertices_nopca=np.array([[-.5, -.5, 0], [.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0]], dtype=np.float32),
    )
    rows = patches_to_daft_rows([patch], "test_mesh", "chair", "objaverse")

    assert len(rows["mesh_id"]) == 1
    assert rows["mesh_id"][0] == "test_mesh"
    assert rows["n_faces"][0] == 2
    assert rows["n_verts"][0] == 4
    assert isinstance(rows["faces"][0], list)
    assert len(rows["faces"][0]) == 6  # (2,3) flattened
    assert len(rows["vertices"][0]) == 12  # (4,3) flattened
    assert len(rows["centroid"][0]) == 3
    assert len(rows["principal_axes"][0]) == 9  # (3,3) flattened
    assert isinstance(rows["scale"][0], float)


def test_build_patch_dataframe_schema():
    """build_patch_dataframe should produce a Daft DataFrame with correct schema."""
    import daft
    from src.daft_utils import patches_to_daft_rows, build_patch_dataframe
    from src.patch_segment import MeshPatch

    patch = MeshPatch(
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        global_face_indices=np.array([0], dtype=np.int32),
        boundary_vertices=[0, 1, 2],
        centroid=np.zeros(3, dtype=np.float32),
        principal_axes=np.eye(3, dtype=np.float32),
        scale=1.0,
        local_vertices=np.zeros((3, 3), dtype=np.float32),
        local_vertices_nopca=np.zeros((3, 3), dtype=np.float32),
    )
    rows = patches_to_daft_rows([patch], "m1", "chair", "objaverse")
    df = build_patch_dataframe(rows)

    schema = df.schema()
    assert schema["scale"].dtype == daft.DataType.float32()
    assert "vertices" in schema.column_names()
    assert "faces" in schema.column_names()
    assert df.count_rows() == 1

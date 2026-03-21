"""Tests for parquet_loader — HF Parquet to NPZ conversion."""
import numpy as np
import pytest
from pathlib import Path


def _make_fake_row(mesh_id="abc123", patch_idx=5, n_verts=10, n_faces=6):
    """Create a fake HF dataset row with flat arrays."""
    rng = np.random.default_rng(42)
    return {
        "mesh_id": mesh_id,
        "patch_idx": patch_idx,
        "category": "chair",
        "source": "shapenet",
        "n_verts": n_verts,
        "n_faces": n_faces,
        "vertices": rng.standard_normal(n_verts * 3).astype(np.float32).tolist(),
        "local_vertices": rng.standard_normal(n_verts * 3).astype(np.float32).tolist(),
        "local_vertices_nopca": rng.standard_normal(n_verts * 3).astype(np.float32).tolist(),
        "faces": list(range(n_faces * 3)),  # flat int list
        "centroid": [0.1, 0.2, 0.3],
        "principal_axes": list(np.eye(3, dtype=np.float32).flatten()),
        "scale": 1.5,
        "boundary_vertices": [0, 2, 4],
        "global_face_indices": [10, 11, 12, 13, 14, 15],
    }


class TestParquetRowToNpz:
    """Tests for parquet_row_to_npz."""

    def test_creates_npz_with_correct_filename(self, tmp_path):
        from src.parquet_loader import parquet_row_to_npz

        row = _make_fake_row(mesh_id="mesh42", patch_idx=7)
        path = parquet_row_to_npz(row, tmp_path)

        assert path.name == "mesh42_patch_007.npz"
        assert path.exists()

    def test_npz_contains_all_expected_keys(self, tmp_path):
        from src.parquet_loader import parquet_row_to_npz

        row = _make_fake_row()
        path = parquet_row_to_npz(row, tmp_path)
        data = np.load(path)

        expected_keys = {
            "local_vertices", "local_vertices_nopca", "vertices",
            "faces", "centroid", "principal_axes", "scale",
            "boundary_vertices",
        }
        assert set(data.files) == expected_keys

    def test_shapes_are_correct(self, tmp_path):
        from src.parquet_loader import parquet_row_to_npz

        n_verts, n_faces = 12, 8
        row = _make_fake_row(n_verts=n_verts, n_faces=n_faces)
        path = parquet_row_to_npz(row, tmp_path)
        data = np.load(path)

        assert data["local_vertices"].shape == (n_verts, 3)
        assert data["local_vertices_nopca"].shape == (n_verts, 3)
        assert data["vertices"].shape == (n_verts, 3)
        assert data["faces"].shape == (n_faces, 3)
        assert data["centroid"].shape == (3,)
        assert data["principal_axes"].shape == (3, 3)
        assert data["scale"].shape == ()
        assert data["boundary_vertices"].ndim == 1

    def test_dtypes_are_correct(self, tmp_path):
        from src.parquet_loader import parquet_row_to_npz

        row = _make_fake_row()
        path = parquet_row_to_npz(row, tmp_path)
        data = np.load(path)

        assert data["local_vertices"].dtype == np.float32
        assert data["faces"].dtype == np.int32
        assert data["centroid"].dtype == np.float32
        assert data["principal_axes"].dtype == np.float32
        assert data["scale"].dtype == np.float32
        assert data["boundary_vertices"].dtype == np.int32

    def test_values_are_preserved(self, tmp_path):
        from src.parquet_loader import parquet_row_to_npz

        row = _make_fake_row(n_verts=4, n_faces=2)
        path = parquet_row_to_npz(row, tmp_path)
        data = np.load(path)

        np.testing.assert_allclose(data["centroid"], [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_array_equal(data["principal_axes"], np.eye(3))
        assert float(data["scale"]) == pytest.approx(1.5)
        np.testing.assert_array_equal(data["boundary_vertices"], [0, 2, 4])

    def test_creates_output_dir_if_missing(self, tmp_path):
        from src.parquet_loader import parquet_row_to_npz

        nested = tmp_path / "a" / "b" / "c"
        row = _make_fake_row()
        path = parquet_row_to_npz(row, nested)
        assert path.exists()


class TestDownloadSplitPatches:
    """Tests for download_split_patches with monkeypatched load_dataset."""

    def test_creates_npz_files_for_matching_mesh_ids(self, tmp_path, monkeypatch):
        from src import parquet_loader

        # Create fake dataset rows
        rows = [
            _make_fake_row(mesh_id="m1", patch_idx=0),
            _make_fake_row(mesh_id="m1", patch_idx=1),
            _make_fake_row(mesh_id="m2", patch_idx=0),
            _make_fake_row(mesh_id="m3", patch_idx=0),  # not requested
        ]

        import datasets
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: iter(rows))

        result = parquet_loader.download_split_patches(
            mesh_ids=["m1", "m2"],
            output_dir=tmp_path / "patches",
        )

        npz_files = sorted(result.glob("*.npz"))
        stems = {f.stem for f in npz_files}
        assert stems == {"m1_patch_000", "m1_patch_001", "m2_patch_000"}
        assert "m3_patch_000" not in stems

    def test_resume_skips_existing_files(self, tmp_path, monkeypatch):
        from src import parquet_loader

        out_dir = tmp_path / "patches"
        out_dir.mkdir(parents=True)

        # Pre-create one file to simulate resume
        row0 = _make_fake_row(mesh_id="m1", patch_idx=0)
        parquet_loader.parquet_row_to_npz(row0, out_dir)

        rows = [
            _make_fake_row(mesh_id="m1", patch_idx=0),  # should be skipped
            _make_fake_row(mesh_id="m1", patch_idx=1),  # should be written
        ]

        import datasets
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: iter(rows))

        result = parquet_loader.download_split_patches(
            mesh_ids=["m1"],
            output_dir=out_dir,
        )

        npz_files = sorted(result.glob("*.npz"))
        assert len(npz_files) == 2  # both exist
        stems = {f.stem for f in npz_files}
        assert stems == {"m1_patch_000", "m1_patch_001"}

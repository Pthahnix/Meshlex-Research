"""Tests for Objaverse streaming pipeline cleanup behavior."""
from pathlib import Path


def test_cleanup_objaverse_cache_deletes_real_download_paths(tmp_path):
    """Cleanup should delete GLBs using actual objaverse download paths."""
    from scripts.stream_objaverse_daft import cleanup_objaverse_cache

    uid = "abcdef1234567890abcdef1234567890"
    real_dir = tmp_path / "glbs" / "000-090"
    real_dir.mkdir(parents=True)
    real_file = real_dir / f"{uid}.glb"
    real_file.write_bytes(b"glb")

    guessed_dir = tmp_path / "glbs" / uid[:2]
    guessed_dir.mkdir(parents=True)
    guessed_file = guessed_dir / f"{uid}.glb"
    guessed_file.write_bytes(b"wrong")

    cleanup_objaverse_cache({uid: str(real_file)})

    assert not real_file.exists()
    assert guessed_file.exists()

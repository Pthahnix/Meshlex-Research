"""Load MeshLex patch data from HuggingFace Parquet dataset to local NPZ files."""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parquet_row_to_npz(row: dict, output_dir: str | Path) -> Path:
    """Convert a single HuggingFace dataset row to an NPZ file.

    Args:
        row: Dict with keys matching HF dataset columns. Arrays are flat lists.
        output_dir: Directory to write NPZ files into.

    Returns:
        Path to the written NPZ file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_id = row["mesh_id"]
    patch_idx = int(row["patch_idx"])
    n_verts = int(row["n_verts"])
    n_faces = int(row["n_faces"])

    filename = f"{mesh_id}_patch_{patch_idx:03d}.npz"
    output_path = output_dir / filename

    # Reshape flat arrays to proper shapes
    local_vertices = np.array(row["local_vertices"], dtype=np.float32).reshape(n_verts, 3)
    local_vertices_nopca = np.array(row["local_vertices_nopca"], dtype=np.float32).reshape(n_verts, 3)
    vertices = np.array(row["vertices"], dtype=np.float32).reshape(n_verts, 3)
    faces = np.array(row["faces"], dtype=np.int32).reshape(n_faces, 3)
    centroid = np.array(row["centroid"], dtype=np.float32).reshape(3)
    principal_axes = np.array(row["principal_axes"], dtype=np.float32).reshape(3, 3)
    scale = float(row["scale"])
    boundary_vertices = np.array(row["boundary_vertices"], dtype=np.int32)

    np.savez_compressed(
        output_path,
        local_vertices=local_vertices,
        local_vertices_nopca=local_vertices_nopca,
        vertices=vertices,
        faces=faces,
        centroid=centroid,
        principal_axes=principal_axes,
        scale=np.float32(scale),
        boundary_vertices=boundary_vertices,
    )
    return output_path


def download_splits_json(
    hf_repo: str = "Pthahnix/MeshLex-Patches",
    output_path: str | Path = "data/splits.json",
) -> Path:
    """Download splits.json from HuggingFace dataset repo.

    Args:
        hf_repo: HuggingFace dataset repository ID.
        output_path: Local path to save splits.json.

    Returns:
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=hf_repo,
        filename="splits.json",
        repo_type="dataset",
        local_dir=str(output_path.parent),
    )
    downloaded = Path(downloaded)
    # hf_hub_download saves to local_dir/splits.json; rename if needed
    if downloaded != output_path and downloaded.exists():
        downloaded.rename(output_path)
    return output_path


def download_split_patches(
    mesh_ids: list[str] | set[str],
    output_dir: str | Path,
    hf_repo: str = "Pthahnix/MeshLex-Patches",
    batch_size: int = 10000,
) -> Path:
    """Download and convert patches for a set of mesh IDs to local NPZ files.

    Args:
        mesh_ids: Set of mesh IDs to download patches for.
        output_dir: Directory to write NPZ files into.
        hf_repo: HuggingFace dataset repository ID.
        batch_size: Batch size for streaming iteration.

    Returns:
        Path to the output directory.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_ids = set(mesh_ids)

    # Find existing files for resume support
    existing = {p.stem for p in output_dir.glob("*.npz")}

    logger.info(
        "Downloading patches for %d mesh IDs to %s (existing: %d files)",
        len(mesh_ids), output_dir, len(existing),
    )

    ds = load_dataset(hf_repo, split="train", streaming=True)

    written = 0
    skipped = 0
    for row in ds:
        if row["mesh_id"] not in mesh_ids:
            continue

        # Build expected filename stem for resume check
        stem = f"{row['mesh_id']}_patch_{int(row['patch_idx']):03d}"
        if stem in existing:
            skipped += 1
            continue

        parquet_row_to_npz(row, output_dir)
        written += 1

    logger.info("Done: %d written, %d skipped (already existed)", written, skipped)
    return output_dir


def prepare_training_data(
    output_base: str | Path,
    hf_repo: str = "Pthahnix/MeshLex-Patches",
    splits: list[str] | None = None,
) -> Path:
    """Full pipeline: download splits.json, then download patches for each split.

    Args:
        output_base: Base directory for all output (e.g. "data/hf_patches").
        hf_repo: HuggingFace dataset repository ID.
        splits: Which splits to download. Defaults to ["seen_train", "seen_test"].

    Returns:
        Path to output_base.
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["seen_train", "seen_test"]

    # Step 1: download splits.json
    splits_path = output_base / "splits.json"
    if not splits_path.exists():
        logger.info("Downloading splits.json...")
        download_splits_json(hf_repo, splits_path)

    with open(splits_path) as f:
        splits_data = json.load(f)

    # Step 2: download patches for each requested split
    for split_name in splits:
        if split_name not in splits_data:
            logger.warning("Split '%s' not found in splits.json, skipping", split_name)
            continue
        mesh_ids = splits_data[split_name]
        split_dir = output_base / split_name
        logger.info("Downloading split '%s' (%d meshes)...", split_name, len(mesh_ids))
        download_split_patches(mesh_ids, split_dir, hf_repo=hf_repo)

    return output_base

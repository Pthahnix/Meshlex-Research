"""Stream-process ShapeNetCore v2 → Daft → HF Parquet, one synset at a time.

Usage:
    python scripts/stream_shapenet_daft.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --work_dir /tmp/meshlex_shapenet
"""
import argparse
import gc
import logging
import shutil
import time
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from src.data_prep import load_and_preprocess_mesh
from src.patch_segment import segment_mesh_to_patches
from src.daft_utils import (
    patches_to_daft_rows, build_patch_dataframe,
    get_hf_io_config, make_empty_rows, accumulate_rows,
)
from src.stream_utils import (
    ProgressTracker, MetadataCollector, SHAPENET_SYNSET_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SHAPENET_HF_REPO = "ShapeNet/ShapeNetCore"


def process_category(
    synset_id: str, cat_name: str,
    work_dir: Path, hf_repo: str,
    progress: ProgressTracker, metadata: MetadataCollector,
    io_config, target_faces: int = 1000, sub_batch_size: int = 500,
):
    cat_key = f"shapenet_{synset_id}"
    if progress.is_done(cat_key):
        log.info(f"Skipping {cat_name} ({synset_id}) — already done")
        return

    log.info(f"[{cat_name}] Downloading category {synset_id}.zip...")
    t0 = time.time()
    local_dir = work_dir / "shapenet_raw"
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = hf_hub_download(
            repo_id=SHAPENET_HF_REPO, repo_type="dataset",
            filename=f"{synset_id}.zip",
            local_dir=str(local_dir),
        )
    except Exception as e:
        log.error(f"[{cat_name}] Download failed: {e}")
        progress.mark_done(cat_key, {"error": str(e)})
        progress.save()
        return

    # Extract zip
    extract_dir = local_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(extract_dir))
    except Exception as e:
        log.error(f"[{cat_name}] Extraction failed: {e}")
        progress.mark_done(cat_key, {"error": str(e)})
        progress.save()
        return

    # Delete zip to save disk
    Path(zip_path).unlink(missing_ok=True)
    log.info(f"[{cat_name}] Downloaded + extracted in {time.time()-t0:.0f}s")

    synset_dir = extract_dir / synset_id
    obj_files = sorted(synset_dir.rglob("model_normalized.obj")) if synset_dir.exists() else []
    log.info(f"[{cat_name}] Found {len(obj_files)} models")

    accumulated = make_empty_rows()
    n_ok, n_fail, n_patches_total, mesh_count = 0, 0, 0, 0

    for obj_file in obj_files:
        model_id = obj_file.parent.parent.name
        mesh_id = f"{synset_id}_{model_id}"
        try:
            mesh = load_and_preprocess_mesh(str(obj_file), target_faces=target_faces)
            if mesh is None:
                n_fail += 1
                continue
            patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
            if len(patches) == 0:
                n_fail += 1
                continue

            rows = patches_to_daft_rows(patches, mesh_id, cat_name, "shapenet")
            accumulate_rows(accumulated, rows)
            metadata.add(mesh_id, {
                "category": cat_name, "source": "shapenet",
                "synset_id": synset_id, "n_patches": len(patches),
                "n_faces": int(mesh.faces.shape[0]),
                "n_verts": int(mesh.vertices.shape[0]),
            })
            n_ok += 1
            n_patches_total += len(patches)
            mesh_count += 1
        except Exception as e:
            log.warning(f"[{cat_name}] Failed {mesh_id}: {e}")
            n_fail += 1

        # Write sub-batch when full
        if mesh_count >= sub_batch_size:
            log.info(f"[{cat_name}] Writing sub-batch ({len(accumulated['mesh_id'])} patches)...")
            df = build_patch_dataframe(accumulated)
            df.write_huggingface(hf_repo, io_config=io_config)
            accumulated = make_empty_rows()
            mesh_count = 0
            metadata.save()

    # Write remaining
    if len(accumulated["mesh_id"]) > 0:
        df = build_patch_dataframe(accumulated)
        df.write_huggingface(hf_repo, io_config=io_config)

    # Cleanup extracted files + HF cache
    shutil.rmtree(extract_dir, ignore_errors=True)
    # Also remove the downloaded zip from HF cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for cache_dir in hf_cache.glob("datasets--ShapeNet--ShapeNetCore*"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    log.info(f"[{cat_name}] Done: {n_ok} ok, {n_fail} fail, {n_patches_total} patches")
    progress.mark_done(cat_key, {
        "meshes_ok": n_ok, "meshes_fail": n_fail, "patches": n_patches_total,
    })
    progress.save()
    metadata.save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--work_dir", default="/tmp/meshlex_shapenet")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--sub_batch_size", type=int, default=500)
    parser.add_argument("--only_synset", type=str, default=None,
                        help="Process only this synset ID (for dry-run)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    io_config = get_hf_io_config()
    progress = ProgressTracker(str(work_dir / "progress.json"))
    metadata = MetadataCollector(str(work_dir / "metadata.json"))

    synsets = SHAPENET_SYNSET_MAP
    if args.only_synset:
        synsets = {args.only_synset: SHAPENET_SYNSET_MAP[args.only_synset]}

    log.info(f"Processing {len(synsets)} ShapeNet categories")
    for synset_id, cat_name in sorted(synsets.items()):
        process_category(synset_id, cat_name, work_dir, args.hf_repo,
                         progress, metadata, io_config,
                         args.target_faces, args.sub_batch_size)

    HfApi().upload_file(
        path_or_fileobj=str(work_dir / "metadata.json"),
        path_in_repo="metadata_shapenet.json",
        repo_id=args.hf_repo, repo_type="dataset",
    )
    log.info("ShapeNet streaming complete!")


if __name__ == "__main__":
    main()

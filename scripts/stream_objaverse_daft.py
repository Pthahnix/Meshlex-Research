"""Stream-process Objaverse-LVIS objects → Daft → HF Parquet.

Usage:
    python scripts/stream_objaverse_daft.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --batch_size 500 \
        --work_dir /tmp/meshlex_objaverse
"""
import argparse
import gc
import logging
import shutil
import time
from pathlib import Path

import objaverse

from src.data_prep import load_and_preprocess_mesh
from src.patch_segment import segment_mesh_to_patches
from src.daft_utils import (
    patches_to_daft_rows, build_patch_dataframe,
    get_hf_io_config, make_empty_rows, accumulate_rows,
)
from src.stream_utils import ProgressTracker, MetadataCollector, batch_uids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def cleanup_objaverse_cache(objects: dict[str, str | None]):
    """Delete downloaded Objaverse GLBs using the actual returned paths."""
    for glb_path in objects.values():
        if glb_path is None:
            continue
        try:
            Path(glb_path).unlink(missing_ok=True)
        except Exception:
            pass


def process_batch(
    batch_idx: int,
    uids: list[str],
    uid_to_cat: dict[str, str],
    work_dir: Path,
    hf_repo: str,
    progress: ProgressTracker,
    metadata: MetadataCollector,
    io_config,
    download_processes: int = 8,
    target_faces: int = 1000,
):
    batch_id = f"batch_{batch_idx:03d}"
    if progress.is_done(batch_id):
        log.info(f"Skipping {batch_id} (already done)")
        return

    # Download GLBs
    log.info(f"[{batch_id}] Downloading {len(uids)} objects...")
    t0 = time.time()
    objects = objaverse.load_objects(uids=uids, download_processes=download_processes)
    log.info(f"[{batch_id}] Downloaded in {time.time()-t0:.0f}s")

    accumulated = make_empty_rows()
    n_ok, n_fail, n_patches_total = 0, 0, 0

    for uid in uids:
        glb_path = objects.get(uid)
        if glb_path is None:
            n_fail += 1
            continue
        try:
            mesh = load_and_preprocess_mesh(glb_path, target_faces=target_faces)
            if mesh is None:
                n_fail += 1
                continue
            patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
            if len(patches) == 0:
                n_fail += 1
                continue

            category = uid_to_cat.get(uid, "unknown")
            rows = patches_to_daft_rows(patches, uid, category, "objaverse")
            accumulate_rows(accumulated, rows)

            metadata.add(uid, {
                "category": category, "source": "objaverse",
                "n_patches": len(patches),
                "n_faces": int(mesh.faces.shape[0]),
                "n_verts": int(mesh.vertices.shape[0]),
            })
            n_ok += 1
            n_patches_total += len(patches)
        except Exception as e:
            log.warning(f"[{batch_id}] Failed {uid}: {e}")
            n_fail += 1

    log.info(f"[{batch_id}] Processed: {n_ok} ok, {n_fail} fail, {n_patches_total} patches")

    # Write to HF via Daft
    if n_ok > 0:
        log.info(f"[{batch_id}] Writing {n_patches_total} patches to HF...")
        df = build_patch_dataframe(accumulated)
        df.write_huggingface(hf_repo, io_config=io_config)

    # Cleanup objaverse caches
    cleanup_objaverse_cache(objects)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for cache_dir in hf_cache.glob("models--allenai--objaverse*"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    progress.mark_done(batch_id, {
        "meshes_ok": n_ok, "meshes_fail": n_fail, "patches": n_patches_total,
    })
    progress.save()
    metadata.save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--download_processes", type=int, default=8)
    parser.add_argument("--work_dir", default="/tmp/meshlex_objaverse")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Max batches to process (-1=all, for dry-run use 1 or 2)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    io_config = get_hf_io_config()

    progress = ProgressTracker(str(work_dir / "progress.json"))
    metadata = MetadataCollector(str(work_dir / "metadata.json"))

    log.info("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    uid_to_cat = {}
    all_uids = []
    for cat_name, uids in sorted(lvis.items()):
        for uid in uids:
            if uid not in uid_to_cat:
                uid_to_cat[uid] = cat_name
                all_uids.append(uid)
    log.info(f"Total UIDs: {len(all_uids)}")

    batches = list(batch_uids(all_uids, batch_size=args.batch_size))
    if args.max_batches > 0:
        batches = batches[:args.max_batches]
    log.info(f"Processing {len(batches)} batches")

    for i, batch in enumerate(batches):
        process_batch(i, batch, uid_to_cat, work_dir, args.hf_repo,
                      progress, metadata, io_config,
                      args.download_processes, args.target_faces)

    # Upload metadata JSON
    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=str(work_dir / "metadata.json"),
        path_in_repo="metadata_objaverse.json",
        repo_id=args.hf_repo, repo_type="dataset",
    )
    log.info("Objaverse streaming complete!")


if __name__ == "__main__":
    main()

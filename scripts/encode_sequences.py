"""Encode all meshes through trained VQ-VAE → save per-mesh sequence NPZ.

This bridges the VQ-VAE (Phase 1/2) and AR (Phase 3) training stages.
For each mesh, loads its patches, runs them through the trained VQ-VAE encoder
+ quantizer, then saves centroids, scales, and codebook indices as a single
NPZ file.

Usage:
    # From NPZ directories (original):
    python scripts/encode_sequences.py \
        --patch_dirs data/patches/lvis_wide/seen_train \
        --checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir data/sequences/lvis_wide \
        --mode rvq

    # From Arrow dataset + mmap features (full-scale):
    python scripts/encode_sequences.py \
        --arrow_dir /data/.../splits/seen_train \
        --feature_dir /data/.../features/seen_train \
        --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
        --output_dir data/sequences/rvq_full_pca \
        --mode rvq --batch_size 4096
"""
import argparse
import gc
import time
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.patch_dataset import PatchGraphDataset
from src.model import MeshLexVQVAE
from src.model_rvq import MeshLexRVQVAE


def encode_from_npz(model, args, device, out):
    """Original NPZ-based encoding path."""
    for patch_dir in args.patch_dirs:
        patch_dir = Path(patch_dir)
        npz_files = sorted(patch_dir.glob("*.npz"))
        print(f"Processing {len(npz_files)} patches from {patch_dir.name}...")

        # Group files by mesh_id (filename format: {mesh_id}_patch_{N}.npz)
        mesh_groups = {}
        for f in npz_files:
            stem = f.stem
            if "_patch_" in stem:
                mesh_id = stem.rsplit("_patch_", 1)[0]
            elif "_patch" in stem:
                mesh_id = stem.rsplit("_patch", 1)[0]
            else:
                mesh_id = stem
            mesh_groups.setdefault(mesh_id, []).append(f)

        # Load dataset and encode all patches
        dataset = PatchGraphDataset(str(patch_dir), use_nopca=args.nopca)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_tokens = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z = model.encoder(batch.x, batch.edge_index, batch.batch)
                if args.mode == "rvq":
                    _, indices = model.rvq(z)  # (B, n_levels)
                else:
                    _, indices = model.codebook(z)  # (B,)
                all_tokens.append(indices.cpu().numpy())

        tokens = np.concatenate(all_tokens, axis=0)

        # Extract centroids and scales from NPZ files directly
        idx = 0
        for mesh_id, patch_files in sorted(mesh_groups.items()):
            out_path = out / f"{mesh_id}_sequence.npz"
            if out_path.exists():
                idx += len(patch_files)
                continue

            n = len(patch_files)
            mesh_tokens = tokens[idx:idx + n]

            centroids = []
            scales = []
            pca_axes = []
            for pf in sorted(patch_files):
                data = np.load(str(pf))
                centroids.append(data["centroid"])
                scale_val = data["scale"]
                scales.append(float(scale_val[0]) if scale_val.ndim > 0 else float(scale_val))
                if "principal_axes" in data:
                    pca_axes.append(data["principal_axes"])
                else:
                    pca_axes.append(np.eye(3, dtype=np.float32))

            mesh_centroids = np.array(centroids, dtype=np.float32)
            mesh_scales = np.array(scales, dtype=np.float32)
            mesh_pca_axes = np.array(pca_axes, dtype=np.float32)
            idx += n

            np.savez(out_path,
                     centroids=mesh_centroids,
                     scales=mesh_scales,
                     tokens=mesh_tokens,
                     principal_axes=mesh_pca_axes)

        gc.collect()
        torch.cuda.empty_cache()


def encode_from_arrow(model, args, device, out):
    """Arrow dataset + mmap features encoding path (full-scale).

    Strategy: load the mmap feature dataset for GPU encoding,
    then read metadata (mesh_id, centroid, scale, principal_axes) from Arrow
    to group tokens into per-mesh sequence NPZ files.
    """
    from src.patch_dataset import MmapPatchDataset
    from datasets import load_from_disk

    # Step 1: Load Arrow dataset for metadata
    print(f"Loading Arrow dataset from {args.arrow_dir}...")
    ds = load_from_disk(args.arrow_dir)
    N = len(ds)
    print(f"  {N} patches total")

    # Step 2: Load mmap features for fast encoding
    print(f"Loading mmap features from {args.feature_dir}...")
    dataset = MmapPatchDataset(args.feature_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True,
                        prefetch_factor=4)

    # Step 3: Encode all patches through VQ-VAE
    print(f"Encoding {N} patches (batch_size={args.batch_size})...")
    all_tokens = []
    t0 = time.time()
    done = 0

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        for batch in loader:
            batch = batch.to(device)
            z = model.encoder(batch.x, batch.edge_index, batch.batch)
            if args.mode == "rvq":
                _, indices = model.rvq(z)
            else:
                _, indices = model.codebook(z)
            all_tokens.append(indices.cpu().numpy())
            done += len(batch.x) if hasattr(batch, 'x') and batch.x.dim() == 2 else batch.n_faces.shape[0]
            if done % (args.batch_size * 50) < args.batch_size:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (N - done) / rate if rate > 0 else 0
                print(f"  Encoded {done}/{N} ({done/N*100:.1f}%) | {rate:.0f}/sec | ETA {eta/60:.1f}min")

    tokens = np.concatenate(all_tokens, axis=0)
    elapsed = time.time() - t0
    print(f"  Encoding done: {N} patches in {elapsed/60:.1f}min ({N/elapsed:.0f}/sec)")

    # Step 4: Read ALL metadata in one sequential pass (avoids slow random access)
    print("Reading metadata and grouping by mesh_id...")
    t1 = time.time()

    # Pre-allocate metadata arrays
    all_mesh_ids = []
    all_patch_idx = []
    all_centroids = np.zeros((N, 3), dtype=np.float32)
    all_scales = np.zeros(N, dtype=np.float32)
    all_pca_axes = np.zeros((N, 3, 3), dtype=np.float32)

    bs = 10000
    for start in range(0, N, bs):
        end = min(start + bs, N)
        batch = ds[start:end]
        actual = end - start
        all_mesh_ids.extend(batch["mesh_id"])
        all_patch_idx.extend(batch["patch_idx"])
        for i in range(actual):
            all_centroids[start + i] = np.array(batch["centroid"][i], dtype=np.float32)
            all_scales[start + i] = float(batch["scale"][i])
            if batch.get("principal_axes") and batch["principal_axes"][i]:
                all_pca_axes[start + i] = np.array(batch["principal_axes"][i],
                                                     dtype=np.float32).reshape(3, 3)
            else:
                all_pca_axes[start + i] = np.eye(3, dtype=np.float32)

        if (start + actual) % (bs * 50) < bs:
            print(f"  Metadata: {start + actual}/{N} ({(start + actual)/N*100:.1f}%)")

    # Build mesh_id → sorted patch indices
    mesh_patches = {}
    for gidx in range(N):
        mesh_id = all_mesh_ids[gidx]
        patch_idx = all_patch_idx[gidx]
        if mesh_id not in mesh_patches:
            mesh_patches[mesh_id] = []
        mesh_patches[mesh_id].append((gidx, patch_idx))

    print(f"  {len(mesh_patches)} unique meshes, metadata read in {time.time()-t1:.1f}s")

    # Step 5: For each mesh, extract tokens + metadata and save
    print("Saving per-mesh sequence NPZ files...")
    t2 = time.time()
    saved = 0
    skipped = 0

    for mesh_id, patch_list in sorted(mesh_patches.items()):
        out_path = out / f"{mesh_id}_sequence.npz"
        if out_path.exists():
            skipped += 1
            continue

        # Sort by patch_idx within each mesh
        patch_list.sort(key=lambda x: x[1])
        global_indices = [p[0] for p in patch_list]

        np.savez(out_path,
                 centroids=all_centroids[global_indices],
                 scales=all_scales[global_indices],
                 tokens=tokens[global_indices],
                 principal_axes=all_pca_axes[global_indices])
        saved += 1
        if saved % 5000 == 0:
            print(f"  Saved {saved} meshes...")

    elapsed2 = time.time() - t2
    print(f"  Saved {saved} meshes, skipped {skipped} existing ({elapsed2/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dirs", nargs="+", default=None,
                        help="Directories with patch NPZ files")
    parser.add_argument("--arrow_dir", type=str, default=None,
                        help="Arrow dataset directory (full-scale)")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Mmap feature directory (used with --arrow_dir)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained VQ-VAE checkpoint")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save per-mesh sequence NPZs")
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nopca", action="store_true",
                        help="Use non-PCA-normalized vertices for encoding")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load VQ-VAE
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.mode == "rvq":
        model = MeshLexRVQVAE()
    else:
        model = MeshLexVQVAE()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    if args.arrow_dir and args.feature_dir:
        encode_from_arrow(model, args, device, out)
    elif args.patch_dirs:
        encode_from_npz(model, args, device, out)
    else:
        raise ValueError("Must specify either --patch_dirs or --arrow_dir + --feature_dir")

    print(f"All sequences saved to {out}")


if __name__ == "__main__":
    main()

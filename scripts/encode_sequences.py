"""Encode all meshes through trained VQ-VAE → save per-mesh sequence NPZ.

This bridges the VQ-VAE (Phase 1/2) and AR (Phase 3) training stages.
For each mesh, loads its patches, runs them through the trained VQ-VAE encoder
+ quantizer, then saves centroids, scales, and codebook indices as a single
NPZ file.

Usage:
    python scripts/encode_sequences.py \
        --patch_dirs data/patches/lvis_wide/seen_train \
        --checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir data/sequences/lvis_wide \
        --mode rvq
"""
import argparse
import gc
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.patch_dataset import PatchGraphDataset
from src.model import MeshLexVQVAE
from src.model_rvq import MeshLexRVQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dirs", nargs="+", required=True,
                        help="Directories with patch NPZ files")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained VQ-VAE checkpoint")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save per-mesh sequence NPZs")
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--batch_size", type=int, default=256)
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
        dataset = PatchGraphDataset(str(patch_dir))
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

            # Load centroid and scale from each patch NPZ
            centroids = []
            scales = []
            for pf in sorted(patch_files):
                data = np.load(str(pf))
                centroids.append(data["centroid"])
                scale_val = data["scale"]
                scales.append(float(scale_val[0]) if scale_val.ndim > 0 else float(scale_val))

            mesh_centroids = np.array(centroids, dtype=np.float32)
            mesh_scales = np.array(scales, dtype=np.float32)
            idx += n

            np.savez(out_path,
                     centroids=mesh_centroids,
                     scales=mesh_scales,
                     tokens=mesh_tokens)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"All sequences saved to {out}")


if __name__ == "__main__":
    main()

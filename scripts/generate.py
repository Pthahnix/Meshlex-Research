"""Generate meshes from trained AR model + VQ-VAE decoder.

Usage:
    python scripts/generate.py \
        --ar_checkpoint data/checkpoints/ar/checkpoint_final.pt \
        --vqvae_checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir results/generated --n_meshes 10 --mode rvq
"""
import argparse
import torch
import numpy as np
import trimesh
from pathlib import Path

from src.ar_model import PatchGPT
from src.model_rvq import MeshLexRVQVAE
from src.model import MeshLexVQVAE
from src.stitching import infer_adjacency, merge_boundary_vertices


def decode_token_sequence(
    sequence: np.ndarray,
    mode: str,
    n_pos_bins: int = 256,
    n_scale_bins: int = 64,
):
    """Decode flat token sequence back to patch parameters."""
    tokens_per_patch = 5 if mode == "simvq" else 7
    offset_y = n_pos_bins
    offset_z = 2 * n_pos_bins
    offset_scale = 3 * n_pos_bins
    offset_code = 3 * n_pos_bins + n_scale_bins

    n_patches = len(sequence) // tokens_per_patch
    patches = []

    for i in range(n_patches):
        base = i * tokens_per_patch
        pos_x = int(sequence[base + 0])
        pos_y = int(sequence[base + 1]) - offset_y
        pos_z = int(sequence[base + 2]) - offset_z
        scale = int(sequence[base + 3]) - offset_scale

        if mode == "simvq":
            tok = int(sequence[base + 4]) - offset_code
            patches.append({"pos": [pos_x, pos_y, pos_z], "scale": scale, "tokens": tok})
        else:
            tok1 = int(sequence[base + 4]) - offset_code
            tok2 = int(sequence[base + 5]) - offset_code
            tok3 = int(sequence[base + 6]) - offset_code
            patches.append({
                "pos": [pos_x, pos_y, pos_z],
                "scale": scale,
                "tokens": [tok1, tok2, tok3],
            })

    return patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--vqvae_checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/generated")
    parser.add_argument("--n_meshes", type=int, default=10)
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load AR model
    ar_ckpt = torch.load(args.ar_checkpoint, map_location=device, weights_only=False)
    ar_config = ar_ckpt.get("config", {})
    ar_model = PatchGPT(**ar_config).to(device)
    ar_model.load_state_dict(ar_ckpt["model_state_dict"])
    ar_model.eval()

    # Load VQ-VAE
    vq_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    if args.mode == "rvq":
        vqvae = MeshLexRVQVAE().to(device)
    else:
        vqvae = MeshLexVQVAE().to(device)
    vqvae.load_state_dict(vq_ckpt["model_state_dict"], strict=False)
    vqvae.eval()

    for mesh_idx in range(args.n_meshes):
        print(f"Generating mesh {mesh_idx + 1}/{args.n_meshes}...")

        # Generate token sequence
        with torch.no_grad():
            seq = ar_model.generate(
                max_len=910 if args.mode == "rvq" else 650,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        seq_np = seq.cpu().numpy()
        patch_params = decode_token_sequence(seq_np, mode=args.mode)
        print(f"  Generated {len(patch_params)} patches")

        # Decode each patch through VQ-VAE decoder
        all_verts = []
        boundary_verts_list = []

        for p in patch_params:
            with torch.no_grad():
                if args.mode == "rvq":
                    tok_indices = torch.tensor(p["tokens"], dtype=torch.long, device=device)
                    z_hat = vqvae.rvq.decode_indices(tok_indices.unsqueeze(0))
                else:
                    tok_idx = torch.tensor([p["tokens"]], dtype=torch.long, device=device)
                    cw = vqvae.codebook.get_quant_codebook()
                    z_hat = cw[tok_idx]

                n_verts = torch.tensor([30], device=device)
                local_verts = vqvae.decoder(z_hat, n_verts)
                local_verts = local_verts[0, :30].cpu().numpy()

            # Inverse transform to world space
            pos = np.array(p["pos"], dtype=np.float32) / 255.0
            scale = max(p["scale"] / 63.0, 0.01)
            world_verts = local_verts * scale + pos

            all_verts.append(world_verts)
            boundary_verts_list.append(world_verts)

        # Assemble: for now save as point cloud
        if all_verts:
            combined_verts = np.concatenate(all_verts, axis=0)
            mesh = trimesh.PointCloud(combined_verts)
            mesh.export(str(out / f"mesh_{mesh_idx:03d}.ply"))

        # Save raw sequence for analysis
        np.savez(
            out / f"mesh_{mesh_idx:03d}_sequence.npz",
            sequence=seq_np,
            n_patches=len(patch_params),
        )

    print(f"Generated {args.n_meshes} meshes to {out}")


if __name__ == "__main__":
    main()

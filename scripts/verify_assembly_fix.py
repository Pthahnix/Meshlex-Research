"""Verify the PCA inverse rotation fix for VQ-VAE patch assembly.

Loads real patches, encodes through VQ-VAE, decodes, and compares:
  - Buggy: world_verts = local_verts * scale + centroid (no rotation)
  - Fixed: world_verts = (local_verts * scale) @ Vt + centroid

Outputs before/after comparison visualizations and Chamfer Distance.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/verify_assembly_fix.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset, compute_face_features, build_face_edge_index
from src.losses import chamfer_distance


def load_patch_data(npz_path):
    """Load a single patch NPZ and return all fields."""
    data = np.load(str(npz_path))
    return {
        "faces": data["faces"],
        "vertices": data["vertices"],  # original world-space
        "local_vertices": data["local_vertices"],  # PCA-aligned, unit-scaled
        "centroid": data["centroid"],
        "principal_axes": data["principal_axes"],  # Vt from SVD
        "scale": float(data["scale"][0]) if data["scale"].ndim > 0 else float(data["scale"]),
        "boundary_vertices": data["boundary_vertices"],
    }


def decode_patch(local_verts_decoded, scale, centroid, pca_axes, use_pca_inverse=True):
    """Apply inverse normalization to go from local to world space."""
    aligned = local_verts_decoded * scale
    if use_pca_inverse:
        centered = aligned @ pca_axes  # Vt: (3,3)
    else:
        centered = aligned
    world = centered + centroid
    return world


def compute_cd(pts_a, pts_b):
    """Compute Chamfer Distance between two point sets."""
    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_a2b, _ = tree_a.query(pts_b)
    d_b2a, _ = tree_b.query(pts_a)
    return float(np.mean(d_a2b ** 2) + np.mean(d_b2a ** 2))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("results/assembly_fix")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VQ-VAE
    ckpt_path = "data/checkpoints/rvq_lvis/checkpoint_final.pt"
    print(f"Loading VQ-VAE from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MeshLexRVQVAE()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Find patches from a few meshes
    patch_dir = Path("data/patches/lvis_wide/seen_train")
    npz_files = sorted(patch_dir.glob("*.npz"))

    # Group by mesh
    mesh_groups = {}
    for f in npz_files:
        stem = f.stem
        if "_patch_" in stem:
            mesh_id = stem.rsplit("_patch_", 1)[0]
        else:
            mesh_id = stem
        mesh_groups.setdefault(mesh_id, []).append(f)

    # Pick 5 meshes for testing
    mesh_ids = list(mesh_groups.keys())[:5]
    print(f"Testing {len(mesh_ids)} meshes: {mesh_ids}")

    all_cd_buggy = []
    all_cd_fixed = []

    for mesh_idx, mesh_id in enumerate(mesh_ids):
        patch_files = sorted(mesh_groups[mesh_id])
        print(f"\n{'='*50}")
        print(f"Mesh {mesh_idx+1}: {mesh_id} ({len(patch_files)} patches)")

        original_world_pts = []
        buggy_world_pts = []
        fixed_world_pts = []

        for pf in patch_files:
            patch = load_patch_data(pf)

            # Encode through VQ-VAE
            face_feats = compute_face_features(patch["local_vertices"], patch["faces"])
            edge_index = build_face_edge_index(patch["faces"])

            x = torch.tensor(face_feats, dtype=torch.float32).to(device)
            ei = torch.tensor(edge_index, dtype=torch.long).to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            with torch.no_grad():
                z = model.encoder(x, ei, batch)
                z_q, indices = model.rvq(z)
                n_verts = torch.tensor([min(patch["local_vertices"].shape[0], 128)], device=device)
                decoded_local = model.decoder(z_q, n_verts)
                decoded_local = decoded_local[0, :n_verts[0].item()].cpu().numpy()

            # Buggy: no PCA inverse
            buggy = decode_patch(decoded_local, patch["scale"], patch["centroid"],
                                 patch["principal_axes"], use_pca_inverse=False)
            # Fixed: with PCA inverse
            fixed = decode_patch(decoded_local, patch["scale"], patch["centroid"],
                                 patch["principal_axes"], use_pca_inverse=True)

            original_world_pts.append(patch["vertices"])
            buggy_world_pts.append(buggy)
            fixed_world_pts.append(fixed)

        # Assemble full mesh point clouds
        orig_all = np.concatenate(original_world_pts, axis=0)
        buggy_all = np.concatenate(buggy_world_pts, axis=0)
        fixed_all = np.concatenate(fixed_world_pts, axis=0)

        # Compute CD
        cd_buggy = compute_cd(orig_all, buggy_all)
        cd_fixed = compute_cd(orig_all, fixed_all)
        improvement = (cd_buggy - cd_fixed) / cd_buggy * 100 if cd_buggy > 0 else 0

        all_cd_buggy.append(cd_buggy)
        all_cd_fixed.append(cd_fixed)

        print(f"  CD (buggy, no PCA inverse): {cd_buggy:.6f}")
        print(f"  CD (fixed, with PCA inverse): {cd_fixed:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        # Visualize: 3 columns (original, buggy, fixed) × 2 views
        fig = plt.figure(figsize=(18, 10))
        views = [(30, 45), (30, 135)]

        for vi, (elev, azim) in enumerate(views):
            for ci, (pts, label, color) in enumerate([
                (orig_all, f"Original ({len(orig_all)} pts)", "steelblue"),
                (buggy_all, f"Buggy (CD={cd_buggy:.4f})", "coral"),
                (fixed_all, f"Fixed (CD={cd_fixed:.4f})", "seagreen"),
            ]):
                ax = fig.add_subplot(2, 3, vi * 3 + ci + 1, projection='3d')
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5, c=color)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(label, fontsize=10)

        plt.suptitle(
            f"Assembly Fix — {mesh_id}\n"
            f"Buggy CD={cd_buggy:.6f} → Fixed CD={cd_fixed:.6f} "
            f"({improvement:.1f}% improvement)",
            fontsize=13)
        plt.tight_layout()
        plt.savefig(output_dir / f"compare_{mesh_idx:02d}_{mesh_id}.png", dpi=150)
        plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    avg_buggy = np.mean(all_cd_buggy)
    avg_fixed = np.mean(all_cd_fixed)
    avg_improvement = (avg_buggy - avg_fixed) / avg_buggy * 100 if avg_buggy > 0 else 0
    print(f"Average CD (buggy): {avg_buggy:.6f}")
    print(f"Average CD (fixed): {avg_fixed:.6f}")
    print(f"Average improvement: {avg_improvement:.1f}%")

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_pos = np.arange(len(mesh_ids))
    width = 0.35
    axes[0].bar(x_pos - width/2, all_cd_buggy, width, label='Buggy (no PCA inv)', color='coral')
    axes[0].bar(x_pos + width/2, all_cd_fixed, width, label='Fixed (PCA inv)', color='seagreen')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"M{i}" for i in range(len(mesh_ids))], fontsize=9)
    axes[0].set_ylabel("Chamfer Distance")
    axes[0].set_title("Per-Mesh CD Comparison")
    axes[0].legend()

    improvements = [(b - f) / b * 100 for b, f in zip(all_cd_buggy, all_cd_fixed)]
    axes[1].bar(x_pos, improvements, color='steelblue')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"M{i}" for i in range(len(mesh_ids))], fontsize=9)
    axes[1].set_ylabel("Improvement (%)")
    axes[1].set_title(f"CD Improvement (avg: {avg_improvement:.1f}%)")
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle("Assembly Fix: PCA Inverse Rotation", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_comparison.png", dpi=150)
    plt.close()

    # Save results
    import json
    results = {
        "n_meshes": len(mesh_ids),
        "mesh_ids": mesh_ids,
        "cd_buggy": all_cd_buggy,
        "cd_fixed": all_cd_fixed,
        "improvements_pct": improvements,
        "avg_cd_buggy": float(avg_buggy),
        "avg_cd_fixed": float(avg_fixed),
        "avg_improvement_pct": float(avg_improvement),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write report
    with open(output_dir / "REPORT.md", "w") as f:
        f.write("# Assembly Fix — PCA Inverse Rotation Verification\n\n")
        f.write(f"**Date**: 2026-03-21\n\n")
        f.write("## Bug Description\n\n")
        f.write("VQ-VAE reconstruction was missing the PCA inverse rotation step.\n")
        f.write("The normalization does: `centered → PCA rotate → scale` (forward).\n")
        f.write("The inverse should be: `scale → PCA inverse rotate → translate` (backward).\n")
        f.write("But the buggy code only did: `scale → translate` (missing rotation).\n\n")
        f.write("## Fix\n\n")
        f.write("Added `aligned @ Vt` step where `Vt` = `principal_axes` from SVD.\n")
        f.write("Modified:\n")
        f.write("- `scripts/encode_sequences.py` — now saves `principal_axes` in sequence NPZ\n")
        f.write("- `scripts/visualize_mesh_comparison.py` — applies PCA inverse in decode\n\n")
        f.write("## Results\n\n")
        f.write(f"| Mesh | CD (Buggy) | CD (Fixed) | Improvement |\n")
        f.write(f"|------|-----------|-----------|-------------|\n")
        for i, mid in enumerate(mesh_ids):
            f.write(f"| {mid[:12]}... | {all_cd_buggy[i]:.6f} | {all_cd_fixed[i]:.6f} | {improvements[i]:.1f}% |\n")
        f.write(f"| **Average** | **{avg_buggy:.6f}** | **{avg_fixed:.6f}** | **{avg_improvement:.1f}%** |\n\n")
        f.write("## Visualizations\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write("| `compare_XX_*.png` | Original vs Buggy vs Fixed, 2 views per mesh |\n")
        f.write("| `summary_comparison.png` | Bar chart of CD comparison + improvement |\n")
        f.write("| `results.json` | Raw numerical results |\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

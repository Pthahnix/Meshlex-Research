"""MeshLex v2 — Mesh Reconstruction & Comparison Visualization.

Two modes:
  1. Reconstruction: original mesh → encode → decode → surface recon → side-by-side
  2. Generation: AR-generated tokens → decode → surface recon → render

Uses Open3D Ball Pivoting for surface reconstruction from decoded point clouds.

Usage:
    python scripts/visualize_mesh_comparison.py \
        --vqvae_checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --ar_checkpoint data/checkpoints/ar_v2/checkpoint_final.pt \
        --mesh_dir data/meshes/lvis_wide \
        --seq_dir data/sequences/rvq_lvis \
        --output_dir results/mesh_comparison \
        --n_recon 5 --n_gen 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ar_model import PatchGPT
from src.model_rvq import MeshLexRVQVAE
from src.patch_sequence import compute_vocab_size


# ── Surface reconstruction ──────────────────────────────────────────

def pointcloud_to_mesh_bpa(points, normals=None):
    """Reconstruct a triangle mesh from a point cloud using Ball Pivoting."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Ball pivoting with multiple radii
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * f for f in [0.5, 1.0, 1.5, 2.0, 3.0]]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))

    if len(mesh.triangles) == 0:
        # Fallback: Poisson reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return verts, faces


def pointcloud_to_mesh_alpha(points, alpha=0.05):
    """Reconstruct mesh using Alpha Shapes (simpler, more robust)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return verts, faces


# ── Rendering ────────────────────────────────────────────────────────

def render_mesh_matplotlib(verts, faces, ax, color='steelblue', alpha=0.6, title=''):
    """Render a triangle mesh on a matplotlib 3D axis."""
    if len(faces) > 0:
        triangles = verts[faces]
        poly = Poly3DCollection(triangles, alpha=alpha, linewidths=0.1,
                                edgecolors='gray', facecolors=color)
        ax.add_collection3d(poly)

    # Set axis limits from vertices
    if len(verts) > 0:
        center = verts.mean(axis=0)
        extent = max(verts.max(axis=0) - verts.min(axis=0)) / 2 * 1.2
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)


def render_pointcloud_matplotlib(points, ax, color='coral', s=1, title=''):
    """Render a point cloud on a matplotlib 3D axis."""
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=s, alpha=0.5)
    if len(points) > 0:
        center = points.mean(axis=0)
        extent = max(points.max(axis=0) - points.min(axis=0)) / 2 * 1.2
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)


# ── Decode helpers ───────────────────────────────────────────────────

def decode_sequence_to_patches(sequence, vqvae, device, n_pos_bins=256, n_scale_bins=64):
    """Decode a token sequence into world-space vertices per patch."""
    tokens_per_patch = 7
    offset_y = n_pos_bins
    offset_z = 2 * n_pos_bins
    offset_scale = 3 * n_pos_bins
    offset_code = 3 * n_pos_bins + n_scale_bins

    n_patches = len(sequence) // tokens_per_patch
    all_world_verts = []
    patch_infos = []

    for i in range(n_patches):
        base = i * tokens_per_patch
        pos_x = int(sequence[base + 0]) / 255.0
        pos_y = (int(sequence[base + 1]) - offset_y) / 255.0
        pos_z = (int(sequence[base + 2]) - offset_z) / 255.0
        scale_tok = int(sequence[base + 3]) - offset_scale
        scale = max(scale_tok / 63.0, 0.01)
        tok1 = int(sequence[base + 4]) - offset_code
        tok2 = int(sequence[base + 5]) - offset_code
        tok3 = int(sequence[base + 6]) - offset_code

        with torch.no_grad():
            tok_indices = torch.tensor([[tok1, tok2, tok3]], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok_indices)
            n_verts = torch.tensor([30], device=device)
            local_verts = vqvae.decoder(z_hat, n_verts)[0, :30].cpu().numpy()

        pos = np.array([pos_x, pos_y, pos_z])
        world_verts = local_verts * scale + pos
        all_world_verts.append(world_verts)
        patch_infos.append({"pos": pos, "scale": scale, "tokens": [tok1, tok2, tok3]})

    return all_world_verts, patch_infos


def decode_training_sequence(seq_path, vqvae, device):
    """Decode a training sequence NPZ (centroids, scales, tokens, principal_axes) to world-space vertices."""
    data = np.load(seq_path)
    centroids = data["centroids"]  # (N, 3)
    scales = data["scales"]        # (N,)
    tokens = data["tokens"]        # (N, 3)
    # Load PCA axes if available (for correct inverse rotation)
    if "principal_axes" in data:
        pca_axes = data["principal_axes"]  # (N, 3, 3) — Vt from SVD
    else:
        pca_axes = None

    all_world_verts = []
    for i in range(len(centroids)):
        with torch.no_grad():
            tok_indices = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok_indices)
            n_verts = torch.tensor([30], device=device)
            local_verts = vqvae.decoder(z_hat, n_verts)[0, :30].cpu().numpy()

        # Inverse PCA transform: scale → rotate → translate
        aligned = local_verts * max(scales[i], 0.01)
        if pca_axes is not None:
            # local_verts = centered @ Vt.T / scale, so centered = aligned @ Vt
            centered = aligned @ pca_axes[i]
        else:
            centered = aligned
        world_verts = centered + centroids[i]
        all_world_verts.append(world_verts)

    return all_world_verts


# ── Main comparison functions ────────────────────────────────────────

def do_reconstruction_comparison(mesh_dir, seq_dir, vqvae, device, output_dir, n_samples=5):
    """Compare original meshes with their VQ-VAE reconstructions."""
    import trimesh

    mesh_dir = Path(mesh_dir)
    seq_dir = Path(seq_dir)
    out = Path(output_dir) / "reconstruction"
    out.mkdir(parents=True, exist_ok=True)

    # Find matching pairs: mesh_id → (mesh_path, seq_path)
    seq_files = sorted(seq_dir.glob("*_sequence.npz"))
    obj_files = {}
    for obj in mesh_dir.rglob("*.obj"):
        obj_files[obj.stem] = obj

    pairs = []
    for sf in seq_files:
        mesh_id = sf.stem.replace("_sequence", "")
        if mesh_id in obj_files:
            pairs.append((obj_files[mesh_id], sf, mesh_id))
    pairs = pairs[:n_samples]

    print(f"\nReconstruction comparison: {len(pairs)} pairs found")

    for idx, (obj_path, seq_path, mesh_id) in enumerate(pairs):
        print(f"  [{idx+1}/{len(pairs)}] {mesh_id}")

        # Load original mesh
        orig_mesh = trimesh.load(str(obj_path), force='mesh')
        orig_verts = np.array(orig_mesh.vertices)
        orig_faces = np.array(orig_mesh.faces)

        # Decode through VQ-VAE
        all_world_verts = decode_training_sequence(seq_path, vqvae, device)
        recon_points = np.concatenate(all_world_verts, axis=0)

        # Surface reconstruction
        try:
            recon_verts, recon_faces = pointcloud_to_mesh_bpa(recon_points)
        except Exception as e:
            print(f"    BPA failed ({e}), trying alpha shapes...")
            try:
                bbox = recon_points.max(axis=0) - recon_points.min(axis=0)
                alpha = max(bbox) * 0.15
                recon_verts, recon_faces = pointcloud_to_mesh_alpha(recon_points, alpha=alpha)
            except Exception as e2:
                print(f"    Alpha shapes also failed ({e2}), skipping mesh recon")
                recon_verts, recon_faces = recon_points, np.zeros((0, 3), dtype=int)

        # Create comparison figure: 4 views
        # Row 1: Original mesh (2 views)
        # Row 2: Reconstructed mesh (2 views)
        fig = plt.figure(figsize=(20, 16))

        views = [(30, 45), (30, 135)]

        for vi, (elev, azim) in enumerate(views):
            # Original mesh
            ax = fig.add_subplot(2, 2, vi + 1, projection='3d')
            render_mesh_matplotlib(orig_verts, orig_faces, ax,
                                   color='steelblue', alpha=0.5,
                                   title=f'Original ({len(orig_faces)} faces)')
            ax.view_init(elev=elev, azim=azim)

            # Reconstructed mesh
            ax = fig.add_subplot(2, 2, vi + 3, projection='3d')
            if len(recon_faces) > 0:
                render_mesh_matplotlib(recon_verts, recon_faces, ax,
                                       color='coral', alpha=0.5,
                                       title=f'Reconstructed ({len(recon_faces)} faces, '
                                             f'{len(all_world_verts)} patches)')
            else:
                render_pointcloud_matplotlib(recon_points, ax, color='coral', s=2,
                                             title=f'Reconstructed (point cloud, '
                                                   f'{len(all_world_verts)} patches)')
            ax.view_init(elev=elev, azim=azim)

        plt.suptitle(f'Reconstruction Comparison — {mesh_id}\n'
                     f'Original: {len(orig_verts)}V/{len(orig_faces)}F → '
                     f'Encoded: {len(all_world_verts)} patches → '
                     f'Decoded: {len(recon_points)} points',
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(out / f"recon_{idx:03d}_{mesh_id}.png", dpi=150)
        plt.close()

        # Save stats
        stats = {
            "mesh_id": mesh_id,
            "original_verts": len(orig_verts),
            "original_faces": len(orig_faces),
            "n_patches": len(all_world_verts),
            "recon_points": len(recon_points),
            "recon_faces": len(recon_faces),
        }
        with open(out / f"recon_{idx:03d}_{mesh_id}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    print(f"  Saved to {out}")


def do_generation_comparison(ar_model, vqvae, device, output_dir,
                             n_meshes=5, temperatures=(0.8, 1.0)):
    """Generate meshes and render them with surface reconstruction."""
    out = Path(output_dir) / "generation"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nGeneration visualization: {n_meshes} meshes × {len(temperatures)} temps")

    for temp in temperatures:
        temp_dir = out / f"temp_{temp:.1f}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for mesh_idx in range(n_meshes):
            print(f"  T={temp}, mesh {mesh_idx+1}/{n_meshes}")

            # Generate
            with torch.no_grad():
                seq = ar_model.generate(max_len=910, temperature=temp, top_k=50)
            seq_np = seq[0].cpu().numpy()

            # Decode
            all_world_verts, patch_infos = decode_sequence_to_patches(
                seq_np, vqvae, device)
            if not all_world_verts:
                print("    No patches decoded, skipping")
                continue

            combined_points = np.concatenate(all_world_verts, axis=0)

            # Surface reconstruction
            try:
                recon_verts, recon_faces = pointcloud_to_mesh_bpa(combined_points)
            except Exception:
                try:
                    bbox = combined_points.max(axis=0) - combined_points.min(axis=0)
                    alpha = max(bbox) * 0.15
                    recon_verts, recon_faces = pointcloud_to_mesh_alpha(
                        combined_points, alpha=alpha)
                except Exception:
                    recon_verts, recon_faces = combined_points, np.zeros((0, 3), dtype=int)

            # Create figure: point cloud vs mesh, 3 views
            fig = plt.figure(figsize=(24, 8))
            views = [(30, 45), (30, 135), (0, 90)]

            for vi, (elev, azim) in enumerate(views):
                # Point cloud (colored by patch)
                ax = fig.add_subplot(2, 3, vi + 1, projection='3d')
                cmap = plt.colormaps.get_cmap('tab20')
                for pi, wv in enumerate(all_world_verts):
                    c = cmap(pi % 20)
                    ax.scatter(wv[:, 0], wv[:, 1], wv[:, 2],
                               c=[c], s=1, alpha=0.5)
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(f'Point Cloud ({len(combined_points)} pts)', fontsize=9)

                # Reconstructed mesh
                ax = fig.add_subplot(2, 3, vi + 4, projection='3d')
                if len(recon_faces) > 0:
                    render_mesh_matplotlib(recon_verts, recon_faces, ax,
                                           color='coral', alpha=0.5,
                                           title=f'Mesh ({len(recon_faces)} faces)')
                else:
                    render_pointcloud_matplotlib(combined_points, ax, s=2,
                                                 title='Mesh recon failed')
                ax.view_init(elev=elev, azim=azim)

            plt.suptitle(
                f'Generated Mesh {mesh_idx:03d} (T={temp}) — '
                f'{len(all_world_verts)} patches, {len(combined_points)} points → '
                f'{len(recon_faces)} faces',
                fontsize=12)
            plt.tight_layout()
            plt.savefig(temp_dir / f"gen_{mesh_idx:03d}.png", dpi=150)
            plt.close()

            # Save mesh as OBJ
            if len(recon_faces) > 0:
                save_obj(recon_verts, recon_faces,
                         temp_dir / f"gen_{mesh_idx:03d}.obj")

            # Save PLY point cloud
            try:
                import trimesh
                pc = trimesh.PointCloud(combined_points)
                pc.export(str(temp_dir / f"gen_{mesh_idx:03d}_pointcloud.ply"))
            except Exception:
                pass

    print(f"  Saved to {out}")


def save_obj(verts, faces, path):
    """Save mesh as OBJ file."""
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_checkpoint", required=True)
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--mesh_dir", default="data/meshes/lvis_wide")
    parser.add_argument("--seq_dir", default="data/sequences/rvq_lvis")
    parser.add_argument("--output_dir", default="results/mesh_comparison")
    parser.add_argument("--n_recon", type=int, default=5)
    parser.add_argument("--n_gen", type=int, default=5)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MeshLex v2 — Mesh Comparison Visualization")
    print("=" * 60)
    print(f"Device: {device}")

    # Load VQ-VAE
    print("\nLoading RVQ VQ-VAE...")
    vq_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    vqvae = MeshLexRVQVAE().to(device)
    vqvae.load_state_dict(vq_ckpt["model_state_dict"], strict=False)
    vqvae.eval()

    # Load AR model
    print("Loading AR v2 model...")
    ar_ckpt = torch.load(args.ar_checkpoint, map_location=device, weights_only=False)
    ar_config = ar_ckpt.get("config", {})
    ar_model = PatchGPT(**ar_config).to(device)
    ar_model.load_state_dict(ar_ckpt["model_state_dict"])
    ar_model.eval()
    print(f"  AR: {sum(p.numel() for p in ar_model.parameters())/1e6:.1f}M params")

    # 1. Reconstruction comparison
    do_reconstruction_comparison(
        args.mesh_dir, args.seq_dir, vqvae, device,
        args.output_dir, n_samples=args.n_recon)

    # 2. Generation visualization
    do_generation_comparison(
        ar_model, vqvae, device, args.output_dir,
        n_meshes=args.n_gen, temperatures=args.temperatures)

    print("\nDone!")


if __name__ == "__main__":
    main()

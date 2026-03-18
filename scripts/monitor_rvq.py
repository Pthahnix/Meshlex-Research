"""Monitor RVQ training: generate progress reports + mesh visualizations.

Reads training log, generates loss/utilization curves, reconstructs
sample meshes through the latest checkpoint, and saves everything
to results/rvq_training/.

Usage:
    python scripts/monitor_rvq.py \
        --log_file logs/train_rvq.log \
        --checkpoint_dir data/checkpoints/rvq_lvis \
        --patch_dir data/patches/lvis_wide/seen_test \
        --output_dir results/rvq_training \
        --n_samples 5
"""
import argparse
import json
import re
import torch
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset


def parse_training_log(log_file: str) -> list[dict]:
    """Parse epoch metrics from training log."""
    epochs = []
    pattern = re.compile(
        r"Epoch (\d+).*loss ([\d.]+).*recon ([\d.]+).*commit ([\d.]+).*embed ([\d.]+).*util ([\d.]+)%.*lr ([\d.e+-]+).*?([\d.]+)s"
    )
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append({
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "recon_loss": float(m.group(3)),
                    "commit_loss": float(m.group(4)),
                    "embed_loss": float(m.group(5)),
                    "utilization": float(m.group(6)) / 100,
                    "lr": float(m.group(7)),
                    "time_sec": float(m.group(8)),
                })
    return epochs


def plot_training_curves(epochs: list[dict], output_path: str):
    """Plot loss and utilization curves."""
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ep = [e["epoch"] for e in epochs]

    # Total loss
    axes[0, 0].plot(ep, [e["loss"] for e in epochs], "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Component losses
    axes[0, 1].plot(ep, [e["recon_loss"] for e in epochs], "r-", label="Recon", linewidth=1.5)
    axes[0, 1].plot(ep, [e["commit_loss"] for e in epochs], "g-", label="Commit", linewidth=1.5)
    axes[0, 1].plot(ep, [e["embed_loss"] for e in epochs], "m-", label="Embed", linewidth=1.5)
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Component Losses")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Utilization
    axes[1, 0].plot(ep, [e["utilization"] * 100 for e in epochs], "g-", linewidth=1.5)
    axes[1, 0].axhline(90, color="r", linestyle="--", alpha=0.5, label="90% target")
    axes[1, 0].set_ylabel("Utilization (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_title("Codebook Utilization")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(ep, [e["lr"] for e in epochs], "k-", linewidth=1.5)
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].grid(True, alpha=0.3)

    latest = epochs[-1]
    fig.suptitle(
        f"RVQ-VAE Training — Epoch {latest['epoch']} | "
        f"Loss {latest['loss']:.4f} | Util {latest['utilization']:.1%}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def reconstruct_samples(
    checkpoint_path: str,
    patch_dir: str,
    output_dir: str,
    n_samples: int = 5,
    device: str = "cuda",
):
    """Load checkpoint, reconstruct sample patches, save OBJ + preview."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MeshLexRVQVAE()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    dataset = PatchGraphDataset(patch_dir)
    # Pick evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    samples = [dataset[i] for i in indices]
    loader = DataLoader(samples, batch_size=n_samples, shuffle=False)

    batch = next(iter(loader))
    batch = batch.to(device)

    with torch.no_grad():
        z = model.encoder(batch.x, batch.edge_index, batch.batch)
        z_q, tok_indices = model.rvq(z)
        recon = model.decoder(z_q, batch.n_vertices)

    # Save each sample as OBJ + comparison plot
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples),
                             subplot_kw={"projection": "3d"})
    if n_samples == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_samples):
        nv = batch.n_vertices[i].item()
        gt = batch.gt_vertices[i, :nv].cpu().numpy()
        pred = recon[i, :nv].detach().cpu().numpy()

        # Save OBJ
        gt_cloud = trimesh.PointCloud(gt)
        pred_cloud = trimesh.PointCloud(pred)
        gt_cloud.export(str(out / f"sample_{i:02d}_gt.ply"))
        pred_cloud.export(str(out / f"sample_{i:02d}_pred.ply"))

        # Plot GT
        axes[i, 0].scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=2, c="blue", alpha=0.6)
        axes[i, 0].set_title(f"GT ({nv} verts)")
        axes[i, 0].set_aspect("equal")

        # Plot Pred
        axes[i, 1].scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=2, c="red", alpha=0.6)
        cd = np.mean(np.min(
            np.sum((gt[:, None] - pred[None, :]) ** 2, axis=-1), axis=1
        ))
        axes[i, 1].set_title(f"Recon (CD={cd:.4f})")
        axes[i, 1].set_aspect("equal")

    plt.suptitle("GT vs Reconstructed Patches", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "gt_vs_recon.png", dpi=150)
    plt.close()

    return tok_indices.cpu().numpy()


def write_progress_report(
    epochs: list[dict],
    output_dir: str,
    checkpoint_path: str = None,
):
    """Write markdown progress report."""
    out = Path(output_dir)
    latest = epochs[-1] if epochs else {}
    now = datetime.now().strftime("%Y%m%d_%H%M")

    report = f"""# RVQ-VAE Training Progress — {now}

## Summary
- Epochs completed: {latest.get('epoch', 0) + 1}
- Latest loss: {latest.get('loss', 'N/A'):.4f}
- Recon loss: {latest.get('recon_loss', 'N/A'):.4f}
- Commit loss: {latest.get('commit_loss', 'N/A'):.4f}
- Embed loss: {latest.get('embed_loss', 'N/A'):.4f}
- Codebook utilization: {latest.get('utilization', 0):.1%}
- Learning rate: {latest.get('lr', 'N/A'):.2e}
- Time per epoch: {latest.get('time_sec', 0):.1f}s

## Training Curves
![Training Curves](training_curves.png)

## Reconstruction Samples
![GT vs Recon](gt_vs_recon.png)

## Epoch History (last 10)
| Epoch | Loss | Recon | Commit | Embed | Util | LR |
|-------|------|-------|--------|-------|------|----|
"""
    for e in epochs[-10:]:
        report += f"| {e['epoch']:3d} | {e['loss']:.4f} | {e['recon_loss']:.4f} | {e['commit_loss']:.4f} | {e['embed_loss']:.4f} | {e['utilization']:.1%} | {e['lr']:.2e} |\n"

    report_path = out / f"{now}_progress_epoch{latest.get('epoch', 0):03d}.md"
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="logs/train_rvq.log")
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/rvq_lvis")
    parser.add_argument("--patch_dir", default="data/patches/lvis_wide/seen_test")
    parser.add_argument("--output_dir", default="results/rvq_training")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parse log
    epochs = parse_training_log(args.log_file)
    if not epochs:
        print("No epochs found in log yet.")
        return

    print(f"Parsed {len(epochs)} epochs from log")

    # Plot curves
    plot_training_curves(epochs, str(out / "training_curves.png"))
    print("Training curves saved")

    # Find latest checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    if ckpts:
        latest_ckpt = str(ckpts[-1])
        print(f"Using checkpoint: {latest_ckpt}")

        # Reconstruct samples
        reconstruct_samples(
            latest_ckpt, args.patch_dir, args.output_dir,
            n_samples=args.n_samples,
        )
        print("Reconstruction samples saved")
    else:
        print("No checkpoint found yet, skipping reconstruction")
        latest_ckpt = None

    # Write report
    report_path = write_progress_report(epochs, args.output_dir, latest_ckpt)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()

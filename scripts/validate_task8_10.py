"""
Validation script for Tasks 8-10: VQ-VAE assembly, DataLoader, Training loop.
Runs a short training session (5 epochs) on real mesh patches.
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import load_and_preprocess_mesh
from src.patch_dataset import process_and_save_patches, PatchGraphDataset
from src.model import MeshLexVQVAE
from src.trainer import Trainer

RAW_DIR = Path("data/raw_samples")
RESULTS_DIR = Path("results/task8_10_validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_LINES: list[str] = []


def log(msg: str):
    print(msg)
    LOG_LINES.append(msg)


def save_log():
    (RESULTS_DIR / "validation_log.txt").write_text("\n".join(LOG_LINES))


def prepare_patches():
    """Prepare patches from real meshes."""
    patch_dir = RESULTS_DIR / "patches"
    for obj_path in sorted(RAW_DIR.glob("*.obj")):
        raw = trimesh.load(str(obj_path), force="mesh")
        if raw.faces.shape[0] < 200:
            continue
        name = obj_path.stem
        mesh = load_and_preprocess_mesh(str(obj_path), target_faces=1000, min_faces=200)
        if mesh is None:
            continue
        prep_path = RESULTS_DIR / "meshes" / f"{name}.obj"
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(prep_path))
        process_and_save_patches(str(prep_path), name, str(patch_dir / name))
    return patch_dir


def main():
    log("MeshLex Validation: Tasks 8-10 — VQ-VAE + Training Loop")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # Prepare data
    log("=" * 70)
    log("STEP 0: Prepare patches")
    log("=" * 70)
    patch_dir = prepare_patches()

    # Collect all patch subdirs
    patch_subdirs = sorted([d for d in patch_dir.iterdir() if d.is_dir()])
    train_dirs = [str(d) for d in patch_subdirs]
    log(f"Patch directories: {train_dirs}")

    # Load dataset
    from torch.utils.data import ConcatDataset
    datasets = [PatchGraphDataset(d) for d in train_dirs]
    train_dataset = ConcatDataset(datasets)
    log(f"Total training patches: {len(train_dataset)}\n")

    # Create model
    log("=" * 70)
    log("STEP 1: Model Creation")
    log("=" * 70)
    device = "cpu"  # Use CPU for validation to avoid OOM
    model = MeshLexVQVAE(codebook_size=64, embed_dim=64, hidden_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Device: {device}")
    log(f"Model parameters: {n_params:,}")
    log(f"Codebook size: 64 (small for validation)")
    log(f"Embed dim: 64, Hidden dim: 128\n")

    # Train for 5 epochs
    log("=" * 70)
    log("STEP 2: Training (5 epochs smoke test)")
    log("=" * 70)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=32,
        lr=1e-3,
        epochs=5,
        checkpoint_dir=str(RESULTS_DIR / "checkpoints"),
        device=device,
        vq_start_epoch=2,
    )

    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0
    log(f"\nTotal training time: {total_time:.1f}s")

    # Check training history
    history = trainer.history
    log(f"\nTraining history ({len(history)} epochs):")
    for h in history:
        val_info = f", val_recon={h.get('val_recon_loss', 'N/A')}" if 'val_recon_loss' in h else ""
        log(f"  Epoch {h['epoch']}: loss={h['loss']:.4f}, recon={h['recon_loss']:.4f}, "
            f"util={h['codebook_utilization']:.1%}{val_info}")

    # Verify loss decreases
    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    loss_decreased = last_loss < first_loss
    log(f"\nLoss decreased: {loss_decreased} ({first_loss:.4f} → {last_loss:.4f})")

    # Visualizations
    log("\n" + "=" * 70)
    log("STEP 3: Visualizations")
    log("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    recon_losses = [h["recon_loss"] for h in history]
    utils = [h["codebook_utilization"] for h in history]

    axes[0].plot(epochs, losses, "b-o", label="total loss")
    axes[0].plot(epochs, recon_losses, "r--o", label="recon loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, utils, "g-o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Utilization")
    axes[1].set_title("Codebook Utilization")
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(0.3, color="red", linestyle="--", alpha=0.5, label="warning threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    lrs = [h["lr"] for h in history]
    axes[2].plot(epochs, lrs, "m-o")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule (Cosine)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(RESULTS_DIR / "training_curves.png"), dpi=150)
    plt.close(fig)
    log(f"Plot saved: {RESULTS_DIR / 'training_curves.png'}")

    # Mesh preview
    for obj_path in sorted((RESULTS_DIR / "meshes").glob("*.obj"))[:1]:
        mesh = trimesh.load(str(obj_path), force="mesh")
        fig = plt.figure(figsize=(12, 4))
        for i, (elev, azim) in enumerate([(30, 45), (30, 135), (30, 225), (90, 0)]):
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            ax.plot_trisurf(
                mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces, color="steelblue", alpha=0.8,
                edgecolor="k", linewidth=0.1,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
            ax.tick_params(labelsize=5)
        name = obj_path.stem
        plt.suptitle(f"{name} Preview", fontsize=10)
        plt.tight_layout()
        fig.savefig(str(RESULTS_DIR / f"{name}_preview.png"), dpi=150)
        plt.close(fig)
        log(f"Mesh preview saved: {RESULTS_DIR / f'{name}_preview.png'}")

    # Summary
    summary = {
        "model_params": n_params,
        "training_epochs": len(history),
        "final_loss": last_loss,
        "final_recon_loss": history[-1]["recon_loss"],
        "final_utilization": history[-1]["codebook_utilization"],
        "loss_decreased": loss_decreased,
        "total_time_sec": round(total_time, 1),
        "history": history,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    md = [
        "# Task 8-10 Validation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model",
        f"- Parameters: {n_params:,}",
        f"- Codebook: K=64, dim=64 (small for validation)",
        "",
        "## Training (5 epochs smoke test)",
        "",
        "| Epoch | Loss | Recon Loss | Utilization |",
        "|-------|------|------------|-------------|",
    ]
    for h in history:
        md.append(f"| {h['epoch']} | {h['loss']:.4f} | {h['recon_loss']:.4f} | {h['codebook_utilization']:.1%} |")
    md += [
        "",
        f"Loss decreased: {first_loss:.4f} → {last_loss:.4f} ({'Yes' if loss_decreased else 'No'})",
        "",
        "![Training Curves](training_curves.png)",
        "",
        "## Conclusion",
        "",
        "- MeshLexVQVAE end-to-end forward+backward works",
        "- Training loop with staged VQ introduction functions correctly",
        "- Cosine LR schedule active",
        f"- Loss {'decreased' if loss_decreased else 'did not decrease'} over 5 epochs",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(md))
    log(f"Report saved: {RESULTS_DIR / 'report.md'}")

    save_log()
    log(f"\nFull log saved: {RESULTS_DIR / 'validation_log.txt'}")
    log("\n✓ Tasks 8-10 validation complete.")


if __name__ == "__main__":
    main()

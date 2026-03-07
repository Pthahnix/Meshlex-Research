"""Codebook visualization: t-SNE, utilization histogram, training curves."""
import argparse
import json
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset


def plot_utilization_histogram(code_counts: Counter, K: int, save_path: str):
    """Plot histogram of code usage frequencies."""
    counts = np.zeros(K)
    for code_id, freq in code_counts.items():
        counts[code_id] = freq

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sorted frequency bar chart
    sorted_counts = np.sort(counts)[::-1]
    axes[0].bar(range(K), sorted_counts, width=1.0, color="steelblue")
    axes[0].set_xlabel("Code rank")
    axes[0].set_ylabel("Usage frequency")
    axes[0].set_title(f"Codebook Usage (K={K}, active={np.sum(counts > 0)}/{K})")

    # Right: cumulative coverage
    total = sorted_counts.sum()
    if total > 0:
        cumulative = np.cumsum(sorted_counts) / total
    else:
        cumulative = np.zeros(K)
    axes[1].plot(range(K), cumulative, color="coral")
    axes[1].axhline(y=0.9, color="gray", linestyle="--", label="90% coverage")
    idx_90 = np.searchsorted(cumulative, 0.9)
    axes[1].axvline(x=idx_90, color="gray", linestyle="--")
    axes[1].set_xlabel("Top-N codes")
    axes[1].set_ylabel("Cumulative coverage")
    axes[1].set_title(f"Top-{idx_90} codes cover 90% of patches")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_codebook_tsne(model, save_path: str):
    """t-SNE of codebook embeddings."""
    from sklearn.manifold import TSNE

    embeddings = model.codebook.codebook.weight.detach().cpu().numpy()  # (K, dim)
    tsne = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0] - 1), random_state=42)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.6, c="steelblue")
    plt.title(f"Codebook t-SNE (K={embeddings.shape[0]})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(history_path: str, save_path: str):
    """Plot training loss, recon loss, and utilization over epochs."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    recon = [h["recon_loss"] for h in history]
    util = [h["codebook_utilization"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, loss, label="Total loss")
    axes[0].plot(epochs, recon, label="Recon loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, util, color="coral")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", label="50% threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Utilization")
    axes[1].set_title("Codebook Utilization")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, recon, color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Recon CD")
    axes[2].set_title("Reconstruction Chamfer Distance")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history", type=str, default="data/checkpoints/training_history.json")
    parser.add_argument("--patch_dirs", nargs="+", default=None,
                        help="Patch dirs for utilization histogram")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MeshLexVQVAE(codebook_size=args.codebook_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # 1. Training curves
    if Path(args.history).exists():
        plot_training_curves(args.history, str(out / "training_curves.png"))

    # 2. Codebook t-SNE
    plot_codebook_tsne(model, str(out / "codebook_tsne.png"))

    # 3. Utilization histogram (if patch dirs provided)
    if args.patch_dirs:
        ds = ConcatDataset([PatchGraphDataset(d) for d in args.patch_dirs])
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        all_indices = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                result = model(
                    batch.x, batch.edge_index, batch.batch,
                    batch.n_vertices, batch.gt_vertices,
                )
                all_indices.append(result["indices"].cpu())

        all_idx = torch.cat(all_indices)
        code_counts = Counter(all_idx.numpy().tolist())
        plot_utilization_histogram(code_counts, model.codebook.K, str(out / "utilization_histogram.png"))

    print("All visualizations complete.")


if __name__ == "__main__":
    main()

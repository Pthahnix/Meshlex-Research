"""Initialize SimVQ codebook with K-means on encoder embeddings."""
import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pre-VQ checkpoint (from first 20 epochs encoder-only)")
    parser.add_argument("--patch_dirs", nargs="+", required=True)
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output", type=str, required=True,
                        help="Output checkpoint with initialized codebook")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MeshLexVQVAE(
        codebook_size=args.codebook_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Collect encoder embeddings
    ds = ConcatDataset([PatchGraphDataset(d) for d in args.patch_dirs])
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    embeddings = []
    print("Collecting encoder embeddings...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode_only(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu().numpy())

    all_embeddings = np.concatenate(embeddings, axis=0)
    print(f"Collected {all_embeddings.shape[0]} embeddings of dim {all_embeddings.shape[1]}")

    # K-means clustering
    n_samples = all_embeddings.shape[0]
    effective_k = min(args.codebook_size, n_samples)
    print(f"Running K-means (K={effective_k}, samples={n_samples})...")
    kmeans = MiniBatchKMeans(
        n_clusters=effective_k,
        batch_size=min(4096, n_samples),
        max_iter=100,
        random_state=42,
    )
    kmeans.fit(all_embeddings)
    centers = kmeans.cluster_centers_  # (effective_k, dim)
    print(f"K-means done. Inertia: {kmeans.inertia_:.4f}")

    # If fewer clusters than codebook, pad remaining with random perturbations
    if effective_k < args.codebook_size:
        print(f"Padding {args.codebook_size - effective_k} remaining codes with random perturbations")
        extra = args.codebook_size - effective_k
        noise = np.random.randn(extra, centers.shape[1]) * 0.01 + centers[np.random.randint(0, effective_k, extra)]
        centers = np.concatenate([centers, noise], axis=0)

    # Initialize codebook: set C = W^T(centroids) so CW ≈ centroids
    with torch.no_grad():
        centroids_tensor = torch.tensor(centers, dtype=torch.float32).to(device)
        model.codebook.init_from_z(centroids_tensor)

    # Verify alignment
    with torch.no_grad():
        cw = model.codebook.get_quant_codebook()
        alignment_error = (cw.cpu() - torch.tensor(centers, dtype=torch.float32)).norm(dim=1).mean()
        print(f"CW-centroid alignment error (L2): {alignment_error:.4f}")

    # Verify utilization with encoder outputs
    with torch.no_grad():
        sample_z = torch.tensor(all_embeddings[:1000], dtype=torch.float32).to(device)
        _, sample_idx = model.codebook(sample_z)
        util = sample_idx.unique().numel() / args.codebook_size
        print(f"Post-init utilization (sample 1000): {util:.1%}")

    # Save
    ckpt["model_state_dict"] = model.state_dict()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"Codebook-initialized checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()

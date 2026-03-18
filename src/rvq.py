"""Residual Vector Quantization (RVQ) built on SimVQCodebook levels."""
import torch
import torch.nn as nn

from src.model import SimVQCodebook


class ResidualVQ(nn.Module):
    """Multi-level residual vector quantization.

    Each level quantizes the residual from the previous level.
    All levels use SimVQCodebook with use_rotation=False.
    """

    def __init__(self, n_levels: int = 3, K: int = 1024, dim: int = 128):
        super().__init__()
        self.n_levels = n_levels
        self.K = K
        self.dim = dim
        self.levels = nn.ModuleList([
            SimVQCodebook(K=K, dim=dim, use_rotation=False)
            for _ in range(n_levels)
        ])

    def forward(self, z: torch.Tensor):
        """Iterative residual quantization.

        Args:
            z: (B, dim) encoder output

        Returns:
            z_hat: (B, dim) sum of quantized vectors across all levels
            indices: (B, n_levels) codebook indices per level
        """
        residual = z
        z_hat = torch.zeros_like(z)
        all_indices = []

        for level in self.levels:
            quantized_st, idx = level(residual)
            z_hat = z_hat + quantized_st
            # Next level quantizes the residual
            residual = residual - quantized_st
            all_indices.append(idx)

        indices = torch.stack(all_indices, dim=-1)  # (B, n_levels)
        return z_hat, indices

    def compute_loss(self, z: torch.Tensor, indices: torch.Tensor):
        """Compute commit + embed loss summed across all levels.

        Args:
            z: (B, dim) original encoder output
            indices: (B, n_levels) codebook indices

        Returns:
            commit_loss: scalar, sum of commit losses across levels
            embed_loss: scalar, sum of embed losses across levels
        """
        residual = z
        total_commit = torch.tensor(0.0, device=z.device)
        total_embed = torch.tensor(0.0, device=z.device)

        for i, level in enumerate(self.levels):
            idx = indices[:, i]
            quantized_st, _ = level(residual)

            # Compute losses for this level
            commit, embed = level.compute_loss(residual, quantized_st, idx)
            total_commit = total_commit + commit
            total_embed = total_embed + embed

            # Advance residual
            residual = residual - quantized_st

        return total_commit, total_embed

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct z_hat from indices alone.

        Args:
            indices: (B, n_levels) codebook indices

        Returns:
            z_hat: (B, dim) reconstructed embedding
        """
        B = indices.shape[0]
        z_hat = torch.zeros(B, self.dim, device=indices.device)

        for i, level in enumerate(self.levels):
            idx = indices[:, i]
            quant_codebook = level.linear(level.codebook.weight)  # (K, dim)
            z_hat = z_hat + quant_codebook[idx]

        return z_hat

    @torch.no_grad()
    def get_utilization(self, indices: torch.Tensor) -> list[float]:
        """Fraction of codes used per level.

        Args:
            indices: (B, n_levels) codebook indices

        Returns:
            List of utilization fractions, one per level.
        """
        utils = []
        for i, level in enumerate(self.levels):
            idx = indices[:, i]
            utils.append(idx.unique().numel() / self.K)
        return utils

    @torch.no_grad()
    def init_from_z(self, all_z: torch.Tensor):
        """K-means init per level on residuals.

        Level 0 is initialized from all_z directly.
        Subsequent levels are initialized from the residual after
        quantizing with the previous levels.

        Args:
            all_z: (N, dim) encoder outputs for initialization
        """
        from sklearn.cluster import MiniBatchKMeans

        residual = all_z.clone()

        for i, level in enumerate(self.levels):
            z_np = residual.numpy()
            n_samples = len(z_np)
            effective_k = min(self.K, n_samples)

            print(f"  RVQ level {i}: K-means {n_samples} samples → {effective_k} clusters...")
            kmeans = MiniBatchKMeans(
                n_clusters=effective_k,
                batch_size=min(4096, n_samples),
                max_iter=100,
                random_state=42 + i,
            )
            kmeans.fit(z_np)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

            # Pad if fewer clusters than K
            if effective_k < self.K:
                extra = self.K - effective_k
                pad_idx = torch.randint(0, effective_k, (extra,))
                noise = torch.randn(extra, centroids.shape[1]) * 0.01
                centroids = torch.cat([centroids, centroids[pad_idx] + noise])

            level.init_from_z(centroids)

            # Compute residual for next level using CW lookup
            quant_codebook = level.linear(level.codebook.weight)  # (K, dim)
            distances = torch.cdist(residual, quant_codebook)
            nearest = distances.argmin(dim=-1)
            quantized = quant_codebook[nearest]
            residual = residual - quantized

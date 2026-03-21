"""Residual Vector Quantization (RVQ) built on SimVQCodebook levels."""
import torch
import torch.nn as nn

from src.model import SimVQCodebook


class VanillaVQ(nn.Module):
    """Standard VQ with straight-through gradient estimator."""

    def __init__(self, K: int = 1024, dim: int = 128):
        super().__init__()
        self.K = K
        self.dim = dim
        self.codebook = nn.Embedding(K, dim)
        nn.init.uniform_(self.codebook.weight, -1 / K, 1 / K)
        # Identity linear for compatibility with SimVQCodebook interface
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.linear.weight)
        self.linear.weight.requires_grad = False  # not used for VanillaVQ

    def forward(self, z):
        dists = torch.cdist(z.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)
        z_q = z + (z_q - z).detach()  # straight-through
        return z_q, indices

    def compute_loss(self, z, quantized_st, indices):
        quantized = self.codebook(indices)
        commit_loss = ((z - quantized.detach()) ** 2).mean()
        embed_loss = ((z.detach() - quantized) ** 2).mean()
        return commit_loss, embed_loss

    def get_quant_codebook(self):
        return self.codebook.weight

    @torch.no_grad()
    def init_from_z(self, centroids):
        self.codebook.weight.data.copy_(centroids)


class EMAVQ(nn.Module):
    """EMA-updated VQ codebook with straight-through gradient."""

    def __init__(self, K: int = 1024, dim: int = 128, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K = K
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.codebook = nn.Embedding(K, dim)
        nn.init.uniform_(self.codebook.weight, -1 / K, 1 / K)
        # Identity linear for compatibility with SimVQCodebook interface
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.linear.weight)
        self.linear.weight.requires_grad = False  # not used for EMAVQ
        self.register_buffer('ema_count', torch.zeros(K))
        self.register_buffer('ema_weight', self.codebook.weight.clone())

    def forward(self, z):
        dists = torch.cdist(z.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)

        if self.training:
            one_hot = torch.zeros(z.size(0), self.K, device=z.device)
            one_hot.scatter_(1, indices.unsqueeze(1), 1)
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * one_hot.sum(0)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * (one_hot.T @ z)
            n = self.ema_count.sum()
            count = (self.ema_count + self.eps) / (n + self.K * self.eps) * n
            self.codebook.weight.data = self.ema_weight / count.unsqueeze(1)

        z_q = z + (z_q - z).detach()  # straight-through
        return z_q, indices

    def compute_loss(self, z, quantized_st, indices):
        quantized = self.codebook(indices)
        commit_loss = ((z - quantized.detach()) ** 2).mean()
        embed_loss = ((z.detach() - quantized) ** 2).mean()
        return commit_loss, embed_loss

    def get_quant_codebook(self):
        return self.codebook.weight

    @torch.no_grad()
    def init_from_z(self, centroids):
        self.codebook.weight.data.copy_(centroids)
        self.ema_weight.copy_(centroids)


class ResidualVQ(nn.Module):
    """Multi-level residual vector quantization.

    Each level quantizes the residual from the previous level.
    Supports SimVQ (default), VanillaVQ, and EMAVQ codebook methods.
    """

    def __init__(self, n_levels: int = 3, K: int = 1024, dim: int = 128, vq_method: str = "simvq"):
        super().__init__()
        self.n_levels = n_levels
        self.K = K
        self.dim = dim
        self.vq_method = vq_method
        levels = []
        for _ in range(n_levels):
            if vq_method == "vanilla":
                levels.append(VanillaVQ(K=K, dim=dim))
            elif vq_method == "ema":
                levels.append(EMAVQ(K=K, dim=dim))
            else:
                levels.append(SimVQCodebook(K=K, dim=dim, use_rotation=False))
        self.levels = nn.ModuleList(levels)

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

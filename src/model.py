"""MeshLex model components: GNN Encoder, SimVQ Codebook, Patch Decoder."""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class PatchEncoder(nn.Module):
    """4-layer SAGEConv encoder: face features → patch embedding.

    Input: per-face features (F_total, 15) across all patches in batch
    Output: per-patch embedding (B, out_dim)
    """

    def __init__(self, in_dim: int = 15, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(out_dim)

        self.act = nn.GELU()

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: (N_total, in_dim) face features for all patches in batch
            edge_index: (2, E_total) face adjacency edges
            batch: (N_total,) batch assignment vector
        Returns:
            (B, out_dim) patch embeddings
        """
        x = self.act(self.norm1(self.conv1(x, edge_index)))
        x = self.act(self.norm2(self.conv2(x, edge_index)))
        x = self.act(self.norm3(self.conv3(x, edge_index)))
        x = self.act(self.norm4(self.conv4(x, edge_index)))

        # Global mean pooling per patch
        return global_mean_pool(x, batch)  # (B, out_dim)


class SimVQCodebook(nn.Module):
    """SimVQ codebook with frozen C and learnable linear W.

    Reference: SimVQ (ICCV 2025) — linear transform prevents codebook collapse.
    Official: https://github.com/youngsheen/SimVQ

    Key design (aligned with official):
    - C (codebook embedding) is FROZEN — never updated by gradient
    - W (linear layer) is the ONLY learnable parameter
    - Distance: z to CW (not z_proj to C)
    - Quantized: taken from CW (not from C)
    """

    def __init__(self, K: int = 4096, dim: int = 128):
        super().__init__()
        self.K = K
        self.dim = dim
        self.codebook = nn.Embedding(K, dim)
        self.linear = nn.Linear(dim, dim, bias=False)

        # Official SimVQ initialization
        nn.init.normal_(self.codebook.weight, mean=0, std=dim ** -0.5)
        nn.init.orthogonal_(self.linear.weight)

        # Freeze C — only W is learnable (SimVQ paper Remark 1)
        self.codebook.weight.requires_grad = False

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, dim) encoder output
        Returns:
            quantized_st: (B, dim) quantized embedding (straight-through)
            indices: (B,) codebook indices
        """
        # Compute CW — transformed codebook
        quant_codebook = self.linear(self.codebook.weight)  # (K, dim)

        # Distance: z to CW
        distances = torch.cdist(
            z.unsqueeze(0), quant_codebook.unsqueeze(0),
        ).squeeze(0)  # (B, K)

        indices = distances.argmin(dim=-1)  # (B,)

        # Quantized from CW (not from C)
        quantized = quant_codebook[indices]  # (B, dim)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        return quantized_st, indices

    def compute_loss(self, z: torch.Tensor, quantized_st: torch.Tensor, indices: torch.Tensor):
        """Compute commitment + embedding losses in CW space.

        Returns:
            commit_loss: ||z - sg(CW[idx])||²
            embed_loss: ||sg(z) - CW[idx]||²
        """
        quant_codebook = self.linear(self.codebook.weight)
        quantized = quant_codebook[indices]
        commit_loss = torch.mean((z - quantized.detach()) ** 2)
        embed_loss = torch.mean((z.detach() - quantized) ** 2)
        return commit_loss, embed_loss

    @torch.no_grad()
    def get_utilization(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in given indices."""
        return indices.unique().numel() / self.K

    @torch.no_grad()
    def get_quant_codebook(self) -> torch.Tensor:
        """Return CW — the effective codebook in encoder output space."""
        return self.linear(self.codebook.weight)


class PatchDecoder(nn.Module):
    """MLP decoder: codebook embedding → per-vertex coordinates.

    Uses learnable vertex query positions + cross-attention from codebook embedding,
    with masking for variable vertex counts.
    """

    def __init__(self, embed_dim: int = 128, max_vertices: int = 128):
        super().__init__()
        self.max_vertices = max_vertices

        # Learnable positional queries for each vertex slot
        self.vertex_queries = nn.Parameter(torch.randn(max_vertices, embed_dim) * 0.02)

        # Cross-attention: vertex queries attend to patch embedding
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # MLP to decode xyz
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, z: torch.Tensor, n_vertices: torch.Tensor):
        """
        Args:
            z: (B, embed_dim) codebook embedding
            n_vertices: (B,) actual vertex count per patch
        Returns:
            (B, max_vertices, 3) predicted vertex coordinates (masked beyond n_vertices)
        """
        B = z.shape[0]

        # Expand queries for batch: (B, max_V, D)
        queries = self.vertex_queries.unsqueeze(0).expand(B, -1, -1)

        # Key/Value = patch embedding repeated: (B, 1, D)
        kv = z.unsqueeze(1)

        # Cross-attention
        attn_out, _ = self.cross_attn(queries, kv, kv)
        attn_out = self.norm(attn_out + queries)  # residual

        # Decode to xyz
        coords = self.mlp(attn_out)  # (B, max_V, 3)

        # Mask: zero out positions beyond actual vertex count
        mask = torch.arange(self.max_vertices, device=z.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        coords = coords * mask.unsqueeze(-1).float()

        return coords


class MeshLexVQVAE(nn.Module):
    """Full MeshLex VQ-VAE: Encoder → SimVQ → Decoder.

    Combines PatchEncoder, SimVQCodebook, PatchDecoder into end-to-end model.
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        codebook_size: int = 4096,
        max_vertices: int = 128,
        lambda_commit: float = 1.0,
        lambda_embed: float = 1.0,
    ):
        super().__init__()
        self.encoder = PatchEncoder(in_dim, hidden_dim, embed_dim)
        self.codebook = SimVQCodebook(codebook_size, embed_dim)
        self.decoder = PatchDecoder(embed_dim, max_vertices)
        self.max_vertices = max_vertices
        self.lambda_commit = lambda_commit
        self.lambda_embed = lambda_embed

    def forward(self, x, edge_index, batch, n_vertices, gt_vertices):
        """
        Args:
            x: (N_total, in_dim) face features
            edge_index: (2, E_total) face adjacency
            batch: (N_total,) batch vector
            n_vertices: (B,) actual vertex count per patch
            gt_vertices: (B, max_V, 3) ground truth local vertices (padded)
        Returns:
            dict with recon_vertices, total_loss, recon_loss, commit_loss, embed_loss, indices
        """
        from src.losses import chamfer_distance

        # Encode
        z = self.encoder(x, edge_index, batch)  # (B, embed_dim)

        # Quantize
        z_q, indices = self.codebook(z)  # (B, embed_dim), (B,)

        # Decode
        recon = self.decoder(z_q, n_vertices)  # (B, max_V, 3)

        # Losses
        mask = torch.arange(self.max_vertices, device=x.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        recon_loss = chamfer_distance(recon, gt_vertices, mask)
        commit_loss, embed_loss = self.codebook.compute_loss(z, z_q, indices)

        total_loss = recon_loss + self.lambda_commit * commit_loss + self.lambda_embed * embed_loss

        return {
            "recon_vertices": recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "embed_loss": embed_loss,
            "indices": indices,
            "z": z,
        }

    def encode_only(self, x, edge_index, batch):
        """Encode patches without decoding (for codebook init and eval)."""
        return self.encoder(x, edge_index, batch)

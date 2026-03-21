"""MeshLex VQ-VAE with RVQ quantizer.

Same encoder and decoder as v1, but replaces SimVQ with 3-level RVQ.
"""
import torch
import torch.nn as nn

from src.model import PatchEncoder, PatchDecoder
from src.rvq import ResidualVQ
from src.losses import chamfer_distance


class MeshLexRVQVAE(nn.Module):
    """MeshLex VQ-VAE with Residual Vector Quantization.

    Encoder → RVQ (3-level) → Decoder.
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        codebook_size: int = 1024,
        n_levels: int = 3,
        max_vertices: int = 128,
        lambda_commit: float = 1.0,
        lambda_embed: float = 1.0,
        num_kv_tokens: int = 4,
        vq_method: str = "simvq",
    ):
        super().__init__()
        self.encoder = PatchEncoder(in_dim, hidden_dim, embed_dim)
        self.rvq = ResidualVQ(n_levels=n_levels, K=codebook_size, dim=embed_dim, vq_method=vq_method)
        self.decoder = PatchDecoder(embed_dim, max_vertices, num_kv_tokens=num_kv_tokens)
        # Expose first-level codebook for Trainer compatibility
        # (Trainer accesses model.codebook.K, .init_from_z, .get_quant_codebook, etc.)
        self.codebook = self.rvq.levels[0]
        self.max_vertices = max_vertices
        self.lambda_commit = lambda_commit
        self.lambda_embed = lambda_embed

    def forward(self, x, edge_index, batch, n_vertices, gt_vertices):
        z = self.encoder(x, edge_index, batch)
        z_q, indices = self.rvq(z)
        recon = self.decoder(z_q, n_vertices)

        mask = torch.arange(self.max_vertices, device=x.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        recon_loss = chamfer_distance(recon, gt_vertices, mask)
        commit_loss, embed_loss = self.rvq.compute_loss(z, indices)

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
        return self.encoder(x, edge_index, batch)


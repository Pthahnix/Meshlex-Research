import torch
from torch_geometric.data import Data, Batch
from src.model_rvq import MeshLexRVQVAE


def test_rvqvae_forward():
    """Full forward: graph → RVQ quantize → reconstruct → losses."""
    max_verts = 60
    model = MeshLexRVQVAE(codebook_size=64, n_levels=3, embed_dim=128, max_vertices=max_verts)

    graphs = []
    for _ in range(4):
        nf = 30
        x = torch.randn(nf, 15)
        ei = torch.stack([torch.randint(0, nf, (60,)), torch.randint(0, nf, (60,))])
        graphs.append(Data(x=x, edge_index=ei))

    batch = Batch.from_data_list(graphs)
    n_vertices = torch.tensor([20, 25, 18, 30])
    gt_vertices = torch.randn(4, max_verts, 3)

    result = model(batch.x, batch.edge_index, batch.batch, n_vertices, gt_vertices)

    assert result["recon_vertices"].shape == (4, max_verts, 3)
    assert result["indices"].shape == (4, 3)  # 3 RVQ levels
    assert result["total_loss"].requires_grad


def test_rvqvae_encode_only():
    """encode_only should return encoder embeddings."""
    model = MeshLexRVQVAE(codebook_size=64, n_levels=3, embed_dim=128)
    nf = 20
    x = torch.randn(nf, 15)
    ei = torch.stack([torch.randint(0, nf, (40,)), torch.randint(0, nf, (40,))])
    data = Data(x=x, edge_index=ei)
    batch = Batch.from_data_list([data])
    z = model.encode_only(batch.x, batch.edge_index, batch.batch)
    assert z.shape == (1, 128)

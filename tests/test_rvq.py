"""Tests for ResidualVQ module."""
import pytest
import torch

from src.rvq import ResidualVQ


def test_rvq_output_shape():
    """z_hat should be (8, 128), indices should be (8, 3)."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=128)
    z = torch.randn(8, 128)
    z_hat, indices = rvq(z)
    assert z_hat.shape == (8, 128)
    assert indices.shape == (8, 3)
    assert indices.min() >= 0
    assert indices.max() < 64


def test_rvq_residual_reduction():
    """Reconstruction error should be less than original norm."""
    torch.manual_seed(42)
    rvq = ResidualVQ(n_levels=3, K=256, dim=64)
    z = torch.randn(32, 64)

    z_hat, _ = rvq(z)
    recon_error = (z - z_hat).detach().norm(dim=-1).mean()
    original_norm = z.norm(dim=-1).mean()

    assert recon_error < original_norm, (
        f"Recon error {recon_error:.4f} should be < original norm {original_norm:.4f}"
    )


def test_rvq_gradient_flow():
    """Gradients should flow through RVQ via straight-through estimator."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=128)
    z = torch.randn(4, 128, requires_grad=True)
    z_hat, _ = rvq(z)
    loss = z_hat.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_rvq_compute_loss():
    """Commit and embed losses should be non-negative scalars."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=128)
    z = torch.randn(8, 128, requires_grad=True)
    z_hat, indices = rvq(z)
    commit_loss, embed_loss = rvq.compute_loss(z, indices)
    assert commit_loss.shape == ()
    assert embed_loss.shape == ()
    assert commit_loss.item() >= 0
    assert embed_loss.item() >= 0


def test_rvq_utilization():
    """Each level should use at least 2 codes with diverse input."""
    torch.manual_seed(123)
    rvq = ResidualVQ(n_levels=3, K=32, dim=16)
    z = torch.randn(256, 16)
    _, indices = rvq(z)
    utils = rvq.get_utilization(indices)
    assert len(utils) == 3
    for i, u in enumerate(utils):
        n_used = int(u * 32)
        assert n_used >= 2, f"Level {i} only used {n_used} codes"


def test_rvq_decode_indices():
    """decode_indices should match the z_hat from forward."""
    torch.manual_seed(0)
    rvq = ResidualVQ(n_levels=3, K=64, dim=32)
    z = torch.randn(8, 32)

    z_hat, indices = rvq(z)
    z_decoded = rvq.decode_indices(indices)

    assert torch.allclose(z_hat.detach(), z_decoded, atol=1e-5), (
        f"Max diff: {(z_hat.detach() - z_decoded).abs().max():.6f}"
    )

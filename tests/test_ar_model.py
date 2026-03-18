"""Tests for src.ar_model — PatchGPT autoregressive Transformer."""

import torch
import pytest
from src.ar_model import PatchGPT


@pytest.fixture
def small_model():
    """Small PatchGPT for fast testing."""
    return PatchGPT(
        vocab_size=2000, d_model=64, n_heads=4, n_layers=2,
        max_seq_len=256, dropout=0.0,
    )


def test_patchgpt_forward_shape(small_model):
    """Logits shape must be (B, T, vocab_size)."""
    tokens = torch.randint(0, 2000, (2, 100))
    logits = small_model(tokens)
    assert logits.shape == (2, 100, 2000)


def test_patchgpt_causal(small_model):
    """Causal masking: first 10 positions must match between full and prefix."""
    full_tokens = torch.randint(0, 2000, (1, 50))
    prefix_tokens = full_tokens[:, :10]

    logits_full = small_model(full_tokens)
    logits_prefix = small_model(prefix_tokens)

    # First 10 positions should produce identical logits
    assert torch.allclose(logits_full[:, :10], logits_prefix[:, :10], atol=1e-4)


def test_patchgpt_loss(small_model):
    """Loss must be positive and have gradients."""
    tokens = torch.randint(0, 2000, (2, 50))
    loss = small_model.compute_loss(tokens)
    assert loss.item() > 0
    assert loss.requires_grad


def test_patchgpt_generate(small_model):
    """Generated sequence must be <= max_len with values in [0, vocab_size)."""
    max_len = 30
    out = small_model.generate(max_len=max_len, temperature=1.0, top_k=50)
    assert out.shape[0] == 1
    assert out.shape[1] <= max_len
    assert (out >= 0).all()
    assert (out < 2000).all()

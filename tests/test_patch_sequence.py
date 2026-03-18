"""Tests for src.patch_sequence — patch sequence encoding utilities."""

import torch
import pytest
from src.patch_sequence import (
    quantize_position,
    quantize_scale,
    morton_code_3d,
    patches_to_token_sequence,
    compute_vocab_size,
)


def test_quantize_position_range():
    """Quantized positions must be in [0, n_bins)."""
    positions = torch.randn(50, 3) * 10  # arbitrary scale
    n_bins = 256
    qpos = quantize_position(positions, n_bins)
    assert qpos.shape == (50, 3)
    assert qpos.dtype == torch.int64
    assert (qpos >= 0).all()
    assert (qpos < n_bins).all()


def test_quantize_scale_range():
    """Quantized scales must be in [0, n_bins)."""
    scales = torch.rand(50) * 5 + 0.1  # positive scales
    n_bins = 64
    qs = quantize_scale(scales, n_bins)
    assert qs.shape == (50,)
    assert qs.dtype == torch.int64
    assert (qs >= 0).all()
    assert (qs < n_bins).all()


def test_morton_code_ordering():
    """Morton codes must be deterministic and produce a unique ordering."""
    positions = torch.randn(20, 3)
    codes_a = morton_code_3d(positions, n_bins=256)
    codes_b = morton_code_3d(positions, n_bins=256)
    # Deterministic
    assert torch.equal(codes_a, codes_b)
    # Sorting by morton code gives a valid permutation
    order = codes_a.argsort()
    assert order.shape == (20,)
    assert set(order.tolist()) == set(range(20))


def test_patches_to_sequence_simvq():
    """SimVQ mode: 10 patches -> 10*5 = 50 tokens."""
    M = 10
    centroids = torch.randn(M, 3)
    scales = torch.rand(M) + 0.1
    codebook_tokens = torch.randint(0, 1024, (M,))
    seq = patches_to_token_sequence(
        centroids, scales, codebook_tokens, mode="simvq",
        n_pos_bins=256, n_scale_bins=64,
    )
    assert seq.shape == (M * 5,)
    # All tokens should be non-negative
    assert (seq >= 0).all()


def test_patches_to_sequence_rvq():
    """RVQ mode: 10 patches -> 10*7 = 70 tokens."""
    M = 10
    centroids = torch.randn(M, 3)
    scales = torch.rand(M) + 0.1
    codebook_tokens = torch.randint(0, 1024, (M, 3))
    seq = patches_to_token_sequence(
        centroids, scales, codebook_tokens, mode="rvq",
        n_pos_bins=256, n_scale_bins=64,
    )
    assert seq.shape == (M * 7,)
    assert (seq >= 0).all()

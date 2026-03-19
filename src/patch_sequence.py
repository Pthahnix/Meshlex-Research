"""MeshLex v2 — Patch Sequence Encoding.

Converts patch-level data (centroids, scales, codebook tokens) into flat
token sequences suitable for autoregressive generation, with Morton/Z-order
spatial sorting.
"""

import torch
import numpy as np


def quantize_position(positions: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """Normalize positions to [0,1] per axis, then bin into [0, n_bins).

    Args:
        positions: (M, 3) float tensor of patch centroids.
        n_bins: number of quantization bins per axis.

    Returns:
        (M, 3) int64 tensor with values in [0, n_bins).
    """
    # Per-axis min/max normalization to [0, 1]
    mins = positions.min(dim=0).values  # (3,)
    maxs = positions.max(dim=0).values  # (3,)
    span = (maxs - mins).clamp(min=1e-8)
    normed = (positions - mins) / span  # (M, 3) in [0, 1]
    # Bin and clamp
    binned = (normed * n_bins).long().clamp(0, n_bins - 1)
    return binned


def quantize_scale(scales: torch.Tensor, n_bins: int = 64) -> torch.Tensor:
    """Normalize scales to [0,1], then bin into [0, n_bins).

    Args:
        scales: (M,) float tensor of patch scales.
        n_bins: number of quantization bins.

    Returns:
        (M,) int64 tensor with values in [0, n_bins).
    """
    s_min = scales.min()
    s_max = scales.max()
    span = (s_max - s_min).clamp(min=1e-8)
    normed = (scales - s_min) / span  # [0, 1]
    binned = (normed * n_bins).long().clamp(0, n_bins - 1)
    return binned


def morton_code_3d(positions: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """Compute Morton (Z-order) codes for 3D quantized positions.

    Interleaves bits of (x, y, z) coordinates to produce a single 30-bit code
    (10 bits per axis, supporting n_bins up to 1024).

    Args:
        positions: (M, 3) float tensor of patch centroids.
        n_bins: number of bins per axis for quantization.

    Returns:
        (M,) int64 tensor of Morton codes.
    """
    qpos = quantize_position(positions, n_bins)  # (M, 3) int
    x = qpos[:, 0].cpu().numpy().astype(np.uint32)
    y = qpos[:, 1].cpu().numpy().astype(np.uint32)
    z = qpos[:, 2].cpu().numpy().astype(np.uint32)

    def spread_bits(v: np.ndarray) -> np.ndarray:
        """Spread 10 low bits of v into every-3rd bit position."""
        v = v & 0x3FF  # keep 10 bits
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    code = (spread_bits(x) << 2) | (spread_bits(y) << 1) | spread_bits(z)
    return torch.from_numpy(code.astype(np.int64)).to(positions.device)


def patches_to_token_sequence(
    centroids: torch.Tensor,
    scales: torch.Tensor,
    codebook_tokens: torch.Tensor,
    mode: str = "rvq",
    n_pos_bins: int = 256,
    n_scale_bins: int = 64,
) -> torch.Tensor:
    """Convert patch data into a flat token sequence sorted by Morton code.

    Token layout per patch:
        pos_x  : [0, n_pos_bins)
        pos_y  : [n_pos_bins, 2*n_pos_bins)
        pos_z  : [2*n_pos_bins, 3*n_pos_bins)
        scale  : [3*n_pos_bins, 3*n_pos_bins + n_scale_bins)
        codebook: offset by (3*n_pos_bins + n_scale_bins)

    SimVQ mode: 1 codebook token  -> 5 tokens/patch
    RVQ mode:   3 codebook tokens -> 7 tokens/patch

    Args:
        centroids: (M, 3) float patch centroids.
        scales: (M,) float patch scales.
        codebook_tokens: (M,) for SimVQ or (M, 3) for RVQ.
        mode: "simvq" or "rvq".
        n_pos_bins: position quantization bins.
        n_scale_bins: scale quantization bins.

    Returns:
        (M * tokens_per_patch,) int64 flat token sequence.
    """
    # Ensure torch tensors (MeshSequenceDataset passes numpy arrays)
    if not isinstance(centroids, torch.Tensor):
        centroids = torch.tensor(centroids, dtype=torch.float32)
    if not isinstance(scales, torch.Tensor):
        scales = torch.tensor(scales, dtype=torch.float32)
    if not isinstance(codebook_tokens, torch.Tensor):
        codebook_tokens = torch.tensor(codebook_tokens, dtype=torch.long)

    M = centroids.shape[0]
    cb_offset = 3 * n_pos_bins + n_scale_bins

    # Quantize spatial attributes
    qpos = quantize_position(centroids, n_pos_bins)  # (M, 3)
    qscale = quantize_scale(scales, n_scale_bins)  # (M,)

    # Sort by Morton code
    morton = morton_code_3d(centroids, n_pos_bins)  # (M,)
    order = morton.argsort()

    qpos = qpos[order]
    qscale = qscale[order]
    codebook_tokens = codebook_tokens[order]

    # Build per-patch token tuples with offsets
    pos_x = qpos[:, 0]                          # offset 0
    pos_y = qpos[:, 1] + n_pos_bins             # offset n_pos_bins
    pos_z = qpos[:, 2] + 2 * n_pos_bins         # offset 2*n_pos_bins
    scale = qscale + 3 * n_pos_bins              # offset 3*n_pos_bins

    if mode == "simvq":
        # codebook_tokens: (M,)
        cb = codebook_tokens.long() + cb_offset
        # Stack: (M, 5)
        tokens = torch.stack([pos_x, pos_y, pos_z, scale, cb], dim=1)
    elif mode == "rvq":
        # codebook_tokens: (M, 3)
        cb = codebook_tokens.long() + cb_offset
        tokens = torch.stack(
            [pos_x, pos_y, pos_z, scale, cb[:, 0], cb[:, 1], cb[:, 2]], dim=1
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simvq' or 'rvq'.")

    return tokens.reshape(-1)


def compute_vocab_size(
    n_pos_bins: int = 256,
    n_scale_bins: int = 64,
    codebook_K: int = 1024,
) -> int:
    """Total vocabulary size for the AR model.

    Layout: 3*n_pos_bins (xyz) + n_scale_bins + codebook_K
    """
    return 3 * n_pos_bins + n_scale_bins + codebook_K

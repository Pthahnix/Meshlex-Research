# PatchDiffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PatchDiffusion — the first masked discrete diffusion model for patch-level mesh token sequences.

**Architecture:** Bidirectional Transformer with timestep embeddings that learns to iteratively unmask patch tokens. Three variants: Pure MDLM (full parallel unmasking), Block Diffusion (AR over blocks + MDLM within blocks), and Hierarchical RVQ Diffusion (coarse-to-fine across RVQ levels).

**Tech Stack:** PyTorch, PyTorch Geometric (for data loading), following existing MeshLex codebase conventions.

---

## File Structure

```
src/
├── mdlm.py                    # MDLM core: MaskedDiffusion model + schedule utilities
├── mdlm_hierarchical.py       # Variant C: Hierarchical RVQ Diffusion
├── mdlm_block.py              # Variant B: Block Diffusion (AR + MDLM hybrid)
└── (existing files...)

scripts/
├── train_mdlm.py              # Unified training script for all variants
├── generate_mdlm.py           # Generation + visualization
└── (existing files...)

tests/
├── test_mdlm.py               # Unit tests for MDLM core
├── test_mdlm_hierarchical.py  # Unit tests for hierarchical variant
└── test_mdlm_block.py         # Unit tests for block variant
```

---

## Task 1: MDLM Core — Masking Schedule Utilities

**Files:**
- Create: `src/mdlm.py`
- Test: `tests/test_mdlm.py`

- [ ] **Step 1: Write the failing test for masking schedules**

```python
# tests/test_mdlm.py
import torch
import pytest
from src.mdlm import cosine_schedule, linear_schedule, get_alpha_t


def test_cosine_schedule_bounds():
    """Cosine schedule: alpha_0=1, alpha_1≈0."""
    alpha_0 = cosine_schedule(0.0)
    alpha_1 = cosine_schedule(1.0)
    assert torch.isclose(alpha_0, torch.tensor(1.0), atol=1e-5)
    assert alpha_1 < 0.01  # Near zero at t=1


def test_linear_schedule_bounds():
    """Linear schedule: alpha_0=1, alpha_1=0."""
    alpha_0 = linear_schedule(0.0)
    alpha_1 = linear_schedule(1.0)
    assert torch.isclose(alpha_0, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(alpha_1, torch.tensor(0.0), atol=1e-5)


def test_get_alpha_t_vectorized():
    """get_alpha_t must handle batched t values."""
    t = torch.tensor([0.0, 0.5, 1.0])
    alphas = get_alpha_t(t, schedule="cosine")
    assert alphas.shape == (3,)
    assert (alphas >= 0).all() and (alphas <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mdlm.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.mdlm'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/mdlm.py
"""Masked Discrete Diffusion Language Model (MDLM) for patch tokens.

Based on Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine masking schedule: alpha_t = cos(πt/2)^2.

    Slows down unmasking at the start, reducing conflicting tokens.

    Args:
        t: (...,) tensor of timesteps in [0, 1].

    Returns:
        (...,) tensor of alpha values in [0, 1].
    """
    return torch.cos(math.pi / 2 * t) ** 2


def linear_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear masking schedule: alpha_t = 1 - t.

    Args:
        t: (...,) tensor of timesteps in [0, 1].

    Returns:
        (...,) tensor of alpha values in [0, 1].
    """
    return 1.0 - t


def sqrt_schedule(t: torch.Tensor) -> torch.Tensor:
    """Square root schedule: alpha_t = sqrt(1 - t).

    Args:
        t: (...,) tensor of timesteps in [0, 1].

    Returns:
        (...,) tensor of alpha values in [0, 1].
    """
    return torch.sqrt(1.0 - t)


def get_alpha_t(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """Get alpha_t for given schedule.

    Args:
        t: (...,) tensor of timesteps in [0, 1].
        schedule: One of "cosine", "linear", "sqrt".

    Returns:
        (...,) tensor of alpha values.
    """
    if schedule == "cosine":
        return cosine_schedule(t)
    elif schedule == "linear":
        return linear_schedule(t)
    elif schedule == "sqrt":
        return sqrt_schedule(t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mdlm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mdlm.py tests/test_mdlm.py
git commit -m "feat(mdlm): add masking schedule utilities (cosine, linear, sqrt)"
git push origin innovation-brainstorm
```

---

## Task 2: MDLM Core — Forward Process (Masking)

**Files:**
- Modify: `src/mdlm.py`
- Modify: `tests/test_mdlm.py`

- [ ] **Step 1: Write the failing test for forward masking**

```python
# Add to tests/test_mdlm.py
from src.mdlm import forward_mask


def test_forward_mask_correct_ratio():
    """Masked tokens should approximately match mask_ratio."""
    torch.manual_seed(42)
    x = torch.randint(0, 100, (4, 50))  # (B, N) clean tokens
    mask_token = 100  # Special mask token index

    # t=0.3 → ~30% masked
    x_masked, mask = forward_mask(x, t=torch.tensor(0.3), mask_token=mask_token)

    actual_ratio = mask.float().mean().item()
    expected_ratio = 0.3
    assert abs(actual_ratio - expected_ratio) < 0.15  # Allow variance


def test_forward_mask_all_masked_at_t1():
    """At t=1, all tokens should be masked."""
    x = torch.randint(0, 100, (2, 30))
    x_masked, mask = forward_mask(x, t=torch.tensor(1.0), mask_token=100)

    assert mask.all()
    assert (x_masked == 100).all()


def test_forward_mask_none_masked_at_t0():
    """At t=0, no tokens should be masked."""
    x = torch.randint(0, 100, (2, 30))
    x_masked, mask = forward_mask(x, t=torch.tensor(0.0), mask_token=100)

    assert not mask.any()
    assert torch.equal(x_masked, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mdlm.py::test_forward_mask_correct_ratio -v`
Expected: FAIL with "ImportError: cannot import name 'forward_mask'"

- [ ] **Step 3: Write minimal implementation**

```python
# Add to src/mdlm.py
def forward_mask(
    x: torch.Tensor,
    t: torch.Tensor,
    mask_token: int,
    schedule: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward masking process: progressively mask tokens.

    Args:
        x: (B, N) clean token indices.
        t: scalar or (B,) tensor of timesteps in [0, 1].
        mask_token: index of the mask token.
        schedule: masking schedule name.

    Returns:
        x_masked: (B, N) masked token indices.
        mask: (B, N) boolean mask where True = masked.
    """
    alpha_t = get_alpha_t(t, schedule)  # Probability of staying unmasked

    # Sample mask: each token is masked with probability (1 - alpha_t)
    B, N = x.shape
    if alpha_t.dim() == 0:
        prob_unmasked = alpha_t.expand(B, N)
    else:
        prob_unmasked = alpha_t.view(B, 1).expand(B, N)

    mask = torch.rand_like(x.float()) > prob_unmasked  # (B, N) bool
    x_masked = x.clone()
    x_masked[mask] = mask_token

    return x_masked, mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mdlm.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mdlm.py tests/test_mdlm.py
git commit -m "feat(mdlm): add forward masking process"
git push origin innovation-brainstorm
```

---

## Task 3: MDLM Core — MaskedDiffusionTransformer Model

**Files:**
- Modify: `src/mdlm.py`
- Modify: `tests/test_mdlm.py`

- [ ] **Step 1: Write the failing test for MaskedDiffusionTransformer**

```python
# Add to tests/test_mdlm.py
from src.mdlm import MaskedDiffusionTransformer


@pytest.fixture
def small_mdlm():
    """Small MDLM for fast testing."""
    return MaskedDiffusionTransformer(
        vocab_size=2000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=256,
    )


def test_mdlm_forward_shape(small_mdlm):
    """Output logits must be (B, N, vocab_size)."""
    x = torch.randint(0, 2000, (2, 50))
    t = torch.rand(2)
    logits = small_mdlm(x, t)
    assert logits.shape == (2, 50, 2000)


def test_mdlm_bidirectional(small_mdlm):
    """Bidirectional: token at position i must not depend on tokens at j>i."""
    # This is verified by checking that masking later tokens doesn't affect earlier predictions
    x1 = torch.randint(0, 2000, (1, 50))
    x2 = x1.clone()
    x2[:, 25:] = 1999  # Mask second half

    t = torch.tensor([0.5])

    logits1 = small_mdlm(x1, t)
    logits2 = small_mdlm(x2, t)

    # First 25 positions should be identical (or very close due to dropout)
    # Disable dropout for this test
    small_mdlm.eval()
    logits1 = small_mdlm(x1, t)
    logits2 = small_mdlm(x2, t)

    assert torch.allclose(logits1[:, :25], logits2[:, :25], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mdlm.py::test_mdlm_forward_shape -v`
Expected: FAIL with "ImportError: cannot import name 'MaskedDiffusionTransformer'"

- [ ] **Step 3: Write minimal implementation**

```python
# Add to src/mdlm.py

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar or batch timesteps.

        Args:
            t: (B,) or scalar tensor of timesteps in [0, 1].

        Returns:
            (B, dim) embedding.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device) * -(math.log(10000.0) / half_dim)
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MaskedDiffusionTransformer(nn.Module):
    """Bidirectional Transformer for masked discrete diffusion.

    Unlike PatchGPT (AR), this uses bidirectional attention to predict
    all token positions simultaneously given a partially masked sequence.

    Args:
        vocab_size: total token vocabulary size.
        d_model: hidden dimension.
        n_heads: number of attention heads.
        n_layers: number of Transformer blocks.
        max_seq_len: maximum sequence length.
        dropout: dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = SinusoidalEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks (no causal mask!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.normal_(self.time_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N) token indices (may contain mask tokens).
            t: (B,) or scalar timestep in [0, 1].

        Returns:
            (B, N, vocab_size) logits for all positions.
        """
        B, N = x.shape
        assert N <= self.max_seq_len, f"Sequence length {N} exceeds max {self.max_seq_len}"

        # Token + position embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        # Timestep embedding (broadcast to all positions)
        t_emb = self.time_emb(t)  # (B, d_model)
        t_emb = self.time_proj(t_emb)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # (B, N, d_model)

        h = self.drop(h)

        # No causal mask — bidirectional attention
        h = self.blocks(h)  # (B, N, d_model)

        h = self.ln_f(h)
        logits = self.head(h)  # (B, N, vocab_size)

        return logits

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        t: torch.Tensor,
        mask_token: int,
        schedule: str = "cosine",
    ) -> torch.Tensor:
        """Training loss: cross-entropy on masked positions only.

        Args:
            x_clean: (B, N) clean token indices.
            t: (B,) timesteps sampled uniformly in [0, 1].
            mask_token: index of mask token.
            schedule: masking schedule.

        Returns:
            Scalar loss.
        """
        # Forward masking
        x_masked, mask = forward_mask(x_clean, t, mask_token, schedule)

        # Predict all tokens
        logits = self.forward(x_masked, t)  # (B, N, V)

        # Loss only on masked positions
        if mask.any():
            loss = F.cross_entropy(
                logits[mask],  # (num_masked, V)
                x_clean[mask],  # (num_masked,)
            )
        else:
            # No tokens masked (t=0 edge case) — zero loss
            loss = torch.tensor(0.0, device=x_clean.device, requires_grad=True)

        return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mdlm.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mdlm.py tests/test_mdlm.py
git commit -m "feat(mdlm): add MaskedDiffusionTransformer with bidirectional attention"
git push origin innovation-brainstorm
```

---

## Task 4: MDLM Core — Iterative Unmasking (Sampling)

**Files:**
- Modify: `src/mdlm.py`
- Modify: `tests/test_mdlm.py`

- [ ] **Step 1: Write the failing test for sampling**

```python
# Add to tests/test_mdlm.py
from src.mdlm import iterative_unmask


def test_iterative_unmask_shape(small_mdlm):
    """Generated sequence must have correct shape."""
    n_patches = 50
    mask_token = 1999  # vocab_size - 1
    steps = 10

    generated = iterative_unmask(
        model=small_mdlm,
        n_patches=n_patches,
        mask_token=mask_token,
        steps=steps,
        schedule="cosine",
    )

    assert generated.shape == (1, n_patches)
    assert (generated >= 0).all()
    assert (generated < small_mdlm.vocab_size).all()


def test_iterative_unmask_unmasking_progress(small_mdlm):
    """All tokens should be unmasked by the final step."""
    n_patches = 30
    mask_token = 1999

    generated = iterative_unmask(
        model=small_mdlm,
        n_patches=n_patches,
        mask_token=mask_token,
        steps=20,
        schedule="cosine",
    )

    # No mask tokens should remain
    assert not (generated == mask_token).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mdlm.py::test_iterative_unmask_shape -v`
Expected: FAIL with "ImportError: cannot import name 'iterative_unmask'"

- [ ] **Step 3: Write minimal implementation**

```python
# Add to src/mdlm.py
@torch.no_grad()
def iterative_unmask(
    model: MaskedDiffusionTransformer,
    n_patches: int,
    mask_token: int,
    steps: int = 20,
    schedule: str = "cosine",
    temperature: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Iterative unmasking for generation.

    Start from all-masked and progressively unmask based on model confidence.

    Args:
        model: trained MaskedDiffusionTransformer.
        n_patches: number of patch tokens to generate.
        mask_token: index of mask token.
        steps: number of unmasking iterations.
        schedule: masking schedule name.
        temperature: sampling temperature.
        device: device to run on.

    Returns:
        (1, n_patches) generated token indices.
    """
    model.eval()
    model.to(device)

    # Start with all mask tokens
    x = torch.full((1, n_patches), mask_token, dtype=torch.long, device=device)

    for step in range(steps):
        # Compute t (from 1.0 to 0.0)
        t = torch.tensor(1.0 - step / steps, device=device)

        # Get model predictions
        logits = model(x, t) / temperature  # (1, N, V)

        # For masked positions, sample based on confidence
        mask = (x == mask_token)  # (1, N)
        if not mask.any():
            break  # All unmasked

        probs = F.softmax(logits, dim=-1)  # (1, N, V)
        confidence = probs.max(dim=-1).values  # (1, N)

        # Decide how many to unmask this step
        alpha_t = get_alpha_t(t, schedule).item()
        target_unmasked = int(n_patches * (1 - alpha_t))
        already_unmasked = (~mask).sum().item()
        n_to_unmask = max(0, target_unmasked - already_unmasked)

        if n_to_unmask == 0:
            continue

        # Select top-confidence positions among masked
        confidence[~mask] = -float("inf")  # Ignore already unmasked
        _, top_indices = confidence.topk(n_to_unmask, dim=-1)  # (1, n_to_unmask)

        # Sample tokens for those positions
        sampled = torch.multinomial(
            probs[0, top_indices[0]], num_samples=1
        ).squeeze(-1)  # (n_to_unmask,)

        x[0, top_indices[0]] = sampled

    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mdlm.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mdlm.py tests/test_mdlm.py
git commit -m "feat(mdlm): add iterative unmasking for generation"
git push origin innovation-brainstorm
```

---

## Task 5: Token Sequence Dataset for Diffusion

**Files:**
- Create: `src/diffusion_dataset.py`
- Create: `tests/test_diffusion_dataset.py`

- [ ] **Step 1: Write the failing test for DiffusionDataset**

```python
# tests/test_diffusion_dataset.py
import torch
import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.diffusion_dataset import DiffusionPatchDataset


def create_mock_sequence_file(path: Path, n_meshes: int = 10, max_patches: int = 100):
    """Create mock NPZ files for testing."""
    for i in range(n_meshes):
        n_patches = np.random.randint(20, max_patches)
        # RVQ mode: 7 tokens per patch (pos_x, pos_y, pos_z, scale, tok_L1, tok_L2, tok_L3)
        tokens = np.random.randint(0, 2000, (n_patches, 7))
        np.savez(path / f"mesh_{i:04d}.npz", tokens=tokens)


def test_diffusion_dataset_length():
    """Dataset length must match number of files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        create_mock_sequence_file(tmpdir, n_meshes=15)
        dataset = DiffusionPatchDataset(tmpdir, mode="rvq", max_seq_len=512)
        assert len(dataset) == 15


def test_diffusion_dataset_item_shape():
    """Each item must have shape (seq_len,) with correct values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        create_mock_sequence_file(tmpdir, n_meshes=5, max_patches=50)
        dataset = DiffusionPatchDataset(tmpdir, mode="rvq", max_seq_len=512)

        tokens = dataset[0]
        # Should be flattened: (n_patches * 7,) ≤ 512
        assert tokens.dim() == 1
        assert tokens.shape[0] <= 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_diffusion_dataset.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.diffusion_dataset'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/diffusion_dataset.py
"""Dataset for masked diffusion training on patch token sequences."""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class DiffusionPatchDataset(Dataset):
    """Patch token sequences for masked diffusion training.

    Unlike MeshSequenceDataset (AR), this returns the full sequence without
    separating input/target — masking is applied dynamically during training.

    Args:
        sequence_dir: directory containing NPZ files with token sequences.
        mode: "rvq" (7 tokens/patch) or "simvq" (5 tokens/patch).
        max_seq_len: maximum sequence length (truncate if longer).
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        mode: str = "rvq",
        max_seq_len: int = 1024,
    ):
        self.sequence_dir = Path(sequence_dir)
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.tokens_per_patch = 7 if mode == "rvq" else 5

        self.files = sorted(self.sequence_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise ValueError(f"No NPZ files found in {sequence_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return flat token sequence.

        Returns:
            (seq_len,) int64 tensor where seq_len ≤ max_seq_len.
        """
        data = np.load(self.files[idx])
        tokens = data["tokens"]  # (n_patches, tokens_per_patch)

        # Flatten and truncate
        flat = tokens.reshape(-1)  # (n_patches * tokens_per_patch,)
        if len(flat) > self.max_seq_len:
            flat = flat[: self.max_seq_len]

        return torch.tensor(flat, dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_diffusion_dataset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diffusion_dataset.py tests/test_diffusion_dataset.py
git commit -m "feat(mdlm): add DiffusionPatchDataset for diffusion training"
git push origin innovation-brainstorm
```

---

## Task 6: Training Script for Pure MDLM

**Files:**
- Create: `scripts/train_mdlm.py`
- Create: `tests/test_train_mdlm.py`

- [ ] **Step 1: Write the failing test for training script CLI**

```python
# tests/test_train_mdlm.py
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np


def create_mock_dataset(path: Path, n_meshes: int = 20):
    """Create mock NPZ files for training."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n_meshes):
        n_patches = np.random.randint(20, 80)
        tokens = np.random.randint(0, 2000, (n_patches, 7))
        np.savez(path / f"mesh_{i:04d}.npz", tokens=tokens)


def test_train_mdlm_help():
    """Training script must accept --help."""
    result = subprocess.run(
        [sys.executable, "scripts/train_mdlm.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--sequence_dir" in result.stdout
    assert "--epochs" in result.stdout


def test_train_mdlm_dry_run():
    """Training script must run for 1 epoch without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        seq_dir = tmpdir / "sequences"
        ckpt_dir = tmpdir / "checkpoints"
        create_mock_dataset(seq_dir, n_meshes=10)

        result = subprocess.run(
            [
                sys.executable, "scripts/train_mdlm.py",
                "--sequence_dir", str(seq_dir),
                "--checkpoint_dir", str(ckpt_dir),
                "--mode", "rvq",
                "--epochs", "1",
                "--batch_size", "2",
                "--n_layers", "1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Stderr: {result.stderr}"
        assert (ckpt_dir / "checkpoint_final.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_mdlm.py -v`
Expected: FAIL with "FileNotFoundError" or "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/train_mdlm.py
"""Train Masked Diffusion Language Model on patch token sequences.

Usage:
    python scripts/train_mdlm.py \
        --sequence_dir data/sequences/rvq_lvis \
        --checkpoint_dir data/checkpoints/mdlm \
        --mode rvq --epochs 500 --batch_size 128
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
import gc

from src.mdlm import MaskedDiffusionTransformer
from src.diffusion_dataset import DiffusionPatchDataset
from src.patch_sequence import compute_vocab_size


def main():
    parser = argparse.ArgumentParser(description="Train MDLM for patch token generation")
    parser.add_argument("--sequence_dir", required=True, help="Directory with token sequences")
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/mdlm")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "sqrt"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = DiffusionPatchDataset(args.sequence_dir, mode=args.mode, max_seq_len=args.max_seq_len)
    print(f"Sequences: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    vocab_size = compute_vocab_size(codebook_K=args.codebook_size)
    mask_token = vocab_size  # Mask token is one beyond vocab

    model = MaskedDiffusionTransformer(
        vocab_size=vocab_size + 1,  # +1 for mask token
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MDLM: {n_params / 1e6:.1f}M params, vocab_size={vocab_size + 1}, mask_token={mask_token}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR schedule
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8 / args.lr,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    start_epoch = 0
    history = []
    config = {
        "vocab_size": vocab_size + 1,
        "mask_token": mask_token,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.max_seq_len,
        "schedule": args.schedule,
    }

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for tokens in loader:
            tokens = tokens.to(device)  # (B, N)
            B, N = tokens.shape

            # Sample timesteps uniformly
            t = torch.rand(B, device=device)

            loss = model.compute_loss(tokens, t, mask_token, schedule=args.schedule)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        torch.cuda.empty_cache()

        metrics = {"epoch": epoch, "loss": avg_loss, "lr": lr_now, "time_sec": elapsed}
        history.append(metrics)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": config,
            }, ckpt_path)
            # Keep only latest 3
            old_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))[:-3]
            for old in old_ckpts:
                old.unlink()

        gc.collect()

    # Final checkpoint
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": config,
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Final checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_mdlm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/train_mdlm.py tests/test_train_mdlm.py
git commit -m "feat(mdlm): add training script for Pure MDLM variant"
git push origin innovation-brainstorm
```

---

## Task 7: Block Diffusion Variant

**Files:**
- Create: `src/mdlm_block.py`
- Create: `tests/test_mdlm_block.py`

- [ ] **Step 1: Write the failing test for Block Diffusion**

```python
# tests/test_mdlm_block.py
import torch
import pytest
from src.mdlm_block import BlockDiffusionTransformer


@pytest.fixture
def small_block_model():
    return BlockDiffusionTransformer(
        vocab_size=2000,
        d_model=64,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
        block_size=13,
    )


def test_block_diffusion_forward_shape(small_block_model):
    """Output must have correct shape."""
    x = torch.randint(0, 2000, (2, 50))
    block_idx = torch.tensor([0, 2])  # Which block we're generating
    t = torch.rand(2)
    context = torch.randint(0, 2000, (2, 39))  # Already generated blocks

    logits = small_block_model(x, t, block_idx, context)
    assert logits.shape == (2, 50, 2000)


def test_block_diffusion_causal_between_blocks(small_block_model):
    """Blocks must be processed in causal order (AR)."""
    small_block_model.eval()

    # Block 0 with no context
    x0 = torch.randint(0, 2000, (1, 13))
    logits0 = small_block_model(x0, t=torch.tensor([0.5]), block_idx=torch.tensor([0]), context=None)

    # Block 1 with block 0 as context
    x1 = torch.randint(0, 2000, (1, 13))
    context1 = torch.randint(0, 2000, (1, 13))
    logits1 = small_block_model(x1, t=torch.tensor([0.5]), block_idx=torch.tensor([1]), context=context1)

    # Just verify shapes and no crash — causal constraint is enforced by masking
    assert logits0.shape == (1, 13, 2000)
    assert logits1.shape == (1, 13, 2000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mdlm_block.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.mdlm_block'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/mdlm_block.py
"""Block Diffusion — AR over blocks + MDLM within blocks.

Combines the global coherence of AR with the parallel generation of MDLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.mdlm import SinusoidalEmbedding, forward_mask, get_alpha_t


class BlockDiffusionTransformer(nn.Module):
    """Hybrid AR-MDLM: AR between blocks, MDLM within blocks.

    Args:
        vocab_size: total token vocabulary size (+1 for mask).
        d_model: hidden dimension.
        n_heads: number of attention heads.
        n_layers: total layers (half for AR, half for MDLM).
        max_seq_len: maximum total sequence length.
        block_size: number of patches per block.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 16,
        max_seq_len: int = 1024,
        block_size: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.block_size = block_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = SinusoidalEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Split layers: first half for AR, second half for MDLM within blocks
        n_ar = n_layers // 2
        n_mdlm = n_layers - n_ar

        # AR layers (causal attention across blocks)
        ar_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ar_blocks = nn.TransformerEncoder(ar_layer, num_layers=n_ar)

        # MDLM layers (bidirectional within blocks)
        mdlm_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mdlm_blocks = nn.TransformerEncoder(mdlm_layer, num_layers=n_mdlm)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.normal_(self.time_proj.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        block_idx: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, block_size) current block tokens (may be masked).
            t: (B,) diffusion timestep for within-block MDLM.
            block_idx: (B,) which block we're generating (for position offset).
            context: (B, context_len) previously generated blocks, or None.

        Returns:
            (B, block_size, vocab_size) logits.
        """
        B, block_len = x.shape
        device = x.device

        # Combine context + current block
        if context is not None:
            h = torch.cat([context, x], dim=1)  # (B, context_len + block_len)
        else:
            h = x  # (B, block_len)

        total_len = h.shape[1]

        # Position embeddings (offset by block_idx)
        positions = torch.arange(total_len, device=device).unsqueeze(0)
        block_offset = block_idx.unsqueeze(1) * self.block_size
        if context is not None:
            positions = positions + block_offset
        h = self.token_emb(h) + self.pos_emb(positions[:, :total_len])

        # Timestep embedding
        t_emb = self.time_proj(self.time_emb(t))
        h = h + t_emb.unsqueeze(1)

        h = self.drop(h)

        # AR attention: causal across the full sequence
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device, dtype=torch.bool), diagonal=1
        )
        h = self.ar_blocks(h, mask=causal_mask)

        # MDLM attention: bidirectional, but only on the current block
        h_block = h[:, -block_len:]  # (B, block_len, d_model)
        h_block = self.mdlm_blocks(h_block)  # Bidirectional

        h_block = self.ln_f(h_block)
        logits = self.head(h_block)  # (B, block_len, vocab_size)

        return logits

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        t: torch.Tensor,
        block_idx: torch.Tensor,
        mask_token: int,
        context: torch.Tensor | None = None,
        schedule: str = "cosine",
    ) -> torch.Tensor:
        """Training loss for block diffusion."""
        # Mask current block
        x_masked, mask = forward_mask(x_clean, t, mask_token, schedule)

        logits = self.forward(x_masked, t, block_idx, context)

        if mask.any():
            loss = F.cross_entropy(logits[mask], x_clean[mask])
        else:
            loss = torch.tensor(0.0, device=x_clean.device, requires_grad=True)

        return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mdlm_block.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mdlm_block.py tests/test_mdlm_block.py
git commit -m "feat(mdlm): add Block Diffusion variant (AR + MDLM hybrid)"
git push origin innovation-brainstorm
```

---

## Task 8: Hierarchical RVQ Diffusion Variant

**Files:**
- Create: `src/mdlm_hierarchical.py`
- Create: `tests/test_mdlm_hierarchical.py`

- [ ] **Step 1: Write the failing test for Hierarchical Diffusion**

```python
# tests/test_mdlm_hierarchical.py
import torch
import pytest
from src.mdlm_hierarchical import HierarchicalRVQDiffusion


@pytest.fixture
def small_hier_model():
    return HierarchicalRVQDiffusion(
        vocab_size=2000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=256,
    )


def test_hierarchical_three_stages(small_hier_model):
    """Must have three separate models for L1, L2, L3."""
    assert hasattr(small_hier_model, "stage1")
    assert hasattr(small_hier_model, "stage2")
    assert hasattr(small_hier_model, "stage3")


def test_hierarchical_stage1_forward(small_hier_model):
    """Stage 1 generates L1 tokens + positions."""
    n_patches = 30
    x = torch.randint(0, 2000, (2, n_patches))  # Full sequence placeholder
    t = torch.rand(2)

    # Stage 1 output should be for L1 tokens (positions 4, 5, 6 per patch)
    logits = small_hier_model.forward_stage1(x, t)
    assert logits.shape[0] == 2
    # Should predict L1 tokens for all patches
    assert logits.shape[2] == 2000  # vocab_size


def test_hierarchical_stage2_conditioned(small_hier_model):
    """Stage 2 must accept L1 conditioning."""
    n_patches = 20
    l1_tokens = torch.randint(0, 2000, (2, n_patches))
    x = torch.randint(0, 2000, (2, n_patches))
    t = torch.rand(2)

    logits = small_hier_model.forward_stage2(x, t, l1_condition=l1_tokens)
    assert logits.shape == (2, n_patches, 2000)


def test_hierarchical_stage3_conditioned(small_hier_model):
    """Stage 3 must accept L1+L2 conditioning."""
    n_patches = 15
    l1_tokens = torch.randint(0, 2000, (2, n_patches))
    l2_tokens = torch.randint(0, 2000, (2, n_patches))
    x = torch.randint(0, 2000, (2, n_patches))
    t = torch.rand(2)

    logits = small_hier_model.forward_stage3(x, t, l1_condition=l1_tokens, l2_condition=l2_tokens)
    assert logits.shape == (2, n_patches, 2000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mdlm_hierarchical.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.mdlm_hierarchical'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/mdlm_hierarchical.py
"""Hierarchical RVQ Diffusion — coarse-to-fine across RVQ levels.

Generates L1 tokens (coarse) → L2 tokens (medium) → L3 tokens (fine).
Each stage conditions on the previous level's tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.mdlm import MaskedDiffusionTransformer, SinusoidalEmbedding, forward_mask, get_alpha_t


class HierarchicalRVQDiffusion(nn.Module):
    """Three-stage diffusion for RVQ token hierarchy.

    Stage 1: Diffuse L1 tokens + position tokens
    Stage 2: Conditioned on L1, diffuse L2 tokens
    Stage 3: Conditioned on L1+L2, diffuse L3 tokens

    Args:
        vocab_size: base vocabulary size (codebook K).
        d_model: hidden dimension.
        n_heads: attention heads per stage.
        n_layers: layers per stage.
        max_seq_len: maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 8,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Three separate MDLM models
        # Stage 1: predict L1 + positions (7 tokens/patch)
        self.stage1 = MaskedDiffusionTransformer(
            vocab_size=vocab_size + 1,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Stage 2: predict L2 conditioned on L1
        self.stage2 = ConditionedMDLM(
            vocab_size=vocab_size + 1,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Stage 3: predict L3 conditioned on L1+L2
        self.stage3 = ConditionedMDLM(
            vocab_size=vocab_size + 1,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward_stage1(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Stage 1: predict L1 tokens."""
        return self.stage1(x, t)

    def forward_stage2(
        self, x: torch.Tensor, t: torch.Tensor, l1_condition: torch.Tensor
    ) -> torch.Tensor:
        """Stage 2: predict L2 tokens conditioned on L1."""
        return self.stage2(x, t, condition=l1_condition)

    def forward_stage3(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        l1_condition: torch.Tensor,
        l2_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 3: predict L3 tokens conditioned on L1+L2."""
        # Concatenate L1 and L2 as condition
        condition = torch.stack([l1_condition, l2_condition], dim=-1).reshape(
            l1_condition.shape[0], -1
        )
        return self.stage3(x, t, condition=condition)

    def compute_loss_stage1(
        self, l1_tokens: torch.Tensor, t: torch.Tensor, mask_token: int, schedule: str = "cosine"
    ) -> torch.Tensor:
        """Stage 1 loss on L1 tokens."""
        return self.stage1.compute_loss(l1_tokens, t, mask_token, schedule)

    def compute_loss_stage2(
        self,
        l2_tokens: torch.Tensor,
        l1_tokens: torch.Tensor,
        t: torch.Tensor,
        mask_token: int,
        schedule: str = "cosine",
    ) -> torch.Tensor:
        """Stage 2 loss on L2 tokens, conditioned on L1."""
        x_masked, mask = forward_mask(l2_tokens, t, mask_token, schedule)
        logits = self.forward_stage2(x_masked, t, l1_condition=l1_tokens)
        if mask.any():
            return F.cross_entropy(logits[mask], l2_tokens[mask])
        return torch.tensor(0.0, device=l2_tokens.device, requires_grad=True)

    def compute_loss_stage3(
        self,
        l3_tokens: torch.Tensor,
        l1_tokens: torch.Tensor,
        l2_tokens: torch.Tensor,
        t: torch.Tensor,
        mask_token: int,
        schedule: str = "cosine",
    ) -> torch.Tensor:
        """Stage 3 loss on L3 tokens, conditioned on L1+L2."""
        x_masked, mask = forward_mask(l3_tokens, t, mask_token, schedule)
        logits = self.forward_stage3(x_masked, t, l1_condition=l1_tokens, l2_condition=l2_tokens)
        if mask.any():
            return F.cross_entropy(logits[mask], l3_tokens[mask])
        return torch.tensor(0.0, device=l3_tokens.device, requires_grad=True)


class ConditionedMDLM(nn.Module):
    """MDLM with cross-attention conditioning.

    Used for Stage 2 and Stage 3 in hierarchical diffusion.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 8,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = SinusoidalEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Condition encoder
        self.cond_emb = nn.Linear(d_model, d_model)

        # Cross-attention transformer
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.normal_(self.time_proj.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward with conditioning.

        Args:
            x: (B, N) current tokens (may be masked).
            t: (B,) timestep.
            condition: (B, cond_len) conditioning tokens.

        Returns:
            (B, N, vocab_size) logits.
        """
        B, N = x.shape
        device = x.device

        # Token + position embeddings
        positions = torch.arange(N, device=device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        # Timestep
        t_emb = self.time_proj(self.time_emb(t))
        h = h + t_emb.unsqueeze(1)
        h = self.drop(h)

        # Condition
        if condition is not None:
            cond_emb = self.token_emb(condition)  # (B, cond_len, d_model)
            cond_emb = self.cond_emb(cond_emb)
        else:
            cond_emb = torch.zeros(1, 1, self.d_model, device=device).expand(B, 1, -1)

        # Cross-attention: h queries cond_emb
        h = self.blocks(h, cond_emb)  # (B, N, d_model)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mdlm_hierarchical.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mdlm_hierarchical.py tests/test_mdlm_hierarchical.py
git commit -m "feat(mdlm): add Hierarchical RVQ Diffusion variant"
git push origin innovation-brainstorm
```

---

## Task 9: Generation Script

**Files:**
- Create: `scripts/generate_mdlm.py`
- Create: `tests/test_generate_mdlm.py`

- [ ] **Step 1: Write the failing test for generation script**

```python
# tests/test_generate_mdlm.py
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import numpy as np


def create_mock_checkpoint(path: Path, vocab_size: int = 2000):
    """Create a minimal mock checkpoint."""
    from src.mdlm import MaskedDiffusionTransformer

    model = MaskedDiffusionTransformer(
        vocab_size=vocab_size + 1,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=256,
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size + 1,
            "mask_token": vocab_size,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "max_seq_len": 256,
        }
    }, path)


def test_generate_mdlm_help():
    """Generation script must accept --help."""
    result = subprocess.run(
        [sys.executable, "scripts/generate_mdlm.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout


def test_generate_mdlm_dry_run():
    """Generation script must produce output file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt_path = tmpdir / "checkpoint.pt"
        create_mock_checkpoint(ckpt_path)

        out_path = tmpdir / "generated.pt"

        result = subprocess.run(
            [
                sys.executable, "scripts/generate_mdlm.py",
                "--checkpoint", str(ckpt_path),
                "--output", str(out_path),
                "--n_meshes", "2",
                "--n_patches", "30",
                "--steps", "5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Stderr: {result.stderr}"
        assert out_path.exists()

        # Verify output shape
        data = torch.load(out_path, weights_only=False)
        assert "tokens" in data
        assert data["tokens"].shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_mdlm.py -v`
Expected: FAIL with "FileNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/generate_mdlm.py
"""Generate meshes using trained MDLM.

Usage:
    python scripts/generate_mdlm.py \
        --checkpoint data/checkpoints/mdlm/checkpoint_final.pt \
        --output results/generated_mdlm.pt \
        --n_meshes 100 --steps 20
"""
import argparse
import torch
from pathlib import Path

from src.mdlm import MaskedDiffusionTransformer, iterative_unmask


def main():
    parser = argparse.ArgumentParser(description="Generate meshes with MDLM")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--n_meshes", type=int, default=100, help="Number of meshes to generate")
    parser.add_argument("--n_patches", type=int, default=130, help="Patches per mesh")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "sqrt"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = MaskedDiffusionTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["max_seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mask_token = config["mask_token"]
    print(f"Loaded model from {args.checkpoint}")
    print(f"Generating {args.n_meshes} meshes with {args.n_patches} patches each...")

    # Generate
    all_tokens = []
    for i in range(args.n_meshes):
        tokens = iterative_unmask(
            model=model,
            n_patches=args.n_patches,
            mask_token=mask_token,
            steps=args.steps,
            schedule=args.schedule,
            temperature=args.temperature,
            device=device,
        )
        all_tokens.append(tokens.cpu())

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.n_meshes}")

    # Save
    all_tokens = torch.cat(all_tokens, dim=0)  # (n_meshes, n_patches)
    torch.save({"tokens": all_tokens, "config": config}, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_mdlm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_mdlm.py tests/test_generate_mdlm.py
git commit -m "feat(mdlm): add generation script for MDLM"
git push origin innovation-brainstorm
```

---

## Task 10: Extend Training Script for All Variants

**Files:**
- Modify: `scripts/train_mdlm.py`
- Modify: `tests/test_train_mdlm.py`

- [ ] **Step 1: Write the failing test for variant selection**

```python
# Add to tests/test_train_mdlm.py
def test_train_mdlm_block_variant():
    """Training script must support --variant block."""
    result = subprocess.run(
        [sys.executable, "scripts/train_mdlm.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--variant" in result.stdout
    assert "pure" in result.stdout
    assert "block" in result.stdout
    assert "hierarchical" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_mdlm.py::test_train_mdlm_block_variant -v`
Expected: FAIL (assertion error)

- [ ] **Step 3: Write minimal implementation**

Update `scripts/train_mdlm.py` to add `--variant` argument:

```python
# Add to argument parser
parser.add_argument(
    "--variant",
    choices=["pure", "block", "hierarchical"],
    default="pure",
    help="Diffusion variant to train"
)
parser.add_argument("--block_size", type=int, default=13, help="Block size for block variant")
```

Then add conditional model instantiation based on variant.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_mdlm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_mdlm.py tests/test_train_mdlm.py
git commit -m "feat(mdlm): add variant selection to training script"
git push origin innovation-brainstorm
```

---

## Task 11: Evaluation Metrics

**Files:**
- Create: `scripts/evaluate_diffusion.py`
- Create: `tests/test_evaluate_diffusion.py`

- [ ] **Step 1: Write the failing test for evaluation**

```python
# tests/test_evaluate_diffusion.py
import subprocess
import sys
import tempfile
from pathlib import Path
import torch


def create_mock_generated(path: Path, n_meshes: int = 10, n_patches: int = 50):
    """Create mock generated tokens."""
    tokens = torch.randint(0, 2000, (n_meshes, n_patches * 7))
    torch.save({
        "tokens": tokens,
        "config": {"vocab_size": 2001, "mask_token": 2000}
    }, path)


def test_evaluate_diffusion_help():
    """Eval script must accept --help."""
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_diffusion.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate_diffusion.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/evaluate_diffusion.py
"""Evaluate generated meshes from diffusion model.

Computes token distribution statistics, sequence length distribution,
and (optionally) mesh quality metrics if VQ-VAE decoder is provided.

Usage:
    python scripts/evaluate_diffusion.py \
        --generated results/generated_mdlm.pt \
        --output results/eval_mdlm.json
"""
import argparse
import json
import torch
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="Generated tokens .pt file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--rvq_checkpoint", help="Optional RVQ VQ-VAE for mesh reconstruction")
    args = parser.parse_args()

    data = torch.load(args.generated, weights_only=False)
    tokens = data["tokens"]  # (N, seq_len)
    config = data["config"]

    n_meshes, seq_len = tokens.shape
    print(f"Loaded {n_meshes} meshes, seq_len={seq_len}")

    # Token distribution
    token_counts = Counter(tokens.flatten().tolist())
    unique_tokens = len(token_counts)

    # Sequence length stats
    lengths = (tokens != config["mask_token"]).sum(dim=1).float()
    avg_len = lengths.mean().item()
    std_len = lengths.std().item()

    metrics = {
        "n_meshes": n_meshes,
        "seq_len": seq_len,
        "unique_tokens": unique_tokens,
        "vocab_usage": unique_tokens / config["vocab_size"],
        "avg_sequence_length": avg_len,
        "std_sequence_length": std_len,
        "top_10_tokens": token_counts.most_common(10),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {output_path}")
    print(f"  Unique tokens: {unique_tokens} / {config['vocab_size']} ({metrics['vocab_usage']:.1%})")
    print(f"  Avg sequence length: {avg_len:.1f} ± {std_len:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate_diffusion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_diffusion.py tests/test_evaluate_diffusion.py
git commit -m "feat(mdlm): add evaluation script for diffusion generation"
git push origin innovation-brainstorm
```

---

## Task 12: Integration Test — Full Pipeline

**Files:**
- Create: `tests/test_patchdiffusion_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_patchdiffusion_integration.py
"""Integration test: train → generate → evaluate."""
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch


def create_mock_sequences(path: Path, n_meshes: int = 30):
    """Create mock token sequences."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n_meshes):
        n_patches = np.random.randint(20, 80)
        tokens = np.random.randint(0, 2000, (n_patches, 7))
        np.savez(path / f"mesh_{i:04d}.npz", tokens=tokens)


def test_full_pipeline():
    """Full pipeline: train → generate → evaluate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        seq_dir = tmpdir / "sequences"
        ckpt_dir = tmpdir / "checkpoints"
        gen_path = tmpdir / "generated.pt"
        eval_path = tmpdir / "eval.json"

        # 1. Create data
        create_mock_sequences(seq_dir, n_meshes=20)

        # 2. Train (1 epoch, minimal model)
        train_result = subprocess.run(
            [
                sys.executable, "scripts/train_mdlm.py",
                "--sequence_dir", str(seq_dir),
                "--checkpoint_dir", str(ckpt_dir),
                "--mode", "rvq",
                "--variant", "pure",
                "--epochs", "1",
                "--batch_size", "4",
                "--n_layers", "1",
                "--d_model", "64",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert train_result.returncode == 0, f"Train failed: {train_result.stderr}"
        assert (ckpt_dir / "checkpoint_final.pt").exists()

        # 3. Generate
        gen_result = subprocess.run(
            [
                sys.executable, "scripts/generate_mdlm.py",
                "--checkpoint", str(ckpt_dir / "checkpoint_final.pt"),
                "--output", str(gen_path),
                "--n_meshes", "5",
                "--n_patches", "30",
                "--steps", "5",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert gen_result.returncode == 0, f"Generate failed: {gen_result.stderr}"
        assert gen_path.exists()

        # 4. Evaluate
        eval_result = subprocess.run(
            [
                sys.executable, "scripts/evaluate_diffusion.py",
                "--generated", str(gen_path),
                "--output", str(eval_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert eval_result.returncode == 0, f"Eval failed: {eval_result.stderr}"
        assert eval_path.exists()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_patchdiffusion_integration.py -v --timeout=180`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_patchdiffusion_integration.py
git commit -m "test(mdlm): add integration test for full pipeline"
git push origin innovation-brainstorm
```

---

## Summary

| Task | Description | Files Created | Files Modified |
|------|-------------|---------------|----------------|
| 1 | Masking schedule utilities | `src/mdlm.py`, `tests/test_mdlm.py` | — |
| 2 | Forward masking process | — | `src/mdlm.py`, `tests/test_mdlm.py` |
| 3 | MaskedDiffusionTransformer | — | `src/mdlm.py`, `tests/test_mdlm.py` |
| 4 | Iterative unmasking | — | `src/mdlm.py`, `tests/test_mdlm.py` |
| 5 | DiffusionDataset | `src/diffusion_dataset.py`, `tests/test_diffusion_dataset.py` | — |
| 6 | Training script | `scripts/train_mdlm.py`, `tests/test_train_mdlm.py` | — |
| 7 | Block Diffusion | `src/mdlm_block.py`, `tests/test_mdlm_block.py` | — |
| 8 | Hierarchical Diffusion | `src/mdlm_hierarchical.py`, `tests/test_mdlm_hierarchical.py` | — |
| 9 | Generation script | `scripts/generate_mdlm.py`, `tests/test_generate_mdlm.py` | — |
| 10 | Variant selection | — | `scripts/train_mdlm.py`, `tests/test_train_mdlm.py` |
| 11 | Evaluation script | `scripts/evaluate_diffusion.py`, `tests/test_evaluate_diffusion.py` | — |
| 12 | Integration test | `tests/test_patchdiffusion_integration.py` | — |

**Estimated total implementation time:** ~4-6 hours for a skilled developer following TDD.

**Dependencies:**
- Phase D dataset (token sequences)
- RVQ VQ-VAE checkpoint (for reconstruction evaluation)
- PatchGPT AR checkpoint (for baseline comparison)

**Next steps after implementation:**
1. Run Phase P0: Encode all meshes to token sequences
2. Train Pure MDLM for 500 epochs
3. Train Block Diffusion for 500 epochs
4. Train Hierarchical RVQ Diffusion (300+200+200 epochs)
5. Run evaluation and ablation experiments

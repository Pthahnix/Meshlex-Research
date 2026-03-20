# MeshLex Theory-Driven Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement theory-driven MeshLex v3 with curvature-aware codebook, validated by phase transition experiments and Lean4 formalization.

**Architecture:** Three-layer pipeline: (1) Theory experiments to measure phase transitions and power law, (2) Curvature-aware non-uniform codebook based on Gauss-Bonnet bound, (3) Full retraining and evaluation.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, PyG, trimesh, Lean4, matplotlib, scipy

**Scope:**
- ✅ C1: Phase transition experiments (Task 4)
- ✅ C2: Power law + curvature correlation (Task 3, 5)
- ✅ C3: Lean4 formalization setup (Task 8)
- ✅ C4: Curvature-aware codebook ablation (Task 9-11)
- ⏳ C5: Generation evaluation (AR model training deferred to separate plan)

**Out of Scope (follow-up plan):**
- AR model full training (~30h GPU)
- Generation evaluation (FID/COV/MMD)
- Paper writing

---

## File Structure

```
src/
├── curvature.py              # NEW: Discrete Gaussian curvature computation
├── model_curvature.py        # NEW: Curvature-aware codebook model
├── theory_analysis.py        # NEW: Phase transition + power law analysis
└── [existing files...]

tests/
├── test_curvature.py         # NEW: Curvature computation tests
├── test_model_curvature.py   # NEW: Curvature-aware model tests
└── test_theory_analysis.py   # NEW: Theory analysis tests

scripts/
├── run_theory_experiments.py # NEW: Phase transition + power law experiments
├── train_curvature_vqvae.py  # NEW: Train curvature-aware VQ-VAE
└── [existing scripts...]

lean/
├── MeshLex/
│   ├── GaussBonnet.lean      # NEW: Discrete Gauss-Bonnet axiom + Markov bound
│   └── MeshLex.lean          # NEW: Main theorem proof

results/
├── theory_experiments/       # NEW: Phase transition plots, power law fits
└── [existing directories...]
```

---

## Phase 1: Curvature Computation Module

### Task 1: Discrete Gaussian Curvature Computation

**Files:**
- Create: `src/curvature.py`
- Test: `tests/test_curvature.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curvature.py
import numpy as np
import pytest
from src.curvature import compute_discrete_gaussian_curvature, compute_patch_curvature


class TestDiscreteGaussianCurvature:
    """Tests for discrete Gaussian curvature computation."""

    def test_flat_plane_curvature_zero(self):
        """Flat plane should have zero curvature at all vertices."""
        import trimesh
        # Create a flat plane mesh (2x2 grid)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 1, 0], [1, 1, 0], [2, 1, 0],
            [0, 2, 0], [1, 2, 0], [2, 2, 0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 3], [1, 4, 3],
            [1, 2, 4], [2, 5, 4],
            [3, 4, 6], [4, 7, 6],
            [4, 5, 7], [5, 8, 7],
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        K = compute_discrete_gaussian_curvature(mesh)
        # All curvatures should be near zero for a flat plane
        assert np.allclose(K, 0, atol=1e-10)

    def test_cube_corner_curvature_positive(self):
        """Cube corner should have positive curvature (~π/2)."""
        import trimesh
        # Create a cube
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        K = compute_discrete_gaussian_curvature(mesh)
        # Corner vertices should have K = π/2 ≈ 1.57
        # For a cube, 8 corners each with K = π/2
        # Total = 4π ≈ 12.57 (Gauss-Bonnet check)
        total_K = np.sum(K)
        assert np.isclose(total_K, 4 * np.pi, rtol=0.01)

    def test_sphere_curvature_distribution(self):
        """Sphere should have total curvature 4π."""
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        K = compute_discrete_gaussian_curvature(mesh)
        total_K = np.sum(K)
        assert np.isclose(total_K, 4 * np.pi, rtol=0.1)


class TestPatchCurvature:
    """Tests for patch-level curvature computation."""

    def test_patch_curvature_from_mesh(self):
        """Test patch curvature computation from mesh + face indices."""
        import trimesh
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        # All faces as one patch
        patch_faces = np.arange(len(mesh.faces))
        K_patch = compute_patch_curvature(mesh, patch_faces)
        assert isinstance(K_patch, float)
        assert K_patch >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curvature.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.curvature'"

- [ ] **Step 3: Write the implementation**

```python
# src/curvature.py
"""Discrete Gaussian curvature computation for mesh patches.

Implements the angle defect formula:
    K_v = 2π - Σ θ_vf

where θ_vf is the interior angle at vertex v in face f.
"""
import numpy as np
import trimesh
from typing import Tuple


def compute_discrete_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute discrete Gaussian curvature (angle defect) for each vertex.

    Args:
        mesh: A trimesh.Trimesh object.

    Returns:
        K: (V,) array of Gaussian curvature at each vertex.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    n_vertices = len(vertices)

    # Initialize angle defect as 2π for each vertex
    K = np.full(n_vertices, 2 * np.pi)

    # Subtract interior angles from each face
    for face in faces:
        v0, v1, v2 = face

        # Edge vectors
        e01 = vertices[v1] - vertices[v0]
        e02 = vertices[v2] - vertices[v0]
        e12 = vertices[v2] - vertices[v1]

        # Edge lengths
        len01 = np.linalg.norm(e01)
        len02 = np.linalg.norm(e02)
        len12 = np.linalg.norm(e12)

        # Interior angles using law of cosines
        # θ0 is the angle at v0 (between edges e01 and e02)
        cos_theta0 = np.dot(e01, e02) / (len01 * len02 + 1e-12)
        theta0 = np.arccos(np.clip(cos_theta0, -1, 1))

        # θ1 is the angle at v1 (between edges -e01 and e12)
        cos_theta1 = np.dot(-e01, e12) / (len01 * len12 + 1e-12)
        theta1 = np.arccos(np.clip(cos_theta1, -1, 1))

        # θ2 is the angle at v2 (use sum of angles = π)
        theta2 = np.pi - theta0 - theta1

        # Subtract from angle defect
        K[v0] -= theta0
        K[v1] -= theta1
        K[v2] -= theta2

    return K


def compute_patch_curvature(
    mesh: trimesh.Trimesh,
    patch_faces: np.ndarray,
    aggregate: str = "mean"
) -> float:
    """Compute curvature statistic for a patch.

    Args:
        mesh: A trimesh.Trimesh object.
        patch_faces: Array of face indices in the patch.
        aggregate: Aggregation method: 'mean', 'max', or 'median'.

    Returns:
        Scalar curvature value for the patch.
    """
    K = compute_discrete_gaussian_curvature(mesh)

    # Get all vertices in the patch
    patch_vertices = set()
    for face_idx in patch_faces:
        patch_vertices.update(mesh.faces[face_idx])

    if len(patch_vertices) == 0:
        return 0.0

    # Get absolute curvature values for vertices in patch
    patch_K = np.array([np.abs(K[v]) for v in patch_vertices])

    if aggregate == "mean":
        return float(np.mean(patch_K))
    elif aggregate == "max":
        return float(np.max(patch_K))
    elif aggregate == "median":
        return float(np.median(patch_K))
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")


def compute_all_patch_curvatures(
    mesh: trimesh.Trimesh,
    patch_assignments: np.ndarray
) -> np.ndarray:
    """Compute curvature for all patches in a mesh.

    Args:
        mesh: A trimesh.Trimesh object.
        patch_assignments: (F,) array mapping each face to a patch ID.

    Returns:
        (num_patches,) array of patch curvatures.
    """
    n_patches = patch_assignments.max() + 1
    patch_curvatures = np.zeros(n_patches)

    for patch_id in range(n_patches):
        patch_faces = np.where(patch_assignments == patch_id)[0]
        patch_curvatures[patch_id] = compute_patch_curvature(mesh, patch_faces)

    return patch_curvatures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curvature.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/curvature.py tests/test_curvature.py
git commit -m "feat: add discrete Gaussian curvature computation

Implements angle defect formula for mesh vertices and patches.
Includes tests for flat plane, cube corner, and sphere curvature."
```

---

### Task 2: Curvature Binning and Statistics

**Files:**
- Modify: `src/curvature.py`
- Test: `tests/test_curvature.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_curvature.py

class TestCurvatureBinning:
    """Tests for curvature binning functionality."""

    def test_assign_curvature_bin_flat(self):
        """Flat curvature (0.05) should go to bin 0."""
        from src.curvature import assign_curvature_bin
        bin_id = assign_curvature_bin(0.05)
        assert bin_id == 0

    def test_assign_curvature_bin_extreme(self):
        """Extreme curvature (1.5) should go to bin 4."""
        from src.curvature import assign_curvature_bin
        bin_id = assign_curvature_bin(1.5)
        assert bin_id == 4

    def test_assign_curvature_bin_boundaries(self):
        """Test boundary values."""
        from src.curvature import assign_curvature_bin
        assert assign_curvature_bin(0.0) == 0
        assert assign_curvature_bin(0.099) == 0
        assert assign_curvature_bin(0.1) == 1
        assert assign_curvature_bin(0.299) == 1
        assert assign_curvature_bin(0.3) == 2
        assert assign_curvature_bin(0.599) == 2
        assert assign_curvature_bin(0.6) == 3
        assert assign_curvature_bin(0.999) == 3
        assert assign_curvature_bin(1.0) == 4

    def test_compute_bin_distribution(self):
        """Test bin distribution computation."""
        from src.curvature import compute_bin_distribution
        curvatures = np.array([0.01, 0.05, 0.15, 0.25, 0.35, 0.7, 1.2, 0.5])
        distribution = compute_bin_distribution(curvatures)
        # bin 0: [0.01, 0.05] -> 2
        # bin 1: [0.15, 0.25] -> 2
        # bin 2: [0.35, 0.5] -> 2
        # bin 3: [0.7] -> 1
        # bin 4: [1.2] -> 1
        assert distribution[0] == 2
        assert distribution[1] == 2
        assert distribution[2] == 2
        assert distribution[3] == 1
        assert distribution[4] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curvature.py::TestCurvatureBinning -v`
Expected: FAIL with "cannot import name 'assign_curvature_bin'"

- [ ] **Step 3: Add implementation to src/curvature.py**

```python
# Add to src/curvature.py

# Curvature bin boundaries (mutually exclusive intervals)
CURVATURE_BINS = [
    (0.0, 0.1),   # bin 0: flat
    (0.1, 0.3),   # bin 1: mild
    (0.3, 0.6),   # bin 2: medium
    (0.6, 1.0),   # bin 3: sharp
    (1.0, float('inf')),  # bin 4: extreme
]

# Codeword allocation per bin (total = 512)
CODEWORD_ALLOCATION = [200, 130, 100, 52, 30]


def assign_curvature_bin(curvature: float) -> int:
    """Assign a curvature value to a bin.

    Uses mutually exclusive intervals:
        bin 0: 0.0 <= |K| < 0.1  (flat)
        bin 1: 0.1 <= |K| < 0.3  (mild)
        bin 2: 0.3 <= |K| < 0.6  (medium)
        bin 3: 0.6 <= |K| < 1.0  (sharp)
        bin 4: |K| >= 1.0        (extreme)

    Args:
        curvature: Non-negative curvature value.

    Returns:
        Bin index (0-4).
    """
    abs_k = abs(curvature)
    for bin_id, (low, high) in enumerate(CURVATURE_BINS):
        if low <= abs_k < high:
            return bin_id
    return 4  # Last bin (catch-all for >= 1.0)


def compute_bin_distribution(curvatures: np.ndarray) -> np.ndarray:
    """Compute the distribution of curvatures across bins.

    Args:
        curvatures: Array of curvature values.

    Returns:
        (5,) array with count in each bin.
    """
    distribution = np.zeros(5, dtype=np.int64)
    for k in curvatures:
        bin_id = assign_curvature_bin(k)
        distribution[bin_id] += 1
    return distribution


def get_codewords_for_bin(bin_id: int) -> int:
    """Get the number of codewords allocated to a bin.

    Args:
        bin_id: Bin index (0-4).

    Returns:
        Number of codewords for that bin.
    """
    return CODEWORD_ALLOCATION[bin_id]


def get_cumulative_codeword_offset(bin_id: int) -> int:
    """Get the cumulative codeword offset for a bin.

    Args:
        bin_id: Bin index (0-4).

    Returns:
        Starting index for codewords in this bin.
    """
    return sum(CODEWORD_ALLOCATION[:bin_id])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curvature.py::TestCurvatureBinning -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/curvature.py tests/test_curvature.py
git commit -m "feat: add curvature binning with 5-level allocation

Implements mutually exclusive bin intervals:
- bin 0 (flat): [0, 0.1) -> 200 codewords
- bin 1 (mild): [0.1, 0.3) -> 130 codewords
- bin 2 (medium): [0.3, 0.6) -> 100 codewords
- bin 3 (sharp): [0.6, 1.0) -> 52 codewords
- bin 4 (extreme): [1.0, ∞) -> 30 codewords"
```

---

## Phase 2: Theory Analysis Module

### Task 3: Power Law Fitting

**Files:**
- Create: `src/theory_analysis.py`
- Test: `tests/test_theory_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_theory_analysis.py
import numpy as np
import pytest
from src.theory_analysis import fit_power_law, compute_zipf_plot


class TestPowerLawFitting:
    """Tests for power law distribution fitting."""

    def test_fit_power_law_synthetic(self):
        """Test power law fitting on synthetic data."""
        # Generate synthetic power law: f(r) = 1000 * r^(-1.5)
        np.random.seed(42)
        ranks = np.arange(1, 101)
        true_alpha = 1.5
        frequencies = 1000 * ranks ** (-true_alpha)
        frequencies = frequencies.astype(int)

        alpha, r_squared = fit_power_law(frequencies)

        # Should recover alpha close to 1.5
        assert 1.3 < alpha < 1.7, f"Expected alpha ~1.5, got {alpha}"
        assert r_squared > 0.95, f"R² too low: {r_squared}"

    def test_compute_zipf_plot(self):
        """Test Zipf plot computation."""
        frequencies = np.array([1000, 500, 333, 250, 200, 167, 143, 125])
        log_ranks, log_freqs = compute_zipf_plot(frequencies)

        # Check shapes
        assert len(log_ranks) == len(frequencies)
        assert len(log_freqs) == len(frequencies)

        # First point should be log(1) = 0, log(1000)
        assert np.isclose(log_ranks[0], 0.0)
        assert np.isclose(log_freqs[0], np.log(1000))

    def test_fit_power_law_uniform_distribution(self):
        """Uniform distribution should have poor power law fit."""
        uniform_freqs = np.ones(100) * 50
        alpha, r_squared = fit_power_law(uniform_freqs)

        # R² should be low for uniform distribution
        assert r_squared < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_theory_analysis.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# src/theory_analysis.py
"""Theory analysis tools: power law fitting, phase transitions, etc."""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Optional


def fit_power_law(
    frequencies: np.ndarray,
    min_rank: int = 1
) -> Tuple[float, float]:
    """Fit a power law distribution to frequency data.

    Models: f(r) = C * r^(-alpha)

    Args:
        frequencies: Array of frequencies (sorted descending).
        min_rank: Minimum rank to include in fit (default 1).

    Returns:
        alpha: Power law exponent.
        r_squared: Goodness of fit (R²).
    """
    # Filter out zeros
    valid_mask = frequencies > 0
    freqs = frequencies[valid_mask]

    if len(freqs) < 10:
        return 0.0, 0.0

    # Sort descending and create ranks
    freqs = np.sort(freqs)[::-1]
    ranks = np.arange(1, len(freqs) + 1)

    # Apply min_rank filter
    mask = ranks >= min_rank
    ranks = ranks[mask]
    freqs = freqs[mask]

    # Take logs
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    # Linear regression: log(f) = log(C) - alpha * log(r)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks, log_freqs
    )

    # alpha is the negative slope
    alpha = -slope
    r_squared = r_value ** 2

    return alpha, r_squared


def compute_zipf_plot(
    frequencies: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute log-log coordinates for Zipf plot.

    Args:
        frequencies: Array of frequencies.

    Returns:
        log_ranks: Log of ranks (x-axis).
        log_freqs: Log of frequencies (y-axis).
    """
    # Sort descending
    freqs = np.sort(frequencies)[::-1].astype(float)

    # Filter zeros
    valid_mask = freqs > 0
    freqs = freqs[valid_mask]

    ranks = np.arange(1, len(freqs) + 1)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    return log_ranks, log_freqs


def detect_phase_transitions(
    K_values: np.ndarray,
    D_values: np.ndarray
) -> np.ndarray:
    """Detect phase transitions in Rate-Distortion curve.

    A phase transition is where the curve slope changes abruptly.

    Args:
        K_values: Codebook sizes (x-axis).
        D_values: Distortion values (y-axis).

    Returns:
        Array of K values where phase transitions occur.
    """
    # Compute second derivative (curvature)
    # First, interpolate to regular grid
    K_log = np.log(K_values)
    D_log = np.log(D_values + 1e-10)

    # Compute numerical gradient
    grad = np.gradient(D_log, K_log)

    # Phase transitions: where gradient changes significantly
    grad_change = np.abs(np.diff(grad))

    # Normalize
    grad_change_norm = grad_change / (np.max(grad_change) + 1e-10)

    # Threshold: points where normalized gradient change > 0.3
    threshold = 0.3
    transition_indices = np.where(grad_change_norm > threshold)[0]

    # Return K values at transitions
    return K_values[transition_indices + 1]  # +1 because of diff
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_theory_analysis.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/theory_analysis.py tests/test_theory_analysis.py
git commit -m "feat: add power law fitting and phase transition detection

Includes:
- fit_power_law: Linear regression on log-log Zipf plot
- compute_zipf_plot: Generate plotting coordinates
- detect_phase_transitions: Find slope changes in R-D curve"
```

---

### Task 4: Rate-Distortion Experiment Runner

**Files:**
- Create: `scripts/run_theory_experiments.py`
- Create: `results/theory_experiments/` directory

- [ ] **Step 1: Write the experiment script**

```python
# scripts/run_theory_experiments.py
#!/usr/bin/env python
"""Run theory experiments: Rate-Distortion curve and power law analysis.

Phase 1: Train VQ-VAE with varying K and measure reconstruction CD.
Phase 2: Analyze token frequency distribution and fit power law.
Phase 3: Correlate token frequency with patch curvature.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchDataset
from src.trainer import Trainer
from src.curvature import compute_all_patch_curvatures
from src.theory_analysis import fit_power_law, compute_zipf_plot, detect_phase_transitions


def run_rate_distortion_experiment(
    data_path: str,
    K_values: list,
    output_dir: Path,
    epochs: int = 100,
    device: str = "cuda"
):
    """Run Rate-Distortion experiment: train VQ-VAE with different K.

    Args:
        data_path: Path to patch data.
        K_values: List of codebook sizes to try.
        output_dir: Output directory for results.
        epochs: Training epochs per K.
        device: Device to use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"K": [], "D": [], "utilization": []}

    for K in tqdm(K_values, desc="Rate-Distortion sweep"):
        print(f"\n=== Training with K={K} ===")

        # Create model with this K
        model = MeshLexVQVAE(
            in_dim=15,
            hidden_dim=256,
            embed_dim=128,
            codebook_size=K,
            max_vertices=128,
        ).to(device)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_path=data_path,
            val_path=data_path,  # Use same for quick experiment
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            device=device,
        )

        # Train
        history = trainer.train()

        # Record final distortion (reconstruction loss)
        final_D = history["val_loss"][-1]
        results["K"].append(K)
        results["D"].append(final_D)

        # Record codebook utilization
        # Count unique token IDs used in last epoch
        all_indices = []
        with torch.no_grad():
            for batch in trainer.val_loader:
                output = model(batch["x"].to(device), batch["edge_index"].to(device),
                              batch["batch"].to(device), batch["n_vertices"].to(device),
                              batch["gt_vertices"].to(device))
                all_indices.append(output["indices"].cpu().numpy())
        all_indices = np.concatenate(all_indices)
        util = len(np.unique(all_indices)) / K
        results["utilization"].append(util)

        print(f"  K={K}: D={final_D:.4f}, util={util:.1%}")

        # Save intermediate results
        with open(output_dir / "rd_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_power_law_analysis(
    model_path: str,
    data_path: str,
    output_dir: Path,
    device: str = "cuda"
):
    """Analyze token frequency distribution.

    Args:
        model_path: Path to trained model checkpoint.
        data_path: Path to patch data.
        output_dir: Output directory.
        device: Device to use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MeshLexVQVAE(codebook_size=1024).to(device)  # Adjust K as needed
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    dataset = PatchDataset(data_path)

    # Encode all patches and count token frequencies
    token_counts = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Encoding patches"):
            batch = dataset[i]
            # Move to device
            x = batch["x"].unsqueeze(0).to(device)
            edge_index = batch["edge_index"].unsqueeze(0).to(device)
            batch_idx = torch.zeros(1, dtype=torch.long, device=device)
            n_vertices = batch["n_vertices"].unsqueeze(0).to(device)
            gt_vertices = batch["gt_vertices"].unsqueeze(0).to(device)

            # Forward pass to get token indices
            output = model(x, edge_index, batch_idx, n_vertices, gt_vertices)
            token_id = output["indices"].item()

            # Count token frequency
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

    # Fit power law
    frequencies = np.array(sorted(token_counts.values(), reverse=True))
    alpha, r_squared = fit_power_law(frequencies)

    print(f"\nPower law fit: alpha={alpha:.3f}, R²={r_squared:.3f}")

    # Generate Zipf plot
    log_ranks, log_freqs = compute_zipf_plot(frequencies)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(token_counts.keys(), token_counts.values())
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.title("Token Frequency Distribution")

    plt.subplot(1, 2, 2)
    plt.plot(log_ranks, log_freqs, 'o-', markersize=2)
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Frequency)")
    plt.title(f"Zipf Plot (α={alpha:.2f}, R²={r_squared:.2f})")

    plt.tight_layout()
    plt.savefig(output_dir / "power_law_analysis.png", dpi=150)

    # Save results
    results = {
        "alpha": alpha,
        "r_squared": r_squared,
        "frequencies": frequencies.tolist(),
    }
    with open(output_dir / "power_law_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return alpha, r_squared


def main():
    parser = argparse.ArgumentParser(description="Run theory experiments")
    parser.add_argument("--mode", choices=["rd", "powerlaw", "all"], default="all")
    parser.add_argument("--data", type=str, required=True, help="Path to patch data")
    parser.add_argument("--model", type=str, help="Path to trained model (for powerlaw mode)")
    parser.add_argument("--output", type=str, default="results/theory_experiments")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ["rd", "all"]:
        print("=" * 50)
        print("Phase 1: Rate-Distortion Experiment")
        print("=" * 50)
        rd_results = run_rate_distortion_experiment(
            args.data, args.K_values, output_dir / "rd", args.epochs, args.device
        )

        # Detect phase transitions
        K_arr = np.array(rd_results["K"])
        D_arr = np.array(rd_results["D"])
        transitions = detect_phase_transitions(K_arr, D_arr)
        print(f"\nPhase transitions detected at K = {transitions}")

    if args.mode in ["powerlaw", "all"]:
        print("\n" + "=" * 50)
        print("Phase 2: Power Law Analysis")
        print("=" * 50)
        if args.model is None:
            print("ERROR: --model required for powerlaw mode")
            return
        alpha, r2 = run_power_law_analysis(
            args.model, args.data, output_dir / "powerlaw", args.device
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create output directory**

```bash
mkdir -p results/theory_experiments
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_theory_experiments.py
git commit -m "feat: add theory experiment runner script

Includes:
- Rate-Distortion sweep with varying K
- Power law analysis on token frequencies
- Phase transition detection"
```

---

### Task 5: Curvature Correlation Analysis

**Files:**
- Modify: `scripts/run_theory_experiments.py`
- Modify: `src/theory_analysis.py`

- [ ] **Step 1: Add curvature correlation function to theory_analysis.py**

```python
# Add to src/theory_analysis.py

def compute_curvature_frequency_correlation(
    token_frequencies: np.ndarray,
    patch_curvatures: np.ndarray,
    token_assignments: np.ndarray
) -> Dict[str, float]:
    """Correlate token frequency with patch curvature.

    Args:
        token_frequencies: (K,) frequency of each token
        patch_curvatures: (N,) curvature of each patch
        token_assignments: (N,) token ID assigned to each patch

    Returns:
        Dict with correlation statistics.
    """
    # Rank tokens by frequency
    sorted_indices = np.argsort(token_frequencies)[::-1]
    n_tokens = len(token_frequencies)

    # Get average curvature for each token
    token_avg_curvature = np.zeros(n_tokens)
    for tok_id in range(n_tokens):
        mask = token_assignments == tok_id
        if mask.sum() > 0:
            token_avg_curvature[tok_id] = np.mean(patch_curvatures[mask])

    # Group by frequency rank
    top_10_pct = int(n_tokens * 0.1)
    middle_40_pct = int(n_tokens * 0.5)
    bottom_50_pct = n_tokens

    results = {
        "top_10_avg_curvature": float(np.mean(
            [token_avg_curvature[i] for i in sorted_indices[:top_10_pct]]
        )),
        "middle_40_avg_curvature": float(np.mean(
            [token_avg_curvature[i] for i in sorted_indices[top_10_pct:middle_40_pct]]
        )),
        "bottom_50_avg_curvature": float(np.mean(
            [token_avg_curvature[i] for i in sorted_indices[middle_40_pct:]]
        )),
        "spearman_correlation": float(stats.spearmanr(
            token_frequencies, token_avg_curvature
        )[0]),
    }
    return results
```

- [ ] **Step 2: Add correlation analysis mode to run_theory_experiments.py**

```python
# Add mode to scripts/run_theory_experiments.py
from scipy import stats

def run_curvature_correlation_analysis(
    model_path: str,
    data_path: str,
    output_dir: Path,
    device: str = "cuda"
):
    """Run curvature-frequency correlation analysis.

    Generates table matching spec Section 4.3:
    - Top 10% (highest freq) → avg curvature ~0 (flat)
    - Middle 40% → avg curvature small positive
    - Bottom 50% → avg curvature large positive
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MeshLexVQVAE(codebook_size=1024).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    dataset = PatchDataset(data_path)

    # Collect token assignments and patch curvatures
    token_assignments = []
    patch_curvatures = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Collecting data"):
            batch = dataset[i]

            x = batch["x"].unsqueeze(0).to(device)
            edge_index = batch["edge_index"].unsqueeze(0).to(device)
            batch_idx = torch.zeros(1, dtype=torch.long, device=device)
            n_vertices = batch["n_vertices"].unsqueeze(0).to(device)
            gt_vertices = batch["gt_vertices"].unsqueeze(0).to(device)

            # Get token assignment
            output = model(x, edge_index, batch_idx, n_vertices, gt_vertices)
            token_id = output["indices"].item()
            token_assignments.append(token_id)

            # Get patch curvature (from precomputed or compute on-the-fly)
            if "curvature" in batch:
                patch_curvatures.append(batch["curvature"].item())
            else:
                # Compute on-the-fly if not precomputed
                mesh_path = dataset.metadata[i]["mesh_path"]
                face_indices = dataset.metadata[i]["face_indices"]
                mesh = trimesh.load(mesh_path)
                K = compute_patch_curvature(mesh, face_indices)
                patch_curvatures.append(K)

    # Convert to arrays
    token_assignments = np.array(token_assignments)
    patch_curvatures = np.array(patch_curvatures)

    # Compute token frequencies
    n_tokens = model.codebook.K
    token_frequencies = np.zeros(n_tokens)
    for tok_id in token_assignments:
        token_frequencies[tok_id] += 1

    # Compute correlation
    from src.theory_analysis import compute_curvature_frequency_correlation
    results = compute_curvature_frequency_correlation(
        token_frequencies, patch_curvatures, token_assignments
    )

    # Print results table
    print("\n" + "=" * 60)
    print("Curvature-Frequency Correlation Results")
    print("=" * 60)
    print(f"{'Token Rank':<20} {'Avg Curvature':<15}")
    print("-" * 60)
    print(f"{'Top 10% (highest freq)':<20} {results['top_10_avg_curvature']:<15.4f}")
    print(f"{'Middle 40%':<20} {results['middle_40_avg_curvature']:<15.4f}")
    print(f"{'Bottom 50%':<20} {results['bottom_50_avg_curvature']:<15.4f}")
    print("-" * 60)
    print(f"Spearman correlation: {results['spearman_correlation']:.4f}")

    # Save results
    with open(output_dir / "curvature_correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
```

- [ ] **Step 3: Commit**

```bash
git add src/theory_analysis.py scripts/run_theory_experiments.py
git commit -m "feat: add curvature-frequency correlation analysis"
```

---

### Task 6: Cross-Dataset Universality Experiment

**Files:**
- Modify: `scripts/run_theory_experiments.py`

- [ ] **Step 1: Add universality experiment function**

```python
# Add to scripts/run_theory_experiments.py

def run_universality_experiment(
    train_data_path: str,  # Objaverse (for reference)
    test_data_path: str,   # ShapeNet
    model_path: str,
    output_dir: Path,
    device: str = "cuda"
):
    """Test if power law distribution transfers across datasets.

    Per spec Section 4.2:
    1. Load model trained on Objaverse
    2. Encode ShapeNet patches (zero fine-tuning)
    3. Compare power law exponents (α)
    4. If α values are similar → evidence of universality
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Objaverse-trained model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = MeshLexVQVAE(codebook_size=1024).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load ShapeNet test data
    print(f"Loading test data from {test_data_path}")
    test_dataset = PatchDataset(test_data_path)

    # Encode ShapeNet patches and count token frequencies
    test_token_counts = {}

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Encoding test patches"):
            batch = test_dataset[i]
            x = batch["x"].unsqueeze(0).to(device)
            edge_index = batch["edge_index"].unsqueeze(0).to(device)
            batch_idx = torch.zeros(1, dtype=torch.long, device=device)
            n_vertices = batch["n_vertices"].unsqueeze(0).to(device)
            gt_vertices = batch["gt_vertices"].unsqueeze(0).to(device)

            output = model(x, edge_index, batch_idx, n_vertices, gt_vertices)
            token_id = output["indices"].item()
            test_token_counts[token_id] = test_token_counts.get(token_id, 0) + 1

    # Fit power law on test data
    test_frequencies = np.array(sorted(test_token_counts.values(), reverse=True))
    test_alpha, test_r2 = fit_power_law(test_frequencies)

    # Load training data power law (from checkpoint history or recompute)
    train_alpha = checkpoint.get("power_law_alpha", None)
    if train_alpha is None:
        # Recpute from training data
        train_dataset = PatchDataset(train_data_path)
        train_token_counts = {}
        with torch.no_grad():
            for i in tqdm(range(len(train_dataset)), desc="Encoding train patches"):
                batch = train_dataset[i]
                x = batch["x"].unsqueeze(0).to(device)
                edge_index = batch["edge_index"].unsqueeze(0).to(device)
                batch_idx = torch.zeros(1, dtype=torch.long, device=device)
                n_vertices = batch["n_vertices"].unsqueeze(0).to(device)
                gt_vertices = batch["gt_vertices"].unsqueeze(0).to(device)
                output = model(x, edge_index, batch_idx, n_vertices, gt_vertices)
                token_id = output["indices"].item()
                train_token_counts[token_id] = train_token_counts.get(token_id, 0) + 1
        train_frequencies = np.array(sorted(train_token_counts.values(), reverse=True))
        train_alpha, train_r2 = fit_power_law(train_frequencies)
    else:
        train_r2 = checkpoint.get("power_law_r2", 0)

    # Compare
    print("\n" + "=" * 60)
    print("Universality Experiment Results")
    print("=" * 60)
    print(f"{'Dataset':<20} {'α':<10} {'R²':<10}")
    print("-" * 60)
    print(f"{'Objaverse (train)':<20} {train_alpha:<10.3f} {train_r2:<10.3f}")
    print(f"{'ShapeNet (test)':<20} {test_alpha:<10.3f} {test_r2:<10.3f}")
    print("-" * 60)

    # Check universality criterion
    alpha_diff = abs(test_alpha - train_alpha)
    if alpha_diff < 0.2:
        print(f"✓ α difference = {alpha_diff:.3f} < 0.2 → UNIVERSALITY SUPPORTED")
    else:
        print(f"✗ α difference = {alpha_diff:.3f} >= 0.2 → Universality NOT supported")

    # Save results
    results = {
        "train_alpha": float(train_alpha),
        "train_r2": float(train_r2),
        "test_alpha": float(test_alpha),
        "test_r2": float(test_r2),
        "alpha_difference": float(alpha_diff),
        "universality_supported": alpha_diff < 0.2,
    }
    with open(output_dir / "universality_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
```

- [ ] **Step 2: Add universality mode to argparse**

```python
# Add to argparse in main()
parser.add_argument("--test-data", type=str,
                    help="Test data for universality experiment (e.g., ShapeNet)")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_theory_experiments.py
git commit -m "feat: add cross-dataset universality experiment"
```

---

## Phase 3: Curvature-Aware Codebook Model

### Task 7: Curvature-Aware VQ-VAE Model

**Files:**
- Create: `src/model_curvature.py`
- Test: `tests/test_model_curvature.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_curvature.py
import pytest
import torch
import numpy as np
from src.model_curvature import CurvatureAwareVQVAE


class TestCurvatureAwareVQVAE:
    """Tests for curvature-aware VQ-VAE model."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 4
        n_faces = 32
        n_vertices = 16

        return {
            "x": torch.randn(n_faces * batch_size, 15),
            "edge_index": torch.randint(0, n_faces * batch_size, (2, 100)),
            "batch": torch.repeat_interleave(torch.arange(batch_size), n_faces),
            "n_vertices": torch.tensor([n_vertices] * batch_size),
            "gt_vertices": torch.randn(batch_size, 128, 3),
            "curvature": torch.tensor([0.05, 0.2, 0.4, 0.8]),  # Different bins
        }

    def test_model_forward(self, sample_batch):
        """Test forward pass."""
        model = CurvatureAwareVQVAE(
            in_dim=15,
            hidden_dim=256,
            embed_dim=128,
            total_codewords=512,
            max_vertices=128,
        )

        output = model(
            sample_batch["x"],
            sample_batch["edge_index"],
            sample_batch["batch"],
            sample_batch["n_vertices"],
            sample_batch["gt_vertices"],
            sample_batch["curvature"],
        )

        assert "total_loss" in output
        assert "recon_loss" in output
        assert output["total_loss"].item() >= 0

    def test_curvature_bin_routing(self, sample_batch):
        """Test that patches are routed to correct codebooks."""
        model = CurvatureAwareVQVAE(
            in_dim=15,
            hidden_dim=256,
            embed_dim=128,
            total_codewords=512,
            max_vertices=128,
        )

        # Check bin assignment
        curvatures = torch.tensor([0.05, 0.15, 0.35, 0.7, 1.2])
        expected_bins = [0, 1, 2, 3, 4]

        for curv, expected_bin in zip(curvatures, expected_bins):
            bin_id = model.assign_bin(curv.item())
            assert bin_id == expected_bin, f"Curvature {curv} -> bin {bin_id}, expected {expected_bin}"

    def test_codebook_sizes(self):
        """Test that total codewords match allocation."""
        model = CurvatureAwareVQVAE(total_codewords=512)

        # Check that sum of sub-codebook sizes equals total
        total = sum(cb.K for cb in model.codebooks)
        assert total == 512, f"Total codewords: {total}, expected 512"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_curvature.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# src/model_curvature.py
"""Curvature-Aware VQ-VAE with non-uniform codebook allocation.

Key innovation: Instead of a single uniform codebook, we use 5 separate
sub-codebooks, one per curvature bin. Each sub-codebook has a different size
based on the theoretical allocation:
    bin 0 (flat):     200 codewords
    bin 1 (mild):     130 codewords
    bin 2 (medium):   100 codewords
    bin 3 (sharp):     52 codewords
    bin 4 (extreme):   30 codewords
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.model import PatchEncoder, PatchDecoder, SimVQCodebook
from src.curvature import (
    CURVATURE_BINS, CODEWORD_ALLOCATION,
    assign_curvature_bin, get_cumulative_codeword_offset
)
from src.losses import chamfer_distance


class CurvatureAwareVQVAE(nn.Module):
    """VQ-VAE with curvature-aware non-uniform codebook.

    Architecture:
        - Single encoder (shared across all bins)
        - 5 sub-codebooks (one per curvature bin)
        - Single decoder (shared across all bins)

    During forward:
        1. Encode patch to latent z
        2. Determine curvature bin from input curvature
        3. Quantize using the appropriate sub-codebook
        4. Decode quantized z to reconstruct vertices
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        total_codewords: int = 512,
        max_vertices: int = 128,
        lambda_commit: float = 1.0,
        lambda_embed: float = 1.0,
        num_kv_tokens: int = 4,
    ):
        super().__init__()

        # Shared encoder
        self.encoder = PatchEncoder(in_dim, hidden_dim, embed_dim)

        # Curvature-aware sub-codebooks
        self.codebooks = nn.ModuleList([
            SimVQCodebook(K=n_codewords, dim=embed_dim, use_rotation=False)
            for n_codewords in CODEWORD_ALLOCATION
        ])
        self.total_codewords = total_codewords

        # Shared decoder
        self.decoder = PatchDecoder(embed_dim, max_vertices, num_kv_tokens=num_kv_tokens)

        self.max_vertices = max_vertices
        self.lambda_commit = lambda_commit
        self.lambda_embed = lambda_embed

    def assign_bin(self, curvature: float) -> int:
        """Assign a curvature value to a bin (0-4)."""
        return assign_curvature_bin(curvature)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        curvatures: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with curvature-aware quantization.

        Args:
            x: (N_total, in_dim) face features
            edge_index: (2, E_total) edges
            batch: (N_total,) batch assignment
            n_vertices: (B,) vertices per patch
            gt_vertices: (B, max_vertices, 3) ground truth vertices
            curvatures: (B,) curvature value per patch

        Returns:
            Dict with losses and outputs.
        """
        # Encode
        z = self.encoder(x, edge_index, batch)  # (B, embed_dim)

        # Quantize with curvature-aware routing
        batch_size = z.shape[0]
        z_q = torch.zeros_like(z)
        indices = torch.zeros(batch_size, dtype=torch.long, device=z.device)

        # Group by bin for efficiency
        bin_assignments = [self.assign_bin(c.item()) for c in curvatures]

        for bin_id in range(5):
            mask = torch.tensor([b == bin_id for b in bin_assignments], device=z.device)
            if not mask.any():
                continue

            z_bin = z[mask]
            z_q_bin, indices_bin = self.codebooks[bin_id](z_bin)

            # Offset indices to global space
            offset = get_cumulative_codeword_offset(bin_id)
            indices[mask] = indices_bin + offset
            z_q[mask] = z_q_bin

        # Decode
        recon = self.decoder(z_q, n_vertices)

        # Compute losses
        mask = torch.arange(self.max_vertices, device=x.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        recon_loss = chamfer_distance(recon, gt_vertices, mask)

        # Commitment loss (aggregate from all codebooks)
        commit_loss = torch.tensor(0.0, device=z.device)
        embed_loss = torch.tensor(0.0, device=z.device)

        for bin_id in range(5):
            mask = torch.tensor([b == bin_id for b in bin_assignments], device=z.device)
            if not mask.any():
                continue
            z_bin = z[mask]
            _, idx_bin = self.codebooks[bin_id](z_bin)
            c_loss, e_loss = self.codebooks[bin_id].compute_loss(z_bin, idx_bin)
            commit_loss = commit_loss + c_loss * mask.sum()
            embed_loss = embed_loss + e_loss * mask.sum()

        commit_loss = commit_loss / batch_size
        embed_loss = embed_loss / batch_size

        total_loss = recon_loss + self.lambda_commit * commit_loss + self.lambda_embed * embed_loss

        return {
            "recon_vertices": recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "embed_loss": embed_loss,
            "indices": indices,
            "z": z,
            "z_q": z_q,
            "bin_assignments": bin_assignments,
        }

    def encode_only(self, x, edge_index, batch, curvatures):
        """Encode to indices only (for inference)."""
        z = self.encoder(x, edge_index, batch)
        batch_size = z.shape[0]

        indices = torch.zeros(batch_size, dtype=torch.long, device=z.device)
        bin_assignments = [self.assign_bin(c.item()) for c in curvatures]

        for bin_id in range(5):
            mask = torch.tensor([b == bin_id for b in bin_assignments], device=z.device)
            if not mask.any():
                continue

            _, indices_bin = self.codebooks[bin_id](z[mask])
            offset = get_cumulative_codeword_offset(bin_id)
            indices[mask] = indices_bin + offset

        return indices, bin_assignments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_curvature.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model_curvature.py tests/test_model_curvature.py
git commit -m "feat: add curvature-aware VQ-VAE with non-uniform codebook

Key features:
- 5 sub-codebooks with sizes [200, 130, 100, 52, 30]
- Curvature-based routing to appropriate codebook
- Shared encoder/decoder, separate quantizers"
```

---

## Phase 4: Lean4 Formalization

### Task 8: Lean4 Project Setup and Proof Skeleton

**Note:** This task sets up the project structure and proof skeleton. The actual proof completion (removing `sorry` statements) is a separate, substantial effort estimated at 2-3 weeks per the spec.

**Files:**
- Create: `lean/MeshLex/` directory structure
- Create: `lean/MeshLex/lakefile.lean`
- Create: `lean/MeshLex/MeshLex.lean`
- Create: `lean/MeshLex/README.md`

- [ ] **Step 1: Create Lean4 project structure**

```bash
mkdir -p lean/MeshLex
```

- [ ] **Step 2: Create lakefile.lean**

```lean
-- lean/MeshLex/lakefile.lean
import Lake
open Lake DSL

package «MeshLex» where
  version := v!"0.1.0"

lean_lib «MeshLex» where
  -- Add library configuration options here

@[default_target]
lean_exe «meshlex» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "master"
```

- [ ] **Step 3: Create main theorem file**

```lean
-- lean/MeshLex/MeshLex.lean
/-
MeshLex Theory Formalization

Main theorem: High-curvature vertex bound from Gauss-Bonnet

For a closed triangular mesh M, the number of vertices with
angle defect > κ is bounded by 2π|χ(M)| / κ, independent of
the number of faces in the mesh.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Order.Ring.Lemmas
import Mathlib.Data.Nat.Basic

namespace MeshLex

/-! ## Discrete Gauss-Bonnet Axiom

We take the discrete Gauss-Bonnet theorem as an axiom, citing
the classical result from Descartes (1637) and the modern
polyhedral formulation by Banchoff (1970).

For a closed triangulated 2-manifold M:
  Σ K_v = 2π · χ(M)

where K_v is the angle defect at vertex v.
-/

axiom discreteGaussBonnet {M : Type*} [Fintype M]
    (K : M → ℝ) (χ : ℤ) (h_closed : True) :
    (∑ v : M, K v) = 2 * Real.pi * χ

/-! ## Finite Sum Markov Inequality

For a finite set with non-negative values, the count of
elements exceeding κ is bounded by the total sum divided by κ.
-/

theorem finite_sum_markov {α : Type*} (s : Finset α)
    (f : α → ℝ) (hf : ∀ x ∈ s, 0 ≤ f x) (κ : ℝ) (hκ : κ > 0) :
    (s.filter (fun x => f x > κ)).card ≤ ⌈(s.sum f) / κ⌉₊ := by
  -- Get the filtered set
  let filtered := s.filter (fun x => f x > κ)
  -- Key insight: for each x in filtered, f x > κ
  -- So: sum over filtered > κ * |filtered|
  have h_filter_sum : κ * filtered.card < (filtered.sum f) := by
    calc κ * filtered.card
        = (filtered.sum (fun _ => κ)) := by
          simp [Finset.sum_const, Finset.card_filter]
        _ < filtered.sum f := by
          apply Finset.sum_lt_sum
          · intro x hx
            simp only [Finset.mem_filter] at hx
            exact hx.2
          · intro x hx
            simp only [Finset.mem_filter] at hx
            linarith [hx.2]
  -- And filtered.sum f ≤ s.sum f (sub-monoid property)
  have h_sum_le : filtered.sum f ≤ s.sum f := by
    apply Finset.sum_le_sum_of_subset
    exact Finset.filter_subset _ _
  -- Combine: κ * |filtered| < s.sum f
  calc filtered.card ≤ ⌈(filtered.sum f) / κ⌉₊ := by
      have : κ > 0 := hκ
      have h_div : (filtered.sum f) / κ ≥ filtered.card := by
        calc (filtered.sum f) / κ
            > κ * filtered.card / κ := by
              apply div_lt_div_of_lt_right h_filter_sum
              linarith
            _ = filtered.card := by ring
        sorry -- This direction needs adjustment
      sorry
    _ ≤ ⌈(s.sum f) / κ⌉₊ := by
      apply Nat.ceil_le_ceil
      · exact div_nonneg (Finset.sum_nonneg hf) hκ.le
      · exact div_le_div_of_le_right h_sum_le hκ.le

/-! ## Main Theorem: High Curvature Vertex Bound -/

theorem high_curvature_vertex_bound {M : Type*} [Fintype M]
    (K : M → ℝ) (χ : ℤ) (hχ : χ > 0) (κ : ℝ) (hκ : κ > 0)
    (h_nonneg : ∀ v : M, 0 ≤ K v) :
    (Finset.univ.filter (fun v : M => K v > κ)).card ≤
    Int.natAbs ⌈(2 * Real.pi * χ) / κ⌉ := by
  -- Apply Gauss-Bonnet
  have h_sum : ∑ v : M, K v = 2 * Real.pi * χ := by
    apply discreteGaussBonnet K χ trivial
  -- Apply Markov inequality
  have h_markov := finite_sum_markov Finset.univ K h_nonneg κ hκ
  -- Substitute the sum
  have h_sum' : (Finset.univ.sum K) / κ = (2 * Real.pi * χ) / κ := by
    rw [h_sum]
  calc (Finset.univ.filter (fun v => K v > κ)).card
      ≤ ⌈(Finset.univ.sum K) / κ⌉₊ := h_markov
    _ ≤ Int.natAbs ⌈(2 * Real.pi * χ) / κ⌉ := by
      rw [h_sum']
      sorry -- Coercion details

end MeshLex
```

- [ ] **Step 4: Create README for Lean4**

```markdown
# MeshLex Lean4 Formalization

This directory contains Lean4 formalization of the theoretical results
in the MeshLex paper.

## Building

```bash
cd lean/MeshLex
lake build
```

## Main Theorem

`high_curvature_vertex_bound`: For a closed triangulated 2-manifold M,
the number of vertices with angle defect exceeding κ is bounded by
`2π|χ(M)| / κ`, independent of the number of faces.

## Axioms

- `discreteGaussBonnet`: The polyhedral Gauss-Bonnet theorem, citing
  Descartes (1637) and Banchoff (1970).
```

- [ ] **Step 5: Commit**

```bash
git add lean/
git commit -m "feat: add Lean4 formalization structure

Includes:
- Project setup with Mathlib dependency
- Discrete Gauss-Bonnet as axiom
- Finite sum Markov inequality (in progress)
- Main theorem skeleton"
```

---

## Phase 5: Training and Evaluation Scripts

### Task 9: Curvature-Aware Training Script

**Files:**
- Create: `scripts/train_curvature_vqvae.py`

- [ ] **Step 1: Write training script**

```python
# scripts/train_curvature_vqvae.py
#!/usr/bin/env python
"""Train curvature-aware VQ-VAE.

Usage:
    python scripts/train_curvature_vqvae.py \
        --data data/patches/full \
        --output data/checkpoints/curvature_vqvae \
        --epochs 200 \
        --batch-size 32
"""
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_curvature import CurvatureAwareVQVAE
from src.patch_dataset import PatchDataset
from src.curvature import compute_patch_curvature
import trimesh


class CurvatureAwareDataset(torch.utils.data.Dataset):
    """Dataset that includes precomputed curvature values."""

    def __init__(self, patch_dir: str):
        self.base_dataset = PatchDataset(patch_dir)
        self.curvatures = self._precompute_curvatures(patch_dir)

    def _precompute_curvatures(self, patch_dir: str) -> list:
        """Precompute curvature for all patches."""
        print("Precomputing patch curvatures...")
        curvatures = []

        for i in tqdm(range(len(self.base_dataset))):
            sample = self.base_dataset[i]
            # Get mesh path from metadata
            mesh_path = self.base_dataset.metadata[i]["mesh_path"]
            face_indices = self.base_dataset.metadata[i]["face_indices"]

            # Load mesh and compute curvature
            mesh = trimesh.load(mesh_path)
            K_patch = compute_patch_curvature(mesh, face_indices)
            curvatures.append(K_patch)

        return curvatures

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample["curvature"] = torch.tensor(self.curvatures[idx])
        return sample


def train(args):
    """Main training loop."""
    device = torch.device(args.device)

    # Create dataset
    print(f"Loading dataset from {args.data}")
    dataset = CurvatureAwareDataset(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = CurvatureAwareVQVAE(
        in_dim=15,
        hidden_dim=256,
        embed_dim=128,
        total_codewords=512,
        max_vertices=128,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    history = {"train_loss": [], "recon_loss": [], "commit_loss": []}

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_commit = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Move to device
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch"].to(device)
            n_vertices = batch["n_vertices"].to(device)
            gt_vertices = batch["gt_vertices"].to(device)
            curvatures = batch["curvature"].to(device)

            # Forward
            optimizer.zero_grad()
            output = model(
                x, edge_index, batch_idx,
                n_vertices, gt_vertices, curvatures
            )

            # Backward
            output["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Log
            epoch_loss += output["total_loss"].item()
            epoch_recon += output["recon_loss"].item()
            epoch_commit += output["commit_loss"].item()

            pbar.set_postfix({
                "loss": f"{output['total_loss'].item():.4f}",
                "recon": f"{output['recon_loss'].item():.4f}",
            })

        scheduler.step()

        # Record history
        n_batches = len(dataloader)
        history["train_loss"].append(epoch_loss / n_batches)
        history["recon_loss"].append(epoch_recon / n_batches)
        history["commit_loss"].append(epoch_commit / n_batches)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
    }, output_dir / "checkpoint_final.pt")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train curvature-aware VQ-VAE")
    parser.add_argument("--data", type=str, required=True, help="Path to patch data")
    parser.add_argument("--output", type=str, default="data/checkpoints/curvature_vqvae")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_curvature_vqvae.py
git commit -m "feat: add curvature-aware VQ-VAE training script

Features:
- Precomputes patch curvatures
- Uses curvature-aware dataset wrapper
- Saves checkpoints and training history"
```

---

### Task 10: Uniform Baseline Training

**Files:**
- Use existing: `scripts/train.py` or `scripts/train_rvq.py`

- [ ] **Step 1: Train uniform 512 baseline**

```bash
# Train uniform baseline with K=512
python scripts/train.py \
    --data data/patches/full \
    --output data/checkpoints/uniform_512 \
    --codebook-size 512 \
    --epochs 200 \
    --device cuda
```

- [ ] **Step 2: Train uniform 1024 upper bound (optional)**

```bash
# Train uniform upper bound with K=1024
python scripts/train.py \
    --data data/patches/full \
    --output data/checkpoints/uniform_1024 \
    --codebook-size 1024 \
    --epochs 200 \
    --device cuda
```

- [ ] **Step 3: Commit checkpoint locations**

Document checkpoint paths in `data/checkpoints/checkpoint_manifest.json`:
```json
{
    "curvature_512": "data/checkpoints/curvature_vqvae/checkpoint_final.pt",
    "uniform_512": "data/checkpoints/uniform_512/checkpoint_final.pt",
    "uniform_1024": "data/checkpoints/uniform_1024/checkpoint_final.pt"
}
```

---

### Task 11: Reconstruction Evaluation

**Files:**
- Create: `scripts/evaluate_curvature_vqvae.py`

- [ ] **Step 1: Write evaluation script**

```python
# scripts/evaluate_curvature_vqvae.py
#!/usr/bin/env python
"""Evaluate curvature-aware VQ-VAE.

Compares:
1. Uniform baseline (512 tokens)
2. Curvature-aware (512 tokens, non-uniform)
3. Upper bound (1024 tokens, uniform)

Reports CD, F-Score, Normal Consistency, Codebook Utilization.
"""
import argparse
import sys
from pathlib import Path
import json

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import MeshLexVQVAE
from src.model_curvature import CurvatureAwareVQVAE
from src.patch_dataset import PatchDataset
from src.metrics import compute_all_metrics


def evaluate_model(model, dataset, device, model_name="Model"):
    """Evaluate a model on a dataset."""
    model.eval()

    all_cd = []
    all_nc = []
    all_fscore = []

    codebook_usage = {i: 0 for i in range(model.total_codewords)}

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
            sample = dataset[i]

            # Move to device
            x = sample["x"].unsqueeze(0).to(device)
            edge_index = sample["edge_index"].unsqueeze(0).to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            n_vertices = sample["n_vertices"].unsqueeze(0).to(device)
            gt_vertices = sample["gt_vertices"].unsqueeze(0).to(device)

            # Forward
            if isinstance(model, CurvatureAwareVQVAE):
                curv = sample["curvature"].unsqueeze(0).to(device)
                output = model(x, edge_index, batch, n_vertices, gt_vertices, curv)
            else:
                output = model(x, edge_index, batch, n_vertices, gt_vertices)

            # Compute metrics
            recon = output["recon_vertices"][0].cpu().numpy()
            gt = gt_vertices[0].cpu().numpy()

            metrics = compute_all_metrics(recon, gt)
            all_cd.append(metrics["chamfer_distance"])
            all_nc.append(metrics["normal_consistency"])
            all_fscore.append(metrics["f_score_01"])

            # Track codebook usage
            idx = output["indices"][0].item()
            codebook_usage[idx] = codebook_usage.get(idx, 0) + 1

    # Compute statistics
    results = {
        "chamfer_distance_mean": float(np.mean(all_cd)),
        "chamfer_distance_std": float(np.std(all_cd)),
        "normal_consistency_mean": float(np.mean(all_nc)),
        "f_score_01_mean": float(np.mean(all_fscore)),
        "codebook_utilization": len([v for v in codebook_usage.values() if v > 0]) / model.total_codewords,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate curvature-aware VQ-VAE")
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--uniform-512", type=str, help="Path to uniform baseline checkpoint")
    parser.add_argument("--curvature-512", type=str, help="Path to curvature-aware checkpoint")
    parser.add_argument("--uniform-1024", type=str, help="Path to upper bound checkpoint")
    parser.add_argument("--output", type=str, default="results/evaluation")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load dataset
    print(f"Loading dataset from {args.data}")
    dataset = PatchDataset(args.data)

    results = {}

    # Evaluate uniform baseline
    if args.uniform_512:
        print("\n=== Evaluating Uniform Baseline (512) ===")
        model = MeshLexVQVAE(codebook_size=512).to(device)
        model.load_state_dict(torch.load(args.uniform_512, map_location=device)["model_state_dict"])
        results["uniform_512"] = evaluate_model(model, dataset, device, "Uniform-512")

    # Evaluate curvature-aware
    if args.curvature_512:
        print("\n=== Evaluating Curvature-Aware (512) ===")
        model = CurvatureAwareVQVAE(total_codewords=512).to(device)
        model.load_state_dict(torch.load(args.curvature_512, map_location=device)["model_state_dict"])
        results["curvature_512"] = evaluate_model(model, dataset, device, "Curvature-512")

    # Evaluate upper bound
    if args.uniform_1024:
        print("\n=== Evaluating Upper Bound (1024) ===")
        model = MeshLexVQVAE(codebook_size=1024).to(device)
        model.load_state_dict(torch.load(args.uniform_1024, map_location=device)["model_state_dict"])
        results["uniform_1024"] = evaluate_model(model, dataset, device, "Uniform-1024")

    # Print comparison table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'CD (↓)':<12} {'NC (↑)':<12} {'F-Score (↑)':<12} {'Util (↑)':<10}")
    print("-" * 80)

    for name, r in results.items():
        print(f"{name:<20} {r['chamfer_distance_mean']:<12.4f} {r['normal_consistency_mean']:<12.4f} "
              f"{r['f_score_01_mean']:<12.4f} {r['codebook_utilization']:<10.1%}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/evaluate_curvature_vqvae.py
git commit -m "feat: add curvature-aware VQ-VAE evaluation script

Evaluates and compares:
- Uniform baseline (512 tokens)
- Curvature-aware (512 tokens)
- Upper bound (1024 tokens)

Reports CD, NC, F-Score, and codebook utilization."
```

---

## Summary

### Task Count

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: Curvature Computation | 2 (Task 1-2) | ~4 hours |
| Phase 2: Theory Analysis | 4 (Task 3-6) | ~10 hours |
| Phase 3: Curvature-Aware Model | 1 (Task 7) | ~4 hours |
| Phase 4: Lean4 Formalization | 1 (Task 8) | ~2 weeks (setup) |
| Phase 5: Training & Evaluation | 3 (Task 9-11) | ~6 hours |

**Total**: 11 tasks, ~3-4 weeks (including Lean4 setup)

### Dependencies

```
Task 1 (Curvature) ──┬──> Task 7 (Model)
                     │
Task 2 (Binning) ────┘

Task 3 (Power Law) ────> Task 4 (RD Experiment)
                              │
                              ├──> Task 5 (Curvature Correlation)
                              │
                              └──> Task 6 (Universality)
                                       │
                                       v
Task 8 (Lean4) ───────────────────────────────────> Paper writing

Task 9 (Curvature Training) ──┐
                               ├──> Task 11 (Evaluation)
Task 10 (Baseline Training) ──┘
```

### Success Criteria

| Criterion | How to Verify |
|-----------|---------------|
| C1: Phase transitions | Run `run_theory_experiments.py --mode rd`, check for ≥2 transition points |
| C2: Power law fit | R² > 0.9 on Zipf plot |
| C3: Lean4 proof | `lake build` succeeds |
| C4: Curvature-aware > baseline | CD_curvature_512 < CD_uniform_512 |
| C5: Generation quality | Deferred to follow-up plan (AR model training required) |

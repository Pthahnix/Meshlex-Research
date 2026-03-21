"""Phase 3b: Theory-driven analysis orchestrator.

Three sub-analyses:
  3b-1: Codebook K scaling analysis (K=512, 1024, 2048)
  3b-2: VQ method comparison (SimVQ vs VanillaVQ vs EMA)
  3b-3: Curvature-frequency correlation (Gauss-Bonnet prediction)

Usage:
    # Run all sub-analyses:
    PYTHONPATH=. python scripts/run_fullscale_analysis.py \
        --output_dir results/fullscale_theory

    # Run individual sub-analyses:
    PYTHONPATH=. python scripts/run_fullscale_analysis.py \
        --analysis k_ablation \
        --seq_dirs_k512 data/sequences/rvq_full_pca_k512 \
        --seq_dirs_k1024 data/sequences/rvq_full_pca \
        --seq_dirs_k2048 data/sequences/rvq_full_pca_k2048 \
        --output_dir results/fullscale_theory

    PYTHONPATH=. python scripts/run_fullscale_analysis.py \
        --analysis vq_comparison \
        --simvq_seq_dir data/sequences/rvq_full_pca \
        --vanilla_seq_dir data/sequences/rvq_full_vanilla \
        --ema_seq_dir data/sequences/rvq_full_ema \
        --output_dir results/fullscale_theory

    PYTHONPATH=. python scripts/run_fullscale_analysis.py \
        --analysis curvature \
        --seq_dir data/sequences/rvq_full_pca \
        --patch_dir data/patches_full/seen_train \
        --output_dir results/fullscale_theory
"""
import argparse
import json
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ──────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────

def gini_coefficient(values):
    """Compute Gini coefficient for a 1-D array of non-negative values."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    cumsum = np.cumsum(sorted_vals)
    if cumsum[-1] == 0:
        return 0.0
    return float(
        (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * cumsum[-1]))
        - (n + 1) / n
    )


def fit_distributions(token_freqs):
    """Fit Zipf (power law) and lognormal to token frequency data.

    Args:
        token_freqs: 1-D array of counts per token ID (length = codebook size).

    Returns:
        Dict with zipf_alpha, zipf_r2, lognormal_sigma, lognormal_mu,
        entropy_bits, entropy_ratio, gini, utilization.
    """
    freqs_sorted = np.sort(token_freqs)[::-1]
    nonzero = freqs_sorted[freqs_sorted > 0]
    if len(nonzero) < 10:
        return None

    ranks = np.arange(1, len(nonzero) + 1)

    # ── Zipf: log(freq) = -alpha * log(rank) + c ──
    log_ranks = np.log(ranks)
    log_freqs = np.log(nonzero.astype(float))
    slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_freqs)
    zipf_alpha = -slope
    zipf_r2 = r_value ** 2

    # Zipf KS statistic
    pred_freqs = np.exp(intercept) * ranks ** slope
    zipf_ks, _ = stats.ks_2samp(
        nonzero / nonzero.sum(), pred_freqs / pred_freqs.sum()
    )

    # ── Lognormal fit ──
    try:
        shape, loc, scale = stats.lognorm.fit(nonzero.astype(float), floc=0)
        lognorm_ks, lognorm_p = stats.kstest(
            nonzero.astype(float), "lognorm", args=(shape, loc, scale)
        )
        lognorm_sigma = float(shape)
        lognorm_mu = float(np.log(scale))
    except Exception:
        lognorm_ks, lognorm_p = 1.0, 0.0
        lognorm_sigma, lognorm_mu = 0.0, 0.0

    # ── Entropy ──
    total = freqs_sorted.sum()
    probs = freqs_sorted / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(len(token_freqs))

    return {
        "zipf_alpha": round(float(zipf_alpha), 4),
        "zipf_r2": round(float(zipf_r2), 4),
        "zipf_ks": round(float(zipf_ks), 4),
        "lognormal_sigma": round(float(lognorm_sigma), 4),
        "lognormal_mu": round(float(lognorm_mu), 4),
        "lognormal_ks": round(float(lognorm_ks), 4),
        "lognormal_p": round(float(lognorm_p), 4),
        "better_fit": "Lognormal" if lognorm_ks < zipf_ks else "Zipf",
        "entropy_bits": round(float(entropy), 4),
        "entropy_ratio": round(float(entropy / max_entropy), 4) if max_entropy > 0 else 0.0,
        "gini": round(float(gini_coefficient(freqs_sorted)), 4),
        "utilization": round(float(np.sum(freqs_sorted > 0) / len(freqs_sorted)), 4),
    }


def collect_tokens_from_dir(seq_dir, n_levels=3):
    """Load all sequence NPZs from a directory and return per-level token lists.

    Returns:
        Dict mapping level index (0,1,2) to list of ints.
    """
    seq_dir = Path(seq_dir)
    npz_files = sorted(seq_dir.glob("*_sequence.npz"))
    print(f"  Loading {len(npz_files)} sequence files from {seq_dir}")

    all_tokens = {level: [] for level in range(n_levels)}
    for f in npz_files:
        data = np.load(str(f))
        tokens = data["tokens"]  # (N, 3)
        for level in range(min(n_levels, tokens.shape[1] if tokens.ndim == 2 else 1)):
            if tokens.ndim == 2:
                all_tokens[level].extend(tokens[:, level].tolist())
            else:
                all_tokens[0].extend(tokens.tolist())
    return all_tokens


# ──────────────────────────────────────────────
# 3b-1: Codebook K Scaling Analysis
# ──────────────────────────────────────────────

def k_ablation_analysis(seq_dirs, k_values, output_dir):
    """Compare token distributions across codebook sizes.

    Args:
        seq_dirs: Dict mapping K (int) -> sequence directory path (str).
        k_values: List of K values, e.g. [512, 1024, 2048].
        output_dir: Root output directory.

    Returns:
        Dict of results keyed by str(K).
    """
    print("\n" + "=" * 60)
    print("Phase 3b-1: Codebook K Scaling Analysis")
    print("=" * 60)

    out = Path(output_dir) / "k_ablation"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for K in k_values:
        seq_dir = seq_dirs[K]
        if not Path(seq_dir).exists():
            print(f"  WARNING: {seq_dir} does not exist, skipping K={K}")
            continue

        all_tokens = collect_tokens_from_dir(seq_dir)

        # Fit per level
        level_results = {}
        for level in range(3):
            if not all_tokens[level]:
                continue
            freqs = np.bincount(all_tokens[level], minlength=K)
            fit = fit_distributions(freqs)
            if fit is not None:
                level_results[f"L{level + 1}"] = fit

        if level_results:
            results[str(K)] = level_results
            sigmas = [
                level_results.get(f"L{l + 1}", {}).get("lognormal_sigma", float("nan"))
                for l in range(3)
            ]
            print(
                f"  K={K}: sigma=[{sigmas[0]:.3f}, {sigmas[1]:.3f}, {sigmas[2]:.3f}]"
            )

        gc.collect()

    if not results:
        print("  ERROR: No results computed (missing sequence dirs?)")
        return {}

    # ── Save JSON ──
    with open(out / "k_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot: K vs sigma / alpha / entropy ratio ──
    available_k = [K for K in k_values if str(K) in results]
    if len(available_k) < 2:
        print("  WARNING: Need at least 2 K values to plot, skipping plot")
        return results

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["lognormal_sigma", "zipf_alpha", "entropy_ratio"]
    titles = ["Lognormal Sigma", "Zipf Alpha", "Entropy Ratio"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for level in ["L1", "L2", "L3"]:
            vals = []
            ks_valid = []
            for K in available_k:
                v = results[str(K)].get(level, {}).get(metric)
                if v is not None:
                    vals.append(v)
                    ks_valid.append(K)
            if vals:
                axes[i].plot(ks_valid, vals, "o-", label=level, linewidth=2, markersize=6)
        axes[i].set_xlabel("Codebook K")
        axes[i].set_ylabel(metric)
        axes[i].set_title(title)
        axes[i].legend()
        axes[i].set_xscale("log", base=2)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Phase 3b-1: Codebook K Scaling", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "k_scaling.png", dpi=150)
    plt.close()

    # ── Plot: Rank-frequency curves for each K ──
    fig, axes = plt.subplots(1, len(available_k), figsize=(6 * len(available_k), 5))
    if len(available_k) == 1:
        axes = [axes]

    for ax, K in zip(axes, available_k):
        all_tokens = collect_tokens_from_dir(seq_dirs[K])
        for level in range(3):
            if not all_tokens[level]:
                continue
            freqs = np.bincount(all_tokens[level], minlength=K)
            freqs_sorted = np.sort(freqs)[::-1]
            nonzero = freqs_sorted[freqs_sorted > 0]
            ranks = np.arange(1, len(nonzero) + 1)
            ax.loglog(ranks, nonzero, label=f"L{level + 1}", alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"K={K}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Rank-Frequency Curves by Codebook Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "rank_frequency_curves.png", dpi=150)
    plt.close()

    print(f"  K ablation results saved to {out}")
    return results


# ──────────────────────────────────────────────
# 3b-2: VQ Method Comparison
# ──────────────────────────────────────────────

def vq_comparison_analysis(method_seq_dirs, output_dir, codebook_size=1024):
    """Compare token distributions across VQ methods.

    Directly tests FM1: Is lognormal a SimVQ artifact?

    Args:
        method_seq_dirs: Dict mapping method name -> sequence directory path.
            Expected keys: "simvq", "vanilla", "ema".
        output_dir: Root output directory.
        codebook_size: Codebook size (default 1024).

    Returns:
        Dict of results keyed by method name.
    """
    print("\n" + "=" * 60)
    print("Phase 3b-2: VQ Method Comparison (FM1 Test)")
    print("=" * 60)

    out = Path(output_dir) / "vq_comparison"
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method, seq_dir in method_seq_dirs.items():
        if not Path(seq_dir).exists():
            print(f"  WARNING: {seq_dir} does not exist, skipping {method}")
            continue

        print(f"\n  Analyzing {method}...")
        all_tokens = collect_tokens_from_dir(seq_dir)

        level_results = {}
        for level in range(3):
            if not all_tokens[level]:
                continue
            freqs = np.bincount(all_tokens[level], minlength=codebook_size)
            fit = fit_distributions(freqs)
            if fit is not None:
                level_results[f"L{level + 1}"] = fit

        if level_results:
            all_results[method] = level_results
            for lname, lr in level_results.items():
                print(
                    f"    {lname}: sigma={lr['lognormal_sigma']:.3f}, "
                    f"alpha={lr['zipf_alpha']:.3f}, fit={lr['better_fit']}"
                )

        gc.collect()

    if not all_results:
        print("  ERROR: No results computed")
        return {}

    # ── Save JSON ──
    with open(out / "vq_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── FM1 verdict ──
    fm1_verdict = _compute_fm1_verdict(all_results)
    with open(out / "fm1_verdict.json", "w") as f:
        json.dump(fm1_verdict, f, indent=2)
    print(f"\n  FM1 Verdict: {fm1_verdict['verdict']}")

    # ── Plot: method comparison bar chart ──
    methods_available = list(all_results.keys())
    metrics = ["lognormal_sigma", "zipf_alpha", "entropy_ratio", "gini"]
    titles = ["Lognormal Sigma", "Zipf Alpha", "Entropy Ratio", "Gini"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(methods_available))
    bar_width = 0.25

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for li, level in enumerate(["L1", "L2", "L3"]):
            vals = [
                all_results[m].get(level, {}).get(metric, 0)
                for m in methods_available
            ]
            axes[i].bar(x + li * bar_width, vals, bar_width,
                        label=level, alpha=0.85)
        axes[i].set_xticks(x + bar_width)
        axes[i].set_xticklabels(methods_available, fontsize=10)
        axes[i].set_title(title)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Phase 3b-2: VQ Method Distribution Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "vq_comparison.png", dpi=150)
    plt.close()

    # ── Plot: overlay rank-frequency curves ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"simvq": "steelblue", "vanilla": "darkorange", "ema": "seagreen"}

    for level in range(3):
        ax = axes[level]
        for method in methods_available:
            seq_dir = method_seq_dirs[method]
            all_tokens = collect_tokens_from_dir(seq_dir)
            if not all_tokens[level]:
                continue
            freqs = np.bincount(all_tokens[level], minlength=codebook_size)
            freqs_sorted = np.sort(freqs)[::-1]
            nonzero = freqs_sorted[freqs_sorted > 0]
            ranks = np.arange(1, len(nonzero) + 1)
            ax.loglog(ranks, nonzero, label=method,
                      color=colors.get(method, None), alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"L{level + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Rank-Frequency Curves by VQ Method", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "rank_frequency_by_method.png", dpi=150)
    plt.close()

    print(f"  VQ comparison results saved to {out}")
    return all_results


def _compute_fm1_verdict(all_results):
    """Compute FM1 (SimVQ artifact) verdict from VQ method comparison results.

    FM1 hypothesis: lognormal distribution is a SimVQ artifact.
    If vanilla/EMA also show lognormal, FM1 is rejected -> geometry-driven.
    """
    verdict = {
        "hypothesis": "FM1: Lognormal is a SimVQ artifact",
        "test": "Compare lognormal sigma across SimVQ, VanillaVQ, EMAVQ",
    }

    methods = list(all_results.keys())
    sigmas_per_method = {}
    fits_per_method = {}

    for method in methods:
        method_sigmas = []
        method_fits = []
        for level in ["L1", "L2", "L3"]:
            lr = all_results[method].get(level, {})
            if "lognormal_sigma" in lr:
                method_sigmas.append(lr["lognormal_sigma"])
            if "better_fit" in lr:
                method_fits.append(lr["better_fit"])
        if method_sigmas:
            sigmas_per_method[method] = round(float(np.mean(method_sigmas)), 4)
        fits_per_method[method] = method_fits

    verdict["mean_sigma_per_method"] = sigmas_per_method
    verdict["fits_per_method"] = fits_per_method

    # If all methods show lognormal, FM1 is rejected
    all_lognorm = all(
        "Lognormal" in fits
        for fits in fits_per_method.values()
    )

    if "simvq" in sigmas_per_method and len(sigmas_per_method) >= 2:
        other_sigmas = [
            v for k, v in sigmas_per_method.items() if k != "simvq"
        ]
        simvq_sigma = sigmas_per_method["simvq"]
        mean_other = np.mean(other_sigmas)
        delta = abs(simvq_sigma - mean_other)

        verdict["simvq_sigma"] = simvq_sigma
        verdict["mean_other_sigma"] = round(float(mean_other), 4)
        verdict["sigma_delta"] = round(float(delta), 4)

        if delta < 0.1 and all_lognorm:
            verdict["verdict"] = "REJECTED — Lognormal is GEOMETRY-DRIVEN (not SimVQ artifact)"
        elif delta < 0.2:
            verdict["verdict"] = "WEAK — Small difference, likely geometry-driven"
        else:
            verdict["verdict"] = "SUPPORTED — SimVQ may induce different distribution shape"
    else:
        verdict["verdict"] = "INCONCLUSIVE — insufficient methods to compare"

    return verdict


# ──────────────────────────────────────────────
# 3b-3: Curvature-Frequency Correlation
# ──────────────────────────────────────────────

def angle_deficit_curvature(vertices, faces):
    """Compute discrete Gaussian curvature via angle deficit method.

    For each interior vertex: K_v = 2pi - sum(angles at v).
    Returns mean absolute curvature across all vertices.
    """
    n_verts = len(vertices)
    angle_sum = np.zeros(n_verts)

    for face in faces:
        for i in range(3):
            v0 = vertices[face[i]]
            v1 = vertices[face[(i + 1) % 3]]
            v2 = vertices[face[(i + 2) % 3]]
            e1 = v1 - v0
            e2 = v2 - v0
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            if norm1 < 1e-10 or norm2 < 1e-10:
                continue
            cos_angle = np.clip(np.dot(e1, e2) / (norm1 * norm2), -1, 1)
            angle_sum[face[i]] += np.arccos(cos_angle)

    curvature = 2 * np.pi - angle_sum
    return float(np.mean(np.abs(curvature)))


def curvature_frequency_analysis(patch_dir, seq_dir, output_dir,
                                 codebook_size=1024, max_meshes=5000):
    """Compute curvature per patch, map to token, correlate with frequency.

    Tests the Gauss-Bonnet theoretical prediction:
      - High-frequency tokens -> low curvature (flat patches)
      - Low-frequency tokens -> high curvature (curved patches)

    Args:
        patch_dir: Directory containing patch NPZ files (may have subdirs).
        seq_dir: Directory containing sequence NPZ files.
        output_dir: Root output directory.
        codebook_size: Codebook size (default 1024).
        max_meshes: Max meshes to process (for speed).

    Returns:
        Dict with correlation results.
    """
    print("\n" + "=" * 60)
    print("Phase 3b-3: Curvature-Frequency Correlation")
    print("=" * 60)

    out = Path(output_dir) / "curvature"
    out.mkdir(parents=True, exist_ok=True)
    patch_dir = Path(patch_dir)
    seq_dir = Path(seq_dir)

    if not seq_dir.exists():
        print(f"  ERROR: {seq_dir} does not exist")
        return {}

    # ── Step 1: Collect token frequencies (L1) ──
    token_counts = np.zeros(codebook_size, dtype=np.int64)
    seq_files = sorted(seq_dir.glob("*_sequence.npz"))
    print(f"  Found {len(seq_files)} sequence files")

    for sf in seq_files:
        data = np.load(str(sf))
        tokens = data["tokens"]
        l1 = tokens[:, 0] if tokens.ndim == 2 else tokens
        for t in l1:
            token_counts[t] += 1

    total_tokens = int(token_counts.sum())
    print(f"  Total L1 tokens: {total_tokens}")

    # ── Step 2: Compute curvature per patch, map to token ──
    token_curvatures = defaultdict(list)
    processed = 0
    skipped = 0

    for sf in seq_files[:max_meshes]:
        mesh_id = sf.stem.replace("_sequence", "")
        data = np.load(str(sf))
        tokens = data["tokens"]
        l1_tokens = tokens[:, 0] if tokens.ndim == 2 else tokens

        # Find corresponding patch files (may be in subdirs)
        patch_files = sorted(patch_dir.glob(f"**/{mesh_id}_patch_*.npz"))
        if not patch_files:
            skipped += 1
            continue

        for i, pf in enumerate(patch_files):
            if i >= len(l1_tokens):
                break
            try:
                pd = np.load(str(pf))
                # Try local_vertices first, fall back to vertices
                if "local_vertices" in pd:
                    verts = pd["local_vertices"]
                elif "vertices" in pd:
                    verts = pd["vertices"]
                else:
                    continue
                faces = pd["faces"]
                if len(faces) < 1 or len(verts) < 3:
                    continue
                curv = angle_deficit_curvature(verts, faces)
                if np.isfinite(curv):
                    token_curvatures[int(l1_tokens[i])].append(curv)
            except Exception:
                continue

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{min(len(seq_files), max_meshes)} meshes")

    print(f"  Processed {processed} meshes, skipped {skipped} (no patches found)")

    # ── Step 3: Compute mean curvature per token ──
    token_mean_curv = np.zeros(codebook_size)
    token_std_curv = np.zeros(codebook_size)
    token_n_samples = np.zeros(codebook_size, dtype=int)

    for tok in range(codebook_size):
        curvs = token_curvatures[tok]
        if curvs:
            token_mean_curv[tok] = np.mean(curvs)
            token_std_curv[tok] = np.std(curvs)
            token_n_samples[tok] = len(curvs)

    # ── Step 4: Correlation ──
    min_samples = 10
    valid = (token_counts > 0) & (token_n_samples >= min_samples)
    n_valid = int(valid.sum())
    print(f"  Valid tokens (freq>0 & n_samples>={min_samples}): {n_valid}/{codebook_size}")

    if n_valid < 10:
        print("  ERROR: Too few valid tokens for correlation")
        results = {
            "error": "Too few valid tokens",
            "n_valid": n_valid,
            "n_processed": processed,
        }
        with open(out / "curvature_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    log_freq = np.log(token_counts[valid].astype(float) + 1)
    mean_curv = token_mean_curv[valid]

    rho, p_value = stats.spearmanr(log_freq, mean_curv)
    pearson_r, pearson_p = stats.pearsonr(log_freq, mean_curv)

    print(f"  Spearman rho: {rho:.4f} (p={p_value:.2e})")
    print(f"  Pearson r:    {pearson_r:.4f} (p={pearson_p:.2e})")

    # Gauss-Bonnet prediction: negative correlation (high freq = flat)
    if rho < -0.1 and p_value < 0.05:
        prediction_result = "CONFIRMED — High-freq tokens have lower curvature"
    elif rho < 0 and p_value < 0.05:
        prediction_result = "WEAK_CONFIRMATION — Weak negative correlation"
    elif p_value >= 0.05:
        prediction_result = "INCONCLUSIVE — Not statistically significant"
    else:
        prediction_result = "REJECTED — Positive correlation (unexpected)"

    results = {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_value), 6),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "n_valid_tokens": n_valid,
        "n_meshes_processed": processed,
        "n_meshes_skipped": skipped,
        "total_curvature_samples": sum(len(v) for v in token_curvatures.values()),
        "prediction": "Negative correlation (high freq = low curvature)",
        "result": prediction_result,
    }

    with open(out / "curvature_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot 1: scatter (log freq vs mean curvature) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    sc = ax.scatter(log_freq, mean_curv, alpha=0.4, s=15, c="steelblue",
                    edgecolors="none")
    # Linear fit line
    z = np.polyfit(log_freq, mean_curv, 1)
    p = np.poly1d(z)
    x_line = np.linspace(log_freq.min(), log_freq.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
            label=f"Linear fit (slope={z[0]:.4f})")
    ax.set_xlabel("log(Token Frequency + 1)")
    ax.set_ylabel("Mean |Gaussian Curvature|")
    ax.set_title(f"Curvature vs Frequency\nSpearman rho={rho:.3f}, p={p_value:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: histogram of per-token mean curvature ──
    ax = axes[1]
    ax.hist(mean_curv, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Mean |Gaussian Curvature|")
    ax.set_ylabel("Token Count")
    ax.set_title(f"Curvature Distribution (n={n_valid} tokens)")
    ax.grid(True, alpha=0.3)

    # ── Plot 3: curvature by frequency quartile ──
    ax = axes[2]
    freq_order = np.argsort(log_freq)
    n_q = len(freq_order) // 4
    quartile_labels = ["Q1 (lowest freq)", "Q2", "Q3", "Q4 (highest freq)"]
    quartile_curvs = []
    for qi in range(4):
        start = qi * n_q
        end = (qi + 1) * n_q if qi < 3 else len(freq_order)
        indices = freq_order[start:end]
        quartile_curvs.append(mean_curv[indices])

    bp = ax.boxplot(quartile_curvs, labels=quartile_labels, patch_artist=True)
    colors_q = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]
    for patch, color in zip(bp["boxes"], colors_q):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean |Gaussian Curvature|")
    ax.set_title("Curvature by Frequency Quartile")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=15)

    plt.suptitle("Phase 3b-3: Curvature-Frequency Correlation", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "curvature_frequency.png", dpi=150)
    plt.close()

    # ── Plot 4: curvature std vs frequency (heteroscedasticity check) ──
    std_curv = token_std_curv[valid]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_freq, std_curv, alpha=0.4, s=15, c="darkorange", edgecolors="none")
    ax.set_xlabel("log(Token Frequency + 1)")
    ax.set_ylabel("Std of |Gaussian Curvature|")
    ax.set_title("Curvature Variability vs Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "curvature_std_vs_freq.png", dpi=150)
    plt.close()

    print(f"  Curvature analysis saved to {out}")
    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b: Theory-driven analysis (K ablation, VQ comparison, curvature)"
    )
    parser.add_argument(
        "--analysis",
        choices=["k_ablation", "vq_comparison", "curvature", "all"],
        default="all",
        help="Which sub-analysis to run (default: all)",
    )
    parser.add_argument("--output_dir", default="results/fullscale_theory")
    parser.add_argument("--codebook_size", type=int, default=1024)

    # K ablation args
    parser.add_argument("--seq_dirs_k512", default="data/sequences/rvq_full_pca_k512",
                        help="Sequence dir for K=512")
    parser.add_argument("--seq_dirs_k1024", default="data/sequences/rvq_full_pca",
                        help="Sequence dir for K=1024")
    parser.add_argument("--seq_dirs_k2048", default="data/sequences/rvq_full_pca_k2048",
                        help="Sequence dir for K=2048")

    # VQ comparison args
    parser.add_argument("--simvq_seq_dir", default="data/sequences/rvq_full_pca",
                        help="Sequence dir for SimVQ")
    parser.add_argument("--vanilla_seq_dir", default="data/sequences/rvq_full_vanilla",
                        help="Sequence dir for VanillaVQ")
    parser.add_argument("--ema_seq_dir", default="data/sequences/rvq_full_ema",
                        help="Sequence dir for EMAVQ")

    # Curvature args
    parser.add_argument("--seq_dir", default="data/sequences/rvq_full_pca",
                        help="Sequence dir for curvature analysis")
    parser.add_argument("--patch_dir", default="data/patches_full/seen_train",
                        help="Patch dir for curvature analysis")
    parser.add_argument("--max_meshes", type=int, default=5000,
                        help="Max meshes for curvature analysis")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 3b-1: K ablation ──
    if args.analysis in ("k_ablation", "all"):
        seq_dirs = {
            512: args.seq_dirs_k512,
            1024: args.seq_dirs_k1024,
            2048: args.seq_dirs_k2048,
        }
        results["k_ablation"] = k_ablation_analysis(
            seq_dirs, [512, 1024, 2048], args.output_dir
        )

    # ── 3b-2: VQ method comparison ──
    if args.analysis in ("vq_comparison", "all"):
        method_seq_dirs = {
            "simvq": args.simvq_seq_dir,
            "vanilla": args.vanilla_seq_dir,
            "ema": args.ema_seq_dir,
        }
        results["vq_comparison"] = vq_comparison_analysis(
            method_seq_dirs, args.output_dir, args.codebook_size
        )

    # ── 3b-3: Curvature-frequency ──
    if args.analysis in ("curvature", "all"):
        results["curvature"] = curvature_frequency_analysis(
            args.patch_dir, args.seq_dir, args.output_dir,
            args.codebook_size, args.max_meshes
        )

    # ── Save combined summary ──
    with open(output_dir / "phase3b_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PHASE 3b COMPLETE")
    print("=" * 60)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

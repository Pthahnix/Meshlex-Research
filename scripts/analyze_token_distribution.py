"""Analyze RVQ token frequency distribution: Zipf vs lognormal fitting.

Loads encoded sequence NPZs, extracts token IDs (L1, L2, L3),
fits Zipf power-law and lognormal distributions, generates visualizations.

Usage:
    python scripts/analyze_token_distribution.py \
        --seq_dir data/sequences/rvq_lvis \
        --output_dir results/token_distribution
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def load_all_tokens(seq_dir: str) -> dict[str, np.ndarray]:
    """Load token IDs from all sequence NPZs.

    Returns dict with keys 'L1', 'L2', 'L3', each (N_total,) array.
    """
    seq_dir = Path(seq_dir)
    npz_files = sorted(seq_dir.glob("*_sequence.npz"))
    print(f"Found {len(npz_files)} sequence files")

    all_tokens = {"L1": [], "L2": [], "L3": []}
    for f in npz_files:
        data = np.load(str(f))
        tokens = data["tokens"]  # (N_patches, 3) for RVQ 3-level
        if tokens.ndim == 1:
            # Single level
            all_tokens["L1"].append(tokens)
        else:
            all_tokens["L1"].append(tokens[:, 0])
            all_tokens["L2"].append(tokens[:, 1])
            all_tokens["L3"].append(tokens[:, 2])

    result = {}
    for k, v in all_tokens.items():
        if v:
            result[k] = np.concatenate(v)
    return result


def compute_frequency(tokens: np.ndarray, codebook_size: int = 1024):
    """Compute rank-frequency data from token array."""
    counts = np.bincount(tokens, minlength=codebook_size)
    # Sort descending
    sorted_counts = np.sort(counts)[::-1]
    # Remove zeros
    nonzero_mask = sorted_counts > 0
    sorted_counts = sorted_counts[nonzero_mask]
    ranks = np.arange(1, len(sorted_counts) + 1)
    freqs = sorted_counts / sorted_counts.sum()
    return ranks, freqs, counts


def fit_zipf(ranks, freqs):
    """Fit Zipf power law: f(r) = C * r^(-alpha)."""
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
    alpha = -slope
    C = np.exp(intercept)

    # Predicted
    pred_freqs = C * ranks ** (-alpha)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(freqs, pred_freqs / pred_freqs.sum())

    return {
        "alpha": float(alpha),
        "C": float(C),
        "r_squared": float(r_value ** 2),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "std_err": float(std_err),
        "pred_freqs": pred_freqs,
    }


def fit_lognormal(counts: np.ndarray):
    """Fit lognormal distribution to non-zero token counts."""
    nonzero = counts[counts > 0].astype(float)
    # Fit lognormal
    shape, loc, scale = stats.lognorm.fit(nonzero, floc=0)
    sigma = shape
    mu = np.log(scale)

    # KS test
    ks_stat, ks_p = stats.kstest(nonzero, "lognorm", args=(shape, loc, scale))

    return {
        "sigma": float(sigma),
        "mu": float(mu),
        "loc": float(loc),
        "scale": float(scale),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


def compute_entropy(freqs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    f = freqs[freqs > 0]
    return float(-np.sum(f * np.log2(f)))


def compute_gini(counts: np.ndarray) -> float:
    """Gini coefficient for measuring inequality."""
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    return float((2.0 * np.sum((np.arange(1, n + 1) * sorted_counts)) / (n * cumulative[-1])) - (n + 1) / n)


def plot_rank_frequency(ranks, freqs, zipf_fit, level_name, output_dir):
    """Plot rank-frequency on log-log with Zipf fit line."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(ranks, freqs, s=8, alpha=0.6, label="Observed", color="steelblue")

    # Zipf fit line
    alpha = zipf_fit["alpha"]
    C = zipf_fit["C"]
    fit_line = C * ranks ** (-alpha)
    ax.plot(ranks, fit_line, "r-", linewidth=2,
            label=f"Zipf fit: α={alpha:.3f} (R²={zipf_fit['r_squared']:.4f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Rank-Frequency Plot — {level_name} Tokens", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"rank_frequency_{level_name}.png", dpi=150)
    plt.close()


def plot_histogram(counts, lognorm_fit, level_name, output_dir):
    """Plot frequency histogram with lognormal fit."""
    nonzero = counts[counts > 0].astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.hist(nonzero, bins=50, density=True, alpha=0.6, color="steelblue", label="Observed")

    # Lognormal fit
    x = np.linspace(nonzero.min(), nonzero.max(), 200)
    pdf = stats.lognorm.pdf(x, lognorm_fit["sigma"], lognorm_fit["loc"], lognorm_fit["scale"])
    ax.plot(x, pdf, "r-", linewidth=2,
            label=f"Lognormal (σ={lognorm_fit['sigma']:.3f}, μ={lognorm_fit['mu']:.3f})")

    ax.set_xlabel("Token Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Count Distribution — {level_name} Tokens", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"histogram_{level_name}.png", dpi=150)
    plt.close()


def plot_comparison(all_results, output_dir):
    """Plot 3-level comparison of rank-frequency."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"L1": "steelblue", "L2": "darkorange", "L3": "seagreen"}

    for i, (level, res) in enumerate(all_results.items()):
        ax = axes[i]
        ranks = res["ranks"]
        freqs = res["freqs"]
        zipf = res["zipf"]

        ax.scatter(ranks, freqs, s=6, alpha=0.5, color=colors[level], label="Observed")
        fit_line = zipf["C"] * ranks ** (-zipf["alpha"])
        ax.plot(ranks, fit_line, "r-", linewidth=2,
                label=f"α={zipf['alpha']:.3f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{level} (Entropy={res['entropy']:.2f} bits)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("RVQ Token Distribution: 3-Level Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_3levels.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ccdf(all_results, output_dir):
    """Plot complementary CDF (survival function) for heavy-tail analysis."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = {"L1": "steelblue", "L2": "darkorange", "L3": "seagreen"}

    for level, res in all_results.items():
        counts = res["counts"]
        nonzero = np.sort(counts[counts > 0])[::-1]
        ccdf = np.arange(1, len(nonzero) + 1) / len(nonzero)
        ax.plot(nonzero, ccdf, label=f"{level} (α={res['zipf']['alpha']:.3f})",
                color=colors[level], linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token Count", fontsize=12)
    ax.set_ylabel("P(X ≥ x)", fontsize=12)
    ax.set_title("Complementary CDF — Heavy Tail Analysis", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ccdf_heavy_tail.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokens
    tokens = load_all_tokens(args.seq_dir)
    print(f"Loaded levels: {list(tokens.keys())}")
    for k, v in tokens.items():
        print(f"  {k}: {len(v)} tokens")

    all_results = {}

    for level, tok_arr in tokens.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {level}...")
        print(f"{'='*50}")

        ranks, freqs, counts = compute_frequency(tok_arr, args.codebook_size)
        print(f"  Active codes: {len(ranks)} / {args.codebook_size}")
        print(f"  Top-10 tokens cover: {freqs[:10].sum()*100:.1f}%")
        print(f"  Top-50 tokens cover: {freqs[:50].sum()*100:.1f}%")

        # Fit Zipf
        zipf = fit_zipf(ranks, freqs)
        print(f"  Zipf α = {zipf['alpha']:.4f} (R² = {zipf['r_squared']:.4f})")
        print(f"  Zipf KS stat = {zipf['ks_stat']:.4f}, p = {zipf['ks_p']:.4e}")

        # Fit lognormal
        lognorm = fit_lognormal(counts)
        print(f"  Lognormal σ = {lognorm['sigma']:.4f}, μ = {lognorm['mu']:.4f}")
        print(f"  Lognormal KS stat = {lognorm['ks_stat']:.4f}, p = {lognorm['ks_p']:.4e}")

        # Entropy and Gini
        entropy = compute_entropy(freqs)
        gini = compute_gini(counts)
        print(f"  Entropy = {entropy:.4f} bits (max = {np.log2(args.codebook_size):.2f})")
        print(f"  Gini = {gini:.4f}")

        all_results[level] = {
            "ranks": ranks,
            "freqs": freqs,
            "counts": counts,
            "zipf": zipf,
            "lognorm": lognorm,
            "entropy": entropy,
            "gini": gini,
        }

        # Individual plots
        plot_rank_frequency(ranks, freqs, zipf, level, output_dir)
        plot_histogram(counts, lognorm, level, output_dir)

    # Comparison plots
    if len(all_results) > 1:
        plot_comparison(all_results, output_dir)
        plot_ccdf(all_results, output_dir)

    # Determine which fit is better per level
    print(f"\n{'='*60}")
    print("SUMMARY: Distribution Classification")
    print(f"{'='*60}")

    summary = {}
    for level, res in all_results.items():
        zipf_r2 = res["zipf"]["r_squared"]
        zipf_ks = res["zipf"]["ks_stat"]
        lognorm_ks = res["lognorm"]["ks_stat"]

        # Better fit = lower KS statistic
        if zipf_ks < lognorm_ks:
            better = "Zipf (power-law)"
        else:
            better = "Lognormal"

        alpha = res["zipf"]["alpha"]
        if alpha > 0.5 and zipf_r2 > 0.9:
            tail = "Heavy-tailed (power-law)"
        elif alpha > 0.3:
            tail = "Moderately heavy-tailed"
        else:
            tail = "Light-tailed / near-uniform"

        print(f"\n{level}:")
        print(f"  Better fit: {better}")
        print(f"  Zipf α = {alpha:.3f}, R² = {zipf_r2:.4f}")
        print(f"  Tail type: {tail}")
        print(f"  Entropy = {res['entropy']:.2f} bits ({res['entropy']/np.log2(1024)*100:.1f}% of max)")

        summary[level] = {
            "better_fit": better,
            "zipf_alpha": round(alpha, 4),
            "zipf_r_squared": round(zipf_r2, 4),
            "zipf_ks": round(zipf_ks, 4),
            "lognorm_ks": round(lognorm_ks, 4),
            "lognorm_sigma": round(res["lognorm"]["sigma"], 4),
            "lognorm_mu": round(res["lognorm"]["mu"], 4),
            "entropy_bits": round(res["entropy"], 4),
            "entropy_pct": round(res["entropy"] / np.log2(1024) * 100, 2),
            "gini": round(res["gini"], 4),
            "active_codes": int(len(res["ranks"])),
            "total_tokens": int(res["counts"].sum()),
            "tail_type": tail,
        }

    # Save JSON summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")

    # Overall conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    alphas = [s["zipf_alpha"] for s in summary.values()]
    r2s = [s["zipf_r_squared"] for s in summary.values()]
    avg_alpha = np.mean(alphas)
    avg_r2 = np.mean(r2s)

    if avg_r2 > 0.9 and avg_alpha > 0.5:
        conclusion = "STRONG POWER-LAW: Mesh tokens follow Zipf distribution. Theory-driven direction strongly supported."
    elif avg_r2 > 0.8:
        conclusion = "MODERATE POWER-LAW: Mesh tokens show heavy-tailed behavior. Theory-driven direction viable."
    else:
        lognorm_better = sum(1 for s in summary.values() if "Lognormal" in s["better_fit"])
        if lognorm_better > len(summary) / 2:
            conclusion = "LOGNORMAL: Mesh tokens follow lognormal distribution (like image tokens). Pivot narrative needed."
        else:
            conclusion = "MIXED: No clear distribution dominance. Both directions worth exploring."

    print(f"  Average Zipf α = {avg_alpha:.3f}, Average R² = {avg_r2:.4f}")
    print(f"  → {conclusion}")

    summary["_conclusion"] = {
        "avg_zipf_alpha": round(avg_alpha, 4),
        "avg_zipf_r_squared": round(avg_r2, 4),
        "verdict": conclusion,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

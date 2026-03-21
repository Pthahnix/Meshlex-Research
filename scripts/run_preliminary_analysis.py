"""Preliminary Experiments 1-4: Token analysis suite.

Runs four analysis experiments on existing RVQ token sequences:
  Exp 1: Per-category token distribution
  Exp 2: Token spatial correlation
  Exp 3: Codebook embedding visualization (requires checkpoint)
  Exp 4: RVQ inter-level dependency (mutual information)

Usage:
    PYTHONPATH=. python scripts/run_preliminary_analysis.py \
        --seq_dir data/sequences/rvq_lvis \
        --patch_dir data/patches/lvis_wide \
        --checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir results/preliminary_exp
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

def load_sequences(seq_dir: str):
    """Load all sequence NPZs. Returns list of dicts with centroids, scales, tokens, mesh_id."""
    seq_dir = Path(seq_dir)
    npz_files = sorted(seq_dir.glob("*_sequence.npz"))
    print(f"Found {len(npz_files)} sequence files")

    sequences = []
    for f in npz_files:
        data = np.load(str(f))
        mesh_id = f.stem.replace("_sequence", "")
        sequences.append({
            "mesh_id": mesh_id,
            "centroids": data["centroids"],   # (N, 3)
            "scales": data["scales"],          # (N,)
            "tokens": data["tokens"],          # (N, 3) for 3-level RVQ
        })
    return sequences


def infer_categories(sequences, patch_dir: str):
    """Infer category from patch directory structure.

    Patches are stored as: patch_dir/{split}/{mesh_id}_patch_{i}.npz
    We use the split subdirs to find mesh_ids, but category comes from
    the Objaverse LVIS metadata. As a fallback, group by first 8 chars
    of mesh_id (Objaverse UIDs are random, so this won't help).

    Better approach: check if any category mapping file exists.
    """
    patch_dir = Path(patch_dir)
    mesh_to_split = {}

    for split_dir in patch_dir.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name  # e.g. seen_train, seen_test, unseen
        seen_meshes = set()
        for f in split_dir.glob("*.npz"):
            stem = f.stem
            if "_patch_" in stem:
                mesh_id = stem.rsplit("_patch_", 1)[0]
            else:
                mesh_id = stem
            if mesh_id not in seen_meshes:
                seen_meshes.add(mesh_id)
                mesh_to_split[mesh_id] = split_name

    return mesh_to_split


def compute_distribution_stats(token_counts, codebook_size=1024):
    """Compute Zipf, lognormal, entropy, gini for a token count array."""
    sorted_counts = np.sort(token_counts)[::-1]
    nonzero = sorted_counts[sorted_counts > 0]
    if len(nonzero) < 10:
        return None

    ranks = np.arange(1, len(nonzero) + 1)
    freqs = nonzero / nonzero.sum()

    # Zipf fit
    log_r = np.log(ranks)
    log_f = np.log(freqs)
    slope, intercept, r_value, _, _ = stats.linregress(log_r, log_f)
    zipf_alpha = -slope
    zipf_r2 = r_value ** 2

    # Lognormal fit
    try:
        shape, loc, scale = stats.lognorm.fit(nonzero.astype(float), floc=0)
        lognorm_ks, lognorm_p = stats.kstest(nonzero.astype(float), "lognorm", args=(shape, loc, scale))
        lognorm_sigma = float(shape)
        lognorm_mu = float(np.log(scale))
    except Exception:
        lognorm_ks, lognorm_p = 1.0, 0.0
        lognorm_sigma, lognorm_mu = 0.0, 0.0

    # Zipf KS
    pred_freqs = np.exp(intercept) * ranks ** slope
    zipf_ks, _ = stats.ks_2samp(freqs, pred_freqs / pred_freqs.sum())

    # Entropy
    f = freqs[freqs > 0]
    entropy = float(-np.sum(f * np.log2(f)))

    # Gini
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    if cumulative[-1] > 0:
        gini = float((2.0 * np.sum(np.arange(1, n + 1) * sorted_counts) / (n * cumulative[-1])) - (n + 1) / n)
    else:
        gini = 0.0

    return {
        "zipf_alpha": round(float(zipf_alpha), 4),
        "zipf_r2": round(float(zipf_r2), 4),
        "zipf_ks": round(float(zipf_ks), 4),
        "lognorm_sigma": round(lognorm_sigma, 4),
        "lognorm_mu": round(lognorm_mu, 4),
        "lognorm_ks": round(float(lognorm_ks), 4),
        "lognorm_p": round(float(lognorm_p), 4),
        "better_fit": "Lognormal" if lognorm_ks < zipf_ks else "Zipf",
        "entropy_bits": round(float(entropy), 4),
        "entropy_pct": round(float(entropy / np.log2(codebook_size) * 100), 2),
        "gini": round(float(gini), 4),
        "active_codes": int(len(nonzero)),
        "n_tokens": int(token_counts.sum()),
    }


# ──────────────────────────────────────────────
# Exp 1: Per-category token distribution
# ──────────────────────────────────────────────

def run_exp1(sequences, patch_dir, output_dir, codebook_size=1024):
    """Per-category token distribution analysis."""
    print("\n" + "=" * 60)
    print("Exp 1: Per-category Token Distribution")
    print("=" * 60)

    exp_dir = output_dir / "exp1_per_category"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Infer categories (split-based grouping as proxy)
    mesh_to_split = infer_categories(sequences, patch_dir)

    # Group tokens by split
    split_tokens = defaultdict(lambda: {"L1": [], "L2": [], "L3": []})
    ungrouped_count = 0
    for seq in sequences:
        mid = seq["mesh_id"]
        split = mesh_to_split.get(mid, "unknown")
        if split == "unknown":
            ungrouped_count += 1
        tokens = seq["tokens"]
        if tokens.ndim == 2 and tokens.shape[1] == 3:
            split_tokens[split]["L1"].append(tokens[:, 0])
            split_tokens[split]["L2"].append(tokens[:, 1])
            split_tokens[split]["L3"].append(tokens[:, 2])

    print(f"  Splits found: {list(split_tokens.keys())}")
    print(f"  Ungrouped meshes: {ungrouped_count}")

    # Since LVIS synset mapping may not be available, we do an alternative:
    # Group by token diversity (high-entropy vs low-entropy meshes)
    # AND by mesh size (number of patches)
    mesh_stats = []
    for seq in sequences:
        tokens = seq["tokens"]
        if tokens.ndim == 2:
            l1_tokens = tokens[:, 0]
            n_unique = len(np.unique(l1_tokens))
            n_patches = len(l1_tokens)
            mesh_stats.append({
                "mesh_id": seq["mesh_id"],
                "n_patches": n_patches,
                "n_unique_L1": n_unique,
                "diversity": n_unique / max(n_patches, 1),
            })

    mesh_stats.sort(key=lambda x: x["n_patches"])

    # Group into quartiles by mesh size
    n = len(mesh_stats)
    quartiles = {
        "small (Q1)": mesh_stats[:n // 4],
        "medium-small (Q2)": mesh_stats[n // 4:n // 2],
        "medium-large (Q3)": mesh_stats[n // 2:3 * n // 4],
        "large (Q4)": mesh_stats[3 * n // 4:],
    }

    mesh_id_to_seq = {s["mesh_id"]: s for s in sequences}

    results = {}
    for group_name, group_meshes in quartiles.items():
        group_ids = {m["mesh_id"] for m in group_meshes}
        all_l1 = []
        for seq in sequences:
            if seq["mesh_id"] in group_ids:
                all_l1.append(seq["tokens"][:, 0])
        if not all_l1:
            continue
        all_l1 = np.concatenate(all_l1)
        counts = np.bincount(all_l1, minlength=codebook_size)
        dist_stats = compute_distribution_stats(counts, codebook_size)
        if dist_stats:
            dist_stats["n_meshes"] = len(group_meshes)
            dist_stats["avg_patches"] = round(np.mean([m["n_patches"] for m in group_meshes]), 1)
            results[group_name] = dist_stats
            print(f"  {group_name}: α={dist_stats['zipf_alpha']}, "
                  f"σ={dist_stats['lognorm_sigma']}, "
                  f"fit={dist_stats['better_fit']}, "
                  f"n_meshes={len(group_meshes)}")

    # Also group by token diversity quartiles
    mesh_stats.sort(key=lambda x: x["diversity"])
    diversity_quartiles = {
        "low diversity (Q1)": mesh_stats[:n // 4],
        "med-low diversity (Q2)": mesh_stats[n // 4:n // 2],
        "med-high diversity (Q3)": mesh_stats[n // 2:3 * n // 4],
        "high diversity (Q4)": mesh_stats[3 * n // 4:],
    }

    diversity_results = {}
    for group_name, group_meshes in diversity_quartiles.items():
        group_ids = {m["mesh_id"] for m in group_meshes}
        all_l1 = []
        for seq in sequences:
            if seq["mesh_id"] in group_ids:
                all_l1.append(seq["tokens"][:, 0])
        if not all_l1:
            continue
        all_l1 = np.concatenate(all_l1)
        counts = np.bincount(all_l1, minlength=codebook_size)
        dist_stats = compute_distribution_stats(counts, codebook_size)
        if dist_stats:
            dist_stats["n_meshes"] = len(group_meshes)
            diversity_results[group_name] = dist_stats
            print(f"  {group_name}: α={dist_stats['zipf_alpha']}, "
                  f"σ={dist_stats['lognorm_sigma']}, "
                  f"fit={dist_stats['better_fit']}")

    # Per-split analysis
    split_results = {}
    for split_name, split_toks in split_tokens.items():
        if not split_toks["L1"]:
            continue
        all_l1 = np.concatenate(split_toks["L1"])
        counts = np.bincount(all_l1, minlength=codebook_size)
        dist_stats = compute_distribution_stats(counts, codebook_size)
        if dist_stats:
            dist_stats["n_meshes"] = len(split_toks["L1"])
            split_results[split_name] = dist_stats
            print(f"  Split '{split_name}': α={dist_stats['zipf_alpha']}, "
                  f"σ={dist_stats['lognorm_sigma']}, fit={dist_stats['better_fit']}")

    # Visualization: comparison of lognormal σ across groups
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Size quartiles
    ax = axes[0]
    groups = list(results.keys())
    sigmas = [results[g]["lognorm_sigma"] for g in groups]
    alphas = [results[g]["zipf_alpha"] for g in groups]
    x = np.arange(len(groups))
    ax.bar(x - 0.2, sigmas, 0.35, label="Lognorm σ", color="steelblue")
    ax.bar(x + 0.2, alphas, 0.35, label="Zipf α", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([g.split(" ")[0] for g in groups], fontsize=9)
    ax.set_title("By Mesh Size (# patches)")
    ax.legend()
    ax.set_ylabel("Parameter value")

    # Diversity quartiles
    ax = axes[1]
    groups = list(diversity_results.keys())
    sigmas = [diversity_results[g]["lognorm_sigma"] for g in groups]
    alphas = [diversity_results[g]["zipf_alpha"] for g in groups]
    x = np.arange(len(groups))
    ax.bar(x - 0.2, sigmas, 0.35, label="Lognorm σ", color="steelblue")
    ax.bar(x + 0.2, alphas, 0.35, label="Zipf α", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([g.split(" ")[0] for g in groups], fontsize=9)
    ax.set_title("By Token Diversity")
    ax.legend()

    # Splits
    ax = axes[2]
    if split_results:
        groups = list(split_results.keys())
        sigmas = [split_results[g]["lognorm_sigma"] for g in groups]
        alphas = [split_results[g]["zipf_alpha"] for g in groups]
        x = np.arange(len(groups))
        ax.bar(x - 0.2, sigmas, 0.35, label="Lognorm σ", color="steelblue")
        ax.bar(x + 0.2, alphas, 0.35, label="Zipf α", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9, rotation=15)
        ax.set_title("By Data Split")
        ax.legend()

    plt.suptitle("Exp 1: Per-group Distribution Parameter Comparison (L1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / "comparison.png", dpi=150)
    plt.close()

    # Better fit pie chart
    all_groups = {**results, **diversity_results, **split_results}
    fit_counts = defaultdict(int)
    for g, s in all_groups.items():
        fit_counts[s["better_fit"]] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(fit_counts.keys())
    sizes = list(fit_counts.values())
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=["steelblue", "coral"])
    ax.set_title(f"Distribution Fit Winner Across {len(all_groups)} Groups")
    plt.savefig(exp_dir / "fit_winner.png", dpi=150)
    plt.close()

    # Save
    summary = {
        "by_mesh_size": results,
        "by_diversity": diversity_results,
        "by_split": split_results,
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Key conclusion
    all_sigmas = [s["lognorm_sigma"] for s in all_groups.values()]
    sigma_std = np.std(all_sigmas) if all_sigmas else 0
    n_lognorm = sum(1 for s in all_groups.values() if s["better_fit"] == "Lognormal")
    n_total = len(all_groups)

    conclusion = {
        "lognorm_sigma_mean": round(float(np.mean(all_sigmas)), 4) if all_sigmas else 0,
        "lognorm_sigma_std": round(float(sigma_std), 4),
        "lognorm_wins": n_lognorm,
        "total_groups": n_total,
        "verdict": "CONSISTENT" if sigma_std < 0.15 and n_lognorm > n_total * 0.7 else "INCONSISTENT",
    }
    summary["_conclusion"] = conclusion
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Conclusion: σ_std={sigma_std:.4f}, lognorm wins {n_lognorm}/{n_total}")
    print(f"  Verdict: {conclusion['verdict']}")
    return summary


# ──────────────────────────────────────────────
# Exp 2: Token spatial correlation
# ──────────────────────────────────────────────

def run_exp2(sequences, output_dir):
    """Token spatial correlation analysis."""
    print("\n" + "=" * 60)
    print("Exp 2: Token Spatial Correlation")
    print("=" * 60)

    exp_dir = output_dir / "exp2_spatial"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_correlations = []
    all_distances = []
    all_matches = []

    # Sample meshes for pairwise computation (full dataset too many pairs)
    sample_indices = np.random.RandomState(42).choice(
        len(sequences), min(500, len(sequences)), replace=False
    )

    for idx in sample_indices:
        seq = sequences[idx]
        centroids = seq["centroids"]  # (N, 3)
        tokens = seq["tokens"]        # (N, 3)
        n = len(centroids)
        if n < 3:
            continue

        # Pairwise distances
        dists = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)

        # Token match (L1 level)
        l1_tokens = tokens[:, 0] if tokens.ndim == 2 else tokens
        match = (l1_tokens[:, None] == l1_tokens[None, :]).astype(float)

        # Upper triangle only (exclude diagonal)
        iu = np.triu_indices(n, k=1)
        d_flat = dists[iu]
        m_flat = match[iu]

        if len(d_flat) > 2:
            corr, p_val = stats.spearmanr(d_flat, m_flat)
            if not np.isnan(corr):
                all_correlations.append(corr)

            # Store for global scatter (subsample)
            if len(d_flat) > 100:
                sub_idx = np.random.choice(len(d_flat), 100, replace=False)
                all_distances.extend(d_flat[sub_idx].tolist())
                all_matches.extend(m_flat[sub_idx].tolist())
            else:
                all_distances.extend(d_flat.tolist())
                all_matches.extend(m_flat.tolist())

    print(f"  Analyzed {len(all_correlations)} meshes")

    if not all_correlations:
        print("  ERROR: No correlations computed")
        return {}

    mean_corr = np.mean(all_correlations)
    median_corr = np.median(all_correlations)
    frac_negative = np.mean(np.array(all_correlations) < 0)

    print(f"  Mean Spearman ρ: {mean_corr:.4f}")
    print(f"  Median Spearman ρ: {median_corr:.4f}")
    print(f"  Fraction with negative correlation: {frac_negative:.2%}")

    # Moran's I (global, aggregated)
    # Simplified: compute on binned distance-match data
    all_distances = np.array(all_distances)
    all_matches = np.array(all_matches)

    # Bin by distance
    n_bins = 20
    dist_bins = np.linspace(all_distances.min(), all_distances.max(), n_bins + 1)
    bin_centers = []
    bin_match_rates = []
    for i in range(n_bins):
        mask = (all_distances >= dist_bins[i]) & (all_distances < dist_bins[i + 1])
        if mask.sum() > 10:
            bin_centers.append((dist_bins[i] + dist_bins[i + 1]) / 2)
            bin_match_rates.append(all_matches[mask].mean())

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Correlation distribution
    ax = axes[0]
    ax.hist(all_correlations, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(mean_corr, color="red", linestyle="--", linewidth=2, label=f"Mean={mean_corr:.3f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Spearman ρ (distance vs L1 token match)")
    ax.set_ylabel("Count")
    ax.set_title("Per-mesh Spatial Correlation")
    ax.legend()

    # Distance vs match rate (binned)
    ax = axes[1]
    if bin_centers:
        ax.plot(bin_centers, bin_match_rates, "o-", color="steelblue", markersize=4)
        ax.set_xlabel("Centroid Distance")
        ax.set_ylabel("L1 Token Match Rate")
        ax.set_title("Match Rate vs Distance (binned)")
        ax.grid(True, alpha=0.3)

    # Scatter sample
    ax = axes[2]
    sample_n = min(5000, len(all_distances))
    si = np.random.choice(len(all_distances), sample_n, replace=False)
    ax.scatter(all_distances[si], all_matches[si], s=1, alpha=0.1, color="steelblue")
    ax.set_xlabel("Centroid Distance")
    ax.set_ylabel("Same L1 Token (0/1)")
    ax.set_title(f"Spatial Correlation (n={sample_n})")

    plt.suptitle("Exp 2: Token Spatial Correlation", fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / "spatial_correlation.png", dpi=150)
    plt.close()

    summary = {
        "n_meshes_analyzed": len(all_correlations),
        "mean_spearman": round(float(mean_corr), 4),
        "median_spearman": round(float(median_corr), 4),
        "std_spearman": round(float(np.std(all_correlations)), 4),
        "frac_negative": round(float(frac_negative), 4),
        "n_distance_bins": len(bin_centers),
        "verdict": "SIGNIFICANT_NEGATIVE" if mean_corr < -0.05 else
                   "SIGNIFICANT_POSITIVE" if mean_corr > 0.05 else "WEAK/NONE",
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ──────────────────────────────────────────────
# Exp 3: Codebook embedding visualization
# ──────────────────────────────────────────────

def run_exp3(sequences, checkpoint_path, output_dir, codebook_size=1024):
    """Codebook embedding UMAP visualization."""
    print("\n" + "=" * 60)
    print("Exp 3: Codebook Embedding Visualization")
    print("=" * 60)

    exp_dir = output_dir / "exp3_codebook_viz"
    exp_dir.mkdir(parents=True, exist_ok=True)

    import torch
    try:
        from umap import UMAP
    except ImportError:
        print("  Installing umap-learn...")
        import subprocess
        subprocess.check_call(["pip", "install", "umap-learn"])
        from umap import UMAP

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Extract codebook embeddings for each level
    # SimVQ: actual embeddings = linear(codebook.weight) = W @ C
    level_embeddings = []
    for i in range(3):
        cb_key = f"rvq.levels.{i}.codebook.weight"
        linear_key = f"rvq.levels.{i}.linear.weight"
        if cb_key in state and linear_key in state:
            C = state[cb_key]  # (K, dim)
            W = state[linear_key]  # (dim, dim)
            CW = C @ W.T  # (K, dim) — transformed codebook
            level_embeddings.append(CW.numpy())
            print(f"  Level {i}: codebook shape {CW.shape}")
        else:
            print(f"  WARNING: Level {i} keys not found")

    if not level_embeddings:
        print("  ERROR: No codebook embeddings found")
        return {}

    # Compute token frequencies for coloring
    all_tokens = {"L1": [], "L2": [], "L3": []}
    for seq in sequences:
        tokens = seq["tokens"]
        if tokens.ndim == 2 and tokens.shape[1] == 3:
            all_tokens["L1"].append(tokens[:, 0])
            all_tokens["L2"].append(tokens[:, 1])
            all_tokens["L3"].append(tokens[:, 2])

    freq_maps = {}
    for level_name, tok_list in all_tokens.items():
        if tok_list:
            all_t = np.concatenate(tok_list)
            counts = np.bincount(all_t, minlength=codebook_size)
            freq_maps[level_name] = counts

    # UMAP per level
    level_names = ["L1", "L2", "L3"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (emb, level_name) in enumerate(zip(level_embeddings, level_names)):
        print(f"  Computing UMAP for {level_name}...")
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(emb)

        freq = freq_maps.get(level_name, np.ones(codebook_size))
        log_freq = np.log1p(freq)

        ax = axes[i]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=log_freq, s=8,
                        cmap="viridis", alpha=0.7)
        ax.set_title(f"{level_name} Codebook (colored by log freq)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.colorbar(sc, ax=ax, label="log(1 + count)")

    plt.suptitle("Exp 3: Codebook Embedding Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / "umap_per_level.png", dpi=150)
    plt.close()

    # Combined UMAP (all 3 levels together)
    combined = np.concatenate(level_embeddings, axis=0)
    labels = np.concatenate([np.full(len(e), i) for i, e in enumerate(level_embeddings)])
    print("  Computing combined UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    combined_coords = reducer.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["steelblue", "darkorange", "seagreen"]
    for i, name in enumerate(level_names[:len(level_embeddings)]):
        mask = labels == i
        ax.scatter(combined_coords[mask, 0], combined_coords[mask, 1],
                   s=4, alpha=0.5, c=colors[i], label=name)
    ax.legend(fontsize=12)
    ax.set_title("Combined Codebook UMAP (3 RVQ levels)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(exp_dir / "umap_combined.png", dpi=150)
    plt.close()

    summary = {
        "n_levels": len(level_embeddings),
        "embed_dim": level_embeddings[0].shape[1],
        "codebook_size": codebook_size,
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ──────────────────────────────────────────────
# Exp 4: RVQ inter-level dependency
# ──────────────────────────────────────────────

def run_exp4(sequences, output_dir, codebook_size=1024):
    """RVQ inter-level mutual information analysis."""
    print("\n" + "=" * 60)
    print("Exp 4: RVQ Inter-level Dependency")
    print("=" * 60)

    exp_dir = output_dir / "exp4_rvq_dependency"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Collect all token pairs
    all_l1, all_l2, all_l3 = [], [], []
    for seq in sequences:
        tokens = seq["tokens"]
        if tokens.ndim == 2 and tokens.shape[1] == 3:
            all_l1.append(tokens[:, 0])
            all_l2.append(tokens[:, 1])
            all_l3.append(tokens[:, 2])

    L1 = np.concatenate(all_l1)
    L2 = np.concatenate(all_l2)
    L3 = np.concatenate(all_l3)
    N = len(L1)
    print(f"  Total tokens: {N}")

    def entropy(x, K):
        counts = np.bincount(x, minlength=K)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def conditional_entropy(x, y, K):
        """H(Y|X) = Σ_x P(x) H(Y|X=x)"""
        h_cond = 0.0
        x_counts = np.bincount(x, minlength=K)
        x_total = x_counts.sum()
        for xi in range(K):
            if x_counts[xi] == 0:
                continue
            p_x = x_counts[xi] / x_total
            mask = x == xi
            y_given_x = y[mask]
            if len(y_given_x) > 0:
                h_cond += p_x * entropy(y_given_x, K)
        return h_cond

    def mutual_info(x, y, K):
        return entropy(y, K) - conditional_entropy(x, y, K)

    def nmi(x, y, K):
        mi = mutual_info(x, y, K)
        hx = entropy(x, K)
        hy = entropy(y, K)
        denom = min(hx, hy)
        return mi / denom if denom > 0 else 0.0

    K = codebook_size
    H_L1 = entropy(L1, K)
    H_L2 = entropy(L2, K)
    H_L3 = entropy(L3, K)

    H_L2_given_L1 = conditional_entropy(L1, L2, K)
    H_L3_given_L1 = conditional_entropy(L1, L3, K)
    H_L3_given_L2 = conditional_entropy(L2, L3, K)

    MI_L1_L2 = H_L2 - H_L2_given_L1
    MI_L1_L3 = H_L3 - H_L3_given_L1
    MI_L2_L3 = H_L3 - H_L3_given_L2

    NMI_L1_L2 = nmi(L1, L2, K)
    NMI_L1_L3 = nmi(L1, L3, K)
    NMI_L2_L3 = nmi(L2, L3, K)

    print(f"  H(L1) = {H_L1:.4f} bits")
    print(f"  H(L2) = {H_L2:.4f} bits")
    print(f"  H(L3) = {H_L3:.4f} bits")
    print(f"  H(L2|L1) = {H_L2_given_L1:.4f} bits")
    print(f"  H(L3|L1) = {H_L3_given_L1:.4f} bits")
    print(f"  H(L3|L2) = {H_L3_given_L2:.4f} bits")
    print(f"  I(L1;L2) = {MI_L1_L2:.4f} bits, NMI = {NMI_L1_L2:.4f}")
    print(f"  I(L1;L3) = {MI_L1_L3:.4f} bits, NMI = {NMI_L1_L3:.4f}")
    print(f"  I(L2;L3) = {MI_L2_L3:.4f} bits, NMI = {NMI_L2_L3:.4f}")

    # Visualize: co-occurrence matrix (subsampled for visibility)
    # Full 1024x1024 is too large to see, so show top-50 most frequent tokens
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [("L1", "L2", L1, L2), ("L1", "L3", L1, L3), ("L2", "L3", L2, L3)]
    mi_values = [MI_L1_L2, MI_L1_L3, MI_L2_L3]
    nmi_values = [NMI_L1_L2, NMI_L1_L3, NMI_L2_L3]

    for ax, (name_x, name_y, x, y), mi, nmi_v in zip(axes, pairs, mi_values, nmi_values):
        # Top 50 tokens per axis
        x_top = np.argsort(np.bincount(x, minlength=K))[::-1][:50]
        y_top = np.argsort(np.bincount(y, minlength=K))[::-1][:50]

        cooc = np.zeros((50, 50))
        for xi_idx, xi in enumerate(x_top):
            mask = x == xi
            y_sub = y[mask]
            for yi_idx, yi in enumerate(y_top):
                cooc[xi_idx, yi_idx] = np.sum(y_sub == yi)

        # Normalize rows
        row_sums = cooc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cooc_norm = cooc / row_sums

        im = ax.imshow(cooc_norm, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xlabel(f"{name_y} token (top 50)")
        ax.set_ylabel(f"{name_x} token (top 50)")
        ax.set_title(f"I({name_x};{name_y})={mi:.3f} bits\nNMI={nmi_v:.4f}")
        plt.colorbar(im, ax=ax, label="P(Y|X)")

    plt.suptitle("Exp 4: RVQ Inter-level Dependency", fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / "dependency_matrix.png", dpi=150)
    plt.close()

    # Bar chart of MI/NMI
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pair_labels = ["L1→L2", "L1→L3", "L2→L3"]
    axes[0].bar(pair_labels, mi_values, color="steelblue")
    axes[0].set_ylabel("Mutual Information (bits)")
    axes[0].set_title("MI between RVQ levels")

    axes[1].bar(pair_labels, nmi_values, color="darkorange")
    axes[1].set_ylabel("Normalized MI")
    axes[1].set_title("NMI between RVQ levels")
    axes[1].axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Threshold (0.1)")
    axes[1].legend()

    plt.suptitle("Exp 4: RVQ Inter-level Dependency Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / "mi_summary.png", dpi=150)
    plt.close()

    summary = {
        "n_tokens": int(N),
        "H_L1": round(H_L1, 4),
        "H_L2": round(H_L2, 4),
        "H_L3": round(H_L3, 4),
        "H_L2_given_L1": round(H_L2_given_L1, 4),
        "H_L3_given_L1": round(H_L3_given_L1, 4),
        "H_L3_given_L2": round(H_L3_given_L2, 4),
        "MI_L1_L2": round(MI_L1_L2, 4),
        "MI_L1_L3": round(MI_L1_L3, 4),
        "MI_L2_L3": round(MI_L2_L3, 4),
        "NMI_L1_L2": round(NMI_L1_L2, 4),
        "NMI_L1_L3": round(NMI_L1_L3, 4),
        "NMI_L2_L3": round(NMI_L2_L3, 4),
        "verdict": "STRONG_DEPENDENCY" if max(nmi_values) > 0.1 else
                   "MODERATE_DEPENDENCY" if max(nmi_values) > 0.05 else "NEAR_INDEPENDENT",
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preliminary experiments 1-4")
    parser.add_argument("--seq_dir", required=True, help="Path to sequence NPZs")
    parser.add_argument("--patch_dir", required=True, help="Path to patch directories")
    parser.add_argument("--checkpoint", required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--output_dir", default="results/preliminary_exp")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--skip_exp3", action="store_true", help="Skip UMAP (no GPU needed)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (shared across exp 1-4)
    print("Loading sequences...")
    sequences = load_sequences(args.seq_dir)
    print(f"Loaded {len(sequences)} sequences\n")

    results = {}

    # Exp 1
    results["exp1"] = run_exp1(sequences, args.patch_dir, output_dir, args.codebook_size)

    # Exp 2
    results["exp2"] = run_exp2(sequences, output_dir)

    # Exp 3 (needs checkpoint + umap)
    if not args.skip_exp3:
        results["exp3"] = run_exp3(sequences, args.checkpoint, output_dir, args.codebook_size)

    # Exp 4
    results["exp4"] = run_exp4(sequences, output_dir, args.codebook_size)

    # Overall summary
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)

    with open(output_dir / "all_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

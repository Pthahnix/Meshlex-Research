"""Phase 0: BPE Feasibility Validation.

Loads preprocessed meshes, builds dual graphs, runs Graph BPE,
computes discretization MI, analyzes within-token normal variance,
and produces Go/No-Go decision.

Usage:
    python scripts/run_phase0.py --mesh_dir data/meshes/lvis_wide \
        --output_dir results/phase0 --n_meshes 200
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import trimesh
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dual_graph import build_labeled_dual_graph
from src.graph_bpe import GraphBPE
from src.discretize import compute_discretization_mi
from src.patch_segment import segment_mesh_to_patches


def load_meshes(mesh_dir: str, n_meshes: int):
    """Load OBJ/GLB meshes from directory."""
    mesh_dir = Path(mesh_dir)
    mesh_files = sorted(list(mesh_dir.rglob("*.obj")) + list(mesh_dir.rglob("*.glb")))[:n_meshes]
    meshes = []
    for f in tqdm(mesh_files, desc="Loading meshes"):
        try:
            m = trimesh.load(str(f), force="mesh")
            if len(m.faces) >= 20:
                meshes.append(m)
        except Exception:
            continue
    return meshes


def compute_metis_normal_variance(meshes, n_sample=50):
    """Compute METIS patch within-patch normal variance as baseline."""
    variances = []
    for mesh in meshes[:n_sample]:
        try:
            patches = segment_mesh_to_patches(mesh)
        except Exception:
            continue
        for patch in patches:
            if len(patch.faces) < 2:
                continue
            normals = []
            for f in patch.faces:
                v0, v1, v2 = patch.vertices[f[0]], patch.vertices[f[1]], patch.vertices[f[2]]
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                if norm > 1e-8:
                    normals.append(n / norm)
            if len(normals) >= 2:
                normals_arr = np.array(normals)
                var = np.var(normals_arr, axis=0).sum()
                variances.append(var)
    return np.median(variances) if variances else 1.0


def analyze_bpe_normal_variance(meshes, patches_per_mesh, metis_median):
    """Compute fraction of BPE tokens with normal variance < METIS median.

    Only considers tokens with >=10 faces (H1a criterion).
    """
    n_pass = 0
    n_total = 0
    all_variances = []
    patch_sizes = []

    for mesh, patches in zip(meshes, patches_per_mesh):
        for patch in patches:
            patch_sizes.append(len(patch.face_indices))
            if len(patch.face_indices) < 10:
                continue
            n_total += 1
            normals = mesh.face_normals[patch.face_indices]
            var = np.var(normals, axis=0).sum()
            all_variances.append(var)
            if var < metis_median:
                n_pass += 1

    ratio = n_pass / max(n_total, 1)
    return ratio, all_variances, patch_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--output_dir", default="results/phase0")
    parser.add_argument("--n_meshes", type=int, default=200)
    parser.add_argument("--target_vocab", type=int, default=2000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load meshes ---
    meshes = load_meshes(args.mesh_dir, args.n_meshes)
    print(f"Loaded {len(meshes)} meshes")

    # --- Test 3 discretization granularities ---
    granularities = {
        "coarse": {"n_normal_bins": 32, "n_area_bins": 4, "n_dihedral_bins": 8},
        "medium": {"n_normal_bins": 64, "n_area_bins": 8, "n_dihedral_bins": 16},
        "fine":   {"n_normal_bins": 128, "n_area_bins": 16, "n_dihedral_bins": 32},
    }

    mi_results = {}
    for name, params in granularities.items():
        print(f"\n--- Discretization: {name} ---")
        all_labels = []
        all_features = []
        for mesh in meshes[:100]:
            dg = build_labeled_dual_graph(mesh, **params)
            all_labels.append(dg.node_labels)
            all_features.append(dg.face_normals)
        labels = np.concatenate(all_labels)
        features = np.concatenate(all_features)
        mi = compute_discretization_mi(labels, features)
        mi_results[name] = mi
        print(f"  MI = {mi:.4f}")

    # --- Choose best granularity ---
    best_gran = max(mi_results, key=mi_results.get)
    best_params = granularities[best_gran]
    print(f"\nBest granularity: {best_gran} (MI={mi_results[best_gran]:.4f})")

    # --- Build dual graphs ---
    dual_graphs = []
    for mesh in tqdm(meshes, desc="Building dual graphs"):
        dg = build_labeled_dual_graph(mesh, **best_params)
        dual_graphs.append(dg)

    # --- Run Graph BPE ---
    t0 = time.time()
    bpe = GraphBPE(target_vocab_size=args.target_vocab)
    vocab = bpe.train(dual_graphs)
    bpe_time = time.time() - t0
    print(f"BPE training: {bpe_time:.1f}s, vocab size: {len(vocab.symbols)}")

    # --- Encode meshes and get patches ---
    all_patches = []
    for dg in tqdm(dual_graphs, desc="BPE encoding"):
        patches = bpe.encode(dg, vocab)
        all_patches.append(patches)

    # --- METIS baseline normal variance ---
    metis_median = compute_metis_normal_variance(meshes)
    print(f"METIS median within-patch normal variance: {metis_median:.6f}")

    # --- BPE normal variance analysis (H1a) ---
    h1a_ratio, bpe_variances, patch_sizes = analyze_bpe_normal_variance(
        meshes, all_patches, metis_median,
    )
    print(f"H1a: {h1a_ratio:.1%} of BPE tokens (>=10 faces) have var < METIS median")

    # --- H5: MI check ---
    h5_pass = any(mi > 0.5 for mi in mi_results.values())

    # --- Patch size distribution ---
    patch_sizes = np.array(patch_sizes)
    size_stats = {
        "mean": float(np.mean(patch_sizes)),
        "std": float(np.std(patch_sizes)),
        "min": int(np.min(patch_sizes)),
        "max": int(np.max(patch_sizes)),
        "p5": float(np.percentile(patch_sizes, 5)),
        "p25": float(np.percentile(patch_sizes, 25)),
        "p50": float(np.percentile(patch_sizes, 50)),
        "p75": float(np.percentile(patch_sizes, 75)),
        "p95": float(np.percentile(patch_sizes, 95)),
    }

    # --- Go/No-Go Decision ---
    h1a_go = h1a_ratio >= 0.60
    h5_go = h5_pass
    decision = "GO" if (h1a_go and h5_go) else "NO-GO"

    report = {
        "n_meshes": len(meshes),
        "granularities_mi": mi_results,
        "best_granularity": best_gran,
        "vocab_size": len(vocab.symbols),
        "base_alphabet_size": vocab.base_alphabet_size,
        "n_merges": len(vocab.merge_rules),
        "bpe_training_time_sec": bpe_time,
        "metis_normal_var_median": float(metis_median),
        "h1a_ratio": h1a_ratio,
        "h1a_go": h1a_go,
        "h5_any_mi_above_0.5": h5_pass,
        "h5_go": h5_go,
        "patch_size_stats": size_stats,
        "decision": decision,
    }

    # --- Save report ---
    with open(out / "phase0_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(mi_results.keys(), mi_results.values())
    axes[0].axhline(0.5, color="r", linestyle="--", label="H5 threshold")
    axes[0].set_ylabel("MI")
    axes[0].set_title("Discretization MI")
    axes[0].legend()

    axes[1].hist(patch_sizes, bins=50, edgecolor="black")
    axes[1].set_xlabel("Faces per BPE token")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"BPE Patch Sizes (mean={size_stats['mean']:.1f})")

    if bpe_variances:
        axes[2].hist(bpe_variances, bins=50, alpha=0.7, label="BPE tokens")
        axes[2].axvline(metis_median, color="r", linestyle="--", label="METIS median")
        axes[2].set_xlabel("Within-token normal variance")
        axes[2].set_title(f"H1a: {h1a_ratio:.0%} below threshold")
        axes[2].legend()

    plt.suptitle(f"Phase 0 BPE Feasibility — Decision: {decision}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "phase0_dashboard.png", dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"Phase 0 BPE Feasibility Report")
    print(f"{'='*60}")
    print(f"H1a (normal variance): {h1a_ratio:.1%} >= 60%? {'YES' if h1a_go else 'NO'}")
    print(f"H5 (MI > 0.5):        {'YES' if h5_go else 'NO'}")
    print(f"Decision:              {decision}")
    print(f"Report saved to:       {out / 'phase0_report.json'}")
    print(f"Dashboard saved to:    {out / 'phase0_dashboard.png'}")


if __name__ == "__main__":
    main()

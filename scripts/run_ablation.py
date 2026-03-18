"""Run 2x2 ablation matrix: Partition(METIS/BPE) x Tokenizer(SimVQ/RVQ).

Evaluates reconstruction quality for all 4 configs using the same
dataset and metrics.

Usage:
    python scripts/run_ablation.py \
        --checkpoints ckpt_c1.pt ckpt_c2.pt ckpt_c3.pt ckpt_c4.pt \
        --data_dirs data/patches/lvis_wide/seen_test \
        --output_dir results/ablation
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.model import MeshLexVQVAE
from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from src.evaluate import evaluate_reconstruction


CONFIGS = {
    "C1_METIS_SimVQ": {
        "model_class": "MeshLexVQVAE",
        "partition": "METIS",
        "tokenizer": "SimVQ",
    },
    "C2_METIS_RVQ": {
        "model_class": "MeshLexRVQVAE",
        "partition": "METIS",
        "tokenizer": "RVQ",
    },
    "C3_BPE_SimVQ": {
        "model_class": "MeshLexVQVAE",
        "partition": "BPE",
        "tokenizer": "SimVQ",
    },
    "C4_BPE_RVQ": {
        "model_class": "MeshLexRVQVAE",
        "partition": "BPE",
        "tokenizer": "RVQ",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs=4, required=True,
                        help="Checkpoint paths for C1 C2 C3 C4")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="Evaluation data directories (METIS-partitioned)")
    parser.add_argument("--bpe_data_dirs", nargs="*", default=None,
                        help="BPE-partitioned data directories (for C3, C4)")
    parser.add_argument("--output_dir", default="results/ablation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    config_names = list(CONFIGS.keys())

    for i, (name, config) in enumerate(CONFIGS.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating {name}")
        print(f"{'='*60}")

        ckpt_path = args.checkpoints[i]
        if not Path(ckpt_path).exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            continue

        # Select data dirs based on partition type
        if config["partition"] == "BPE" and args.bpe_data_dirs:
            data_dirs = args.bpe_data_dirs
        else:
            data_dirs = args.data_dirs

        datasets = [PatchGraphDataset(d) for d in data_dirs]
        dataset = ConcatDataset(datasets)

        # Load model
        if config["model_class"] == "MeshLexRVQVAE":
            model = MeshLexRVQVAE()
        else:
            model = MeshLexVQVAE(num_kv_tokens=4)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(device)

        # Evaluate
        metrics = evaluate_reconstruction(model, dataset, device=device)

        # RVQ: compute per-level utilization
        if config["model_class"] == "MeshLexRVQVAE":
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            all_indices = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    z = model.encoder(batch.x, batch.edge_index, batch.batch)
                    _, indices = model.rvq(z)
                    all_indices.append(indices.cpu())
            all_idx = torch.cat(all_indices, dim=0)
            per_level_util = []
            for lvl in range(all_idx.shape[1]):
                per_level_util.append(all_idx[:, lvl].unique().numel() / model.rvq.K)
            metrics["utilization"] = sum(per_level_util) / len(per_level_util)
            metrics["per_level_utilization"] = per_level_util

        results[name] = metrics
        print(f"  CD: {metrics['mean_cd']:.1f}, Util: {metrics['utilization']:.1%}")

        torch.cuda.empty_cache()

    # Save results (convert non-serializable items)
    serializable = {}
    for name, m in results.items():
        sm = {}
        for k, v in m.items():
            if isinstance(v, dict):
                sm[k] = {str(kk): vv for kk, vv in v.items()}
            elif isinstance(v, list):
                sm[k] = v
            else:
                sm[k] = float(v) if hasattr(v, '__float__') else v
        serializable[name] = sm

    with open(out / "ablation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"2x2 Ablation Results")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'CD':>8} {'Util':>8}")
    print(f"{'-'*36}")
    for name, m in results.items():
        print(f"{name:<20} {m['mean_cd']:8.1f} {m['utilization']:8.1%}")

    print(f"\nResults saved to {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()

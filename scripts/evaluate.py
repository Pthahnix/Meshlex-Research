"""Run full evaluation: same-category CD, cross-category CD, Go/No-Go."""
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset
from src.evaluate import evaluate_reconstruction, compute_go_nogo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--same_cat_dirs", nargs="+", required=True,
                        help="Test patches from training categories (chair/table/airplane)")
    parser.add_argument("--cross_cat_dirs", nargs="+", required=True,
                        help="Test patches from held-out categories (car/lamp)")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MeshLexVQVAE(codebook_size=args.codebook_size, embed_dim=args.embed_dim)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print("Model loaded.")

    # Same-category evaluation
    same_ds = ConcatDataset([PatchGraphDataset(d) for d in args.same_cat_dirs])
    print(f"Same-category test patches: {len(same_ds)}")
    same_results = evaluate_reconstruction(model, same_ds, device)
    print(f"Same-cat CD: {same_results['mean_cd']:.4f} (x10^3)")
    print(f"Codebook utilization: {same_results['utilization']:.1%}")

    # Cross-category evaluation
    cross_ds = ConcatDataset([PatchGraphDataset(d) for d in args.cross_cat_dirs])
    print(f"\nCross-category test patches: {len(cross_ds)}")
    cross_results = evaluate_reconstruction(model, cross_ds, device)
    print(f"Cross-cat CD: {cross_results['mean_cd']:.4f} (x10^3)")

    # Go/No-Go
    decision = compute_go_nogo(same_results["mean_cd"], cross_results["mean_cd"])
    print(f"\n{'='*50}")
    print(f"CD Ratio (cross/same): {decision['ratio']:.2f}x")
    print(f"Decision: {decision['decision']}")
    print(f"Next step: {decision['next_step']}")
    print(f"{'='*50}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "same_category": {k: v for k, v in same_results.items() if k != "code_histogram"},
        "cross_category": {k: v for k, v in cross_results.items() if k != "code_histogram"},
        "go_nogo": decision,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

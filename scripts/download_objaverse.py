"""Download Objaverse-LVIS 3D objects for MeshLex experiments."""
import argparse
import json
import random
from pathlib import Path
from collections import Counter

import objaverse

# 实验 1：5-Category 精确匹配
FIVE_CAT = {
    "chair":    "chair",
    "table":    "table",
    "airplane": "airplane",
    "car":      "car_(automobile)",
    "lamp":     "lamp",
}


def select_5cat(lvis):
    """Select UIDs for 5-category experiment."""
    selected = {}
    for our_name, lvis_tag in FIVE_CAT.items():
        uids = lvis.get(lvis_tag, [])
        selected[our_name] = uids
        print(f"  {our_name} ({lvis_tag}): {len(uids)} objects")
    return selected


def select_lvis_wide(lvis, min_per_cat=10, max_per_cat=10, seed=42):
    """Select UIDs for LVIS-wide experiment: sample from all large-enough categories."""
    rng = random.Random(seed)
    selected = {}
    for cat_name, uids in sorted(lvis.items()):
        if len(uids) >= min_per_cat:
            sampled = rng.sample(uids, min(max_per_cat, len(uids)))
            selected[cat_name] = sampled
    print(f"  {len(selected)} categories selected, "
          f"{sum(len(v) for v in selected.values())} total objects")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["5cat", "lvis_wide"], required=True)
    parser.add_argument("--output_dir", type=str, default="data/objaverse")
    parser.add_argument("--max_per_cat", type=int, default=10,
                        help="Max objects per category (lvis_wide mode)")
    parser.add_argument("--min_per_cat", type=int, default=10,
                        help="Min objects to include a category (lvis_wide mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir) / args.mode
    out.mkdir(parents=True, exist_ok=True)

    print("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    print(f"LVIS: {len(lvis)} categories, {sum(len(v) for v in lvis.values())} objects")

    if args.mode == "5cat":
        selected = select_5cat(lvis)
    else:
        selected = select_lvis_wide(
            lvis, min_per_cat=args.min_per_cat,
            max_per_cat=args.max_per_cat, seed=args.seed,
        )

    # Collect all UIDs
    all_uids = []
    uid_to_cat = {}
    for cat_name, uids in selected.items():
        for uid in uids:
            all_uids.append(uid)
            uid_to_cat[uid] = cat_name

    print(f"\nDownloading {len(all_uids)} objects...")
    objects = objaverse.load_objects(uids=all_uids)

    # Build manifest
    manifest = []
    for uid, glb_path in objects.items():
        manifest.append({
            "uid": uid,
            "category": uid_to_cat[uid],
            "glb_path": str(glb_path),
        })

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path} ({len(manifest)} objects)")

    # Summary
    cat_counts = Counter(m["category"] for m in manifest)
    print(f"\nCategory summary ({len(cat_counts)} categories):")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cat}: {count}")
    if len(cat_counts) > 20:
        print(f"  ... and {len(cat_counts) - 20} more")


if __name__ == "__main__":
    main()

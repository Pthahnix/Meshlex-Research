# MeshLex Experiment Execution: Phase A+B Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Download ShapeNet data from Hugging Face, preprocess all 5 categories into patches, and run a quick training validation (Encoder-Only 20 epochs + K-means init + VQ-VAE 10 epochs) to confirm the full pipeline works before committing to 200-epoch training.

**Architecture:** HuggingFace download → zip extraction → pyfqmr decimation → METIS patch segmentation → NPZ serialization → staged VQ-VAE training (encoder-only → K-means codebook init → full VQ). RTX 4090 24GB, 503GB RAM, 57GB disk.

**Tech Stack:** Python 3.11, huggingface_hub 1.6.0, PyTorch 2.4.1+cu124, torch-geometric, trimesh, pyfqmr, pymetis, sklearn, matplotlib.

**Important context:**
- All `src/` modules are already implemented and tested (17 tests passing)
- `scripts/run_preprocessing.py` exists but needs a fix for ShapeNet model_id extraction
- ShapeNet structure inside zip: `{synset_id}/{model_id}/models/model_normalized.obj`
- The existing script does `mesh_id = obj_file.parent.name` which gives `"models"` instead of the actual model ID — must be fixed to use `obj_file.parent.parent.name`
- Train/test split: Chair/Table/Airplane → 400 train + 100 test each; Car/Lamp → 500 each for cross-category test
- CLAUDE.md rules: commit per functional unit, push after every commit, real data validation with visible outputs saved to `results/`

---

## Task 1: Download ShapeNet from Hugging Face

**Files:**
- Create: `scripts/download_shapenet.py`

**Step 1: Create download script**

The script downloads 5 category zip files from `ShapeNet/ShapeNetCore` and extracts only `model_normalized.obj` files into `data/ShapeNetCore.v2/{synset_id}/{model_id}/models/model_normalized.obj`.

```python
"""Download ShapeNet categories from Hugging Face and extract OBJ files."""
import argparse
import zipfile
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

CATEGORIES = {
    "chair":    "03001627",
    "table":    "04379243",
    "airplane": "02691156",
    "car":      "02958343",
    "lamp":     "03636649",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="data/ShapeNetCore.v2")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()),
                        help="Categories to download (default: all 5)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache directory (default: ~/.cache/huggingface)")
    args = parser.parse_args()

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    for cat_name in args.categories:
        if cat_name not in CATEGORIES:
            print(f"Unknown category: {cat_name}")
            continue
        cat_id = CATEGORIES[cat_name]
        zip_name = f"{cat_id}.zip"

        print(f"\n{'='*60}")
        print(f"Downloading {cat_name} ({cat_id})...")
        print(f"{'='*60}")

        try:
            zip_path = hf_hub_download(
                repo_id="ShapeNet/ShapeNetCore",
                filename=zip_name,
                repo_type="dataset",
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            print(f"ERROR downloading {cat_name}: {e}")
            print("Make sure you have accepted the dataset terms at:")
            print("  https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
            print("And logged in with: huggingface-cli login")
            continue

        # Extract only model_normalized.obj files
        cat_out = output / cat_id
        print(f"Extracting OBJ files to {cat_out}...")
        n_extracted = 0

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith("model_normalized.obj"):
                    zf.extract(member, str(output))
                    n_extracted += 1

        print(f"Extracted {n_extracted} OBJ files for {cat_name}")

    # Summary
    print(f"\n{'='*60}")
    print("Download complete. Summary:")
    for cat_name in args.categories:
        cat_id = CATEGORIES.get(cat_name)
        if cat_id:
            cat_dir = output / cat_id
            n_objs = len(list(cat_dir.rglob("model_normalized.obj"))) if cat_dir.exists() else 0
            print(f"  {cat_name} ({cat_id}): {n_objs} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

**Step 2: User must first login to HuggingFace**

Before running the script, the user needs to:
```bash
huggingface-cli login
# Then accept dataset terms at https://huggingface.co/datasets/ShapeNet/ShapeNetCore
```

**Step 3: Run the download**

```bash
python scripts/download_shapenet.py --output_root data/ShapeNetCore.v2
```

Expected: 5 category directories created under `data/ShapeNetCore.v2/`, each with hundreds of OBJ files. Each category file structure: `{synset_id}/{model_id}/models/model_normalized.obj`.

**Step 4: Verify download**

```bash
python -c "
from pathlib import Path
root = Path('data/ShapeNetCore.v2')
for cat_id in ['02691156','02958343','03001627','03636649','04379243']:
    n = len(list((root / cat_id).rglob('model_normalized.obj'))) if (root / cat_id).exists() else 0
    print(f'{cat_id}: {n} models')
"
```

Expected output (approximate):
```
02691156: ~4000 models  (airplane)
02958343: ~7000 models  (car)
03001627: ~6778 models  (chair)
03636649: ~2318 models  (lamp)
04379243: ~8509 models  (table)
```

**Step 5: Commit**

```bash
git add scripts/download_shapenet.py
git commit -m "feat: add ShapeNet download script from HuggingFace"
git push
```

---

## Task 2: Fix Preprocessing Script for ShapeNet Structure

**Files:**
- Modify: `scripts/run_preprocessing.py` (line 45 — mesh_id extraction)
- Modify: `scripts/run_preprocessing.py` (add train/test split)
- Modify: `src/data_prep.py` (line 70 — same mesh_id bug in `preprocess_shapenet_category`)

**Step 1: Fix mesh_id extraction in `run_preprocessing.py`**

The current code at line 45:
```python
mesh_id = obj_file.parent.name  # BUG: gives "models" for ShapeNet structure
```

ShapeNet path is: `{synset_id}/{model_id}/models/model_normalized.obj`
So `obj_file.parent.name` = `"models"` (wrong), need `obj_file.parent.parent.name`.

Fix to:
```python
# ShapeNet: .../model_id/models/model_normalized.obj → parent.parent.name = model_id
# Fallback: .../model_id/model.obj → parent.name = model_id
if obj_file.parent.name == "models":
    mesh_id = obj_file.parent.parent.name
else:
    mesh_id = obj_file.parent.name
```

Apply same fix in `src/data_prep.py` line 70 (`preprocess_shapenet_category` function).

**Step 2: Add train/test split to preprocessing**

Add arguments to `run_preprocessing.py`:
- `--train_categories` (default: chair table airplane)
- `--test_split_ratio` (default: 0.2 — 80% train, 20% test for train categories)
- `--seed` (default: 42)

After processing each train category, split its patch directory into `data/patches/{category}_train/` and `data/patches/{category}_test/` by mesh_id.

Car and Lamp (cross-category) go entirely to their own directories (no split needed).

```python
import random

# After processing all meshes for a train category, split patches by mesh_id:
def split_patches_by_mesh(patch_dir, category, metadata_entries, test_ratio=0.2, seed=42):
    """Move patches into _train and _test subdirs based on mesh_id split."""
    rng = random.Random(seed)
    mesh_ids = list(set(m["mesh_id"] for m in metadata_entries if m.get("category") == category))
    rng.shuffle(mesh_ids)
    n_test = max(1, int(len(mesh_ids) * test_ratio))
    test_ids = set(mesh_ids[:n_test])
    train_ids = set(mesh_ids[n_test:])

    patch_path = Path(patch_dir) / category
    train_path = Path(patch_dir) / f"{category}_train"
    test_path = Path(patch_dir) / f"{category}_test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for npz_file in patch_path.glob("*.npz"):
        # mesh_id is the part before _patch_XXX.npz
        file_mesh_id = npz_file.stem.rsplit("_patch_", 1)[0]
        if file_mesh_id in test_ids:
            npz_file.rename(test_path / npz_file.name)
        else:
            npz_file.rename(train_path / npz_file.name)

    # Remove now-empty original dir
    if patch_path.exists() and not any(patch_path.glob("*.npz")):
        patch_path.rmdir()

    return len(train_ids), len(test_ids)
```

**Step 3: Run test on existing raw_samples to verify fix**

```bash
python scripts/run_preprocessing.py \
    --shapenet_root data/raw_samples \
    --output_root data/test_fix \
    --target_faces 200 \
    --max_per_category 2
```

Expected: should not crash. Mesh IDs should be meaningful (not "models").

**Step 4: Commit**

```bash
git add scripts/run_preprocessing.py src/data_prep.py
git commit -m "fix: correct mesh_id extraction for ShapeNet dir structure + add train/test split"
git push
```

---

## Task 3: Run Full Preprocessing on ShapeNet

**Files:**
- No code changes — running existing scripts

**Step 1: Run preprocessing (all 5 categories, max 500 each)**

```bash
python scripts/run_preprocessing.py \
    --shapenet_root data/ShapeNetCore.v2 \
    --output_root data \
    --target_faces 1000 \
    --max_per_category 500
```

Expected runtime: ~1-2 hours. Expected output:
- `data/meshes/{category}/` — preprocessed OBJ files
- `data/patches/{category}/` (then split into `_train`/`_test`) — NPZ patches
- `data/patch_metadata.json` — metadata

**Step 2: Split train categories into train/test**

Run the split for chair, table, airplane (400 train / 100 test each).
Car and Lamp stay as-is (cross-category test only).

**Step 3: Verify patch statistics**

```bash
python -c "
import json, numpy as np
meta = json.load(open('data/patch_metadata.json'))
cats = {}
for m in meta:
    c = m['category']
    cats.setdefault(c, []).append(m['n_patches'])
for c, patches in sorted(cats.items()):
    p = np.array(patches)
    total_faces = [sum(fc) for fc in [m['face_counts'] for m in meta if m['category']==c]]
    print(f'{c}: {len(patches)} meshes, {p.sum()} patches, median {np.median(p):.0f}/mesh, mean faces/patch {np.mean([f for fc in [m[\"face_counts\"] for m in meta if m[\"category\"]==c] for f in fc]):.1f}')
"
```

Expected: each category ~400-500 meshes, ~25-35 patches/mesh (for 1000 faces / 35 faces per patch).

**Step 4: Verify train/test split**

```bash
for cat in chair table airplane; do
    echo "=== $cat ==="
    echo "  train: $(ls data/patches/${cat}_train/*.npz 2>/dev/null | wc -l) patches"
    echo "  test:  $(ls data/patches/${cat}_test/*.npz 2>/dev/null | wc -l) patches"
done
echo "=== Cross-category ==="
echo "  car:  $(ls data/patches/car/*.npz 2>/dev/null | wc -l) patches"
echo "  lamp: $(ls data/patches/lamp/*.npz 2>/dev/null | wc -l) patches"
```

**Step 5: Save validation report**

Save a brief markdown report to `results/phase_a_validation/report.md` with:
- Number of meshes and patches per category
- Patch size distribution (histogram plot)
- 3 sample mesh preview images with patch coloring
- Train/test split counts

**Step 6: Commit**

```bash
git add results/phase_a_validation/ data/patch_metadata.json
git commit -m "data: Phase A complete — 5 categories preprocessed, train/test split done"
git push
```

Note: Do NOT commit the large data files (meshes, patches, checkpoints). Add them to `.gitignore` if not already there.

---

## Task 4: Phase B — Encoder-Only Training (20 Epochs)

**Files:**
- No code changes — using existing `scripts/train.py`

**Step 1: Verify .gitignore excludes large data**

Ensure `data/ShapeNetCore.v2/`, `data/meshes/`, `data/patches/`, `data/checkpoints/` are in `.gitignore`.

**Step 2: Run encoder-only training**

```bash
python scripts/train.py \
    --train_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
    --val_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 999 \
    --checkpoint_dir data/checkpoints
```

Expected: 20 epochs, ~5-10 min/epoch on RTX 4090. `recon_loss` should steadily decrease.
Output: `data/checkpoints/checkpoint_epoch019.pt`

**Step 3: Verify training output**

Check that:
- Checkpoint file exists and is loadable
- `recon_loss` decreased over 20 epochs
- No NaN/Inf in losses

```bash
python -c "
import torch
ckpt = torch.load('data/checkpoints/checkpoint_epoch019.pt', map_location='cpu', weights_only=False)
print(f'Epoch: {ckpt[\"epoch\"]}')
history = ckpt['history']
for h in history:
    print(f'  Epoch {h[\"epoch\"]:3d}: recon={h[\"recon_loss\"]:.6f}')
print(f'Loss decreased: {history[0][\"recon_loss\"] > history[-1][\"recon_loss\"]}')
"
```

**Step 4: Commit**

```bash
git add -f data/checkpoints/training_history.json
git commit -m "train: Phase B encoder-only 20 epochs complete"
git push
```

---

## Task 5: K-means Codebook Initialization

**Files:**
- No code changes — using existing `scripts/init_codebook.py`

**Step 1: Run K-means initialization**

```bash
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/checkpoint_epoch019.pt \
    --patch_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
    --codebook_size 4096 \
    --output data/checkpoints/checkpoint_kmeans_init.pt
```

Expected: Collects all encoder embeddings (~50K-75K patches), runs MiniBatchKMeans(K=4096), initializes codebook.
Runtime: ~5-10 min.

**Step 2: Verify**

```bash
python -c "
import torch
ckpt = torch.load('data/checkpoints/checkpoint_kmeans_init.pt', map_location='cpu', weights_only=False)
cb = ckpt['model_state_dict']['codebook.codebook.weight']
print(f'Codebook shape: {cb.shape}')
print(f'Codebook norm range: [{cb.norm(dim=1).min():.4f}, {cb.norm(dim=1).max():.4f}]')
print(f'Unique entries: {cb.unique(dim=0).shape[0]} / {cb.shape[0]}')
"
```

Expected: shape (4096, 128), all unique entries (no duplicates), reasonable norm range.

**Step 3: Commit**

```bash
git commit -m "train: K-means codebook initialization complete"
git push
```

---

## Task 6: Quick VQ-VAE Training (10-20 Epochs)

**Files:**
- No code changes — using existing `scripts/train.py`

**Step 1: Run VQ-VAE training from K-means checkpoint**

```bash
python scripts/train.py \
    --train_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
    --val_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
    --resume data/checkpoints/checkpoint_kmeans_init.pt \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints_vq
```

Expected: VQ loss should converge, codebook utilization should be > 30% (ideally > 50%).
Runtime: ~30-60 min for 20 epochs.

**Step 2: Monitor key metrics**

During training, watch for:
- `recon_loss` trending downward
- `codebook_utilization` > 30% (if < 30%, codebook is collapsing)
- No NaN/Inf

**Step 3: Quick evaluation**

```bash
python scripts/evaluate.py \
    --checkpoint data/checkpoints_vq/checkpoint_final.pt \
    --same_cat_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
    --cross_cat_dirs data/patches/car data/patches/lamp \
    --output results/phase_b_quick_eval.json
```

**Step 4: Quick visualization**

```bash
python scripts/visualize.py \
    --checkpoint data/checkpoints_vq/checkpoint_final.pt \
    --history data/checkpoints_vq/training_history.json \
    --patch_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
    --output_dir results/phase_b_validation
```

**Step 5: Save validation report**

Create `results/phase_b_validation/report.md` with:
- Training curves (loss, utilization over 20 VQ epochs)
- Quick eval results (same-cat CD, cross-cat CD, utilization)
- Preliminary Go/No-Go assessment
- Recommendation: proceed to full 200 epochs or stop

**Step 6: Commit**

```bash
git add results/phase_b_validation/ results/phase_b_quick_eval.json
git commit -m "eval: Phase B quick VQ validation — ready for full training assessment"
git push
```

---

## Decision Point

After Task 6, review the Phase B results:

- **If codebook utilization > 30% and losses converge**: Proceed to Phase C-G (full 200-epoch training + final evaluation + Go/No-Go). This would be a follow-up plan.
- **If codebook collapses (util < 10%) or losses diverge**: Debug before continuing. Possible issues: learning rate too high, K-means init failed, data quality problem.
- **If losses don't decrease at all**: Check data pipeline — patches may be degenerate.

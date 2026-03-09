# MeshLex Validation Experiment — End-to-End Run Guide

本文档记录从 Objaverse-LVIS 数据到最终 Go/No-Go 决策的完整运行流程。

数据源：**Objaverse-LVIS**（46K objects, 1156 categories，无需审批）
实验方案：双实验（5-Category + LVIS-Wide），详见 `context/10_objaverse_migration_design.md`

---

## Prerequisites

```bash
# Python 3.11+, CUDA 12.x
pip install -r requirements.txt
pip install objaverse
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

---

## Experiment 1: 5-Category

### Phase A1: Data Preparation

#### 1. 下载 5 类 Objaverse-LVIS 数据

```bash
python scripts/download_objaverse.py --mode 5cat --output_dir data/objaverse
```

产出：`data/objaverse/5cat/manifest.json`（~847 objects: chair 453, table 101, airplane 112, car 102, lamp 79）

#### 2. 预处理：降面 + 归一化 + Patch 分割

```bash
python scripts/run_preprocessing.py \
    --input_manifest data/objaverse/5cat/manifest.json \
    --experiment_name 5cat \
    --output_root data \
    --target_faces 1000
```

产出：
- `data/meshes/5cat/{category}/` — 预处理后的 OBJ 文件
- `data/patches/5cat/{category}_train/` — 训练 patches（chair/table/airplane）
- `data/patches/5cat/{category}_test/` — 测试 patches（chair/table/airplane）
- `data/patches/5cat/{car,lamp}/` — 跨类别测试 patches
- `data/patch_metadata_5cat.json`

#### 3. 验证 patch 统计

```bash
python -c "
import json, numpy as np
meta = json.load(open('data/patch_metadata_5cat.json'))
cats = {}
for m in meta:
    c = m['category']
    cats.setdefault(c, []).append(m['n_patches'])
for c, patches in sorted(cats.items()):
    p = np.array(patches)
    print(f'{c}: {len(patches)} meshes, {p.sum()} patches, median {np.median(p):.0f} patches/mesh')
"
```

### Phase B1: Encoder-Only Training (20 Epochs)

```bash
python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 999 \
    --checkpoint_dir data/checkpoints/5cat
```

> `vq_start_epoch=999` 表示这 20 epochs 内不启用 VQ loss。

### Phase C1: K-means Codebook Initialization

```bash
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/5cat/checkpoint_epoch019.pt \
    --patch_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --codebook_size 4096 \
    --output data/checkpoints/5cat/checkpoint_kmeans_init.pt
```

### Phase D1: Quick VQ-VAE Training (20 Epochs)

```bash
python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --resume data/checkpoints/5cat/checkpoint_kmeans_init.pt \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints/5cat_vq
```

监控：`recon_loss` 下降，`codebook_utilization` > 30%。

### Phase E1: Evaluate + Visualize

```bash
python scripts/evaluate.py \
    --checkpoint data/checkpoints/5cat_vq/checkpoint_final.pt \
    --same_cat_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --cross_cat_dirs data/patches/5cat/car data/patches/5cat/lamp \
    --output results/exp1_eval.json

python scripts/visualize.py \
    --checkpoint data/checkpoints/5cat_vq/checkpoint_final.pt \
    --history data/checkpoints/5cat_vq/training_history.json \
    --patch_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --output_dir results/exp1_plots
```

---

## Experiment 2: LVIS-Wide

### Phase A2: Data Preparation

#### 1. 下载 LVIS 广采样数据

```bash
python scripts/download_objaverse.py \
    --mode lvis_wide \
    --output_dir data/objaverse \
    --min_per_cat 10 \
    --max_per_cat 10
```

产出：`data/objaverse/lvis_wide/manifest.json`（~500+ 类别，~3000-5000 objects）

#### 2. 预处理（category-holdout split）

```bash
python scripts/run_preprocessing.py \
    --input_manifest data/objaverse/lvis_wide/manifest.json \
    --experiment_name lvis_wide \
    --output_root data \
    --target_faces 1000 \
    --split_mode category_holdout \
    --holdout_categories 50 \
    --seed 42
```

产出：
- `data/patches/lvis_wide/seen_train/` — 训练 patches
- `data/patches/lvis_wide/seen_test/` — seen 类别测试 patches
- `data/patches/lvis_wide/unseen/` — 50 个 unseen 类别 patches

### Phase B2-D2: Training

```bash
# Encoder-Only
python scripts/train.py \
    --train_dirs data/patches/lvis_wide/seen_train \
    --val_dirs data/patches/lvis_wide/seen_test \
    --epochs 20 --batch_size 256 --lr 1e-4 \
    --vq_start_epoch 999 --checkpoint_dir data/checkpoints/lvis_wide

# K-means Init
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/lvis_wide/checkpoint_epoch019.pt \
    --patch_dirs data/patches/lvis_wide/seen_train \
    --codebook_size 4096 \
    --output data/checkpoints/lvis_wide/checkpoint_kmeans_init.pt

# VQ-VAE
python scripts/train.py \
    --train_dirs data/patches/lvis_wide/seen_train \
    --val_dirs data/patches/lvis_wide/seen_test \
    --resume data/checkpoints/lvis_wide/checkpoint_kmeans_init.pt \
    --epochs 20 --batch_size 256 --lr 1e-4 \
    --vq_start_epoch 0 --checkpoint_dir data/checkpoints/lvis_wide_vq
```

### Phase E2: Evaluate + Visualize

```bash
python scripts/evaluate.py \
    --checkpoint data/checkpoints/lvis_wide_vq/checkpoint_final.pt \
    --same_cat_dirs data/patches/lvis_wide/seen_test \
    --cross_cat_dirs data/patches/lvis_wide/unseen \
    --output results/exp2_eval.json

python scripts/visualize.py \
    --checkpoint data/checkpoints/lvis_wide_vq/checkpoint_final.pt \
    --history data/checkpoints/lvis_wide_vq/training_history.json \
    --patch_dirs data/patches/lvis_wide/seen_train \
    --output_dir results/exp2_plots
```

---

## Go/No-Go Decision

综合两组实验，按以下矩阵决策：

| Cross/Same CD Ratio | Utilization | Decision | Action |
|---------------------|-------------|----------|--------|
| < 1.2x | > 50% | **STRONG GO** | 推进完整 MeshLex 论文 |
| 1.2x - 2.0x | > 50% | **WEAK GO** | 调整 story 为 "transferable vocabulary"，继续 |
| < 2.0x | 30% - 50% | **CONDITIONAL GO** | 增大 K 或尝试 RQ-VAE |
| 2.0x - 3.0x | any | **HOLD** | 分析失败原因 |
| > 3.0x | any | **NO-GO** | 核心假设被推翻，pivot |

实验 2 额外关注：unseen 50 categories 的 CD 是否和 seen categories 接近。

---

## Full Training (Phase F, Go 后执行)

选表现好的实验组，训练 200 epochs：

### A-stage 全量训练（已完成）

5-Category:
```bash
PYTHONPATH=. python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --epochs 200 --batch_size 256 --lr 1e-4 \
    --warmup_epochs 5 --dead_code_interval 10 \
    --checkpoint_dir data/checkpoints/5cat_v2
```

LVIS-Wide:
```bash
PYTHONPATH=. python scripts/train.py \
    --train_dirs data/patches/lvis_wide/seen_train \
    --val_dirs data/patches/lvis_wide/seen_test \
    --epochs 200 --batch_size 256 --lr 1e-4 \
    --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 10 \
    --checkpoint_dir data/checkpoints/lvis_wide_A
```

### B-stage 训练（在 A-stage checkpoint 上继续）

5-Category (已完成):
```bash
PYTHONPATH=. python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --epochs 200 --batch_size 256 --lr 1e-4 \
    --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 0 \
    --num_kv_tokens 4 \
    --resume data/checkpoints/5cat_v2/checkpoint_final.pt \
    --checkpoint_dir data/checkpoints/5cat_B
```

LVIS-Wide (待执行):
```bash
PYTHONPATH=. python scripts/train.py \
    --train_dirs data/patches/lvis_wide/seen_train \
    --val_dirs data/patches/lvis_wide/seen_test \
    --epochs 200 --batch_size 256 --lr 1e-4 \
    --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 0 \
    --num_kv_tokens 4 \
    --resume data/checkpoints/lvis_wide_A/checkpoint_final.pt \
    --checkpoint_dir data/checkpoints/lvis_wide_B
```

**注意**: 不要用 `--use_rotation`，rotation trick 与 SimVQ 不兼容（会导致 collapse）。

### 评估

```bash
PYTHONPATH=. python scripts/evaluate.py \
    --checkpoint data/checkpoints/<experiment>/checkpoint_final.pt \
    --same_cat_dirs <same_category_test_dirs> \
    --cross_cat_dirs <cross_category_test_dirs> \
    --output results/<experiment>/eval_results.json
```

### 可视化

```bash
PYTHONPATH=. python scripts/visualize.py \
    --checkpoint data/checkpoints/<experiment>/checkpoint_final.pt \
    --history data/checkpoints/<experiment>/training_history.json \
    --patch_dirs <train_patch_dirs> \
    --output_dir results/<experiment>
```

---

## Disk Budget

| 阶段 | 实验 1 | 实验 2 | 合计 |
|------|--------|--------|------|
| GLB 下载 | ~2GB | ~10GB | ~12GB |
| 预处理 OBJ | ~500MB | ~2GB | ~2.5GB |
| Patches NPZ | ~500MB | ~2GB | ~2.5GB |
| Checkpoints | ~500MB | ~500MB | ~1GB |
| **Total** | **~3.5GB** | **~14.5GB** | **~18GB** |

可用磁盘 60GB，安全余量充足。

---

## Estimated Time

| Phase | Exp 1 (5-cat) | Exp 2 (LVIS-wide) |
|-------|---------------|-------------------|
| Data Prep | ~30min | ~2h |
| Encoder-Only 20ep | ~1h | ~2h |
| K-means Init | ~5min | ~10min |
| VQ-VAE 20ep | ~30min | ~1h |
| Evaluate + Visualize | ~15min | ~30min |
| **Subtotal** | **~2.5h** | **~5.5h** |
| Full 200ep (if Go) | ~6h | ~10h |

GPU：RTX 4090 24GB。

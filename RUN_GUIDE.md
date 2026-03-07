# MeshLex Validation Experiment — End-to-End Run Guide

本文档记录从原始 ShapeNet 数据到最终 Go/No-Go 决策的完整运行流程。

---

## Prerequisites

```bash
# Python 3.11+, CUDA 12.x
pip install -r requirements.txt
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

---

## Phase A: Data Preparation

### 1. 获取 ShapeNet Core v2

将 ShapeNet Core v2 放置在 `data/ShapeNetCore.v2/`，需要以下 5 个类别：

| Category | ShapeNet ID | Role |
|----------|-------------|------|
| Chair | 03001627 | Training |
| Table | 04379243 | Training |
| Airplane | 02691156 | Training |
| Car | 02958343 | Cross-category Test |
| Lamp | 03636649 | Cross-category Test |

### 2. Preprocess: 降面 + 归一化 + Patch 分割

```bash
python scripts/run_preprocessing.py \
    --shapenet_root data/ShapeNetCore.v2 \
    --output_root data \
    --target_faces 1000 \
    --max_per_category 500
```

产出：
- `data/meshes/{category}/` — 预处理后的 OBJ 文件
- `data/patches/{category}/` — 每个 patch 的 NPZ 文件
- `data/patch_metadata.json` — 元数据

### 3. 验证 patch 统计

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
    print(f'{c}: {len(patches)} meshes, {p.sum()} patches, median {np.median(p):.0f} patches/mesh')
"
```

预期：每个类别 ~400-500 meshes，每个 mesh ~28 patches（1000 faces / 35 faces per patch）。

---

## Phase B: Train Encoder-Only (Epochs 0-19)

先只训练 encoder + decoder（不启用 VQ loss），让 encoder 学到有意义的 embedding。

```bash
python scripts/train.py \
    --train_dirs data/patches/chair data/patches/table data/patches/airplane \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 999 \
    --checkpoint_dir data/checkpoints
```

> `vq_start_epoch=999` 表示这 20 epochs 内不启用 VQ loss。

---

## Phase C: K-means Codebook Initialization

用训练好的 encoder 输出做 K-means，初始化 codebook（VQGAN-LC 策略，避免 codebook collapse）。

```bash
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/checkpoint_epoch019.pt \
    --patch_dirs data/patches/chair data/patches/table data/patches/airplane \
    --codebook_size 4096 \
    --output data/checkpoints/checkpoint_kmeans_init.pt
```

---

## Phase D: Full VQ-VAE Training (Epochs 0-200)

从 K-means 初始化的 checkpoint 恢复，启用完整 VQ loss。

```bash
python scripts/train.py \
    --train_dirs data/patches/chair data/patches/table data/patches/airplane \
    --resume data/checkpoints/checkpoint_kmeans_init.pt \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints
```

监控要点：
- `recon_loss` 应持续下降
- `codebook_utilization` 应在 50%+ 以上（低于 30% 说明 collapse）

---

## Phase E: Evaluate

### Same-category test (训练类别的 held-out patches)

需要预先将 chair/table/airplane 的 patches 按 mesh_id 划分为 train/test。

### Cross-category test (Car + Lamp — 训练时从未见过)

```bash
python scripts/evaluate.py \
    --checkpoint data/checkpoints/checkpoint_final.pt \
    --same_cat_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
    --cross_cat_dirs data/patches/car data/patches/lamp \
    --output results/eval_results.json
```

---

## Phase F: Visualize

```bash
python scripts/visualize.py \
    --checkpoint data/checkpoints/checkpoint_final.pt \
    --history data/checkpoints/training_history.json \
    --patch_dirs data/patches/chair data/patches/table data/patches/airplane \
    --output_dir results/plots
```

产出：
- `results/plots/training_curves.png` — Loss + Utilization 曲线
- `results/plots/codebook_tsne.png` — Codebook embedding 的 t-SNE 可视化
- `results/plots/utilization_histogram.png` — Code 使用频率分布

---

## Phase G: Go/No-Go Decision

读取 `results/eval_results.json`，按以下矩阵做决策：

| Cross/Same CD Ratio | Utilization | Decision | Action |
|---------------------|-------------|----------|--------|
| < 1.2x | > 50% | **STRONG GO** | 推进完整 MeshLex 论文 |
| 1.2x - 2.0x | > 50% | **WEAK GO** | 调整 story 为 "transferable vocabulary"，继续 |
| < 2.0x | 30% - 50% | **CONDITIONAL GO** | 增大 K 或尝试 RQ-VAE |
| 2.0x - 3.0x | any | **HOLD** | 分析失败原因 |
| > 3.0x | any | **NO-GO** | 核心假设被推翻，pivot |

---

## Project File Structure

```
src/
├── data_prep.py          # Mesh 加载、降面、归一化
├── patch_segment.py      # METIS Patch 分割 + PCA 归一化
├── patch_dataset.py      # NPZ 序列化 + PyTorch/PyG Dataset
├── model.py              # PatchEncoder, SimVQCodebook, PatchDecoder, MeshLexVQVAE
├── losses.py             # Chamfer Distance loss
├── trainer.py            # Training loop
└── evaluate.py           # Evaluation metrics + Go/No-Go

scripts/
├── run_preprocessing.py  # 批量预处理 ShapeNet
├── train.py              # 训练入口 (支持 --resume)
├── init_codebook.py      # K-means codebook 初始化
├── evaluate.py           # 评估入口
└── visualize.py          # 可视化入口

tests/
├── test_data_prep.py     # 2 tests
├── test_patch_segment.py # 4 tests
├── test_patch_dataset.py # 3 tests
└── test_model.py         # 8 tests
```

---

## Estimated Time

| Phase | Time (estimated) |
|-------|-----------------|
| A: Data Prep | ~3h (download + preprocess) |
| B: Encoder-Only | ~2h (20 epochs) |
| C: K-means Init | ~10min |
| D: Full Training | ~6h (200 epochs) |
| E: Evaluate | ~30min |
| F: Visualize | ~10min |
| **Total** | **~12h** |

GPU 要求：单卡 RTX 3090 / A100 即可。CPU 训练可行但极慢。

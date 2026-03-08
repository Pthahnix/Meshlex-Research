# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-08 16:10
> **当前状态**: Exp3 B-stage 训练中 + LVIS-Wide 下载中（并行）

## 实验进度总览

| # | 实验 | 状态 | 结果 |
|---|------|------|------|
| 1 | A-stage × 5-Category | **完成** | STRONG GO (ratio 1.14x, util 46%) |
| 2 | A-stage × LVIS-Wide | **下载中** (10,560 objects) | — |
| 3 | B-stage × 5-Category | **训练中** (200 epochs) | — |
| 4 | B-stage × LVIS-Wide | 待执行 | — |

## 已完成

### Exp1: A-stage × 5cat — STRONG GO
- Checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt
- 结果: results/exp1_A_5cat/
- CD Ratio: 1.14x, Utilization: 46% (eval) / 99.7% (train)
- Commit: edb86c4

### B-stage 代码实现
- rotation_trick() + use_rotation flag
- PatchDecoder num_kv_tokens (1→4)
- 21 tests pass, +66K params (1.06M→1.13M)
- Commit: ec6e7c3

## 当前进行中

### Exp3: B-stage × 5cat 训练
- 命令: `python scripts/train.py --use_rotation --num_kv_tokens 4 --checkpoint_dir data/checkpoints/5cat_B ...`
- 预计 ~2.5h

### Exp2: LVIS-Wide 数据下载
- 命令: `python scripts/download_objaverse.py --mode lvis_wide`
- 1061 categories, 10,560 objects
- 下载后需要预处理

## 如果 session 中断，恢复步骤

### 1. 检查 Exp3 训练是否完成
```bash
ls data/checkpoints/5cat_B/checkpoint_final.pt
```
如果存在 → 跑评估；如果不存在 → resume 训练

### 2. 检查 LVIS-Wide 数据
```bash
ls data/objaverse/lvis_wide/manifest.json
```
如果存在 → 预处理 → 训练；如果不存在 → 重新下载

### 3. 评估命令
```bash
PYTHONPATH=. python scripts/evaluate.py \
  --checkpoint data/checkpoints/5cat_B/checkpoint_final.pt \
  --same_cat_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
  --cross_cat_dirs data/patches/5cat/car data/patches/5cat/lamp \
  --output results/exp3_B_5cat/eval_results.json
```

### 4. LVIS-Wide 预处理
```bash
PYTHONPATH=. python scripts/run_preprocessing.py \
  --input_manifest data/objaverse/lvis_wide/manifest.json \
  --experiment_name lvis_wide --output_root data --target_faces 1000
```

### 5. LVIS-Wide A-stage 训练
```bash
PYTHONPATH=. python scripts/train.py \
  --train_dirs <lvis_wide 训练目录> \
  --val_dirs <lvis_wide 验证目录> \
  --epochs 200 --batch_size 256 --lr 1e-4 \
  --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 10 \
  --checkpoint_dir data/checkpoints/lvis_wide_A
```

### 6. LVIS-Wide B-stage 训练
同上但加 `--use_rotation --num_kv_tokens 4 --checkpoint_dir data/checkpoints/lvis_wide_B`

## 重要文件
- B-stage 设计: context/19_codebook_collapse_fix_design.md 第四节
- 实验计划: context/11_objaverse_experiment_plan.md
- A-stage checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt

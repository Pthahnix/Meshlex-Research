# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-09 06:00
> **当前状态**: Exp2 LVIS-Wide A-stage 训练中（~9h remaining）

## 实验进度总览

| # | 实验 | 状态 | 结果 |
|---|------|------|------|
| 1 | A-stage × 5-Category | **完成** | STRONG GO (ratio 1.14x, util 46%) |
| 2 | A-stage × LVIS-Wide | **训练中** (epoch 14/200, ~9h left) | util 76% at epoch 14 |
| 3 | B-stage × 5-Category | **完成** | STRONG GO (ratio 1.18x, CD -6.2%, util 99%) |
| 4 | B-stage × LVIS-Wide | 待执行（Exp2 完成后） | — |

## 已完成

### Exp1: A-stage × 5cat — STRONG GO
- Checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt
- 结果: results/exp1_A_5cat/
- CD Ratio: 1.14x, Utilization: 46% (eval) / 99.7% (train)
- Commit: edb86c4

### Exp3: B-stage × 5cat — STRONG GO
- Checkpoint: data/checkpoints/5cat_B/checkpoint_final.pt
- 结果: results/exp3_B_5cat/
- CD Ratio: 1.18x, Same-cat CD: 223.5 (vs A-stage 238.3, -6.2%)
- Train Util: 99.0%, Eval Util: 47.1%
- **关键发现**: rotation trick 与 SimVQ 不兼容（连续 collapse），仅使用 num_kv_tokens=4
- Commit: 06b0ff3

### B-stage 代码实现
- rotation_trick() + use_rotation flag（但实际不可用）
- PatchDecoder num_kv_tokens (1→4)
- 跨阶段 resume 支持 (strict=False)
- 21 tests pass, +66K params (1.06M→1.13M)

### LVIS-Wide 数据准备
- 3183 objects, 844 categories (manifest 中 1061 类, 1467 meshes 成功预处理)
- 71,019 patches total
- Category holdout split: 794 seen + 50 unseen categories
- seen_train: 53,424 patches, seen_test: 13,315 patches, unseen: 4,167 patches

## 当前进行中

### Exp2: LVIS-Wide A-stage 训练
- 命令: `PYTHONPATH=. python scripts/train.py --train_dirs data/patches/lvis_wide/seen_train --val_dirs data/patches/lvis_wide/seen_test --epochs 200 --batch_size 256 --lr 1e-4 --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 10 --checkpoint_dir data/checkpoints/lvis_wide_A`
- 每 epoch ~185s, 预计总 ~10h
- K-means init 完成，post-init util 15.4%, epoch 14 util 76.2%

## 如果 session 中断，恢复步骤

### 1. 检查 Exp2 训练是否完成
```bash
ls data/checkpoints/lvis_wide_A/checkpoint_final.pt
```
如果存在 → 跑评估；如果不存在 → resume 训练

### 2. Resume Exp2 训练
```bash
PYTHONPATH=. python scripts/train.py \
  --train_dirs data/patches/lvis_wide/seen_train \
  --val_dirs data/patches/lvis_wide/seen_test \
  --epochs 200 --batch_size 256 --lr 1e-4 \
  --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 0 \
  --resume data/checkpoints/lvis_wide_A/checkpoint_epoch<LATEST>.pt \
  --checkpoint_dir data/checkpoints/lvis_wide_A
```

### 3. 评估 Exp2
```bash
PYTHONPATH=. python scripts/evaluate.py \
  --checkpoint data/checkpoints/lvis_wide_A/checkpoint_final.pt \
  --same_cat_dirs data/patches/lvis_wide/seen_test \
  --cross_cat_dirs data/patches/lvis_wide/unseen \
  --output results/exp2_A_lvis_wide/eval_results.json
```

### 4. Exp4: B-stage LVIS-Wide
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
**注意**: 不要用 `--use_rotation`，rotation trick 与 SimVQ 不兼容

## 重要文件
- B-stage 设计: context/19_codebook_collapse_fix_design.md 第四节
- 实验计划: context/11_objaverse_experiment_plan.md
- A-stage 5cat checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt
- B-stage 5cat checkpoint: data/checkpoints/5cat_B/checkpoint_final.pt
- Exp3 报告: results/exp3_B_5cat/report.md

# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-09 16:15
> **当前状态**: Exp2 完成 (STRONG GO)，待执行 Exp4 B-stage LVIS-Wide

## 实验进度总览

| # | 实验 | 状态 | 结果 |
|---|------|------|------|
| 1 | A-stage × 5-Category | **完成** | STRONG GO (ratio 1.14x, util 46%) |
| 2 | A-stage × LVIS-Wide | **完成** | **STRONG GO (ratio 1.07x, util 67.8%)** |
| 3 | B-stage × 5-Category | **完成** | STRONG GO (ratio 1.18x, CD -6.2%, util 99%) |
| 4 | B-stage × LVIS-Wide | **待执行** | — |

## 已完成

### Exp1: A-stage × 5cat — STRONG GO
- Checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt
- 结果: results/exp1_A_5cat/
- CD Ratio: 1.14x, Utilization: 46% (eval) / 99.7% (train)
- Commit: edb86c4

### Exp2: A-stage × LVIS-Wide — STRONG GO
- Checkpoint: data/checkpoints/lvis_wide_A/checkpoint_final.pt
- 结果: results/exp2_A_lvis_wide/
- CD Ratio: **1.07x**, Same-cat CD: 217.0, Cross-cat CD: 232.3
- Eval Util: **67.8%** (2779/4096), Train Util: 74.7%
- 训练 200 epochs, ~10h (186s/epoch)
- **关键发现**: 更多类别训练 = 更好泛化（ratio 1.07x vs 5cat 1.14x, util 67.8% vs 46%）
- Commit: eba4b7d

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

## 下一步：Exp4 B-stage LVIS-Wide

### 1. 资源检查
```bash
df -h / && free -h && nvidia-smi
```

### 2. 清理旧 checkpoint（只保留最新 3 个）
```bash
ls -t data/checkpoints/lvis_wide_A/checkpoint_epoch*.pt | tail -n +4 | xargs rm -f
```

### 3. 启动 Exp4 训练
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

### 4. 监控训练（每 30min 写进度报告到 results/exp4_B_lvis_wide/）
- 每 epoch ~186s, 预计总 ~10h
- 监控重点: utilization 是否保持 >30%, recon loss 是否下降

### 5. 评估 Exp4
```bash
PYTHONPATH=. python scripts/evaluate.py \
  --checkpoint data/checkpoints/lvis_wide_B/checkpoint_final.pt \
  --same_cat_dirs data/patches/lvis_wide/seen_test \
  --cross_cat_dirs data/patches/lvis_wide/unseen \
  --output results/exp4_B_lvis_wide/eval_results.json
```

### 6. 可视化 Exp4
```bash
PYTHONPATH=. python scripts/visualize.py \
  --checkpoint data/checkpoints/lvis_wide_B/checkpoint_final.pt \
  --history data/checkpoints/lvis_wide_B/training_history.json \
  --patch_dirs data/patches/lvis_wide/seen_train \
  --output_dir results/exp4_B_lvis_wide
```

### 7. 最终 4 实验综合评估
对比所有 4 组实验结果，写 final report。

### 8. 更新文档
更新 TODO.md, CLAUDE.md, README.md, RUN_GUIDE.md 反映最终结果。

## 重要文件
- B-stage 设计: context/19_codebook_collapse_fix_design.md 第四节
- 实验计划: context/11_objaverse_experiment_plan.md
- A-stage 5cat checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt
- B-stage 5cat checkpoint: data/checkpoints/5cat_B/checkpoint_final.pt
- A-stage LVIS-Wide checkpoint: data/checkpoints/lvis_wide_A/checkpoint_final.pt
- Exp2 报告: results/exp2_A_lvis_wide/report.md
- Exp3 报告: results/exp3_B_5cat/report.md

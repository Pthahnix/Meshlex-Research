# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-14 04:00
> **当前状态**: **可行性验证全部完成** — 4/4 实验 STRONG GO，最终报告已生成。

## 实验进度总览

| # | 实验 | 状态 | 结果 | HF Checkpoint |
|---|------|------|------|---------------|
| 1 | A-stage × 5-Category | **完成** | STRONG GO (ratio 1.14x, util 46%) | `checkpoints/exp1_A_5cat/` |
| 2 | A-stage × LVIS-Wide | **完成** (重训) | STRONG GO (ratio 1.02x, util 95.3%) | `checkpoints/exp2_A_lvis_wide/` (上次版本) |
| 3 | B-stage × 5-Category | **完成** | STRONG GO (ratio 1.18x, util 47%) | `checkpoints/exp3_B_5cat/` |
| 4 | B-stage × LVIS-Wide | **完成** | STRONG GO (ratio 1.02x, util 94.9%) | 待上传 |

**HF Repo**: `Pthahnix/MeshLex-Research` (model repo)

## 已完成步骤

### Step 1: ~~完成 LVIS-Wide 数据下载 + split~~ ✅
### Step 2: ~~训练 Exp2 A-stage LVIS-Wide~~ ✅ (200/200 ep, loss 0.264)
### Step 3: ~~评估 Exp2~~ ✅ (STRONG GO, ratio 1.019x)
### Step 4: ~~训练 Exp4 B-stage LVIS-Wide~~ ✅ (200/200 ep, loss 0.259)
### Step 5: ~~评估 Exp4~~ ✅ (STRONG GO, ratio 1.019x)
### Step 6: ~~可视化 + 最终报告~~ ✅

最终报告: `results/final_comparison/report.md`
可视化: `results/final_comparison/` (5 张对比图)

## 待执行: 下一阶段

### 优先: 上传 Exp2/Exp4 重训 checkpoint 到 HuggingFace
- Exp2: `data/checkpoints/lvis_wide_A/checkpoint_final.pt` → `checkpoints/exp2_A_lvis_wide/`
- Exp4: `data/checkpoints/lvis_wide_B/checkpoint_final.pt` → `checkpoints/exp4_B_lvis_wide/`

### 正式实验设计
1. 更大 codebook (K=8192, 16384) 的 scaling 实验
2. 更深 encoder/decoder 架构探索
3. 与 per-face tokenization baseline 的正式对比
4. 下游任务: text-to-mesh generation with patch vocabulary

### 论文撰写
- 目标: CCF-A (CVPR / NeurIPS / ICCV)

## 已完成实验详情

### Exp1: A-stage × 5cat — STRONG GO
- CD Ratio: 1.145x, Same-cat CD: 238.3, Cross-cat CD: 272.8
- Utilization: 46.0% (eval), Recon loss: 0.241

### Exp2: A-stage × LVIS-Wide — STRONG GO (重训版本)
- CD Ratio: 1.019x, Same-cat CD: 214.3, Cross-cat CD: 218.4
- Utilization: 95.3% (eval), Active codes: 3,903/4,096
- **关键发现**: 更多类别 = 更好泛化，利用率翻倍

### Exp3: B-stage × 5cat — STRONG GO
- CD Ratio: 1.185x, Same-cat CD: 223.5, Cross-cat CD: 264.8
- Utilization: 47.1% (eval), Recon loss: 0.229

### Exp4: B-stage × LVIS-Wide — STRONG GO
- CD Ratio: 1.019x, Same-cat CD: 211.6, Cross-cat CD: 215.8
- Utilization: 94.9% (eval), Active codes: 3,887/4,096
- **最佳结果**: 所有指标最优

## 当前本地状态

### Checkpoints (本地)
- Exp1: `data/checkpoints/5cat_v2/checkpoint_final.pt`
- Exp2: `data/checkpoints/lvis_wide_A/checkpoint_final.pt`
- Exp3: `data/checkpoints/5cat_B/checkpoint_final.pt`
- Exp4: `data/checkpoints/lvis_wide_B/checkpoint_final.pt`

### 数据
- 5-Category: `data/patches/` (chair, table, airplane, car, lamp)
- LVIS-Wide: `data/patches/lvis_wide/` (188,696 train / 45,441 test / 12,655 unseen)

## 重要注意事项
- **Checkpoint 备份**: Exp2/Exp4 重训版本尚未上传 HF，需尽快上传
- **磁盘**: 7.2G/80G 使用，空间充裕
- **恢复训练**: B-stage resume 需 strict=False
- **Rotation trick**: 与 SimVQ 不兼容，不要使用

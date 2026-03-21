# 28 — Preliminary Experiments Results & Strategic Decision

**Date**: 2026-03-21
**Branch**: `quick-validation`
**Data**: 4934 meshes / 246,792 tokens per level (LVIS-Wide, 3-level RVQ K=1024, 5% scale)

## Background

在 context/27 确认 mesh token 服从 lognormal 分布后，设计并执行了 5 个 preliminary experiments 以收集更多信号，用于判断 theory-driven 和 PatchDiffusion 两个子研究方向的可行性。

## Experiment Results

### Exp 1: Per-category Token Distribution — CONSISTENT ✅

**问题**: 全局 lognormal 是否为不同子群 mixture 的假象？

**方法**: 按 mesh 大小 (Q1-Q4)、token diversity (Q1-Q4)、数据 split (train/test/unseen) 分组，独立拟合。

| 分组方式 | 组数 | Lognormal 胜出 | σ 范围 | σ_std |
|---------|-----|---------------|--------|-------|
| Mesh 大小 | 4 | 4/4 | 0.41 ~ 0.81 | — |
| Token 多样性 | 4 | 4/4 | 0.42 ~ 0.57 | — |
| 数据 Split | 3 | 3/3 | 0.42 ~ 0.58 | — |
| **总计** | **11** | **11/11** | — | **0.113** |

**关键发现**:
- 小 mesh (avg 17 patches): σ=0.81, α=0.68 — 最不均匀
- 大 mesh (avg 103 patches): σ=0.48, α=0.43 — 更均匀
- unseen split 与 seen 一致 — 泛化性好
- **结论**: Lognormal 在所有子群上一致，非 mixture artifact

### Exp 2: Token Spatial Correlation — WEAK ⚠️

**问题**: 空间相邻 patch 是否倾向使用相似 token？

| 指标 | 值 |
|------|-----|
| Mean Spearman ρ | -0.0356 |
| Median Spearman ρ | -0.0287 |
| 负相关比例 | 72.6% |
| Verdict | WEAK/NONE |

**结论**: 存在弱负空间相关性（近距离 patch 倾向用不同 token），但效应极小（|ρ| < 0.04）。这反映 METIS 分割的设计特性：相邻 patch 覆盖不同的局部拓扑。

**Implication**: Mesh token 的空间局部性远弱于图像 token，AR 模型的 patch 排列顺序可能不太重要。

### Exp 3: Codebook Embedding Visualization — 观察性

**方法**: 从 VQ-VAE checkpoint 提取 SimVQ codebook (C @ W^T)，UMAP 降维。

**观察**:
- L1 有松散 cluster 结构，但不对应明确的几何类别
- L2 最分散（与 L2 "最具结构性" 的 Exp 1 发现一致 — 更分散意味着更多信息）
- 高频 token 无明显聚集
- 三级在 combined UMAP 中有一定分离但重叠大

### Exp 4: RVQ Inter-level Dependency — STRONG ✅

**问题**: RVQ 三层之间信息依赖程度？

| 对 | H(Y) | H(Y\|X) | MI (bits) | NMI |
|---|------|---------|-----------|-----|
| L1 → L2 | 9.76 | 7.10 | 2.66 | **0.273** |
| L1 → L3 | 9.90 | 7.50 | 2.40 | **0.243** |
| L2 → L3 | 9.90 | 7.55 | 2.35 | **0.241** |

**结论**: NMI 全部 > 0.24，远超 0.1 阈值。三级 RVQ 有显著信息依赖。
- L1→L2 最强（residual 直接相关）
- 条件熵仍 7+ bits — 知道 L1 后 L2 仍有大量不确定性
- **Implication for generation**: 任何生成模型都必须联合建模三级 token，不能分层独立处理

### Exp 5: Toy MDLM Prototype — NOT_FEASIBLE ❌

**架构**: Transformer encoder, 2.76M params, 256d, 4 heads, 4 layers
**训练**: 100 epochs, batch 64, AdamW lr=3e-4, 134s total

| 指标 | MDLM | AR v2 (baseline) |
|------|------|-----------------|
| Val Loss | 6.766 | 1.48 |
| Perplexity | 867.75 | 4.4 |
| Val Acc (t=0.5) | 0.6% | N/A |
| KL divergence | <0.1 nats | N/A |
| 参数量 | 2.76M | 20.4M |

Accuracy 在所有 masking rate (0.2~0.9) 上都约 0.5-0.6%，接近随机 (1/1024 = 0.098%)。Loss 从 7.107 仅降至 6.766（降 5%），严重欠拟合。

**但对比不公平**: 模型差 7.4x (2.76M vs 20.4M)，codebook 接近均匀分布（entropy 98-99%）天然不利于 mask prediction，且数据仅 4934 sequences。此结果仅说明 "toy scale 不可行"，不能排除大规模可行性。

## Signal Confidence Assessment

```
                          数据规模敏感度
                    低（结论稳定）    高（可能翻转）
                   ┌──────────────┬──────────────┐
信  高（强信号）    │ Exp1: 11/11  │              │
号               │ lognormal    │              │
强               │ Exp4: NMI>0.24│              │
度               ├──────────────┼──────────────┤
   低（弱信号）    │ Exp2: ρ=-0.04│ Exp5: MDLM   │
                   │ 弱空间相关    │ PPL=867      │
                   └──────────────┴──────────────┘
```

## Failure Mode Analysis

### FM1 (最高风险): Lognormal 是 SimVQ 的 artifact

SimVQ 用 frozen codebook + linear transform，天然倾向均匀利用 codebook。如果换成 vanilla VQ（有 collapse 倾向），分布可能变为 Zipf 或其他形态。**只能通过不同 VQ 方法的对比实验验证**。

### FM2: 全量 codebook 的分布形态可能不同

当前 codebook 在 4934 meshes 上训练，全量是 32K+ meshes。更多数据 → codebook 可能出现专门化 → 打破均匀性 → 更强重尾。分布参数（σ, μ, α）几乎必然变化。

### FM3: MDLM 不可行是假阴性

Toy model 的失败不能排除 PatchDiffusion 在合理规模（20M+ params, 30K+ sequences, 更强重尾分布）下可行。

## Strategic Decision

### 决策框架 (from spec)

| Exp 1 (per-category) | Exp 5 (MDLM) | Recommended Direction |
|-----------------------|---------------|----------------------|
| ✅ Consistent lognormal | ❌ Not feasible | Theory-driven priority |

### 决定: 等全量数据 + 同步写论文框架 (Plan A+C)

**排除 "做更多 preliminary 实验" (Plan B)**:
- 5% 数据上的分析已经饱和，边际收益递减
- MDLM 放大到 20M 仍然在 4934 sequences 上训练，结论不可靠
- 最高风险点 (FM1: SimVQ artifact) 只能通过 retrain 验证，不能通过更多分析绕过

**等待期任务**:
1. 写 theory-driven 论文的 Related Work + Introduction 框架
2. 整理跨域对比文献 (image VQ token distribution, time series VQ, etc.)
3. 设计全量实验的详细 protocol（多大 codebook、几种 VQ 方法、统计方案）

**全量数据到位后执行**:
1. ShapeNet pipeline 完成 (~12-24h remaining, 35/55 categories)
2. 全量 VQ-VAE retrain (~4-8h on A4000)
3. 重跑 5 experiments on full codebook (~2h, 自动化脚本已有)
4. 根据全量结果做最终方向决策

## Confirmed Takeaways (High Confidence)

1. **Lognormal 是真的** — 11/11 子群一致，非 mixture artifact。可作为 empirical contribution
2. **空间局部性弱** — patch 排列顺序可能不重要，当前 Morton 排序未必是最优
3. **RVQ 层间强依赖** — NMI > 0.24，任何生成模型必须联合建模三级 token
4. **小 mesh token 分布最不均匀** — 可能需要 size-conditional 生成策略

## Pending Verification (Needs Full Data)

1. Lognormal 是否在全量 retrain codebook 上仍成立？
2. Lognormal 是 SimVQ 的属性还是 mesh geometry 的属性？(需不同 VQ 方法对比)
3. PatchDiffusion 在合理规模下是否可行？
4. 全量 codebook 的 RVQ 层间依赖是否保持 NMI > 0.1？

## Files

| Path | Description |
|------|-------------|
| `results/preliminary_exp/REPORT.md` | 远程 CC 生成的原始报告 |
| `results/preliminary_exp/exp1_per_category/` | Per-category 分布分析 |
| `results/preliminary_exp/exp2_spatial/` | 空间相关性分析 |
| `results/preliminary_exp/exp3_codebook_viz/` | Codebook UMAP 可视化 |
| `results/preliminary_exp/exp4_rvq_dependency/` | RVQ 层间互信息 |
| `results/preliminary_exp/exp5_mdlm/` | MDLM 原型训练 |
| `docs/superpowers/specs/2026-03-21-preliminary-experiments-design.md` | 实验设计 spec |

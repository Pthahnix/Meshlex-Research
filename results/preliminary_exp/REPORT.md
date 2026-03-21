# Preliminary Experiments Report

**Date**: 2026-03-21
**Branch**: `quick-validation`
**Data**: 4934 meshes / 246,792 tokens (LVIS-Wide, 3-level RVQ K=1024)

## Exp 1: Per-category Token Distribution

**问题**: 全局 lognormal 是否为不同子群 mixture 的假象？

**方法**: 按 mesh 大小 (Q1-Q4)、token diversity (Q1-Q4)、数据 split (train/test/unseen) 分组，独立拟合分布。

**结果**:

| 分组方式 | 组数 | Lognormal 胜出 | σ 范围 |
|---------|-----|---------------|--------|
| Mesh 大小 | 4 | 4/4 | 0.41 ~ 0.81 |
| Token 多样性 | 4 | 4/4 | 0.42 ~ 0.57 |
| 数据 Split | 3 | 3/3 | 0.42 ~ 0.58 |
| **总计** | **11** | **11/11** | σ_std = 0.113 |

**Verdict**: **CONSISTENT** — Lognormal 在所有分组上一致优于 Zipf，非 mixture artifact。

小 mesh (Q1) 的 σ=0.81 最高、α=0.68 最高，说明小 mesh 的 token 分布最不均匀，但仍然是 lognormal。

---

## Exp 2: Token Spatial Correlation

**问题**: 空间相邻 patch 是否倾向使用相似 token？

**结果**:

| 指标 | 值 |
|------|-----|
| Mean Spearman ρ | **-0.0356** |
| Median Spearman ρ | -0.0287 |
| 负相关比例 | 72.6% |

**Verdict**: **WEAK_NEGATIVE** — 存在弱负空间相关性（近距离 patch 倾向使用 *不同* 的 token），但效应很小（|ρ| < 0.04）。

**解释**: 这可能反映了 METIS 分割的设计——相邻 patch 覆盖不同的局部拓扑，因此自然使用不同 codebook entry。空间局部性不强，意味着 mesh token 的空间结构弱于图像 token。

---

## Exp 3: Codebook Embedding Visualization

**方法**: 从 VQ-VAE checkpoint 提取 3 级 codebook (SimVQ: C @ W^T)，UMAP 降维，按 log(frequency) 着色。

**产出**:
- `exp3_codebook_viz/umap_per_level.png` — 各级 codebook UMAP
- `exp3_codebook_viz/umap_combined.png` — 三级合并 UMAP

**观察**: 需目视检查 UMAP 图，看是否存在 cluster 结构以及高频 token 是否集中。

---

## Exp 4: RVQ Inter-level Dependency

**问题**: RVQ 三层之间信息依赖程度如何？

**结果**:

| 对 | 互信息 (bits) | NMI | 条件熵 (bits) |
|---|-------------|-----|--------------|
| L1 → L2 | 2.66 | **0.273** | H(L2\|L1) = 7.10 |
| L1 → L3 | 2.40 | **0.243** | H(L3\|L1) = 7.50 |
| L2 → L3 | 2.35 | **0.241** | H(L3\|L2) = 7.55 |

**Verdict**: **STRONG_DEPENDENCY** (NMI > 0.1 threshold)

- 三对 NMI 均 > 0.24，远超 0.1 阈值
- L1→L2 依赖最强（NMI=0.273），因为 L2 是 L1 的 residual
- 条件熵仍然较高（7+ bits），说明即使知道 L1，L2 仍有大量不确定性
- **Implication for diffusion**: 必须联合 denoise 三级 token，不能分层独立处理

---

## Exp 5: Toy MDLM Prototype

**架构**: Transformer encoder, 2.76M params, 256d, 4h, 4L
**训练**: 100 epochs, batch 64, AdamW lr=3e-4, 134s total

**结果**:

| 指标 | MDLM | AR v2 (baseline) |
|------|------|-----------------|
| Val Loss | 6.766 | 1.48 |
| Perplexity | **867.75** | **4.4** |
| Val Acc (t=0.5) | 0.6% | N/A |
| KL divergence | <0.1 nats | N/A |
| 参数量 | 2.76M | 20.4M |

**Accuracy vs masking rate**:
| t | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|-----|-----|-----|-----|-----|
| Acc | 0.5% | 0.6% | 0.6% | 0.6% | 0.5% |

**Verdict**: **NOT_FEASIBLE** — 当前 toy MDLM 在 5% 数据上完全无法学习 token 间依赖。

**分析**:
1. PPL 867 接近理论随机下限 (1024)，模型几乎没有学到东西
2. Accuracy 在所有 masking rate 上都约 0.6%，接近随机 (1/1024 = 0.098%)
3. Loss 从 7.107 只降到 6.766（仅降 5%），严重欠拟合
4. 低 KL divergence 是假象——因为 codebook 本身就接近均匀分布
5. 模型太小 (2.76M vs AR 的 20.4M)，且 bidirectional 架构对 codebook-level token 可能不合适

**但注意**: 这是 toy model 在 5% 数据上的结果。MDLM 原论文使用更大模型 + 更多数据 + 更多 epoch。此结果仅说明"小规模不可行"，不能排除大规模可行性。

---

## Decision Framework Application

| Exp 1 (per-category) | Exp 5 (MDLM) | Recommended Direction |
|-----------------------|---------------|----------------------|
| ✅ Consistent lognormal | ❌ MDLM not feasible | **Theory-driven only** |

## Updated Direction Recommendation

### Primary: **Theory-driven (Lognormal Characterization)**

理由：
1. Lognormal 在所有 11/11 子群上一致，非 artifact ✅
2. 跨域对比叙事清晰（image tokens 也是 lognormal）
3. Multiplicative process 理论框架成立
4. RVQ 层间强依赖（NMI > 0.24）提供额外理论探索空间

### Secondary: PatchDiffusion 暂缓

理由：
1. Toy MDLM 完全失败（PPL 867）
2. 虽然 toy model 不能完全否定方向，但在 5% 数据上零信号令人担忧
3. 需要先完成主 pipeline (Phase B-E) 再考虑是否投入更多资源

### Key Insights for Next Steps

1. **空间局部性弱** → AR 的 Morton 排序可能不是最优的序列化方式
2. **RVQ 层间依赖强** → 任何新的生成模型都应联合建模三级 token
3. **小 mesh token 分布最不均匀** → 可能需要 size-conditional 生成策略

---

## Visualization Index

| Directory | Contents |
|-----------|----------|
| `exp1_per_category/` | comparison.png, fit_winner.png, summary.json |
| `exp2_spatial/` | spatial_correlation.png, summary.json |
| `exp3_codebook_viz/` | umap_per_level.png, umap_combined.png, summary.json |
| `exp4_rvq_dependency/` | dependency_matrix.png, mi_summary.png, summary.json |
| `exp5_mdlm/` | training_results.png, checkpoint.pt, generated_tokens.npy, summary.json |

⚠️ **IMPORTANT**: These are 5% scale preliminary results. Do NOT make final direction decisions based solely on these numbers.

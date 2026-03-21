# Preliminary Experiments Design — Token Analysis + MDLM Prototype

**Date**: 2026-03-21
**Branch**: `quick-validation`
**Data**: Existing 4934 meshes / 246K tokens (5% scale, LVIS-Wide, 3-level RVQ K=1024)
**Hardware**: 10.11.219.245 (3x RTX 5090)

## Motivation

Task 4 (token distribution analysis) 发现 mesh RVQ tokens 更接近 lognormal 而非 Zipf。但该结论基于 5% 数据的单次全局分析。在做研究方向决策前，需要更多 preliminary signal：

1. Per-category 分析是否确认 lognormal 不是 mixture artifact？
2. Token 在空间上是否有局部性结构？
3. RVQ 三层之间的依赖关系如何？
4. Masked discrete diffusion 在现有数据上是否可行？

## Experiments

### Exp 1: Per-category Token Distribution

**假设**: 全局 lognormal 可能是不同类别的均匀分布混合后的假象。

**方法**:
- 从 sequence NPZ 文件名提取 mesh_id
- 通过 LVIS synset mapping (从 Objaverse metadata 或 data/patches/ 目录结构) 获取类别
- 对每个 n_meshes ≥ 30 的类别独立计算：
  - Zipf fit (log-log linear regression) → α, R²
  - Lognormal fit (scipy MLE) → σ, μ, KS stat
  - Entropy, Gini coefficient
- 汇总：跨类别的 σ/μ/α 分布

**数据依赖**: `data/sequences/rvq_lvis/`, `data/patches/lvis_wide/`

**产出**:
- `results/preliminary_exp/exp1_per_category/` — per-category stats JSON, heatmap, comparison plot
- 关键判断：per-category lognormal σ 的 variance。低 variance = 一致性强 = 非 artifact

### Exp 2: Token Spatial Correlation

**假设**: 空间相邻的 patches 倾向于使用相似的 token。

**方法**:
- 对每个 mesh 的 sequence NPZ 加载 centroids (N, 3) 和 tokens (N, 3)
- 计算 patch 对之间的欧氏距离 d_ij
- 计算 patch 对的 token match rate（L1 same → 1, else → 0）
- 按距离分 bin，计算每个 bin 的 match rate
- 全局 Spearman correlation (distance vs token_match)
- Moran's I for spatial autocorrelation

**数据依赖**: `data/sequences/rvq_lvis/`

**产出**:
- `results/preliminary_exp/exp2_spatial/` — distance-vs-match scatter, correlation histogram, JSON
- 关键判断：Spearman ρ 的符号和显著性

### Exp 3: Codebook Embedding Visualization

**假设**: Codebook entries 在 embedding 空间中有 cluster 结构。

**方法**:
- 加载 VQ-VAE checkpoint (`data/checkpoints/rvq_lvis/checkpoint_final.pt`)
- 提取 3 级 codebook: `model.rvq.quantizers[i].codebook` (每级 1024 × embed_dim)
  - 注意 SimVQ: 实际 codebook = frozen `C` 经过 linear `W` 变换
- UMAP 降维到 2D (n_neighbors=15, min_dist=0.1)
- 着色：
  - 按 log(frequency) 着色
  - 按 RVQ level 着色 (combined plot)

**数据依赖**: `data/checkpoints/rvq_lvis/checkpoint_final.pt`, Exp 1 的频率统计

**产出**:
- `results/preliminary_exp/exp3_codebook_viz/` — 3 per-level UMAP + 1 combined, JSON
- 关键判断：cluster 是否与几何类型对应？高频 token 是否集中？

### Exp 4: RVQ Inter-level Dependency

**假设**: RVQ L1→L2→L3 之间存在信息依赖。

**方法**:
- 构建 co-occurrence matrix: M[i,j] = count(L1=i AND L2=j), 大小 1024×1024
- 计算条件 entropy: H(L2|L1) = Σ_i P(L1=i) × H(L2|L1=i)
- 互信息: I(L1;L2) = H(L2) - H(L2|L1)
- 同理计算 I(L1;L3), I(L2;L3)
- 归一化互信息: NMI = I(X;Y) / min(H(X), H(Y))

**数据依赖**: `data/sequences/rvq_lvis/`

**产出**:
- `results/preliminary_exp/exp4_rvq_dependency/` — MI matrix, conditional entropy heatmap, JSON
- 关键判断：
  - NMI > 0.1 → 显著依赖，diffusion 需联合 denoise
  - NMI < 0.05 → 近独立，可分层 denoise

### Exp 5: Toy MDLM Prototype

**假设**: Masked discrete diffusion 可以学习 mesh token 分布。

**架构**:
```
Input: token sequence (N_patches, 3) → flatten to (N_patches * 3,)
       + positional encoding (learnable, per-position)
       + level encoding (L1/L2/L3 embedding)

Model: Transformer Encoder
  - d_model = 256
  - n_heads = 4
  - n_layers = 4
  - d_ff = 512
  - dropout = 0.1
  - ~1.5M params

Output: per-position logits over vocab_size=1025 (1024 tokens + MASK)
```

**训练 (MDLM)**:
- Continuous-time masking: 对每个 token 独立，以 t~U(0,1) 的概率 mask
- 目标：预测被 mask 的 token（cross-entropy loss，只在 masked positions 上计算）
- Schedule: linear masking schedule (t=0 → no mask, t=1 → all masked)
- Optimizer: AdamW, lr=3e-4, warmup 500 steps
- Epochs: 100 (数据小，每 epoch 很快)
- Batch size: 64 sequences

**评估**:
- Mask prediction accuracy (t=0.5, 即 50% masked)
- 生成：从 all-masked 开始，逐步 unmask (τ=1.0→0.0, 100 steps)
- 比较生成 token 的 per-level 分布 vs 真实分布 (KL divergence)
- 与 AR v2 baseline 对比 (AR loss 1.48, ppl 4.4)

**数据依赖**: `data/sequences/rvq_lvis/`

**产出**:
- `results/preliminary_exp/exp5_mdlm/` — training curve, generation samples, KL comparison, JSON
- 关键判断：
  - mask prediction acc > 50% at t=0.5 → 模型学到了 token 间的依赖
  - 生成分布 KL < 0.5 nats vs 真实 → 分布学习可行
  - ppl 对比 AR v2 → 量化 diffusion vs AR 的差距

## Execution Plan

```
Phase 1 (并行, CPU, ~5 min):
  Exp 1 + Exp 2 + Exp 4

Phase 2 (GPU needed, ~5 min):
  Exp 3 (UMAP on embeddings)

Phase 3 (GPU, ~1-2 hours):
  Exp 5 (MDLM training)

Phase 4 (汇总):
  综合报告 → results/preliminary_exp/REPORT.md
  commit + push
```

## Decision Framework

实验完成后，根据结果更新方向建议：

| Exp 1 结果 | Exp 5 结果 | 推荐方向 |
|-----------|-----------|---------|
| Per-cat 一致 lognormal | MDLM 收敛 | 两个都做，theory-driven 优先 |
| Per-cat 一致 lognormal | MDLM 不收敛 | Theory-driven only |
| Per-cat 不一致 (mixture) | MDLM 收敛 | PatchDiffusion only |
| Per-cat 不一致 (mixture) | MDLM 不收敛 | 回到 main pipeline，放弃两个子方向 |

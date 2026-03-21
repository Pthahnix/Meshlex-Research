# Token Frequency Distribution Analysis — Report

**Date**: 2026-03-21
**Data**: 4,934 meshes × ~50 patches/mesh = 246,792 tokens (LVIS-Wide, 3-level RVQ)

## Key Findings

### 1. Distribution Type: **Lognormal > Zipf**

所有三级 RVQ token 的频率分布更接近 **对数正态分布**，而非 Zipf 幂律。

| Level | Zipf α | Zipf R² | Zipf KS | Lognorm KS | Better Fit | Entropy |
|-------|--------|---------|---------|------------|------------|---------|
| L1 | 0.360 | 0.736 | 0.157 | 0.052 | Lognormal | 9.89 bits (98.9%) |
| L2 | 0.495 | 0.905 | 0.125 | 0.056 | Lognormal | 9.76 bits (97.6%) |
| L3 | 0.339 | 0.875 | 0.158 | 0.035 | Lognormal | 9.90 bits (99.0%) |

**核心数据**：
- Zipf α 平均 0.398（远低于语言中的 α ≈ 1.0），说明幂律不够强
- Lognormal KS 在所有 level 上都明显低于 Zipf KS，表明对数正态是更好的拟合
- L3 的 Lognormal KS p-value = 0.15（无法拒绝 H0），说明 L3 确实可以被认为服从对数正态

### 2. 极高的 Codebook Utilization

- 所有 1024 个 codebook entries 在三个 level 上全部被使用（100% utilization）
- Entropy 接近理论最大值 10.0 bits（97.6% ~ 99.0%）
- Gini 系数 0.20~0.30，表明 token 使用相当均匀

### 3. 弱重尾特征

- 尽管分布不是完全均匀，但重尾程度远低于语言 token
- Top-10 tokens 仅覆盖 2.6%~4.9%（语言中 Zipf 的 top-10 可覆盖 20%+）
- Top-50 tokens 仅覆盖 10.1%~15.5%

### 4. L2 Layer 最具结构性

- L2 的 Zipf α = 0.495 是三层中最高的
- L2 的 Zipf R² = 0.905 拟合度最好
- L2 的 Gini = 0.30 不均匀程度最大
- **解释**：L2 作为 residual 层，可能捕获了更有区分性的几何细节

## 与相关工作对比

| Domain | Distribution | α (if Zipf) | Reference |
|--------|-------------|-------------|-----------|
| 自然语言 | Zipf (strong) | α ≈ 1.0 | Zipf's law |
| 图像 VQ token | Lognormal | — | "Analyzing the Language of Visual Tokens" (2024) |
| 时间序列 VQ token | Zipf | α ≈ 0.8-1.5 | "The Language of Time" (2025) |
| **3D Mesh RVQ token** | **Lognormal** | α ≈ 0.40 (weak) | **This work** |

**新发现**：3D mesh tokens 的分布更接近图像 token（对数正态），而非时间序列或语言（Zipf）。

## Visualization Index

| File | Description |
|------|-------------|
| `comparison_3levels.png` | 三级 token rank-frequency 对比 |
| `rank_frequency_L1/L2/L3.png` | 各级 rank-frequency plot (log-log) |
| `histogram_L1/L2/L3.png` | 各级频率直方图 + lognormal 拟合 |
| `ccdf_heavy_tail.png` | 互补 CDF，用于重尾分析 |
| `summary.json` | 完整数值结果 |

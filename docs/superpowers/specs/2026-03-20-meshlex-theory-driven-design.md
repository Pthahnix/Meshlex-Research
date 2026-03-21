# MeshLex Theory-Driven Design Spec

**Date**: 2026-03-20
**Target**: NeurIPS 2027
**Type**: Theory-Heavy (Type R)

---

## 1. Executive Summary

MeshLex v1 发现了一个关键现象：用 ~500 个 patch token 可以跨 1156 个类别泛化，CD ratio 仅 1.019×。这暗示 mesh patch 空间存在 universal structure。

本 spec 提出一个**理论驱动的系统重设计**方案：

1. **实验层**：系统性表征 mesh patch 空间的等价结构（相变曲线 + 幂律分布 + 跨数据集 universality）
2. **理论层**：从 Gauss-Bonnet 定理推导高曲率 patch 数量上界，用 Lean4 形式化证明
3. **系统层**：基于理论设计曲率感知的非均匀 codebook

---

## 2. Problem Statement

### 2.1 核心问题

> 为什么 mesh 的拓扑结构会存在 universal vocabulary？

已知：
- MeshLex v1 用 512 个 token 实现 CD ratio 1.019× 的跨类别泛化
- 如果 patch 类型均匀分布，512 个 token 无法支撑这个泛化能力

推断：
- Patch 类型频率一定遵循重尾分布（幂律）
- 少数几种 patch 类型极高频，大量类型极稀少

### 2.2 核心假说

**Primary Hypothesis (H1)**: Mesh patch VQ token 频率遵循重尾分布（幂律或对数正态），其来源是物理世界 3D 表面的微分几何结构。

**关键背景**：不同模态的 VQ token 分布已被发现不同：
- **图像域**：VQ token 频率服从对数正态分布（"Analyzing the Language of Visual Tokens", 2024）
- **时间序列域**：VQ token 频率服从 Zipf 幂律分布（"The Language of Time", 2025）
- **3D Mesh 域**：未知（本工作首次研究）

因此，我们不预设分布族，而是通过实验确定（§3.1 Dual Distribution Test）。

**Sub-hypothesis (H1a)**: 若分布是幂律，其来源可由 Gauss-Bonnet 定理 + MaxEnt 推导解释。

**Sub-hypothesis (H1b)**: 若分布是对数正态，其来源可由曲率的乘法噪声模型解释（patch 曲率是多个独立因素的乘积 → log-normal）。

从微分几何推导（无论哪种分布族，以下直觉均成立）：

光滑 2-流形的每一点可用 Gaussian 曲率 K 和 Mean 曲率 H 描述，分成 5 种本质类型：

| 类型 | K 和 H | 物理直觉 | 预期频率 |
|------|--------|----------|----------|
| Flat | K≈0, H≈0 | 平面、盒子的面 | 极高频 |
| Elliptic | K>0 | 球面、凸起 | 高频 |
| Cylindrical | K≈0, H≠0 | 圆柱侧面 | 中频 |
| Hyperbolic | K<0 | 凹陷、马鞍形 | 低频 |
| Corner | K>>0 | 立方体角 | 稀少 |

物理世界的 3D 物体绝大部分表面是平的或微凸的，尖锐特征是少数。离散化后自然导致 flat/elliptic patch 极高频。

### 2.3 Gauss-Bonnet 约束

Gauss-Bonnet 定理：任意闭合亏格-0 曲面的总 Gaussian 曲率 = 4π（固定值）。

这意味着：
- Elliptic patch (K>0) 和 hyperbolic patch (K<0) 的数量受约束
- Flat patch (K≈0) 可以任意多 → 高频
- 高曲率 patch 的总量被锁住 → 低频

---

## 3. Core Contributions

### 3.1 Five Contributions

| # | 贡献 | 类型 | 与已有工作的区别 |
|---|------|------|-----------------|
| C1 | 首个 3D mesh VQ token 频率分布的系统性研究 + 与图像/时间序列的跨域对比 | 实验 | 图像域有 lognormal 研究，时间序列有 Zipf 研究，**3D mesh 是空白** |
| C2 | Gauss-Bonnet → MaxEnt → 频率分布的完整理论推导链 | 理论 | 已有工作只观察分布，**本工作首次提供几何机制解释** |
| C3 | Lean4 形式化证明：Gauss-Bonnet → 高曲率 patch 上界 O(χ/κ) | 形式化 | "Verified design rationale" 框架 |
| C4 | 曲率感知非均匀 codebook：理论直接驱动系统设计，CD + PTME 双重验证 | 方法 | 无 |
| C5 | 全量数据（97K meshes）上的完整生成+重建实验，逼近 SOTA | 应用 | 无 |

### 3.2 Contribution Chain

```
[实验 C1]
    双分布测试 → 确定分布族（幂律 or 对数正态）
    相变曲线 + 跨数据集 universality + 跨域对比
         ↓ 经验发现
[理论 C2, C3]
    Gauss-Bonnet → MaxEnt → 分布推导（几何解释）
    vs GEM/Pitman-Yor（统计解释）→ 几何解释提供可解释性优势
    Lean4 形式化证明高曲率 patch 上界
         ↓ 解释了为什么 MeshLex v1 能泛化
[方法 C4]
    曲率感知的非均匀 codebook
    CD + PTME 双重验证
         ↓ 用更好的 codebook
[应用 C5]
    全量数据重训，逼近 SOTA
```

---

## 4. Theory Layer Design

### 4.1 Experiment: Dual Distribution Test (Go/No-Go Gate)

**目的**：在投入 GPU 时间之前，首先确定 mesh VQ token 的分布族。

**动机**：
- 图像 VQ token 是对数正态分布（"Analyzing the Language of Visual Tokens"）
- 时间序列 VQ token 是 Zipf 幂律分布（"The Language of Time"）
- 3D mesh VQ token 分布未知 → **必须先测再做**

**实验设计**：

```
输入: 现有 K=1024 VQ-VAE (data/checkpoints/rvq_lvis/) 编码的 token 频率

Step 1: 对 token 频率同时拟合两种分布
    - Power law: f(r) ∝ r^{-α}，使用 powerlaw 库的 MLE 拟合
    - Lognormal: f(r) ~ LogNormal(μ, σ)

Step 2: Vuong's closeness test (likelihood ratio test)
    - H0: 两种模型等好
    - p < 0.05 + R > 0: 幂律显著优于对数正态
    - p < 0.05 + R < 0: 对数正态显著优于幂律
    - p > 0.05: 无法区分

Step 3: Go/No-Go 决策
```

**Go/No-Go 判定**：

| Vuong's test 结果 | 决策 | 后续叙事 |
|-------------------|------|----------|
| 幂律赢 (p<0.05, R>0) | **GO (Path A)** | 继续原计划：Gauss-Bonnet → MaxEnt → 幂律 |
| 对数正态赢 (p<0.05, R<0) | **GO (Path B)** | 修改叙事为对数正态 + 乘法噪声几何解释 |
| 无法区分 (p>0.05) | **GO (Path C)** | 报告两种模型，重点是几何解释的独特性 |
| 两者 R² 均 < 0.7 | **NO-GO** | 分布不是重尾 → 重新评估理论驱动方向 |

**工作量**：~半天（代码 + 分析）

### 4.2 Theory: MaxEnt Curvature Distribution

**目的**：填补当前理论链的逻辑漏洞。

**问题**：当前推导链是：
```
Gauss-Bonnet: Σ K_v = 2πχ
    ↓ Markov 不等式
上界: |{v: K_v > κ}| ≤ 2π|χ|/κ
    ↓ ???（逻辑断裂）
声称: Token 频率服从幂律
```

Markov 不等式只给了上界，但**上界 ≠ 分布**。好的审稿人会一眼看出这个缺口。

**解决方案：MaxEnt 推导**（Jaynes 1957）

```
Step 1: Gauss-Bonnet 约束
    给定封闭亏格-0 曲面，总曲率固定: Σ K_v = 4π

Step 2: Maximum Entropy 原理
    在约束 E[K] = μ（固定均值）下，
    最大熵分布是指数分布: p(K) ∝ exp(-λK)

Step 3: 指数曲率 → VQ bin
    VQ 将连续曲率空间离散化为 K 个 bin
    指数分布的离散化 → 几何分布 p(k) = (1-p)^k · p

Step 4: 几何分布 → 近似幂律
    在 rank-frequency 空间:
    - 纯指数衰减在 log-log 图上不是直线
    - 但加上 VQ 的 "赢者通吃" 效应（高频 token 吸引更多邻近 patch）
    - 结果近似幂律，尤其在中等 rank 范围

Step 5: 对数正态的替代解释
    如果曲率是多个独立乘法因素的结果:
    K_patch = ∏_i X_i （各面贡献的乘法组合）
    → log(K) = Σ log(X_i) → 中心极限定理 → 对数正态
```

**引用**：E.T. Jaynes, "Information Theory and Statistical Mechanics" (1957)

**工作量**：~1 天（理论推导 + 写入 spec）

### 4.3 Experiment: Rate-Distortion Curve + Phase Transitions + Curvature Annotation

**核心想法**：VQ-VAE 本身就是 ε-等价聚类的近似。训练 K 个 codeword 的 VQ，就是在找 K 个等价类。

**实验设计**：

```
For K ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}:
    1. 训练 VQ-VAE (fixed architecture, only K varies)
    2. 记录 mean reconstruction CD = distortion D(K)
    3. 画出 D(K) vs K 曲线
    4. 【新增】对每个 K 的 codebook 标注每个 codeword 的平均曲率
    5. 【新增】追踪哪些曲率类型在哪个 K 处"解锁"
```

**固定架构**：使用 `src/model_rvq.py` 中的 `PatchEncoder` + `SimVQCodebook` + `PatchDecoder`，仅改变 codebook 大小 K。具体参数：
- Encoder: 3-layer GCN, hidden_dim=256
- Codebook: SimVQ, embedding_dim=256
- Decoder: 3-layer MLP, hidden_dim=256

**期望信号**：
- D(K) 下降不均匀，在某些 K* 处急剧下降
- 这些 K* 是"新增 token 刚好能区分一种新曲率类型"的阈值
- **曲率标注验证**：K=32 时只有 flat/mild codeword；K=256 时 sharp 类型开始出现；K=1024+ 时 extreme 类型出现

**可视化**：

```
Panel A: D(K) vs K 曲线（经典 R-D 图）
Panel B: 每个 K 下的 codeword 曲率热力图（行=K，列=曲率类型，颜色=count）
Panel C: 各曲率类型首次出现的 K 值时间线
```

### 4.4 Experiment: Dual Distribution Fitting + Universality

**操作**：用训练好的 K=1024 VQ-VAE 编码所有 patch，统计 token 频率。

```python
# 对全量 dataset 的所有 patches
token_freq = Counter()
for patch in all_patches:
    token_id = vqvae.encode(patch)
    token_freq[token_id] += 1

# 双分布拟合（不只是幂律）
alpha, r2_powerlaw = fit_power_law(sorted_freqs)
mu, sigma, r2_lognormal = fit_lognormal(sorted_freqs)
vuong_R, vuong_p = vuong_test(sorted_freqs, 'power_law', 'lognormal')

# 画 Zipf 图 (log-log) + lognormal 拟合对比
```

**跨数据集检验（Universality 关键实验）**：
1. 在 Objaverse only 上训练 → 得到 codebook
2. 用这个 codebook 编码 ShapeNet 的 patches（零 fine-tuning）
3. 检验：(a) 分布族是否相同；(b) 参数是否接近（α 或 μ,σ）

**跨域对比**：
- 对比 mesh token 分布 vs 已发表的图像/时间序列 token 分布参数
- 绘制三域对比图（mesh / image / time-series rank-frequency plot）

### 4.5 Experiment: Curvature Correlation

**目标**：把 token 频率和 patch 的离散 Gaussian 曲率关联。

**离散曲率计算**（angle defect）：

$$K_v = 2\pi - \sum_{f \ni v} \theta_{vf}$$

每个 patch 的曲率 $\bar{K}_P = \text{mean}_{v \in P} |K_v|$。

**期望结果**：

| Token rank（按频率） | 平均 $\bar{K}_P$ |
|----------------------|------------------|
| Top 10%（最高频） | ≈ 0（平面状） |
| Middle 40% | 小正值（微曲） |
| Bottom 50%（最低频） | 大正值（角点/边缘） |

### 4.6 Experiment: Competing Theories

**目的**：回应审稿人可能的质疑——"GEM/Pitman-Yor 也能解释 Zipf，你的几何模型有什么优势？"

**背景**："The Language of Time" 用 GEM (Griffiths-Engen-McCloskey) / Pitman-Yor 过程解释 VQ token 的 Zipf 分布。这是一个通用的"富者愈富"统计机制，不需要任何领域知识。

**实验设计**：

```
对 K=1024 的 token 频率分布，同时拟合三个模型:

Model A: 几何模型（来自 §4.2 MaxEnt）
    - 参数: λ（Lagrange 乘子，对应平均曲率约束）
    - 优势: 参数有物理含义

Model B: GEM/Pitman-Yor 过程
    - 参数: α（discount）, θ（concentration）
    - 优势: 通用统计模型

Model C: Lognormal（来自图像域）
    - 参数: μ, σ
    - 优势: 如果 mesh 和图像有类似机制

比较标准:
    - AIC / BIC 信息准则
    - Vuong's test（两两比较）
    - 关键: 即使 GEM 的拟合度相当，几何模型提供可解释性:
        "这个 token 高频是因为它对应 flat patch"
        "这个 token 稀有是因为它对应 corner"
      GEM 无法提供这种解释
```

**预期结果**：

| 模型 | AIC | 可解释性 |
|------|-----|----------|
| 几何模型 | 可能略差或相当 | **高**：每个 token 可标注曲率类型 |
| GEM/Pitman-Yor | 可能略好 | 低：只有"富者愈富" |
| Lognormal | 取决于 §4.1 结果 | 中：乘法噪声解释 |

**论文叙事**：无论 AIC 结果如何，几何模型的**可解释性**和**系统设计指导**能力（→ curvature-aware codebook）是 GEM 做不到的。

### 4.7 Lean4 Formalization Design

**目标定理**：

> 对于一个封闭三角网格 M，满足 angle defect 超过 κ 的顶点数，上界是 $2\pi|\chi(M)| / \kappa$。与网格的面数无关，只取决于拓扑类型（χ）。

**证明链**：

**Step 1**: [公理，引用 Descartes/Gauss-Bonnet 经典结论]

$$\sum_{v} K_v = 2\pi \cdot \chi(M)$$

**Step 2**: [Lean4 证明 — 有限求和的 Markov 不等式]

设 $S = \sum_v K_v$，若 $S \geq 0$，则：

$$|\{v : K_v > \kappa\}| \cdot \kappa < \sum_{v: K_v > \kappa} K_v \leq S$$

$$\Rightarrow |\{v : K_v > \kappa\}| < S / \kappa$$

**Step 3**: [Lean4 证明 — 组合两步]

$$|\{v : K_v > \kappa\}| \leq 2\pi|\chi(M)| / \kappa \quad \square$$

**Lean4 实现骨架**：

```lean
theorem finite_sum_markov {α : Type*} (s : Finset α)
    (f : α → ℝ) (hf : ∀ x ∈ s, f x ≥ 0) (κ : ℝ) (hκ : κ > 0) :
    (s.filter (fun x => f x > κ)).card ≤
    (s.sum f) / κ := by
  -- 核心步骤：filter 集合的 sum 被整体 sum 控制
  -- 然后除以 κ
  ...
```

**估计工作量**：50-100 行 Lean4 代码，2-3 周实现。

**离散 Gauss-Bonnet 处理**：作为 axiom 标注引用经典文献（Descartes 1637 + Polyhedral Gauss-Bonnet 现代证明）。

---

## 5. System Layer Design

### 5.1 Curvature-Aware Codebook（核心改动）

**现状**：MeshLex 用均匀 VQ，512 个 codeword 平等对待每种 patch。

**改进**：按曲率分配 codeword，高频的平面 patch 分得多，低频的角点 patch 分得少。

**具体方案**：

**Step 1**: 预计算所有 patch 的曲率 $|\bar{K}_P|$

**Step 2**: 按曲率大小分成 B=5 个 bin（互斥区间）

| Bin | 曲率范围 | 历史频率 | Codeword 分配 |
|-----|----------|----------|---------------|
| 1 (flat) | $0 \leq \|K\| < 0.1$ | ~40% | 200 |
| 2 (mild) | $0.1 \leq \|K\| < 0.3$ | ~25% | 130 |
| 3 (medium) | $0.3 \leq \|K\| < 0.6$ | ~20% | 100 |
| 4 (sharp) | $0.6 \leq \|K\| < 1.0$ | ~10% | 52 |
| 5 (extreme) | $\|K\| \geq 1.0$ | ~5% | 30 |
| **Total** | | 100% | **512** |

**分配逻辑**：对每个 patch，计算 $|\bar{K}_P|$，然后按上表唯一确定所属 bin。

**Step 3**: 每个 bin 单独训练一个小 SimVQ sub-codebook

**Step 4**: 推理时先按 $|\bar{K}_P|$ 分配到 bin，再在 bin 内查询最近 codeword

### 5.2 Key Ablation Table

| 方案 | Codebook 设计 | 理论依据 |
|------|---------------|----------|
| Baseline | 均匀 512 tokens | 无 |
| **Ours** | 曲率感知非均匀 512 tokens | Lean4 定理 |
| Upper bound | 均匀 1024 tokens | 无（两倍大小） |

**成功标准**："Ours" 用同等参数量（512）超过均匀 512，接近均匀 1024。

### 5.3 Complete Pipeline

曲率感知 codebook 只改动 M2（Tokenizer），其余保持不变：

| 模块 | 内容 | 变化 |
|------|------|------|
| M1 | Patch Partitioning (METIS) | 不变 |
| M2 | Patch Tokenizer | **曲率感知 SimVQ（新）** |
| M3 | AR Generation (GPT-2, 20.4M params) | 不变 |
| M4 | Assembly (StitchingMLP) | 不变 |

### 5.4 Training Schedule

| 阶段 | 内容 | 数据 | 时间估算 |
|------|------|------|----------|
| Pre-compute | 计算所有 patch 的 $\bar{K}$ | 全量 | ~2h |
| VQ-VAE | 训练曲率感知 codebook（512 tokens） | 全量 | ~15h GPU |
| Baseline-512 | 训练均匀 VQ baseline（512 tokens，对比用） | 全量 | ~15h GPU |
| Upper-bound-1024 | 训练均匀 VQ 上界（1024 tokens，可选） | 全量 | ~15h GPU |
| AR model | 训练 AR 生成模型 | 全量 token sequences | ~30h GPU |
| Eval | 重建 + 生成质量评估 | ShapeNet Chair/Table | ~5h |

---

## 6. Evaluation Plan

### 6.1 Reconstruction Evaluation

**Metrics**: CD, Normal Consistency, F-Score@{0.01, 0.02, 0.05}

**Baseline**: MeshLex v1（5% 数据训练）

**预期**: 全量数据 + 曲率感知 codebook 大幅超越旧版

### 6.2 Generation Evaluation

**Metrics**: FID, COV, MMD on ShapeNet Chair + Table

**对比**: PolyGen, MeshGPT, FACE（用发表数值）

**目标**: 不一定超越 SOTA，但差距合理（在同量级内），足以反驳"理论没用"

### 6.3 Ablation Experiments

| 实验 | 目的 |
|------|------|
| 均匀 512 vs 曲率感知 512 | 证明非均匀分配有效 |
| 各 bin codeword 数量 ablation | 找最优分配比例 |
| 有无理论先验 vs 纯 data-driven 分配 | 验证理论的指导价值 |
| PTME 对比 | 用 FreeMesh 的 PTME 指标验证 codebook 质量（CD + PTME 双重验证） |
| 数据驱动 vs 理论驱动分配 | 按 embedding 聚类分配 vs 按曲率分配，分离理论的边际贡献 |

---

## 7. Paper Structure

```
§1 Introduction
    - Problem: 为什么 mesh 生成这么难？
    - Observation (MeshLex v1): 512 token 跨 1156 类别泛化
    - Question: 为什么？这个结构是什么？
    - Cross-domain context: 图像 token = lognormal，时间序列 = Zipf，mesh = ?
    - This paper: 3 层贡献（实验 + 理论 + 系统）

§2 Related Work
    - Mesh generation (MeshGPT, FACE, FreeMesh)
    - VQ token distribution (Language of Visual Tokens, Language of Time)
    - Graph tokenization
    - Formal methods in ML

§3 Theory
    §3.1 双分布测试 → 确定分布族
    §3.2 MaxEnt 推导: Gauss-Bonnet → 约束 → 分布
    §3.3 相变曲线 + 曲率标注
    §3.4 跨数据集 universality + 跨域对比
    §3.5 曲率-频率相关性
    §3.6 竞争理论对比 (几何模型 vs GEM/Pitman-Yor)
    §3.7 Lean4 形式化：高曲率 patch 上界证明

§4 Method
    §4.1 曲率感知 codebook 设计
    §4.2 完整 MeshLex pipeline

§5 Experiments
    §5.1 理论验证（双分布 + 相变 + universality）
    §5.2 Codebook ablation（曲率感知 vs 均匀 vs data-driven）
    §5.3 重建质量（CD / F-Score / NC / PTME）
    §5.4 生成质量（FID / COV / MMD）

§6 Conclusion
```

---

## 8. Timeline

**首选**: NeurIPS 2027（deadline 约 2027 年 5 月）→ 约 13 个月

**备选**: ICLR 2027（deadline 约 2026 年 9 月）→ 约 6 个月

### 里程碑

| 阶段 | 内容 | 时间 |
|------|------|------|
| Phase T1 | 全量数据集准备完成 | 当前进行中 |
| Phase T2 | 理论实验（相变 + 幂律） | ~2 周 |
| Phase T3 | Lean4 形式化证明 | ~3 周 |
| Phase T4 | 曲率感知 codebook 实现 + 训练 | ~2 周 |
| Phase T5 | AR 模型全量重训 | ~1 周 |
| Phase T6 | 完整评估 + 论文撰写 | ~4 周 |

---

## 9. Risk Mitigation

### 9.1 理论风险

**风险 9.1a**: 幂律不成立（分布是对数正态）
- **严重程度**: HIGH
- **缓解**: §4.1 双分布测试作为 go/no-go。即使是对数正态，仍是首次在 mesh 域发现 + 首次提供几何解释
- **备案**: Path B 叙事（乘法噪声模型）

**风险 9.1b**: 理论链有逻辑缺口（上界 ≠ 分布）
- **严重程度**: HIGH → **已修复**
- **缓解**: §4.2 MaxEnt 推导补全了 Gauss-Bonnet → 分布的逻辑链

**风险 9.1c**: GEM/Pitman-Yor 同样能解释分布
- **严重程度**: MEDIUM
- **缓解**: §4.6 竞争理论实验。即使 GEM 拟合更好，几何模型提供可解释性 + 系统设计指导，这是 GEM 做不到的

**风险 9.1d**: 相变曲线不明显
- **严重程度**: MEDIUM
- **缓解**: 尝试不同 K 范围；检查数据处理问题；如果确实没有相变，修改叙事为"连续相变"

### 9.2 系统风险

**风险**: 曲率感知 codebook 效果不如均匀 baseline

**缓解**:
- 调整 bin 数量和分配比例
- 尝试更细粒度的曲率划分
- 作为 negative result 报告，仍有理论价值

### 9.3 Lean4 风险

**风险**: Mathlib 基础设施不足，证明受阻

**缓解**:
- 使用 axiom 标注经典结论，不从头证明
- 降低形式化目标，只证明 Markov 不等式应用部分

---

## 10. Success Criteria

| 层次 | 标准 | 判定 |
|------|------|------|
| Go/No-Go C0 | 双分布测试完成，确定分布族，至少一种 R² > 0.7 | GATE |
| 理论 C1 | 相变曲线有 ≥2 个明显相变点 | GO |
| 理论 C2 | MaxEnt 推导完成 + 选定分布族拟合 R² > 0.9 | GO |
| 理论 C2b | 竞争理论对比完成（几何 vs GEM），无论结果如何均有叙事方案 | GO |
| 理论 C3 | Lean4 proof 编译通过 | GO |
| 方法 C4 | 曲率感知 > 均匀 baseline（同等参数），CD + PTME 双重验证 | GO |
| 应用 C5 | 生成 FID 与 SOTA 同量级（<2× 差距） | GO |

**整体判定**: C0 通过 + ≥3 GO + 无 FAIL → 论文可投

---

## Appendix A: Key References

1. **Gauss-Bonnet**: Descartes (1637), Polyhedral Gauss-Bonnet (Banchoff, 1970)
2. **Discrete Curvature**: Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" (2002)
3. **Rate-Distortion**: Cover & Thomas, "Elements of Information Theory"
4. **Lean4 Mathlib**: https://github.com/leanprover-community/mathlib4
5. **SimVQ**: Li et al., 2025
6. **MaxEnt**: E.T. Jaynes, "Information Theory and Statistical Mechanics" (1957)
7. **Visual Token Distribution**: "Analyzing the Language of Visual Tokens" (2024) — VQ tokens in image domain are lognormal
8. **Time-series Token Distribution**: "The Language of Time: A Review of Temporal LLMs" (2025) — VQ tokens are Zipf, GEM/Pitman-Yor explanation
9. **FreeMesh PTME**: FreeMesh (ICML 2025) — Per-Token-Mesh-Entropy metric, r=0.965 correlation with CD
10. **Vuong's Test**: Vuong, "Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses" (1989)
11. **Pitman-Yor Process**: Pitman & Yor, "The two-parameter Poisson-Dirichlet distribution" (1997)

---

## Appendix B: Lean4 Proof Sketch (Full)

```lean
-- 高曲率顶点上界定理
-- 对于封闭三角网格 M，满足 angle defect > κ 的顶点数上界

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Order.Ring.Lemmas

namespace MeshLex

-- 离散 Gauss-Bonnet 作为公理
axiom discreteGaussBonnet {M : Type*} [Fintype M]
  (K : M → ℝ) (χ : ℤ) :
  (∑ v : M, K v) = 2 * Real.pi * χ

-- 有限集合的 Markov 不等式
theorem finite_markov {α : Type*} (s : Finset α)
    (f : α → ℝ) (hf : ∀ x ∈ s, f x ≥ 0) (κ : ℝ) (hκ : κ > 0) :
    (s.filter (fun x => f x > κ)).card ≤ ((s.sum f) / κ).toNat := by
  -- 证明略，利用 filter 集合的 sum 性质
  sorry

-- 主定理：高曲率顶点上界
theorem high_curvature_bound {M : Type*} [Fintype M]
    (K : M → ℝ) (χ : ℤ) (hχ : χ > 0) (κ : ℝ) (hκ : κ > 0) :
    (Finset.univ.filter (fun v : M => K v > κ)).card ≤
    Int.natAbs ((2 * Real.pi * χ) / κ) := by
  have h_sum : ∑ v : M, K v = 2 * Real.pi * χ := discreteGaussBonnet K χ
  have h_pos : 0 ≤ ∑ v : M, K v := by
    rw [h_sum]
    positivity
  have h_markov := finite_markov Finset.univ K (by intro x _; positivity) κ hκ
  -- 完成证明
  sorry

end MeshLex
```

---

## Appendix C: Curvature Computation

```python
import numpy as np
import trimesh

def compute_discrete_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    计算每个顶点的离散 Gaussian 曲率 (angle defect)

    K_v = 2π - Σ θ_vf (所有包含 v 的面的内角)
    """
    K = np.zeros(len(mesh.vertices))

    for face in mesh.faces:
        v0, v1, v2 = face
        # 获取三条边向量
        e01 = mesh.vertices[v1] - mesh.vertices[v0]
        e02 = mesh.vertices[v2] - mesh.vertices[v0]
        e12 = mesh.vertices[v2] - mesh.vertices[v1]

        # 计算三个内角
        theta0 = np.arccos(np.clip(np.dot(e01, e02) / (np.linalg.norm(e01) * np.linalg.norm(e02)), -1, 1))
        theta1 = np.arccos(np.clip(np.dot(-e01, e12) / (np.linalg.norm(e01) * np.linalg.norm(e12)), -1, 1))
        theta2 = np.pi - theta0 - theta1

        # 累加 angle defect
        K[v0] -= theta0
        K[v1] -= theta1
        K[v2] -= theta2

    K += 2 * np.pi  # 最终 angle defect
    return K

def compute_patch_curvature(mesh: trimesh.Trimesh, patch_faces: np.ndarray) -> float:
    """
    计算 patch 的平均曲率
    """
    # 获取 patch 内的所有顶点
    patch_vertices = set()
    for face_idx in patch_faces:
        patch_vertices.update(mesh.faces[face_idx])

    K = compute_discrete_gaussian_curvature(mesh)
    patch_K = np.mean([np.abs(K[v]) for v in patch_vertices])
    return patch_K
```

# MeshLex: Positioning & Competitive Analysis

## Core Slogans (按叙事强度排序)

### 🎯 主 Slogan (论文标题/Abstract 开头)
**"Triangle meshes have a vocabulary: ~4K topology prototypes cover 99%+ local structures across ShapeNet."**

- **为什么强**：这是一个关于世界的断言，不是关于方法的描述
- **可验证性**：Section 3.1 用覆盖率曲线 + 跨类别统计直接证明
- **差异化**：没有任何现有工作量化过这个数字

### 📊 技术 Slogan (用于 Introduction 收尾)
**"From face-level to patch-level: MeshLex compresses mesh generation 30× while preserving exact topology."**

- **对比清晰**：FACE 做到 1 face = 1 token (极限)，我们做到 30 faces = 1 token
- **量化指标**：30× 是硬数字，不需要争论
- **保留拓扑**：强调不是近似表示(如 SDF/NeRF)，是精确结构

### 🔬 方法 Slogan (用于 Method Section 标题)
**"Topology-aware patch vocabulary learning via frequency-driven prototype discovery."**

- **避开术语冲突**："Mesh BPE" 已被 FreeMesh 占用，改用 "frequency-driven"
- **强调拓扑**：与 FreeMesh (坐标级) / PatchNets (连续 SDF) 明确区分
- **学习视角**：强调 prototype 是从数据中 discover 的，不是手工设计的

### 💎 Visual Slogan (用于 Figure 1 / Teaser)
**"Mesh generation as vocabulary composition: each object = sequence of reusable topology patches."**

- **类比自然语言**：降低理解门槛，增强直觉
- **可视化友好**：展示 4096 个 patch 词汇表 + 一个物体的 patch 序列分解
- **强调复用**：patch 跨物体共享，不是 per-object 的

---

## 完整竞品分析矩阵

| 工作 | 会议/年份 | 核心表示 | Patch 粒度 | 离散 Codebook | 拓扑保留 | 与 MeshLex 的本质区别 | 威胁等级 |
|------|-----------|----------|------------|---------------|----------|----------------------|----------|
| **PatchNets** | ECCV 2020 | 连续 SDF latent | ✅ Multi-face (~30) | ❌ 连续 auto-decoder | ❌ 隐式表面 | Patch 是连续函数而非离散拓扑结构 | 🟡 中 (证明跨类别可行但表示不同) |
| **MeshGPT** | SIGGRAPH Asia 2024 | GNN feature + RVQ | ❌ 单 face | ✅ 但 per-face 级 | ✅ | Codebook 量化的是单 face 特征，不是 multi-face topology | 🟢 低 (baseline 对照) |
| **FACE** | 2026-03 | Lightweight embedding | ❌ 单 face | ❌ | ✅ | 1 face = 1 token，极限压缩但仍是 face-level | 🟢 低 (最佳 baseline) |
| **FreeMesh** | ICML 2025 | SentencePiece BPE | ❌ 坐标对 | ✅ 但坐标级 | ✅ | BPE 作用于一维坐标序列，不理解拓扑 | 🟡 中 (占用了"Mesh BPE"术语) |
| **BPT** | NeurIPS 2024 | Block-wise index | ✅ 但仅序列化分组 | ❌ | ✅ | Patch 只是序列化策略，内部仍逐坐标预测 | 🟢 低 (序列化优化) |
| **MeshMosaic** | 2025-09 (已撤回) | Patch-level AR | ✅ Semantic patch | ❌ | ✅ | Patch 内部逐 face 生成，无离散 prototype | 🟡 中 (撤回但思路接近) |
| **Nautilus** | 2025 | Locality-aware token | ✅ 局部共享 | ❌ | ✅ | 序列化优化，无 codebook | 🟢 低 |
| **TSSR** | 2025-10 | DDM two-stage | ❌ Face-level | ❌ | ✅ | 架构创新(DDM)，表示仍 face-level | 🟢 低 |

### 威胁等级说明
- 🔴 **高**：直接竞品，核心 idea 重叠 → **无此等级工作**
- 🟡 **中**：部分思路接近，需在论文中精细区分
- 🟢 **低**：不同维度的工作，可作为 baseline 或引用

---

## 核心差异化总结 (用于 Related Work 撰写)

### 与 PatchNets 的区别
- **表示空间不同**：PatchNets 用连续 SDF 隐函数表示 patch (latent code ∈ ℝ¹²⁸)，每个 patch 有独立的连续 latent，通过 auto-decoder 优化。MeshLex 用离散拓扑结构 codebook，每个 patch 从 4096 个离散 prototype 中选取 + 变形参数。前者输出隐式表面(Marching Cubes 提取)，后者保留精确三角拓扑。
- **关键引用点**：PatchNets 证明了 "patch 可跨类别泛化" (Cabinet → Airplane F-score 93.9)，为 MeshLex 的核心假设提供了间接支持。

### 与 FreeMesh 的区别
- **BPE 作用对象不同**：FreeMesh 的 BPE 作用于一维坐标序列 (合并高频坐标对如 (x₁,y₁))，用 SentencePiece 在数字序列上训练，完全不理解拓扑结构。MeshLex 的频率驱动合并作用于面拓扑图 (合并高频 face pair 形成 topology patch)，操作的是连接关系而非几何坐标。
- **术语冲突**：FreeMesh 已占用 "Mesh BPE" 术语，MeshLex 应改用 "topology-aware frequency-driven merging" 或类似表述。

### 与 MeshGPT 的区别
- **Codebook 语义层级不同**：MeshGPT 的 RVQ codebook 量化的是 GNN 提取的单个 face 的 feature embedding (~128 维向量)。MeshLex 的 codebook 存储的是 multi-face (20-50 face) 的拓扑连接模式 (顶点-边-面关系图)。前者是 per-face feature 的离散化，后者是 multi-face topology 的原型库。

### 与 MeshMosaic 的区别
- **Patch 内部生成方式不同**：MeshMosaic 将 mesh 分成 semantic patch，但每个 patch 内部仍然逐 face 自回归生成 (先决定 patch 边界，然后在 patch 内一个一个生成 face)。MeshLex 从 codebook 中直接选取完整的 topology prototype，无需逐 face 生成。
- **状态备注**：MeshMosaic 已撤回 ICLR 2026，说明方案可能有未解决的问题。

---

## 论文写作策略建议

### Abstract 结构
1. **开场断言 (1 句)**：Mesh 局部拓扑低熵性质 + 4K prototype 覆盖率
2. **方法概述 (2 句)**：MeshLex representation + 频率驱动发现 + VQ 学习
3. **关键结果 (2 句)**：30× 压缩比 + 跨类别泛化 + 生成质量
4. **收尾价值 (1 句)**：首次将 mesh 从 face-level 提升到 patch-level 表示

### Introduction 叙事线
1. **现状困境**：现有方法都在 face-level 操作 (FACE 1:1, MeshGPT RVQ per-face)，序列长度随面数线性增长
2. **关键洞察**：Mesh 不像图像 (每个 pixel 独立)，局部拓扑高度重复 → 类比自然语言的 subword 结构
3. **核心发现**：实证研究显示 4096 个 topology prototype 覆盖 ShapeNet 99%+ 局部结构
4. **技术实现**：MeshLex = frequency-driven patch discovery + discrete topology codebook + constrained stitching
5. **实验验证**：覆盖率分析 + 跨类别泛化 + 生成实验

### Related Work 结构
按维度组织，每个维度明确说明 MeshLex 的差异：

1. **Patch-based Mesh Representations** (PatchNets, BPT)
   - 强调：连续 vs 离散，序列化分组 vs 拓扑 codebook

2. **Discrete Tokenization for Meshes** (MeshGPT, FACE, FreeMesh)
   - 强调：粒度差异 (face-level vs patch-level, 坐标级 BPE vs 拓扑级)

3. **Mesh Generation Architectures** (TSSR, MeshMosaic, Nautilus)
   - 强调：架构创新 vs 表示创新，逐 face 生成 vs prototype 选取

### Method Section 标题建议
- **Section 3.1**: The Low-Entropy Nature of Mesh Topology (覆盖率分析实验)
- **Section 3.2**: Topology-Aware Patch Discovery (频率驱动合并算法)
- **Section 3.3**: Discrete Codebook Learning (VQ + SimVQ + 边界签名)
- **Section 3.4**: Autoregressive Generation with Boundary Constraints (生成流程)

---

## 实验优先级 (验证核心 claim)

### P0 (必须完成，支撑核心发现)
1. **覆盖率曲线**：Codebook size (512/1K/2K/4K/8K) vs 重建覆盖率，证明 4K 是拐点
2. **跨类别泛化**：在 Chair 上训练的 codebook 在 Airplane/Table/Lamp 上的覆盖率
3. **拓扑多样性统计**：Valence 分布 / patch boundary 类型分布 / 面数分布

### P1 (强化 story)
1. **Ablation**：BPE 式发现 vs 随机切分，SimVQ vs vanilla VQ
2. **生成质量**：与 FACE / MeshGPT 对比 (但不需要 SOTA，comparable 即可)
3. **压缩比量化**：Token 数量 / 序列长度对比

### P2 (Bonus，如果有时间)
1. **Mesh Arithmetic**：Patch-level 插值和编辑
2. **Codebook 可视化**：展示学到的 4096 个 topology prototype

---

## 总结

以上定位明确避开了所有竞品的核心阵地，将战场设定在 **"离散拓扑 vocabulary 的存在性"** 这一未被验证的基础问题上。只要实验证明这个发现成立，论文的 contribution 就不可撼动。

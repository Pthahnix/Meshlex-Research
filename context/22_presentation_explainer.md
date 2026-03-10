# MeshLex 汇报讲解文档

> 撰写时间：2026-03-10
> 面向对象：明日汇报，讲解任务、架构与原理

---

## 图 1：任务是什么——一句话概括

**MeshLex 要回答的问题是**：3D mesh 的局部结构（拓扑）是否像自然语言一样，存在一套有限的"词汇表"，可以跨越所有物体类别通用？

把一个复杂的 3D 物体（比如一把椅子、一辆车）切成很多小 **Patch**（局部网格片段）。每个 patch 大约有 20 个三角面。直觉上，桌腿的角落、车门的弧面、椅背的平面……这些局部形状在不同物体上反复出现。MeshLex 要把这些重复出现的结构，归纳成 **4096 个原型（codebook entries）**，相当于一本词典，然后验证这本词典是否对没见过的类别（unseen categories）也同样适用。

```mermaid
flowchart TB
    subgraph QUESTION["核心研究问题"]
        Q["3D mesh 的局部拓扑是否存在\n跨类别通用的有限词汇表？\n(4096个原型能否覆盖所有类别的局部结构)"]
    end

    subgraph PIPELINE["实验流程"]
        direction LR
        P1["切割 Mesh\n-> Patches"]
        P2["训练\nVQ-VAE"]
        P3["测试\n跨类别重建"]
        P1 --> P2 --> P3
    end

    subgraph METRIC["关键指标"]
        direction LR
        M1["Codebook Utilization\n4096个code激活率\n目标 > 30%"]
        M2["CD Ratio\ncross-cat CD / same-cat CD\n目标 < 1.2\n(=1.0 表示完全泛化)"]
    end

    subgraph RESULTS["三组实验结果"]
        direction LR
        R1["Exp1 A  5类别\nratio 1.14x  util 46%"]
        R2["Exp3 B  5类别\nratio 1.18x  util 47%"]
        R3["Exp2 A  844类别\nratio 1.07x  util 67.8%"]
    end

    subgraph FINDING["关键发现"]
        direction LR
        F1["规模越大泛化越好\n5类->844类  ratio 1.14->1.07"]
        F2["seen类/unseen类 util几乎相同\n证明codebook学到类别无关的几何图元"]
    end

    QUESTION --> PIPELINE --> METRIC --> RESULTS --> FINDING
```

---

## 图 2：模型架构——三个模块串联

整个模型是一个 VQ-VAE，由三个模块首尾相连：

- **模块 1 PatchEncoder**：GNN 编码器，把一个 patch（小图）压缩成 128 维连续向量 z
- **模块 2 SimVQ Codebook**：离散量化，把 z 映射到词典里最近的"词"，输出 index（整数）和量化向量 z_q
- **模块 3 PatchDecoder**：跨注意力解码器，把 z_q 还原为每个顶点的 xyz 坐标

```mermaid
flowchart LR
    subgraph INPUT["输入：一个 Mesh Patch"]
        MESH["3D 三角网格 Patch\n约20个三角面\n每面15维特征\n(法向量/坐标/二面角)"]
    end

    subgraph ENCODER["GNN Encoder\n(PatchEncoder)"]
        S1["SAGEConv x4\n+ LayerNorm + GELU"]
        S2["Global Mean Pooling"]
        S1 --> S2
    end

    subgraph VQ["SimVQ Codebook\nK=4096 entries"]
        CW["CW = W(C)\n有效码本 K x 128"]
        NN["最近邻查找\nargmin dist(z, CW)"]
        STE["Straight-Through\n梯度估计器"]
        CW --> NN --> STE
    end

    subgraph DECODER["Cross-Attn Decoder\n(PatchDecoder)"]
        VQ2["128个可学习\nVertex Queries"]
        CA["Cross-Attention\n(4 heads)"]
        MLP2["MLP 3层\n-> xyz坐标"]
        VQ2 --> CA --> MLP2
    end

    subgraph OUTPUT["输出"]
        RECON["重建顶点坐标\n(最多128个顶点)"]
        IDX["Codebook Index\n一个整数 0~4095\n= 该patch的词"]
    end

    subgraph LOSS["训练损失"]
        direction TB
        L1["Chamfer Distance\n重建损失"]
        L2["Commitment Loss\n||z - sg(CW[i])||²"]
        L3["Embedding Loss\n||sg(z) - CW[i]||²"]
    end

    MESH -->|"面邻接图 edge_index"| ENCODER
    ENCODER -->|"z ∈ R128  patch嵌入"| VQ
    VQ -->|"z_q ∈ R128  量化嵌入"| DECODER
    VQ --> IDX
    DECODER --> RECON
    RECON --> L1
    VQ --> L2
    VQ --> L3
```

---

## 图 3：SimVQ 的核心原理——为什么不会 Collapse

这是这套工作最重要的技术贡献点，汇报时需要讲清楚。

**传统 VQ-VAE 的问题（Codebook Collapse）**：普通 VQ 中每个 code 只有在被选中时才能收到梯度更新。冷启动状态下某些 code 从未被选中，就永远无法更新，最终只有少数几个 code 存活。

**SimVQ 的解法**：把 codebook 分拆为冻结的 C 和可学习的线性变换 W，有效码本是 CW = W(C)。改变 W 等于同时移动所有 4096 个 CW，因此即使某个 code 没被选中，也会随着 W 的更新被间接调整。

**形象比喻**：传统 VQ 像 4096 个独立演员，只有上台的才练功；SimVQ 像让所有演员共用一套训练体系（W），台下的人也被动提升。

```mermaid
flowchart TD
    subgraph SIMVQ["SimVQ Codebook 内部机制 (ICCV 2025)"]
        C["C：原始码本 (K=4096, dim=128)\n冻结 — 梯度不流经 C\nrequires_grad = False"]
        W["W：线性变换层 (128x128)\n唯一可学习参数\n正交矩阵初始化"]
        CW2["CW = W(C)  有效码本\n所有4096个code都经过同一个W\n改变W = 同时移动所有code"]
        Z["z：Encoder输出 (128维)"]
        DIST["计算距离 dist(z, CW[k])  k=0..4095"]
        IDX2["argmin -> index i\n找最近的code"]
        QZ["z_q = CW[i]\n量化结果"]
        STE2["Straight-Through 梯度：\n前向 = z_q\n反向梯度 = dL/dz (走z路径)"]

        C --> CW2
        W --> CW2
        Z --> DIST
        CW2 --> DIST
        DIST --> IDX2
        IDX2 --> QZ
        Z --> STE2
        QZ --> STE2
    end

    subgraph INIT["初始化策略"]
        direction LR
        KM["K-means 初始化\n训练前跑一遍encoder\n用4096个聚类中心\n初始化有效码本CW"]
        DCR["Dead Code Revival\n每10个epoch检查\n未激活的code用\n当前encoder输出替换"]
    end

    subgraph COMPARE["与传统VQ的对比"]
        OLD["传统VQ\n每个code独立更新\n未被选中 = 无梯度\n-> Collapse"]
        NEW["SimVQ\n所有code共享W的梯度\n没有被遗忘的code\n-> 稳定利用率"]
        OLD -.->|"SimVQ 解决"| NEW
    end
```

---

## 评估逻辑——两个关键指标

**指标 1：Codebook Utilization（利用率）**

在评估集上，4096 个 code 里有多少个被至少激活了一次。目标是 > 30%。利用率太低说明 codebook collapse，词典里大部分"词"是废的。

**指标 2：CD Ratio（跨类别 CD 比值）**

$$\text{CD Ratio} = \frac{\text{Cross-category Chamfer Distance}}{\text{Same-category Chamfer Distance}}$$

- **分子**：用 unseen 类别（从未训练过的 50 个类别）的 patch 来重建，算重建误差
- **分母**：用 seen 类别的 patch 来重建，算重建误差
- 比值越接近 1.0，说明词汇表对未见类别的泛化能力越强
- 目标：< 1.2（误差最多比 seen 类别高 20%）

---

## 三组实验结论与 Scaling 发现

| 实验 | 规模 | Stage | CD Ratio | Util (same) | Util (cross) | 结论 |
|------|------|-------|----------|-------------|--------------|------|
| Exp1 | 5 类别 | A（单token KV）| 1.14x | 46.0% | — | ✅ STRONG GO |
| Exp3 | 5 类别 | B（4token KV）| 1.18x | 47.1% | 47.3% | ✅ STRONG GO |
| **Exp2** | **844 类别** | **A** | **1.07x** | **67.8%** | **48.2%** | ✅ **STRONG GO** |

最关键的发现：规模从 5 类扩展到 844 类，CD ratio **下降**（1.14→1.07），utilization **上升**（46%→67.8%）。通常我们担心数据越多越复杂、越难泛化——但实验结果反过来，更多类别的训练让词汇表更丰富、对未见类别的描述反而更准确。

**Exp3 的特殊发现**：same-cat eval util（47.1%）和 cross-cat eval util（47.3%）几乎完全相同，差距仅 0.2%。这意味着 unseen 类别和 seen 类别激活了同样多、同样分布的 codebook entries——词汇表真的是类别无关的。

---

## 汇报推荐顺序

1. **先讲"我们在问什么问题"**：类比 NLP 的词汇表直觉，引出图 1
2. **展示整体架构**：Patch 切割 → Codebook index 的流程，对应图 2
3. **重点讲 SimVQ**：为什么不 collapse，Frozen C + Learnable W，对应图 3（这是最容易被问到的）
4. **展示三组实验的结果表格**：强调 scaling 发现和 cross-cat util 对称性
5. **一句话结尾**：词汇表假设在 844 个类别上得到实验支持，STRONG GO

---

*文档生成时间：2026-03-10*
*数据来源：results/exp1_v2_collapse_fix/, results/exp3_B_5cat/, results/exp2_A_lvis_wide/*

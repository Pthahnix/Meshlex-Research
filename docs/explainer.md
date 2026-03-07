# MeshLex 是什么？— 完整讲解

## 我们在解决什么问题？

**任务：让 AI 生成 3D 模型的网格（Mesh）**

3D mesh 是游戏、影视、工业设计里描述三维物体形状的标准格式——本质上是一堆三角形拼成的表面。想象一个椅子的 3D 模型，它的表面是由几千个三角形贴合而成的。

AI 要生成这样的 mesh，就要回答：**这几千个三角形，每一个放在哪里、怎么连接？**

---

## 现有方法怎么做的？

所有现有方法都在做同一件事：**把 mesh 强行压扁成一条序列**，然后像写文章一样一个字一个字生成。

具体来说：一个有 4000 个三角面的 mesh，就被序列化成约 4000 个 token，喂给 Transformer，让它逐 token 预测下一个。这就是 MeshGPT、FACE、Nautilus 等方法的核心思路——区别只在于"怎么排序这条序列"。

**这种做法的根本问题是**：mesh 本质上是一个图（graph），每个三角面和邻居面有连接关系。把它强行压成一条序列，就像把一张地图剪成细条再让人拼回去——信息严重损失，序列极长（4000 个 token），生成速度慢，质量也受限。

### 现有方法对比

| 方法 | 核心思路 | Token 数（4K face mesh） | 问题 |
|------|---------|------------------------|------|
| MeshGPT (CVPR 2024) | 逐 face 自回归，graph conv VQ | ~800 tokens | 序列太长 |
| FACE (ICML 2026) | per-face VQ，1 face = 1 token | ~400 tokens | 压缩已到极限 |
| Nautilus (ICCV 2025) | shell-based 排序，per-face | ~1000 tokens | 仍是逐面生成 |
| MeshMosaic (2025) | 大 patch 分治，patch 内仍逐 face | N/A | 没有 codebook，仍序列化 |
| FreeMesh (ICML 2025) | BPE 合并坐标值 | ~300 tokens | 坐标级 BPE，非拓扑级 |

---

## MeshLex 的核心洞察

我们问了一个不一样的问题：

> **Mesh 真的需要逐面生成吗？**

想想语言模型的进化史。最早的 NLP 模型是字符级的——一个字母一个字母生成。后来有了词汇表（vocabulary）：常见的词直接作为一个单元，"elephant" 是 1 个 token 而不是 8 个字母。效率飙升，质量也更好。

**MeshLex 的核心假设是：mesh 和自然语言一样，存在一个有限的"局部拓扑词汇表"。**

一个椅子腿的截面、一个桌面边缘的过渡、一个圆弧曲面——这些局部形状模式，在成千上万个不同的 3D 模型里反复出现。如果我们能把这些重复出现的局部模式总结成一本"词汇书"（codebook），生成 mesh 就变成了：

> **从词汇书里选词 → 调整位置和形状 → 拼装在一起**

而不是逐面生成。

---

## 方案具体是什么？

### 第一步：把 Mesh 切成小 Patch

把一个完整的 mesh 切成约 30 个三角面的小块（patch）。一个 4000 面的 mesh 切成约 130 个 patch。

```
完整 Mesh (4000 faces)
        ↓ METIS 图分割
130 个 Patch，每个 ~30 faces
```

切割方式使用 METIS 图分割算法，沿着面与面的邻接关系切，切出来的 patch：
- 大小均匀（20–50 faces）
- 切割位置自然落在几何特征线上（棱线、曲率变化处）
- 每个 patch 做 PCA 对齐，归一化到局部坐标系

### 第二步：学一本"拓扑词汇书"（Codebook）

收集大量不同 3D 模型的 patch（几万个），训练一个 VQ-VAE：

```
Patch (30 faces, 15-dim face features)
        ↓ SAGEConv GNN Encoder (4 层)
128-dim embedding
        ↓ SimVQ 量化
Codebook ID（0 ~ 4095 中的一个整数）
        ↓ Cross-attention Decoder
重建顶点坐标
```

- **Encoder**：用图神经网络（SAGEConv GNN）把每个 patch 的几何和拓扑特征提取成 128 维向量
- **Codebook**：4096 个"原型 patch"——相当于词汇表里的 4096 个词
- **量化（SimVQ）**：每个 patch 被分配到最相似的原型，编码成一个 ID
- **Decoder**：从原型 ID 还原出这个 patch 的顶点坐标（cross-attention + MLP）

训练完成后，codebook 就是 mesh 的"词汇书"。

### 第三步：极端压缩

| 表示方式 | Token 数 | 说明 |
|---------|---------|------|
| 逐面序列（FACE） | ~400 tokens | 每 face 1 token |
| **MeshLex** | **~130 tokens** | 每 patch（~30 faces）1 token |

压缩比约为 **30:1**，比当前最好的 FACE 方法高一个数量级。

---

## 关键的科学假设

整个方案能不能成立，取决于一个核心问题：

> **这 4096 个原型 patch，能不能覆盖所有类型 3D 物体的局部结构？**

如果椅子上训练出来的原型，也能用来描述飞机、汽车、灯具——说明局部拓扑结构是"通用的"（universal vocabulary），方案成立。

如果不同类别的物体局部结构差异太大，词汇书就无法泛化。

### 成功标准

| 结果等级 | 跨类别 CD / 同类别 CD | 结论 |
|---------|---------------------|------|
| 强成功 | < 1.2× | Universal vocabulary，冲 CCF-A |
| 弱成功 | 1.2× – 2.0× | Transferable vocabulary，目标 ECCV |
| 边界 | 2.0× – 3.0× | 需修改 story |
| 失败 | > 3.0× | 核心假设不成立，止损 |

### 先行证据

已有论文间接支持这个假设：

- **PatchNets（ECCV 2020）**：仅在 Cabinet 上训练的 patch representation，跨类别 F-score 仅从 94.8 降至 93.9（<1% 降幅），甚至能重建人体模型
- **PatchComplete（NeurIPS 2022）**：明确指出"chairs and tables often share legs"，multi-resolution patch priors 在完全未见类别上实现 shape completion，CD 降低 19.3%
- **Valence 统计**：三角 mesh 的顶点 valence 高度集中在 5-6-7，不同类别、不同来源的 mesh 拓扑分布高度相似

---

## 代码架构

```
src/
├── data_prep.py        # Mesh 加载、降面（pyfqmr）、归一化
├── patch_segment.py    # METIS 分割 + PCA 局部坐标系对齐
├── patch_dataset.py    # NPZ 序列化 + PyG Dataset
├── model.py            # PatchEncoder / SimVQCodebook / PatchDecoder / MeshLexVQVAE
├── losses.py           # Masked Chamfer Distance
├── trainer.py          # 训练循环（staged VQ，codebook 利用率监控）
└── evaluate.py         # 同类/跨类 CD + Go/No-Go 判断

scripts/
├── download_shapenet.py     # 从 HuggingFace 下载 ShapeNet
├── run_preprocessing.py     # 批量预处理（降面 + 分割 + NPZ）
├── train.py                 # 训练入口（支持 --resume）
├── init_codebook.py         # K-means codebook 初始化
├── evaluate.py              # 评估入口
└── visualize.py             # t-SNE / utilization histogram / training curves
```

### 模型核心（model.py）

```python
# Encoder：4 层 SAGEConv GNN
PatchEncoder: (N_faces, 15) → (B, 128)

# Codebook：SimVQ 防 collapse
SimVQCodebook: (B, 128) → codebook_id (B,) + quantized_embedding (B, 128)

# Decoder：Cross-attention 还原顶点
PatchDecoder: (B, 128) → (B, max_vertices, 3)
```

### SimVQ 防 Collapse

标准 VQ-VAE 在 K=4096 时 codebook 利用率可能低至 11%。SimVQ 用一个 learnable linear layer 重参数化 codebook，让整个 codebook 空间都参与梯度更新，利用率接近 100%。

```python
class SimVQCodebook(nn.Module):
    def __init__(self, K=4096, dim=128):
        self.codebook = nn.Embedding(K, dim)
        self.linear = nn.Linear(dim, dim, bias=False)  # 关键：线性重参数化

    def forward(self, z):
        z_proj = self.linear(z)          # 映射到 codebook 空间
        distances = torch.cdist(z_proj, self.codebook.weight)
        indices = distances.argmin(-1)   # 找最近原型
        quantized = self.codebook(indices)
        return z + (quantized - z).detach(), indices  # straight-through
```

---

## 与竞品的本质区别

| | MeshMosaic | FreeMesh | FACE | **MeshLex** |
|---|---|---|---|---|
| 核心贡献 | 分治策略 | 坐标级 BPE | per-face VQ | **拓扑级 codebook** |
| Patch 内部还逐 face 生成？ | ✅ 是 | ✅ 是 | ✅ 是 | ❌ **否** |
| 有 topology codebook？ | ❌ | ❌ | ❌ | ✅ **是** |
| 压缩（4K face mesh） | N/A | ~300 tokens | ~400 tokens | **~130 tokens** |

MeshMosaic 把长序列切成短序列，本质仍是"序列化范式"。MeshLex 跳出这个范式，用词汇书替代逐面生成。

---

## 当前状态

- ✅ 14 个代码模块全部实现
- ✅ 17 个 unit test 全部通过
- ✅ 6 个 task 的真实数据验证产出已保存到 `results/`
- ⏳ **等待 ShapeNet HuggingFace 数据集审批**（已提交申请）
- ⏳ 审批通过后执行 Phase A+B（~5 小时），得出 Go/No-Go 决策

详见 [`RUN_GUIDE.md`](../RUN_GUIDE.md) 和 [`TODO.md`](../TODO.md)。

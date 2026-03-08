# Material: K-means 初始化与 SimVQ linear 层冲突

> 对应缺陷 #2：K-means 初始化覆写 codebook 后，linear 层处于不匹配的表示空间

---

## 1. MeshLex 的分阶段训练流程

```
Phase 1: Encoder-only (epoch 0-19)
  - loss = recon_loss only
  - codebook 参与 forward（选最近邻、straight-through）
  - 但 VQ loss（commit + embed）不加入总 loss
  - linear 层参与 forward，通过 recon_loss 的梯度间接更新

Phase 1.5: K-means 初始化
  - 用训练好的 encoder 编码所有 patches → z
  - 对 z 跑 K-means → 覆写 codebook.weight
  - linear.weight 保持不变

Phase 2: Full VQ-VAE (epoch 20+)
  - loss = recon_loss + commit_loss + embed_loss
  - codebook 和 linear 层同时学习
```

### 问题所在

Phase 1 结束时，`linear.weight` 已经被 encoder 的梯度更新了 20 个 epoch。它学到的变换是相对于 **random normal 初始化的 codebook** 优化的。

K-means 初始化**直接覆写**了 codebook.weight 为完全不同的分布（encoder output 的聚类中心），但 linear.weight 仍然保持 Phase 1 的旧值。

结果：`z_proj = linear(z)` 将 z 投影到一个为旧 codebook 优化的空间，然后在新 codebook（K-means centers）中搜索最近邻 → **搜索结果近乎随机**。

---

## 2. 如果按 SimVQ 官方实现（C frozen），问题如何变化

如果 C 是冻结的，K-means 初始化 C 是有意义的：
- C 冻结后不会通过梯度更新
- K-means 给 C 一个更好的初始位置
- W（linear 层）从 Phase 2 开始学习如何将 C 变换到匹配 encoder output 的空间

**但关键前提**：K-means 初始化 C 后，**W 应该重置为单位矩阵**（`nn.init.eye_(linear.weight)`），因为：
- 如果 W 保持旧值，`CW` 的分布既不是 K-means centers 也不是旧 codebook
- 重置 W = I 意味着 `CW = C`，即初始 codebook 就是 K-means centers
- 然后 W 从 I 开始学习微调整个空间

### SimVQ 论文的默认做法

SimVQ 论文**根本不使用 K-means 初始化**：
- C 从 `normal(0, dim^{-0.5})` 初始化并冻结
- W 从正交矩阵初始化
- 从 epoch 0 开始端到端训练（无分阶段）
- 论文证明这种纯随机初始化 + frozen C + learnable W 就能达到 100% utilization

**这意味着分阶段训练 + K-means 初始化这整套流程，在使用 SimVQ 时可能完全不必要。**

---

## 3. 社区关于 codebook 初始化的讨论

### Reddit r/learnmachinelearning — VQ Codebook Initialization

用户讨论了多种初始化策略：

> **方案 1**："Pre-training the encoder and decoder without quantization and then running KNN to find centroids of the embeddings to initialize the codebook."
>
> 评论：这种方法在 vanilla VQ-VAE 中是标准做法，但在 SimVQ 中因为 C 被冻结，K-means 初始化的价值降低。
>
> **方案 2**："Impose orthogonal regularization on the codebook to discourage code vectors from becoming parallel."
>
> 评论：与 SimVQ 的 orthogonal init for W 有类似思路，但作用在不同的对象上。

### VQGAN-LC (NeurIPS 2024) 的初始化方法

VQGAN-LC 使用预训练 CLIP 模型的特征初始化 codebook：
- 优点：codebook 从一开始就在有意义的特征空间中
- 缺点：依赖外部模型，泛化性差，SimVQ 论文批评这一点

### 标准 VQ-VAE 的 K-means 初始化

传统做法（Van Den Oord et al. 2017 + 后续工作）：
1. 先训练 encoder 若干 epoch
2. 收集 encoder outputs
3. K-means 聚类初始化 codebook
4. 继续训练

这在 vanilla VQ-VAE 中是合理的，因为 codebook 需要与 encoder output 分布对齐。**但 SimVQ 改变了这个前提** — W 的作用就是动态对齐 C 和 encoder output，所以 C 的初始化不再是关键。

---

## 4. FVQ/VQBridge 的观点

FVQ 论文（2025）发现 SimVQ 的线性层存在局限：

> "A linear projector alone is **fragile**: it makes the network **highly sensitive to the learning rate** and insufficiently capable when scaling to larger codebooks."

他们的实验表明：
- 在 262K codebook 下，SimVQ 的线性层无法维持 100% utilization
- 需要更强的 projector（ViT-based VQBridge）

**对 MeshLex 的启示**：
- K=4096 不算大，SimVQ 线性层理论上应该足够
- 但如果 linear 层的初始化有问题（如上述冲突），可能会加剧不稳定性
- 如果修复初始化后仍有问题，可考虑 FVQ 的方案

---

## 5. 修复方案

### 方案 A（推荐）：去掉分阶段训练和 K-means，与 SimVQ 官方对齐

```python
# 不需要 encoder-only 阶段
# 不需要 K-means 初始化
# 从 epoch 0 开始端到端训练全部 loss

# 初始化
codebook.weight = normal(0, dim^{-0.5})  # frozen
linear.weight = orthogonal()              # learnable

# 训练
for epoch in range(200):
    loss = recon_loss + lambda_commit * commit_loss + lambda_embed * embed_loss
```

**理由**：SimVQ 论文证明这种方式可以达到 100% utilization，无需复杂的初始化策略。

### 方案 B：保留分阶段但修复初始化同步

如果因为某种原因需要保留 encoder-only 预训练：

```python
# Phase 1: encoder-only (20 epochs)
# Phase 1.5: K-means 初始化
codebook.weight.data = kmeans_centers     # 覆写 C
linear.weight.data = torch.eye(dim)       # ★ 同步重置 W 为单位矩阵

# Phase 2: VQ-VAE (180 epochs)
# VQ loss 线性 warm-up
```

### 方案 C：K-means 初始化在 CW 空间

```python
# Phase 1 后：
z_all = encode_all_patches()
kmeans_centers = sklearn.cluster.KMeans(K).fit(z_all).cluster_centers_

# 不直接覆写 C，而是计算 W 使得 CW ≈ kmeans_centers
# W = C.pinverse() @ kmeans_centers
# 但这在 C frozen 且随机初始化时可能数值不稳定
```

**不推荐**：过于复杂，不如方案 A 直接。

---

## 6. 来源

- SimVQ 论文 Section 4.2.1 Remark 1 + Toy Experiment（Figure 2, 3）
- SimVQ 官方 README："Optimizing both the codebook C and the linear layer W can work as well"
- FVQ 论文 Section 1："a linear projector alone is fragile"
- Reddit r/learnmachinelearning: VQ Codebook Initialization 讨论
- MeshLex `scripts/init_codebook.py` 源码
- MeshLex `src/trainer.py` 分阶段训练逻辑

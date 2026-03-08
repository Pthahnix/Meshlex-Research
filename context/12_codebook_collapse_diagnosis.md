# MeshLex Codebook Collapse 诊断分析

> **背景**：实验 1（5-Category Objaverse）快速验证结果显示 codebook 利用率仅 0.46%（19/4096），触发本分析。

---

## 结论概要

MeshLex 遇到的 codebook collapse（4096 个码只用了 19 个，利用率 0.46%）**不是根本性的 idea 问题，而是可克服的工程/训练问题**。核心假设（mesh 局部拓扑具有有限词汇表）得到了实验数据的间接支持（cross/same CD ratio = 1.07），而 collapse 的根因可追溯到至少 5 个具体的技术缺陷。

---

## 第一部分：核心假设是否成立？

### 支持 idea 成立的证据

**证据 1：Cross/Same CD Ratio = 1.07**

这是最关键的数据点。跨类别（car + lamp，训练时从未见过）的 Chamfer Distance 与同类别测试集仅相差 7%。这意味着：即使在严重 collapse 的情况下（只用了 19 个码），模型对未见类别的重建误差几乎与已见类别一致。这恰恰说明 **mesh 局部拓扑确实在跨类别之间高度共享**——核心假设的方向是对的。

| 指标 | Same-Cat | Cross-Cat |
|------|----------|-----------|
| Mean CD | 326.17 | 348.63 |
| Utilization | 0.46% | 0.51% |
| Used Codes | 19 / 4096 | 21 / 4096 |
| **CD Ratio** | — | **1.07** |

**证据 2：PatchNets 的跨类别泛化实验**

PatchNets（ECCV 2020）在 ShapeNet 上证明了 patch 可以跨类别泛化：在 Cabinet 上训练的 patch 表示在 Airplane 上达到了 F-score 93.9。虽然 PatchNets 使用的是连续 SDF 表示而非离散拓扑结构，但它为「patch 级别的跨类别通用性」提供了强有力的先验支持。

**证据 3：Mesh 局部拓扑的低熵直觉**

从数学角度看，三角网格的局部拓扑受到严格的几何约束（欧拉公式、顶点度分布集中在 5-7）。这天然导致了拓扑模式的有限性。MeshLex 的核心洞察——4096 个原型可覆盖绝大多数局部结构——与这一数学直觉完全一致。

### 如果 idea 有根本问题，预期会看到什么？

如果「离散拓扑词汇表」这一概念本身不成立，预期会看到：
- Cross/Same CD ratio > 3.0（跨类别完全无法泛化）
- 即使充分训练，不同类别的 patch 也需要完全不同的 codebook entry
- Codebook 利用率很高但跨类别性能极差

**实际观察到的情况恰恰相反**：利用率极低但跨类别性能很好。这是典型的「模型没学到东西，但数据本身支持假设」的信号。

---

## 第二部分：Codebook Collapse 的具体技术原因

### 原因 1：训练 epoch 严重不足（最主要原因）

VQ-VAE 阶段只训练了 **20 个 epoch**（Task 5 快速验证）。这对于学习一个有效的离散码本来说远远不够：

- VQGAN-LC 在 ImageNet（百万级图像）上训练 20 个 epoch，但 MeshLex 只有 13,095 个 training patches——数据量差了两个数量级，但 epoch 数相同
- 标准 VQ-VAE 训练通常需要 100-800 个 epoch，取决于数据集大小和复杂度
- 原始计划（`RUN_GUIDE.md`）明确设计了 200 个 epoch 的完整训练，但 Claude 的快速验证只跑了 20 个 epoch

20 个 epoch 意味着码本刚开始从 K-means 初始化状态调整，encoder 和 codebook 的联合优化远未收敛。

### 原因 2：SimVQ 线性层与 K-means 初始化不协调

`src/model.py` 的 SimVQ 实现存在一个微妙的初始化冲突：

1. 模型创建时，`SimVQ.linear` 用正交矩阵初始化，`codebook.weight` 用 `normal(0, 0.02)` 初始化
2. Encoder-only 训练 20 epoch（此时 `linear` 层参与梯度更新，但 codebook 未参与）
3. K-means 初始化**直接覆盖了 `codebook.weight`**，但**没有重新初始化 `linear` 层**

SimVQ 的核心机制是：`z_proj = linear(z)`，然后在 `z_proj` 空间中做最近邻搜索。如果 `linear` 层是在 encoder-only 阶段训练的旧权重（以 random normal codebook 为参照优化），而 codebook 被 K-means 重新初始化到完全不同的分布，两者处于不同的表示空间，最近邻搜索会退化为近乎随机的匹配。

SimVQ 论文（ICCV 2025）的核心贡献正是通过线性变换让所有 codebook embedding 在梯度更新中产生交互，避免「只有少数 code 被更新」的问题。但在当前实现中，这一机制可能被 K-means 初始化打破了。

**修复方案**：K-means 初始化之后，同时重置 `linear` 层为单位矩阵（`nn.init.eye_`），或者完全跳过 K-means，直接从随机初始化开始 VQ 训练。

### 原因 3：Go/No-Go 评估逻辑缺失利用率检查

`src/evaluate.py` 中的 `compute_go_nogo()` 函数**只检查 CD ratio，完全不检查 codebook utilization**：

```python
# 当前逻辑（有缺陷）
if ratio < 1.2:
    decision = "STRONG GO"  # 即使 utilization = 0.5% 也会判定为 STRONG GO
```

正确的逻辑应该是：只有当 ratio < 1.2 **且** utilization > 30% 时才判定为 STRONG GO。否则应标记为「COLLAPSE DETECTED」并中断实验。

```python
# 修复后的逻辑
if ratio < 1.2 and utilization > 0.30:
    decision = "STRONG GO"
elif utilization < 0.10:
    decision = "COLLAPSE - HALT"  # 新增
    next_step = "Codebook collapsed. Debug VQ training before proceeding."
```

### 原因 4：重建损失数值过大，VQ 损失被淹没

评估结果显示 mean CD = 326（same-cat）。这些数值过大，说明 decoder 的重建误差量级远超 VQ 损失：

- 训练时：`total_loss = recon_loss + 0.25 * commit_loss + 1.0 * embed_loss`
- 如果 `recon_loss` 在 300+ 量级，`commit_loss` 和 `embed_loss` 通常在 0.x-1.x 量级
- 优化器会几乎完全聚焦于减小 recon_loss，VQ 损失对梯度的贡献接近于零
- 结果：encoder 学会了在不关心量化的情况下重建 patch，codebook 永远无法收敛

**修复方案**：对 Chamfer Distance 做归一化（除以点数），或显著增大 `lambda_commit` 和 `lambda_embed`（如各提高 10×）。

### 原因 5：Decoder 架构的瓶颈

`PatchDecoder` 使用交叉注意力机制，但 Key/Value 只有**1 个 token**（patch embedding）：

```python
# src/model.py
kv = z.unsqueeze(1)  # (B, 1, D) — 只有一个 KV token
attn_out, _ = self.cross_attn(queries, kv, kv)
```

60 个 vertex query 全都 attend 同一个 128 维向量。这种架构下，decoder 的表达能力严重受限。Encoder-only 阶段会建立 encoder→decoder 的直接映射，引入 VQ 量化只会增加噪声——decoder 没有足够的容量来利用离散码本的结构化信息。

---

## 第三部分：学术界对 Codebook Collapse 的认知

Codebook collapse 是 VQ-VAE 领域一个**极其常见且被广泛研究**的问题，绝非 MeshLex 特有：

| 方法 | 来源 | 核心思路 |
|------|------|----------|
| **SimVQ** | ICCV 2025 | 线性变换重参数化，让所有 code 同时更新 |
| **VQGAN-LC** | NeurIPS 2024 | 用预训练编码器初始化 codebook |
| **Dead code revival** | 实践经验 | 定期将未使用的 code 重置为编码器输出的扰动版本 |
| **EMA 更新** | VQ-VAE 原论文 | 用指数移动平均替代梯度下降更新 codebook |
| **Rotation Trick** | ICLR 2025 | 旋转编码器输出以对齐 codebook，改善梯度流 |
| **DCVQ** | NeurIPS 2025 | 将高维空间分成多个低维子空间分别量化 |
| **Stochastic VQ (SQVAE)** | ICML 2022 | 随机量化替代确定性最近邻 |

NeurIPS 2025 的研究甚至发现：VQ-VAE 天然倾向于将表示压缩到 4-10 个有效维度（dimensional collapse），这是模型的固有特性而非 bug——需要显式的正则化策略来抵抗它。

---

## 第四部分：具体修复建议

### 🔴 P0 修复（必须，在进行任何后续实验之前）

1. **延长 VQ-VAE 训练至 200 epoch**
   - 按原始 `RUN_GUIDE.md` 的完整计划执行，而非快速验证

2. **修复 `src/evaluate.py` 的 Go/No-Go 逻辑**
   - 加入 utilization 阈值：`utilization < 0.10` → `COLLAPSE - HALT`
   - 加入联合判断：ratio < 1.2 **且** utilization > 30% → `STRONG GO`

3. **修复 K-means 初始化后的 SimVQ linear 层**
   - 选项 A：K-means 初始化完成后，将 `linear.weight` 重置为单位矩阵
   - 选项 B：跳过 K-means，直接从头开始 VQ 训练
   - 选项 C：K-means 初始化 **之前** 冻结 `linear` 层，初始化完成后再解冻

### 🟡 P1 修复（强烈建议）

4. **添加 dead code revival**
   ```python
   # 每 10 epoch 在 trainer 中执行：
   # 找到未使用的 code，用当前 encoder 输出的扰动版本替换
   unused_mask = usage_count == 0
   if unused_mask.any():
       random_encodings = all_z[torch.randint(len(all_z), (unused_mask.sum(),))]
       noise = torch.randn_like(random_encodings) * 0.01
       model.codebook.codebook.weight.data[unused_mask] = random_encodings + noise
   ```

5. **归一化 Chamfer Distance 损失**
   - 当前 `losses.py` 输出的 CD 在 0.x 量级（非 1000x），但乘以 1000 后在 evaluate 中显示为 326
   - 检查训练时的实际 loss 量级，确保 VQ loss 相对可见
   - 或使用 `lambda_commit = 2.5, lambda_embed = 10.0` 提高 VQ 损失权重

6. **VQ 损失预热（Warm-up）**
   - 不要硬切换（第 20 epoch 突然启用 VQ），而是线性预热
   ```python
   vq_weight = min(1.0, (epoch - vq_start_epoch) / 10.0)  # 10 epoch 线性预热
   loss = recon_loss + vq_weight * (lambda_commit * commit_loss + lambda_embed * embed_loss)
   ```

### 🟢 P2 修复（可选优化）

7. **增强 decoder 容量**：2 层交叉注意力，或将 KV 扩展为多个 context token

8. **尝试 Rotation Trick（ICLR 2025）**替代 straight-through estimator，改善梯度流

9. **降低 codebook 维度**：从 128 降到 32-64，配合可以保留更多有效 code

---

## 第五部分：诊断总结

| 维度 | 评估 | 证据 |
|------|------|------|
| **核心假设（拓扑词汇表存在性）** | ✅ 可能成立 | CD ratio 1.07 + PatchNets 先例 |
| **训练充分性** | ❌ 严重不足 | 20 epoch vs 计划的 200 epoch |
| **SimVQ 实现正确性** | ⚠️ 存在初始化冲突 | K-means 覆盖 codebook 但未同步 linear 层 |
| **损失函数量级平衡** | ⚠️ 需要调整 | Recon loss 数量级远大于 VQ loss |
| **评估逻辑** | ❌ 有 bug | Go/No-Go 不检查 utilization |
| **Decoder 架构** | ⚠️ 可能偏弱 | 单 token KV 交叉注意力容量有限 |

**最终判断**：MeshLex 的核心 idea 没有被推翻。Codebook collapse 是 VQ-VAE 领域最常见的训练问题之一，有大量成熟的解决方案。当前的失败是「快速验证」阶段的训练不足和代码缺陷导致的，而非方法论层面的根本缺陷。按照原始计划完成 200 epoch 完整训练，并修复上述 P0 技术问题后，有很大概率看到有效的 codebook 利用率。

---

*分析日期：2026-03-08*

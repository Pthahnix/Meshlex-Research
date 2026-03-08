# Material: SimVQ 实现错误 — straight-through 路径与 codebook 冻结

> 对应缺陷 #1：SimVQ 实现中 straight-through 用 `z` 而非 `z_proj`，且 codebook 未冻结

---

## 1. MeshLex 当前实现

```python
# MeshLex src/model.py SimVQCodebook
class SimVQCodebook(nn.Module):
    def __init__(self, K=4096, dim=128):
        self.codebook = nn.Embedding(K, dim)          # ← 可学习
        self.linear = nn.Linear(dim, dim, bias=False)  # ← 可学习
        nn.init.normal_(self.codebook.weight, std=0.02)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, z):
        z_proj = self.linear(z)                        # (1) 线性变换 encoder output
        distances = torch.cdist(z_proj, self.codebook.weight)  # (2) z_proj 空间搜索
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices)             # (3) 从原始 C 取值（非 CW）
        quantized_st = z + (quantized - z).detach()    # (4) straight-through 用 z
        return quantized_st, indices
```

**三个问题叠加**：
- `codebook` 是可学习的（未冻结）
- 距离在 `z_proj` 空间计算，但 quantized 从原始 `codebook.weight`（C 空间）取
- straight-through `z + (quantized - z).detach()` 的梯度路径完全绕过 `self.linear`

---

## 2. SimVQ 官方实现（youngsheen/SimVQ）

源码位置：`taming/modules/vqvae/quantize.py`

```python
class SimVQ(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, ...):
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)

        # ★ 关键：C 被冻结
        for p in self.embedding.parameters():
            p.requires_grad = False

        # ★ W 是唯一可学习的参数
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

    def forward(self, z, ...):
        z_flattened = z.view(-1, self.e_dim)

        # ★ 计算 CW（变换后的 codebook）
        quant_codebook = self.embedding_proj(self.embedding.weight)  # CW

        # ★ z 到 CW 的距离（不是 z_proj 到 C 的距离）
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)

        # ★ quantized 从 CW 取（不是从 C 取）
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        # commitment loss
        commit_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                      torch.mean((z_q - z.detach()) ** 2)

        # ★ straight-through: z + (CW[idx] - z).detach()
        z_q = z + (z_q - z).detach()

        return (z_q, ...), LossBreakdown(..., commit_loss, ...)
```

### 官方实现的三个核心设计选择

1. **C 冻结**：`embedding.requires_grad = False`
2. **距离在 z vs CW 空间**：`z_flattened` 到 `quant_codebook`（= CW）
3. **quantized 取自 CW**：`F.embedding(indices, quant_codebook)`

---

## 3. 逐项对比

| 方面 | SimVQ 官方 | MeshLex | 差异影响 |
|------|-----------|---------|---------|
| C (codebook) 参数 | `requires_grad = False` | `requires_grad = True` | 论文 Remark 1 证明同时学 C+W 会导致 W 被忽略 |
| 距离搜索空间 | z 到 CW | z_proj (=zW) 到 C | 数学上可能等价（见下文分析），但实现语义不同 |
| quantized 取值 | 从 CW 中取 | 从 C 中取 | **严重 bug**：decoder 收到的向量与距离搜索空间不一致 |
| straight-through 梯度 | 流过 z，VQ loss 通过 CW 更新 W | 流过 z，VQ loss 同时更新 C 和 W | 梯度在 C 上走捷径，W 无效 |
| commitment loss | `\|z_q.detach() - z\|² + β\|z_q - z.detach()\|²` | 同 | 一致 |

### 距离等价性分析

SimVQ 官方：`dist(z, CW) = \|z - CW_i\|²`
MeshLex：`dist(zW, C) = \|zW - C_i\|²`

如果 W 是正交矩阵，`\|zW - C_i\|² = \|W^T(zW - C_i)\|² = \|z - W^T C_i\|²` ≠ `\|z - CW_i\|²`

**所以两种距离计算方式在数学上不等价。** MeshLex 的方式在 z 的变换空间中搜索，官方在 C 的变换空间中搜索。更关键的是，MeshLex 搜索完后从原始 C 取值，而不是从搜索空间 (CW or C) 中取值。

---

## 4. SimVQ 论文的理论依据

### 4.1 为什么 C 必须冻结

SimVQ 论文 Section 4.2.1, **Remark 1**：

> "The simultaneous optimization of the latent basis W and the coefficient matrix C **may lead to the collapse**."

论文提供了 toy experiment 证明：
- 当 C 和 W 同时可学习时，只有被选中的 code 移动，其余不动
- C 直接通过 commitment loss 更新，优化走捷径
- W 的 norm 迅速缩小，等效退化为单位矩阵 → 回到 vanilla VQ
- 论文 Figure 3 展示了 `||W||` 曲线随训练快速下降

**理论解释**：C 的更新公式为 `C^(t+1) = C^(t) - η E[δ_k^T δ_k C^(t)] + η E[δ_k^T z_e]`，只有被选中的行更新。而 W 的更新公式为 `W^(t+1) = (I - η E[C^T δ_k^T δ_k C]) W^(t) + η E[C^T δ_k^T z_e]`，因为 `E[q_k^T q_k]` 近似 I（codes 从高斯初始化），所有 W 的行都会被更新。

**结论**：只有冻结 C，让 W 成为唯一的可学习参数，才能确保整个 codebook 空间被联合优化。

### 4.2 SimVQ 的收敛保证

论文 Eq. (17-18) 证明当 C frozen 时：
- `lim W^(t) = E[q_k^T z_e]`
- `lim q_k W = E[q_k q_k^T e] = E[e]`（当 q ~ N(0,1) 时）

即每个 code 最终会收敛到其最近 encoder output 的期望位置。这个收敛保证**只在 C frozen 时成立**。

### 4.3 官方 README 的注释

> "**Note:** Optimizing both the codebook C and the linear layer W can work as well."

这个注释看似与 Remark 1 矛盾。但论文 Table 4 的 ablation study 表明：
- C frozen + W learnable: **最佳**（100% utilization）
- C learnable + W learnable: **也能工作但不如 frozen**
- C learnable + W frozen (= vanilla VQ): 严重 collapse

README 的注释是说"也能工作"，不是说效果一样好。**对于 MeshLex 的小数据集场景，C frozen 是更安全的选择。**

---

## 5. lucidrains/vector-quantize-pytorch 的实现

lucidrains 的 SimVQ 实现是最被广泛使用的第三方实现（库总 stars 2600+）。

README 中的推荐用法：
```python
from vector_quantize_pytorch import SimVQ

sim_vq = SimVQ(
    dim = 512,
    codebook_size = 1024,
    rotation_trick = True  # ★ 推荐与 rotation trick 组合
)
```

作者注释：
> "The authors claim this setup leads to less codebook collapse as well as easier convergence. I have found this to perform even better when paired with **rotation trick** from Fifty et al., and expanding the linear projection to a **small one layer MLP**."

### Issue #177: SimVQ + rotation trick 导致 commit loss 过大

一位用户报告开启 rotation trick 后 commit loss 异常大。这是因为 rotation trick 改变了梯度传播方式，需要调整 `commitment_weight`。解决方案是降低 commitment weight 或使用 EMA 更新。

---

## 6. 修复方案

### 方案 A：与 SimVQ 官方严格对齐（推荐）

```python
class SimVQCodebook(nn.Module):
    def __init__(self, K=4096, dim=128):
        super().__init__()
        self.K = K
        self.dim = dim
        self.codebook = nn.Embedding(K, dim)
        self.linear = nn.Linear(dim, dim, bias=False)

        # ★ 按官方初始化
        nn.init.normal_(self.codebook.weight, mean=0, std=dim**-0.5)
        nn.init.orthogonal_(self.linear.weight)

        # ★ 冻结 codebook
        self.codebook.weight.requires_grad = False

    def forward(self, z):
        # ★ 计算 CW
        quant_codebook = self.linear(self.codebook.weight)  # (K, dim)

        # ★ z 到 CW 的距离
        distances = torch.cdist(z.unsqueeze(0), quant_codebook.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)

        # ★ 从 CW 取值
        quantized = quant_codebook[indices]

        # ★ straight-through
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices

    def compute_loss(self, z, quantized_st, indices):
        quant_codebook = self.linear(self.codebook.weight)
        quantized = quant_codebook[indices]
        commit_loss = torch.mean((z - quantized.detach()) ** 2)
        embed_loss = torch.mean((z.detach() - quantized) ** 2)
        return commit_loss, embed_loss
```

### 方案 B：使用 lucidrains 库

```python
from vector_quantize_pytorch import SimVQ

# 直接替换 MeshLex 的 SimVQCodebook
self.codebook = SimVQ(
    dim=128,
    codebook_size=4096,
    rotation_trick=True,  # 建议开启
)
```

需要适配接口，但避免了自行实现的风险。

### 方案 C：保留可学习 C 但修复取值 bug（最小改动，不推荐）

```python
def forward(self, z):
    z_proj = self.linear(z)
    distances = torch.cdist(z_proj, self.codebook.weight)
    indices = distances.argmin(dim=-1)
    # ★ 修复：从 z_proj 空间取值（保持一致性）
    quantized = self.codebook(indices)
    # ★ 或者在 z 空间计算距离：
    # quant_codebook = self.linear(self.codebook.weight)
    # distances = torch.cdist(z, quant_codebook)
    # quantized = quant_codebook[indices]
    quantized_st = z + (quantized - z).detach()
    return quantized_st, indices
```

**不推荐**：即使修复了取值不一致，C+W 同时学习仍可能导致 W 被忽略（Remark 1）。

---

## 7. 来源

- SimVQ 论文全文：[arXiv:2411.02038](https://arxiv.org/abs/2411.02038)（ICCV 2025）
- SimVQ 官方代码：https://github.com/youngsheen/SimVQ/blob/main/taming/modules/vqvae/quantize.py
- lucidrains 实现：https://github.com/lucidrains/vector-quantize-pytorch
- lucidrains Issue #177：SimVQ + rotation trick commit loss 问题
- FVQ 论文：[arXiv:2509.10140](https://arxiv.org/abs/2509.10140)（SimVQ 线性层在大 codebook 下的局限性）

# Material: 重建损失量级远大于 VQ 损失

> 对应缺陷 #4：recon_loss 与 VQ loss 的量级不匹配，VQ 梯度信号被淹没

---

## 1. MeshLex 当前状况

### 训练数据（task8_10 smoke test）

| Epoch | total_loss | recon_loss | VQ portion | VQ/recon 比值 |
|-------|-----------|------------|-----------|--------------|
| 2 | 1.0659 | 0.3353 | 0.7306 | 2.18x |
| 3 | 1.0298 | 0.3256 | 0.7042 | 2.16x |
| 4 | 1.0018 | 0.3191 | 0.6827 | 2.14x |

**表面观察**：VQ loss（0.73）反而大于 recon_loss（0.34），比值约 2:1。看起来不是"recon 淹没 VQ"的问题。

**但关键问题不在绝对量级，而在梯度有效性**：

1. 如果 SimVQ 实现有 bug（缺陷 #1），VQ loss 的梯度根本无法有效更新 codebook
2. VQ loss 中 commit_loss 和 embed_loss 的梯度方向可能相互抵消
3. 即使 VQ loss 数值大，如果梯度被 detach() 阻断（straight-through 路径问题），effective gradient 可能远小于数值暗示

**更深层问题**：MeshLex 的 recon_loss 使用 Chamfer Distance，其量级取决于点云归一化方式。如果点云未归一化到 [0,1]，CD 值可能偏大或偏小，影响 loss 比例。

---

## 2. VQ-VAE 社区对 Loss 平衡的核心认识

### 2.1 原始 VQ-VAE 论文的设定

Van Den Oord et al. (2017) 原始公式：

```
L = L_recon + ||sg[z_e] - e||² + β * ||z_e - sg[e]||²
```

- Term 1：重建损失（训练 decoder，通过 STE 传给 encoder）
- Term 2：Codebook loss（移动 codebook 向量靠近 encoder output）
- Term 3：Commitment loss（保持 encoder output 靠近 codebook，权重 β）

原论文 β = 0.25，并声称 "results are not very sensitive to the choice of beta, as the results are similar for all values between 0.1 and 2.0."

### 2.2 实践中 β 的选择

| 实现/论文 | β / commitment_weight | 备注 |
|----------|----------------------|------|
| 原始 VQ-VAE (2017) | 0.25 | 声称 0.1-2.0 都行 |
| lucidrains/vector-quantize-pytorch | **1.0** | 默认值，显著高于原论文 |
| Taming Transformers (VQ-GAN) | 0.25 | 但有 bug：beta 乘错了 term |
| Straightening Out the STE (2023) | alpha=5, β=0.9-0.995 | 大幅高于原论文 |
| LatentQuantize (lucidrains) | 0.1 | 配合不同的量化方式 |

**社区共识**：原始 β=0.25 往往太低，现代实现倾向使用 1.0 或更高。

---

## 3. Loss 量级不匹配的常见原因

### 3.1 Reduction 方式（sum vs mean）

Reddit r/MLQuestions 上的讨论：

> "If you use sum instead of mean [reduction], I wonder if it doesn't lead to much higher numbers relative to the other losses, which makes the optimization essentially ignore them in favor of better reconstruction."

**关键**：recon_loss 如果用 sum reduction（对所有点求和），其量级会远大于 VQ loss（只在 latent space 计算）。必须确保两者使用一致的 reduction 方式。

### 3.2 维度差异

- Recon loss 在高维空间计算（60 vertices × 3 coordinates = 180 维）
- VQ loss 在 latent space 计算（128 维 embedding）
- 即使用 mean reduction，两个空间的量级天然不同

### 3.3 Commitment loss 增长问题

lucidrains GitHub Issue #27：

> "I have been researching to train some VQ-VAE to generate faces from FFHQ 128x128 and I always have the same problem if I use the commitment loss (0.25) and the gamma (0.99) like in the original paper, the commitment loss seems to grow infinitely."

Issue #69 报告 commit_loss 在 epoch 115 突然暴涨。这说明 commitment loss 的动态是不稳定的，需要监控和控制。

### 3.4 去掉 commitment loss 会怎样

Cross Validated 上的实验确认：

> "I tried to remove the commitment loss and train without it and my loss blew up and the training diverged."

解释：没有 commitment loss 时，encoder 可以将 output 推向 ±∞（因为 recon loss 本身不惩罚 latent 的 magnitude）。

---

## 4. Loss 平衡策略

### 4.1 策略 A：手动调 β（最简单）

```python
# 推荐从 1.0 开始，观察 codebook utilization
lambda_commit = 1.0  # 原论文 0.25，但现代实践偏高
lambda_embed = 1.0   # 通常与 commit 相同或稍低
```

**MeshLex 当前**：lambda_commit=0.25, lambda_embed=1.0。建议实验范围：
- lambda_commit: [0.5, 1.0, 2.0, 5.0]
- lambda_embed: [1.0, 2.0]

### 4.2 策略 B：EMA Codebook 更新（消除 codebook loss）

最流行的实践方案。用 EMA 而非梯度更新 codebook：

```python
VectorQuantize(
    dim=256,
    codebook_size=512,
    decay=0.8,       # EMA decay — 越低越快更新
    commitment_weight=1.0
)
```

EMA 完全消除了 codebook loss term，只剩 recon + commit。VQ-VAE-2 和大多数现代实现使用此方案。

**但注意**：SimVQ 中 C 是冻结的，不需要 EMA 更新 C。EMA 的思路可以应用于 W 的初始化或学习率调度。

### 4.3 策略 C：VQ-GAN 自适应梯度权重

VQ-GAN / Taming Transformers 引入了基于梯度的自适应权重：

```python
lambda_GAN = grad(L_perceptual) / (grad(L_GAN) + delta)
```

确保感知损失和对抗损失的梯度贡献大致相等。

Reddit 讨论确认：
> "It kind of makes sense to me in that they want half the update to come from the discriminator and half to come from the perceptual loss."

**应用于 MeshLex**：
```python
# 动态平衡 recon 和 VQ 的梯度贡献
grad_recon = compute_grad_norm(recon_loss, last_decoder_layer)
grad_vq = compute_grad_norm(vq_loss, last_decoder_layer)
lambda_vq = grad_recon / (grad_vq + 1e-6)
loss = recon_loss + lambda_vq * vq_loss
```

### 4.4 策略 D：L2 归一化（消除尺度问题的根源）

Straightening Out the STE (Huh et al., 2023, arXiv:2305.08842)：

> "We find that a primary cause of training instability is the discrepancy between the model embedding and the code-vector distribution."

方案：
1. 对 encoder output 和 codebook embedding 做 L2 归一化
2. 在归一化空间中计算距离和 loss
3. 这使 commitment loss 的量级可预测（bounded by [0, 4]）

```python
z_norm = F.normalize(z, dim=-1)
codebook_norm = F.normalize(codebook.weight, dim=-1)
distances = torch.cdist(z_norm, codebook_norm)
```

### 4.5 策略 E：Uncertainty Weighting（自动学习权重）

Kendall et al. (2018, arXiv:1705.07115) — 多任务学习中的同方差不确定性加权：

```python
# 自动学习每个 loss 的权重
log_var_recon = nn.Parameter(torch.zeros(1))
log_var_vq = nn.Parameter(torch.zeros(1))

loss = (1 / (2 * log_var_recon.exp())) * recon_loss + log_var_recon / 2 + \
       (1 / (2 * log_var_vq.exp())) * vq_loss + log_var_vq / 2
```

log_var 项防止权重趋向零。但有报告称此方法容易过拟合。

### 4.6 策略 F：GradNorm（梯度归一化）

Chen et al. (2018, arXiv:1711.02257)：

> "A gradient normalization (GradNorm) algorithm that automatically balances training in deep multitask models by dynamically tuning gradient magnitudes."

动态调整梯度大小使不同任务的训练速率平衡。单个超参数 alpha 控制平衡强度。

### 4.7 策略 G：FSQ — 完全消除 VQ Loss

Finite Scalar Quantization (Mentzer et al., 2023, arXiv:2309.15505, 447 citations)：

> "FSQ does not suffer from codebook collapse and does not need commitment losses, codebook reseeding, code splitting, entropy penalties."

用标量量化替代向量量化，loss 只有 reconstruction。从根本上消除 loss 平衡问题。

**但对 MeshLex 的适用性有限**：MeshLex 需要离散 token 用于后续生成，FSQ 的 codebook 结构与 MeshLex 的设计不完全兼容。

---

## 5. VP-VAE 的批评

VP-VAE (2026, arXiv:2602.17133) 明确指出 SimVQ 在某些场景下的问题：

> "On LibriSpeech, SimVQ exhibits near-collapse behavior."

VP-VAE 的方案更激进：训练时完全不用 codebook，用 Metropolis-Hastings 采样生成的扰动模拟量化误差。这从根本上避免了 loss 平衡问题，但需要大幅重写训练流程。

---

## 6. 对 MeshLex 的具体建议

### 优先级排序

| 优先级 | 策略 | 改动量 | 预期效果 |
|--------|------|--------|---------|
| **P0** | 先修复 SimVQ bug（缺陷 #1） | 中 | 使 VQ loss 梯度真正有效 |
| **P1** | 调大 lambda_commit 到 1.0 | 小 | 增强 commitment 约束 |
| **P1** | 确认 CD loss 使用 mean reduction | 小 | 排除 reduction 导致的量级偏差 |
| **P2** | L2 归一化 encoder output + codebook | 中 | 消除尺度不匹配的根源 |
| **P2** | 自适应梯度权重（VQ-GAN 风格） | 中 | 动态平衡两个 loss 的梯度贡献 |
| **P3** | Uncertainty Weighting 或 GradNorm | 大 | 自动学习最优权重 |

### 关键判断

**如果修复 SimVQ 实现后 codebook utilization 显著提升**：
- Loss 量级本身可能不是核心问题
- 只需微调 lambda_commit 即可

**如果修复后 utilization 仍低**：
- 考虑 L2 归一化方案
- 或使用自适应梯度权重
- 最后手段：大幅增大 lambda_commit 到 5.0+

---

## 7. 来源

### 核心论文
- Van Den Oord et al. 2017: [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) — VQ-VAE 原始论文，β=0.25
- Huh et al. 2023: [arXiv:2305.08842](https://arxiv.org/abs/2305.08842) — Straightening Out the STE，L2 归一化 + affine reparameterization
- Esser et al. 2021: Taming Transformers (VQ-GAN) — 自适应梯度权重
- Mentzer et al. 2023: [arXiv:2309.15505](https://arxiv.org/abs/2309.15505) — FSQ，消除 VQ loss
- Kendall et al. 2018: [arXiv:1705.07115](https://arxiv.org/abs/1705.07115) — Uncertainty Weighting
- Chen et al. 2018: [arXiv:1711.02257](https://arxiv.org/abs/1711.02257) — GradNorm

### 社区讨论
- lucidrains/vector-quantize-pytorch Issue #27: commitment loss 无限增长
- lucidrains/vector-quantize-pytorch Issue #69: commit_loss 突然暴涨
- Reddit r/MLQuestions: sum vs mean reduction 导致量级偏差
- Cross Validated: 去掉 commitment loss 导致训练发散
- Reddit r/MachineLearning: VQ-GAN 自适应权重讨论
- taming-transformers Issue #57: beta 乘错 term 的 bug

### MeshLex 数据
- task8_10 validation summary.json: VQ loss 量级数据
- exp1_eval.json: CD 值参考

# MeshLex Codebook Collapse 修复设计

> **日期**: 2026-03-08
> **状态**: 待审批
> **目标**: 修复 codebook collapse（0.46% → >30% utilization），验证核心假设

---

## 一、问题诊断总结

5 个技术缺陷的根因是 **SimVQ 实现与官方论文存在 3 处根本性偏差**：

1. C（codebook）未冻结 → W 被优化走捷径忽略（SimVQ 论文 Remark 1 证明）
2. 距离在 z_proj 空间计算，但 quantized 从原始 C 取值 → 空间不一致
3. straight-through 路径完全绕过 linear 层 → SimVQ 机制失效

由此引发的连锁问题：
- K-means 初始化与 linear 层冲突（缺陷 #2）— 正确实现 SimVQ 后不需要 K-means
- VQ loss 硬切换（缺陷 #3）— 正确实现 SimVQ 后不需要分阶段训练
- Loss 量级失衡（缺陷 #4）— SimVQ 梯度路径修复后可能自行缓解
- Decoder 单 token KV（缺陷 #5）— 独立问题，视 A 阶段效果再决定

## 二、整体策略

**两阶段递进**：

- **A 阶段**：修复 SimVQ 核心 + 简化训练流程 + 跑完实验 1（5-Category 200 epoch）
- **B 阶段**：视 A 结果，如需增强则加 rotation trick + multi-token KV decoder + 重跑实验 1

每阶段都是完整实验（数据准备 → 训练 → 评估 → Go/No-Go）。

LVIS-Wide（实验 2）在确认 collapse 解决后再执行。

## 三、A 阶段设计：SimVQ 修复 + 简化训练

### 3.1 重写 `SimVQCodebook`（src/model.py）

与 SimVQ 官方实现严格对齐：

```python
class SimVQCodebook(nn.Module):
    def __init__(self, K=4096, dim=128):
        super().__init__()
        self.K = K
        self.dim = dim
        self.codebook = nn.Embedding(K, dim)
        self.linear = nn.Linear(dim, dim, bias=False)

        # 官方初始化方式
        nn.init.normal_(self.codebook.weight, mean=0, std=dim ** -0.5)
        nn.init.orthogonal_(self.linear.weight)

        # ★ 核心修复 1：冻结 C
        self.codebook.weight.requires_grad = False

    def forward(self, z):
        # ★ 核心修复 2：计算 CW（变换后的 codebook）
        quant_codebook = self.linear(self.codebook.weight)  # (K, dim)

        # ★ 核心修复 3：z 到 CW 的距离（不是 z_proj 到 C）
        distances = torch.cdist(
            z.unsqueeze(0), quant_codebook.unsqueeze(0)
        ).squeeze(0)  # (B, K)

        indices = distances.argmin(dim=-1)  # (B,)

        # ★ 核心修复 4：从 CW 取值（不是从 C 取）
        quantized = quant_codebook[indices]  # (B, dim)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices

    def compute_loss(self, z, quantized_st, indices):
        quant_codebook = self.linear(self.codebook.weight)
        quantized = quant_codebook[indices]
        commit_loss = torch.mean((z - quantized.detach()) ** 2)
        embed_loss = torch.mean((z.detach() - quantized) ** 2)
        return commit_loss, embed_loss

    @torch.no_grad()
    def get_utilization(self, indices):
        return indices.unique().numel() / self.K
```

**4 处修复要点**：
1. `self.codebook.weight.requires_grad = False` — 冻结 C
2. `quant_codebook = self.linear(self.codebook.weight)` — 计算 CW
3. 距离算 z 到 CW — 搜索空间一致
4. `quantized = quant_codebook[indices]` — 从 CW 取值

### 3.2 简化训练流程（src/trainer.py）

**删除分阶段训练逻辑**，从 epoch 0 开始全 loss 端到端训练：

```python
# 旧逻辑（删除）:
# if epoch < self.vq_start_epoch:
#     loss = result["recon_loss"]
# else:
#     loss = result["total_loss"]

# 新逻辑:
loss = result["total_loss"]  # 从 epoch 0 开始全 loss
```

**添加 LR warmup**（SimVQ 官方推荐防止 NaN）：

```python
# 前 5 epoch 线性 warmup
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
combined_scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], milestones=[5])
```

**添加 dead code revival**：

```python
# 每 10 epoch 检查一次，重置 dead codes
# 注意：C 是冻结的，dead code revival 作用于 C 的 embedding
# 但由于 W 的变换，实际效果是在 CW 空间中替换 dead entries
if (epoch + 1) % 10 == 0:
    usage_count = torch.zeros(model.codebook.K)
    for idx in all_indices:
        usage_count.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    dead_mask = usage_count == 0
    if dead_mask.any():
        n_dead = dead_mask.sum().item()
        # 用当前 batch 的 encoder output 重置 dead code 位置
        # 由于 C frozen，需要调整 W 使 CW[dead] ≈ random z
        # 简单做法：直接修改 C[dead]（临时解冻）
        with torch.no_grad():
            random_z = all_z[torch.randint(len(all_z), (n_dead,))]
            noise = torch.randn_like(random_z) * 0.01
            model.codebook.codebook.weight.data[dead_mask] = random_z + noise
        print(f"  Revived {n_dead} dead codes")
```

### 3.3 删除 K-means 初始化步骤

**删除 `scripts/init_codebook.py` 的使用**。SimVQ 论文证明：随机初始化 C + frozen C + learnable W 就能达到 100% utilization，无需 K-means。

训练脚本调用简化为：

```bash
# 旧流程（3 步）：
# 1. encoder-only 20 epoch
# 2. K-means init
# 3. VQ-VAE 20 epoch

# 新流程（1 步）：
python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --checkpoint_dir data/checkpoints/5cat_v2
```

### 3.4 调整 Loss 权重

```python
# 旧值：
lambda_commit = 0.25  # 原论文默认，偏保守
lambda_embed = 1.0

# 新值：
lambda_commit = 1.0   # 现代实践共识，lucidrains 默认值
lambda_embed = 1.0    # 保持一致
```

### 3.5 修复 `src/evaluate.py` Go/No-Go 逻辑

```python
def compute_go_nogo(same_cat_cd, cross_cat_cd, utilization):
    ratio = cross_cat_cd / same_cat_cd

    # ★ 新增 utilization 门控
    if utilization < 0.10:
        return {
            "ratio": ratio,
            "decision": "COLLAPSE - HALT",
            "next_step": "Codebook collapsed (<10% util). Debug VQ training.",
            "utilization": utilization,
        }

    if ratio < 1.2 and utilization > 0.30:
        decision = "STRONG GO"
        next_step = "Proceed to full MeshLex experiment design"
    elif ratio < 1.2 and utilization > 0.10:
        decision = "CONDITIONAL GO"
        next_step = "Utilization suboptimal. Consider decoder enhancement (Phase B)."
    elif ratio < 2.0:
        decision = "WEAK GO"
        next_step = "Adjust story to 'transferable vocabulary', continue"
    elif ratio < 3.0:
        decision = "HOLD"
        next_step = "Analyze failure, consider category-adaptive codebook"
    else:
        decision = "NO-GO"
        next_step = "Core hypothesis falsified. Pivot direction."

    return {
        "ratio": ratio,
        "decision": decision,
        "next_step": next_step,
        "same_cat_cd": same_cat_cd,
        "cross_cat_cd": cross_cat_cd,
        "utilization": utilization,
    }
```

### 3.6 更新 scripts/train.py

- 移除 `--vq_start_epoch` 参数（不再需要）
- 添加 `--lambda_commit` 和 `--lambda_embed` 参数
- 添加 `--warmup_epochs` 参数（默认 5）
- 添加 `--dead_code_interval` 参数（默认 10）

### 3.7 更新 scripts/evaluate.py

- 传递 utilization 到 `compute_go_nogo()`

### 3.8 A 阶段完整执行流程

```
1. 代码修改（上述 3.1-3.7）
2. 运行单元测试确认无破坏
3. 下载数据（如已清理）：python scripts/download_objaverse.py --mode 5cat
4. 预处理：python scripts/run_preprocessing.py --input_manifest ... --experiment_name 5cat
5. 训练 200 epoch：python scripts/train.py --epochs 200 --batch_size 256 --lr 1e-4
6. 评估：python scripts/evaluate.py
7. 可视化：python scripts/visualize.py
8. Go/No-Go 判定
```

### 3.9 A 阶段成功标准

| 指标 | 最低要求 | 理想目标 |
|------|---------|---------|
| Codebook utilization | > 10% | > 30% |
| CD ratio (cross/same) | < 2.0 | < 1.2 |
| Go/No-Go decision | 非 COLLAPSE | STRONG GO 或 CONDITIONAL GO |

### 3.10 A → B 决策矩阵

| A 结果 | 下一步 |
|--------|--------|
| utilization > 30% + ratio < 1.2 | **STRONG GO**，直接跑实验 2（LVIS-Wide） |
| utilization > 30% + CD 质量差 | 进入 B 阶段：增强 decoder |
| utilization 10-30% | 进入 B 阶段：加 rotation trick + dead code revival 增强 |
| utilization < 10% | 重新诊断，可能需要更根本的改动 |

## 四、B 阶段设计：增强优化（视 A 结果决定）

### 4.1 添加 Rotation Trick

替代 straight-through estimator，改善梯度流：

```python
def rotation_trick(z, quantized):
    """ICLR 2025 Fifty et al. — 用旋转变换替代 STE"""
    e_hat = F.normalize(z, dim=-1)
    q_hat = F.normalize(quantized, dim=-1)
    r = F.normalize(e_hat + q_hat, dim=-1)
    lam = quantized.norm(dim=-1, keepdim=True) / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return lam * (z - 2 * (z * r).sum(dim=-1, keepdim=True) * r
                  + 2 * (z * e_hat).sum(dim=-1, keepdim=True) * q_hat)
```

在 `SimVQCodebook.forward()` 中，将 `z + (quantized - z).detach()` 替换为 `rotation_trick(z, quantized)`。

### 4.2 Multi-Token KV Decoder

```python
class PatchDecoder(nn.Module):
    def __init__(self, embed_dim=128, max_vertices=128, num_kv_tokens=4):
        # ...
        self.kv_proj = nn.Linear(embed_dim, num_kv_tokens * embed_dim)
        self.num_kv_tokens = num_kv_tokens

    def forward(self, z, n_vertices):
        B = z.shape[0]
        kv = self.kv_proj(z).reshape(B, self.num_kv_tokens, -1)  # (B, 4, D)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        # ... 其余不变
```

将 1 个 KV token 扩展为 4 个，使 cross-attention 不再退化。

### 4.3 Loss 权重微调

如果 A 阶段显示 VQ loss 仍被淹没：
- `lambda_commit`: 1.0 → 2.0
- `lambda_embed`: 1.0 → 2.0

### 4.4 B 阶段成功标准

| 指标 | 相对 A 的改进目标 |
|------|-----------------|
| Codebook utilization | 比 A 提升 ≥ 50% |
| Mean CD | 比 A 降低 ≥ 20% |
| CD ratio | 维持 < 1.5 |

## 五、需要修改的文件清单

### A 阶段

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/model.py` | **重写** SimVQCodebook | 冻结 C，CW 距离，CW 取值 |
| `src/model.py` | **修改** MeshLexVQVAE | lambda_commit 默认值 0.25 → 1.0 |
| `src/trainer.py` | **修改** Trainer | 删除分阶段逻辑，加 LR warmup + dead code revival |
| `src/evaluate.py` | **修改** compute_go_nogo | 加 utilization 参数和门控 |
| `scripts/train.py` | **修改** | 移除 vq_start_epoch，加 warmup/dead_code 参数 |
| `scripts/evaluate.py` | **修改** | 传 utilization 到 go_nogo |
| `tests/test_model.py` | **更新** | 适配 SimVQ 新接口 |

### B 阶段（额外）

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/model.py` | 添加 `rotation_trick()` | Rotation trick 函数 |
| `src/model.py` | 修改 SimVQCodebook.forward | 用 rotation_trick 替代 STE |
| `src/model.py` | 修改 PatchDecoder | 加 kv_proj, num_kv_tokens |

## 六、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| data/ 已清理，需重新下载 | 磁盘 79GB 可用，充足 |
| 200 epoch 训练时间长（~6h） | RTX 4090 应可胜任，支持 --resume |
| Dead code revival 修改 frozen C | 临时解冻写入后重新冻结 |
| 单元测试可能因接口变化失败 | A 阶段代码修改后立即更新测试 |

---

*设计文档结束。待用户审批后进入实现规划阶段。*

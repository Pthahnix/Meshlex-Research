# 训练进度报告 — 5-Category Exp1 (A-stage Collapse Fix)

> **时间**: 2026-03-08 13:08
> **状态**: 训练进行中 (epoch 39/200)
> **预计完成**: ~2h remaining (~45s/epoch)

## 训练配置

| 参数 | 值 |
|------|------|
| Dataset | 5-Category Objaverse (chair/table/airplane train, car/lamp cross-cat test) |
| Training patches | 12,854 |
| Validation patches | 3,354 |
| Codebook size K | 4,096 |
| Embed dim | 128 |
| Batch size | 256 |
| LR | 1e-4 (warmup 5 epochs + cosine) |
| Lambda commit/embed | 1.0 / 1.0 |
| Encoder warmup | 10 epochs (recon only) |
| Dead code interval | 10 epochs |
| Model params | 1,060,995 total, 536,707 trainable |
| GPU | RTX 4090 (24GB) |

## 关键发现：Codebook Collapse 已修复

### 问题
首次训练（无 encoder warmup）时，utilization 在 epoch 3 后跌至 **0.0%**（仅 1/4096 codes 被使用）。
Dead code revival 替换了 4095/4096 codes 但完全无效 — 模型立即回到使用同一个 code。

### 根因分析
- 4096 个随机 CW 向量在 128 维空间中过于稀疏
- Encoder 输出 z 集中在空间的一个小区域，而 CW 散布在完全不同的区域
- 即使是全新模型，也只有 17/4096 codes 被使用（初始化就已经有问题）
- Dead code revival 无效是因为 W 只学习了映射 1 个 code，新 C 经过 W 变换后依然在错误位置

### 修复方案
1. **Encoder warmup (10 epochs)**: 仅用 recon_loss 训练 encoder，让其学习有意义的 z 表征
2. **K-means codebook init**: warmup 结束后，对所有 z 运行 K-means (k=4096)，设置 C = W^T(centroids) 使得 CW ≈ centroids
3. **CW-aligned dead code revival**: 改用 C[dead] = W^T(z_sample) 替代 C[dead] = z_sample

## 训练曲线

### Encoder Warmup Phase (Epoch 0-9)
```
Epoch 000 [warmup] | recon 0.6649 | util  1.2%
Epoch 004 [warmup] | recon 0.3622 | util  1.7%
Epoch 009 [warmup] | recon 0.3183 | util  1.8%
  → K-means init: 12,854 samples → 4,096 clusters
  → Post-init utilization: 15.9%
```

### VQ Training Phase (Epoch 10+)
```
Epoch 010 | loss 0.5084 | recon 0.4811 | commit 0.0137 | util 62.3%  ← VQ starts!
Epoch 015 | loss 0.3132 | recon 0.2938 | commit 0.0097 | util 52.7%
Epoch 019 | loss 0.3059 | recon 0.2853 | commit 0.0103 | util 55.0%
  → Dead code revival: 1845/4096 replaced
Epoch 025 | loss 0.2977 | recon 0.2760 | commit 0.0109 | util 48.8%
Epoch 029 | loss 0.2947 | recon 0.2719 | commit 0.0114 | util 46.9%
  → Dead code revival: 2173/4096 replaced
Epoch 035 | loss 0.2936 | recon 0.2679 | commit 0.0128 | util 42.6%
Epoch 039 | loss 0.2906 | recon 0.2634 | commit 0.0136 | util 41.8%
  → Dead code revival: 2383/4096 replaced
```

## 指标对比

| 指标 | Exp1 v1 (collapsed) | Exp1 v2 (当前 e39) | 目标 |
|------|-------|--------|------|
| Codebook utilization | 0.46% (19/4096) | **41.8%** (1713/4096) | >30% |
| Recon loss | 0.3260 | **0.2634** | 尽可能低 |
| Commit loss | N/A | 0.0136 | >0 |

## 趋势分析

- **Utilization**: 从 epoch 10 的 62.3% 缓慢下降到 epoch 39 的 41.8%，每轮 dead code revival 替换 ~2000 codes，说明约一半 codes 依然不活跃。趋势需要持续观察。
- **Recon loss**: 持续下降 (0.48 → 0.26)，VQ 没有损害重建质量
- **Commit/Embed loss**: 很小 (~0.01)，说明 z 和 CW 已经很好地对齐
- **LR**: 9.23e-5 (cosine annealing 中)

## 下一步

1. 等待 200 epochs 训练完成 (~2h)
2. 运行评估: same-cat CD vs cross-cat CD
3. Go/No-Go 决策
4. 如果 STRONG GO → 进入 LVIS-Wide 实验
5. 如果 utilization 继续下降 → 考虑 Phase B (rotation trick + decoder 增强)

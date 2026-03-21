# PatchDiffusion Design Spec

**Date**: 2026-03-21
**Target**: CCF-A (CVPR / NeurIPS / ICLR)
**Type**: System/Method-Driven

---

## 1. Executive Summary

PatchDiffusion 是首个在 patch token 序列上使用 masked discrete diffusion 的显式 mesh 生成方法。通过结合 MeshLex 的 patch vocabulary（4000-face mesh → ~130 tokens）和 masked discrete diffusion model (MDLM)，我们实现非自回归、并行化的 mesh 生成，同时保持双向上下文建模的全局一致性。

本 spec 提出三种 diffusion 变体的统一框架：
1. **Pure MDLM**: 全序列并行去噪
2. **Block Diffusion**: block 间 AR + block 内并行
3. **Hierarchical RVQ Diffusion**: 利用 RVQ 3 级结构 coarse-to-fine 生成

---

## 2. Problem Statement

### 2.1 核心问题

> 当前所有显式 mesh 生成方法都是 autoregressive 的。AR 方法有三个根本性限制：

1. **序列长度瓶颈**: 4000 faces → 数千 tokens，受限于 O(n^2) attention
2. **单向上下文**: 生成椅子腿时看不到椅背，缺乏全局一致性
3. **串行推理**: token 逐个生成，无法利用 GPU 并行

### 2.2 现有方法的限制

| 方法 | 生成范式 | Token 粒度 | 序列长度 (4K face) | 限制 |
|------|---------|-----------|-------------------|------|
| MeshGPT (CVPR 2024) | AR | per-vertex/face | ~12K | 序列太长 |
| FACE (arXiv 2026.03) | AR | per-face | ~4K | 单向上下文 |
| DeepMesh (ICCV 2025) | AR + DPO | per-face | ~数千 | 0.5B params, 32×H20 |
| MeshMosaic (arXiv 2025.09) | AR per-patch | per-face within patch | ~数千/patch | patch 内仍 per-face |
| TSSR (arXiv 2025.10) | Discrete Diffusion | per-face | ~几万 | 序列仍然很长 |
| **PatchDiffusion (Ours)** | **Masked Discrete Diffusion** | **per-patch** | **~130** | — |

### 2.3 Key Insight

MeshLex 的 patch vocabulary 将 4000-face mesh 压缩到 ~130 tokens。这个超短序列是 masked discrete diffusion 的 sweet spot：
- 序列够短让 bidirectional attention 高效（130^2 vs 12000^2）
- 每个 token 承载 ~20 faces 的拓扑信息，语义密度极高
- Masked diffusion 的并行去噪在短序列上收敛更快

### 2.4 与 TSSR 的核心差异

TSSR (Topology Sculptor, Shape Refiner) 是目前唯一的 discrete diffusion mesh generation 方法，但它与 PatchDiffusion 有本质区别：

| 维度 | TSSR | PatchDiffusion |
|------|------|----------------|
| Token 粒度 | per-face (每 face 多 token) | per-patch (每 patch 1 token) |
| 序列长度 | ~几万 (10K faces) | ~130 (4K faces) |
| 需要 hourglass 架构 | 是（压缩长序列） | 否（序列本身够短） |
| 重建精度 | 逐 face 精确 | patch 级近似 |
| 推理速度 | 中等（长序列多步） | 快（短序列少步） |
| 全局一致性 | 依赖 hourglass 跨层 | 天然全局 attention |

---

## 3. Architecture

### 3.1 Overall Pipeline

```
Input Mesh
    │
    ▼
[Layer 1: Patch Tokenizer — MeshLex (已有)]
    METIS partition → SimVQ/RVQ encode
    Output: patch token sequence S = [(pos, scale, tok_L1, tok_L2, tok_L3)] × N
    │
    ▼
[Layer 2: Masked Discrete Diffusion — 核心创新]
    Forward:  S_clean → gradually mask → S_masked (all [MASK])
    Reverse:  S_masked → iteratively unmask → S_clean
    三种变体: Pure MDLM / Block Diffusion / Hierarchical RVQ
    │
    ▼
[Layer 3: Assembly — MeshLex (已有)]
    Decode patches → StitchingMLP → Complete mesh
```

### 3.2 Patch Token Format

每个 patch 的 token 表示（沿用 MeshLex RVQ 格式）：

```
patch_token = (pos_x, pos_y, pos_z, scale, tok_L1, tok_L2, tok_L3)
```

- `pos_x, pos_y, pos_z`: patch 中心坐标（连续值，量化为 discrete bins）
- `scale`: patch 尺度（连续值，量化）
- `tok_L1, tok_L2, tok_L3`: RVQ 三级 codebook indices

对于 diffusion，我们 mask 的单位是**整个 patch token**（7 个值一起 mask/unmask），而非逐值 mask。这保证了每个 patch 的内部一致性。

### 3.3 Variant A: Pure MDLM

**训练**:

```python
# Forward process: randomly mask patch tokens
mask_ratio = sample_from_schedule(t)  # t ∈ [0, 1]
mask = Bernoulli(mask_ratio, size=N_patches)
z_t = where(mask, [MASK], x_clean)

# Model predicts all token probabilities
logits = model(z_t, t)  # shape: (N_patches, 7, vocab_sizes)

# Loss: cross entropy on masked positions only
loss = CE(logits[mask], x_clean[mask])
```

**推理** (iterative unmasking):

```python
z = all_mask(N_patches)  # 全 [MASK]
for step in range(T, 0, -1):
    logits = model(z, t=step/T)
    probs = softmax(logits)
    confidence = max(probs, dim=-1)

    # 选择 confidence 最高的 tokens 进行 unmask
    n_unmask = N_patches * (1 - step/T)  # 线性 schedule
    top_indices = topk(confidence, n_unmask)
    z[top_indices] = sample(probs[top_indices])

return z  # clean patch tokens
```

**架构**: Bidirectional Transformer (encoder-only)
- 12 layers, hidden_dim=512, 8 heads
- 无 causal mask → 全局双向 attention
- Timestep embedding (sinusoidal) + token embedding
- ~30M params

### 3.4 Variant B: Block Diffusion

**Block 划分**:
- 将 N 个 patches 按 Z-order curve 排序
- 分成 B=10 blocks，每 block ~13 patches
- Block 间顺序: Z-order (空间局部性)

**生成过程**:

```python
for block_idx in range(B):
    # Context: 已生成的 blocks
    context = generated_blocks[:block_idx]

    # Current block: 全 [MASK]
    current = all_mask(block_size)

    # MDLM 去噪 (block 内并行)
    for step in range(T_inner, 0, -1):
        logits = model(context + current, t=step/T_inner)
        current = unmask_step(current, logits, step)

    generated_blocks.append(current)
```

**架构**: Hybrid Transformer
- Block 间: causal attention (AR)
- Block 内: bidirectional attention (MDLM)
- 8 AR layers + 8 MDLM layers
- ~40M params

### 3.5 Variant C: Hierarchical RVQ Diffusion

**利用 RVQ 的 3 级结构**:

```
Stage 1: Diffuse L1 tokens (coarse geometry)
    Input:  [M] × N → Iterative unmask → [L1_1, ..., L1_N]
    Codebook: K=1024, captures major shape structure

Stage 2: Conditioned on L1, diffuse L2 tokens
    Input:  [L1_1,...,L1_N] + [M] × N → Iterative unmask → [L2_1,...,L2_N]
    Codebook: K=1024, captures medium details

Stage 3: Conditioned on L1+L2, diffuse L3 tokens
    Input:  [L1+L2] + [M] × N → Iterative unmask → [L3_1,...,L3_N]
    Codebook: K=1024, captures fine details
```

**每个 stage 使用独立的 MDLM**（参数不共享）:
- 8 layers, hidden_dim=384
- L2 model 接收 L1 tokens 作为 cross-attention conditioning
- L3 model 接收 L1+L2 tokens 作为 conditioning
- 3 × ~12M = ~35M params total

**位置信息**: pos_x, pos_y, pos_z, scale 在 Stage 1 中与 L1 tokens 一起生成（作为 Stage 1 序列的一部分）。Stage 2/3 接收 Stage 1 的完整输出（含位置）作为 conditioning。

---

## 4. Conditional Generation

### 4.1 Class-Conditioned

```python
class_embedding = embed(class_label)  # (1, dim)
# 加入 transformer 的 [CLS] position
input = [class_embedding] + [patch_tokens]
```

### 4.2 Point Cloud-Conditioned

```python
# Encode point cloud via PointNet
pc_features = pointnet(point_cloud)  # (M, dim)
# Cross-attention in transformer
logits = model(patch_tokens, cross_attn=pc_features)
```

### 4.3 Image-Conditioned

```python
# Encode image via frozen CLIP/DINOv2
img_features = clip_encoder(image)  # (P, dim)
# Cross-attention
logits = model(patch_tokens, cross_attn=img_features)
```

**优先级**: Class-conditioned 先做（最简单），point cloud 和 image 作为扩展。

---

## 5. Training Plan

### 5.1 Data Pipeline

1. 使用 Phase D 统一数据集（Objaverse-LVIS + ShapeNet）
2. 加载已训练的 MeshLex RVQ VQ-VAE
3. 将所有 mesh encode 为 patch token sequences
4. 存储为 `(pos, scale, tok_L1, tok_L2, tok_L3)` 格式的 tensor 文件

### 5.2 Training Schedule

| Phase | 内容 | 数据 | GPU 时间 |
|-------|------|------|----------|
| P0 | Encode all meshes to token sequences | 97K meshes | ~5h |
| P1 | Train Pure MDLM | Token sequences | ~20h |
| P2 | Train Block Diffusion | Token sequences | ~30h |
| P3 | Train Hierarchical RVQ Diffusion | Token sequences | ~25h |
| P4 | Eval + Ablation | Generated meshes | ~10h |
| **Total** | | | **~90h GPU** |

### 5.3 Training Hyperparameters

**Common**:
- Optimizer: AdamW, lr=3e-4, weight_decay=0.01
- Warmup: 1000 steps
- Batch size: 256 (short sequences → large batches possible; 需验证 VRAM，RTX 4090 24GB 应足够，RTX A4000 16GB 可能需降至 128)
- Masking schedule: cosine (MDLM default)
- Gradient clipping: 1.0

**Pure MDLM specific**:
- Epochs: 500
- Diffusion steps (inference): 20 (default), sweep {5, 10, 20, 50, 100}

**Block Diffusion specific**:
- Block size: 13 patches
- Inner diffusion steps: 10
- Outer AR steps: 10 blocks

**Hierarchical RVQ specific**:
- Stage 1 epochs: 300
- Stage 2 epochs: 200 (fine-tune from stage 1 weights)
- Stage 3 epochs: 200
- Each stage diffusion steps: 15

---

## 6. Evaluation Plan

### 6.1 Metrics

**Generation quality**:
- Chamfer Distance (CD) — point cloud distance
- F-Score@{0.01, 0.02, 0.05} — surface accuracy
- Normal Consistency (NC)
- FID (point cloud feature space, PointNet)
- Coverage (COV) — diversity
- MMD — quality × diversity

**Inference efficiency**:
- Wall-clock time per mesh (seconds)
- Number of forward passes
- Tokens generated per second
- GPU memory usage

### 6.2 Baselines

| Baseline | 来源 | 备注 |
|----------|------|------|
| PatchGPT (AR) | MeshLex 现有 | 同 token 空间，AR 对比 |
| MeshGPT | 发表数据 | Per-face AR 代表 |
| FACE | 发表数据 | Per-face SOTA |
| TSSR | 发表数据 | Discrete diffusion per-face |
| DeepMesh | 发表数据 | AR + DPO |

### 6.3 Ablation Experiments

| 实验 | 变量 | 目的 |
|------|------|------|
| A1 | Pure MDLM vs AR (PatchGPT) | Non-AR vs AR 在同等 patch token 空间 |
| A2 | Pure vs Block vs Hierarchical | 三种 diffusion 变体的系统比较 |
| A3 | Diffusion steps {5,10,20,50,100} | 推理步数 vs 质量 trade-off |
| A4 | Patch tokens (130) vs face tokens (~4K) | 序列长度对 MDM 质量影响（使用 TSSR 发表数据对比） |
| A5 | Masking schedule: linear / cosine / sqrt | Schedule 对生成质量的影响 |
| A6 | Unconditional vs class-conditioned | 条件信号的影响 |
| A7 | Confidence-based vs random unmasking | Unmasking 策略对比 |

### 6.4 Key Analysis

**"Why short sequences help MDM"**:

在 NLP 中，MDM 在长序列上仍不如 AR（GPT）。但我们假设在 ~130 tokens 的超短序列上，MDM 可能反超 AR，原因：
- 短序列中全局 attention 无需压缩（无 hourglass 必要）
- 双向上下文在空间结构中比因果上下文更自然（mesh 不像文本有"方向"）
- 并行去噪在短序列上采样更均匀

这是一个可验证的分析性贡献。

---

## 7. Paper Structure

```
§1 Introduction
    - Observation: all explicit mesh generation is AR → three fundamental limitations
    - Key insight: patch vocabulary enables ultra-short token sequences for MDM
    - This paper: PatchDiffusion — first non-AR explicit mesh generation
    - Results preview: comparable quality to AR, N× faster inference

§2 Related Work
    §2.1 Autoregressive mesh generation (MeshGPT → FACE → DeepMesh → MeshMosaic)
    §2.2 Discrete diffusion models (D3PM → MDLM → BD3-LM → TSSR)
    §2.3 Mesh tokenization (FreeMesh, MeshAnything, MeshLex)

§3 Preliminaries
    §3.1 MeshLex patch vocabulary and RVQ
    §3.2 Masked discrete diffusion (MDLM framework)

§4 Method: PatchDiffusion
    §4.1 Overall framework
    §4.2 Pure MDLM on patch tokens
    §4.3 Block diffusion variant
    §4.4 Hierarchical RVQ diffusion variant
    §4.5 Conditional generation

§5 Experiments
    §5.1 Setup (datasets, baselines, metrics, implementation)
    §5.2 Main results: generation quality comparison
    §5.3 Inference speed comparison
    §5.4 Ablation: three diffusion variants
    §5.5 Ablation: diffusion steps vs quality
    §5.6 Ablation: masking schedule and unmasking strategy
    §5.7 Analysis: why short sequences are the sweet spot for MDM

§6 Conclusion and Future Work
```

---

## 8. Success Criteria

| 指标 | 标准 | 判定 |
|------|------|------|
| 生成 CD | 与 PatchGPT (AR) 差距 < 20% | GO |
| 推理速度 | 比 PatchGPT 快 ≥ 3× | GO |
| 三变体对比 | 至少一种变体在质量上匹配或超过 AR | STRONG GO |
| Ablation | Diffusion steps 存在明显 quality-speed trade-off | GO (实验价值) |
| 整体 | 至少 3 GO + 无 FAIL | 论文可投 |

**FAIL 条件**:
- 所有变体 CD 比 AR 差 > 50% → 说明 MDM 不适合 patch tokens
- 推理速度无优势 → 失去核心卖点

---

## 9. Risk Mitigation

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| MDM 生成质量显著差于 AR | 中 | 高 | 增加 diffusion steps; 尝试 classifier-free guidance; 调整 schedule |
| 三种变体差异不显著 | 中 | 低 | 聚焦最好的，其余作 ablation |
| TSSR 发正式版且 SOTA | 低 | 中 | 强调 patch-level vs face-level 本质差异，序列短 100× |
| RVQ codebook 质量限制上限 | 低 | 高 | 可用 theory-driven 曲率感知 codebook (与另一个方向协同) |
| 训练时间超预算 | 中 | 中 | 先训 Pure MDLM (最简单)，确认可行后再做其他变体 |

---

## 10. Dependencies

### 10.1 与其他 MeshLex 工作流的关系

```
Phase D (统一数据集) ──────────────────────┐
                                           ▼
Theory-Driven Design ──→ 曲率感知 codebook ──→ PatchDiffusion
                                           ▲
MeshLex v2 (RVQ VQ-VAE + AR) ─────────────┘
```

- **Phase D 数据集**: PatchDiffusion 需要 encode 后的 token sequences，依赖 Phase D 完成
- **RVQ VQ-VAE**: 复用现有训练好的 checkpoint (`data/checkpoints/rvq_lvis/`)
- **Theory-Driven Codebook**: 如果曲率感知 codebook 效果更好，PatchDiffusion 可直接受益
- **AR v2 (PatchGPT)**: 作为 baseline 对比

### 10.2 前置条件

1. Phase D 统一数据集完成 (Objaverse-LVIS + ShapeNet)
2. RVQ VQ-VAE 在全量数据上重训完成（或使用现有 checkpoint）
3. 全量 mesh → patch token sequence 编码完成

---

## Appendix A: Key References

1. **MDLM**: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
2. **BD3-LM**: Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models" (ICLR 2025)
3. **TSSR**: Song et al., "Topology Sculptor, Shape Refiner: Discrete Diffusion Model for High-Fidelity 3D Mesh Generation" (arXiv 2025.10)
4. **MeshLex**: Our prior work on patch vocabulary
5. **MeshGPT**: Siddiqui et al. (CVPR 2024)
6. **FACE**: (arXiv 2026.03)
7. **MeshMosaic**: Xu et al. (arXiv 2025.09)
8. **DeepMesh**: Zhao et al. (ICCV 2025)
9. **D3PM**: Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces" (NeurIPS 2021)

---

## Appendix B: Naming Rationale

**PatchDiffusion** 名称选择理由：
- "Patch": 直接点明 token 粒度（区别于 per-face 方法）
- "Diffusion": 明确生成范式（区别于 AR）
- 简洁、易记、无已知冲突
- 备选: MeshPatchDiff, PatchMDM, DiffPatch

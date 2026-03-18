# MeshLex v2: Compositional Mesh Generation via a Universal Patch Vocabulary

> **Date**: 2026-03-18
> **Status**: Design Spec
> **Version**: v0.2.0 (exploratory — informing final novel approach)
> **Base**: MeshLex v0.1.0 (4/4 STRONG GO, CD ratio 1.019x, util 95.3%)
> **Target Venue**: CCF-A (CVPR / NeurIPS / ICCV)

---

## 1. Motivation

MeshLex v1 validated that mesh local topology forms a **universal vocabulary** — 4000-face meshes can be represented by ~130 patch tokens with near-perfect cross-category generalization (ratio 1.019x). But v1 is reconstruction-only.

**v2 asks**: Can we use this vocabulary to **generate** new meshes?

**v0.2.0 定位**: 这不是最终投稿方案。v0.2.0 是一组探索性实验，通过模块化组合已有技术（RVQ、BPE、AR Transformer），用 2×2 消融矩阵找出哪些方向有潜力。实验结果将指导后续 v1.0 的自主创新方案设计。

---

## 2. Pipeline Overview

```
Mesh → ①切块 → ②编码 → ③生成 → ④缝合 → New Mesh
```

| Module | What | Variants |
|--------|------|----------|
| **M1: Patch Partitioning** | Decompose mesh into ~130 patches | METIS (v1) / Graph BPE (new) |
| **M2: Patch Tokenizer** | Encode each patch as discrete tokens | SimVQ (v1) / RVQ 3-level (new) |
| **M3: AR Generation** | Autoregressively generate patch token sequences | GPT-2 style Transformer |
| **M4: Assembly & Stitching** | Reassemble patches into watertight mesh | Stitching MLP |

---

## 3. Module 1: Patch Partitioning

### 3.1 METIS (Baseline, v1)

- Face adjacency dual graph with normal-cosine edge weights
- METIS k-way with k = ceil(|F| / 35), targeting ~35 faces/patch
- Post-processing: merge < 15 faces, bisect > 60 faces
- Per-patch PCA normalization + scale normalization

### 3.2 Graph BPE (New)

**Prerequisite**: Phase 0 feasibility validation (~2h CPU).

**Face Feature Discretization**:
- Normal vector → icosphere binning (64 bins default)
- Face area → log-scale binning (8 bins)
- Dihedral angle (edge label) → uniform angular binning (16 bins)
- Combined alphabet: 64 × 8 × 16 = 8,192

**Graph BPE Algorithm**:
1. Initialize vocabulary V = all discrete face labels (base alphabet)
2. Count bigram frequencies (l_u, l_e, l_v) across all training mesh dual graphs
3. Merge most frequent bigram: contract matched node pairs (greedy, ID-ordered)
4. Add new merged symbol to V
5. Repeat 2-4 until |V| reaches target (~2000)

Each BPE token = a group of merged faces = one "data-driven patch".

**Go/No-Go (Phase 0)**:
- H1a: ≥60% of BPE tokens **with ≥10 faces** have within-token normal variance < METIS median → Go
- H5: MI(discrete_label, continuous_features) > 0.5 for at least one granularity → Go
- Patch size distribution: std < 2× METIS std → acceptable
- Any No-Go → fall back to METIS for all subsequent phases

---

## 4. Module 2: Patch Tokenizer (VQ-VAE)

### 4.1 SimVQ (Baseline, v1)

- GCN encoder (4 layers, h=256) → z ∈ R^128
- SimVQ codebook: frozen C ∈ R^{4096×128}, learnable W ∈ R^{128×128}
- Quantized: CW space, STE gradient
- Dead-code revival every 10 epochs, encoder warmup 10 epochs
- Output: 1 token per patch

### 4.2 RVQ 3-Level (New)

Replace single-layer SimVQ with 3-layer Residual Vector Quantization:

```python
# Level 1: coarse shape
z1_hat = simvq_quantize(z, codebook_1)  # K=1024
r1 = z - z1_hat

# Level 2: detail residual
z2_hat = simvq_quantize(r1, codebook_2)  # K=1024
r2 = r1 - z2_hat

# Level 3: fine residual
z3_hat = simvq_quantize(r2, codebook_3)  # K=1024

# Reconstruction
z_hat = z1_hat + z2_hat + z3_hat
```

Each level uses SimVQ internally (frozen C + learnable W) to prevent collapse.

**Output**: 3 tokens per patch (tok_L1, tok_L2, tok_L3).

**Total codebook capacity**: 1024^3 = ~10^9 combinations (vs v1's 4096).

**Training strategy**: Joint — all 3 levels and decoder train simultaneously from scratch. The RVQ gradient flows through all levels via residual connections. This is simpler than progressive training and avoids the decoder retraining problem.

**Alternative** (if joint training is unstable): Progressive with decoder fine-tuning at each level addition. Budget extra ~4h GPU per level.

### 4.3 Decoder

Cross-attention decoder (shared across both variants):
- 60 learnable vertex queries
- KV tokens projected from quantized embedding(s)
- SimVQ: 4 KV tokens (v1 B-stage)
- RVQ: 4 KV tokens from concatenated multi-level embeddings
- Output: R^{60×3} vertex coordinates (masked)

### 4.4 Training Objective

Same as v1:
- L = L_recon (Chamfer Distance) + L_commit + L_embed
- Adam, lr=1e-4, cosine annealing, batch_size=256, 200 epochs

---

## 5. Module 3: AR Generation

### 5.1 Patch Sequence Representation

Each mesh is a sequence of M ≈ 130 patches. Each patch is represented as:

```
(pos_x, pos_y, pos_z, scale, tok_L1, tok_L2, tok_L3)
```

**Prediction order**: Position first, then codebook tokens. This way the model knows WHERE before deciding WHAT — spatial context informs patch type selection.

For SimVQ variant: `(pos_x, pos_y, pos_z, scale, tok)` — 5 tokens/patch.

Position: patch centroid, quantized to 256 levels per axis.
Scale: patch bounding sphere radius, quantized to 64 levels.

### 5.2 Patch Ordering

Z-order (Morton code) on patch centroids for deterministic spatial ordering. Simple, preserves spatial locality, AR-friendly.

### 5.3 Model Architecture

GPT-2 style decoder-only Transformer:
- ~50M parameters
- 12 layers, 768 hidden dim, 12 attention heads
- Context length: 1024 tokens (130 patches × 7 tokens/patch = 910 for RVQ; 130 × 4 = 520 for SimVQ)
- Training: next-token prediction on patch sequences

### 5.4 Conditioning

- **Unconditional**: pure NTP for FID/COV/MMD evaluation
- **Point-cloud conditioned**: cross-attention from point cloud encoder (for comparison with BPT, TreeMeshGPT)

Text conditioning is a stretch goal, not required for the paper.

---

## 6. Module 4: Assembly & Stitching

### 6.1 Problem

Each patch is decoded independently — producing local vertex coordinates but no inter-patch connectivity. To form a complete mesh, we must:
1. Determine which patches are adjacent
2. Align boundary vertices between adjacent patches
3. Create inter-patch face connectivity

### 6.2 Adjacency Recovery

During **training/reconstruction**: adjacency is known from the original mesh partitioning (METIS or BPE records which patches share edges).

During **generation**: the AR model predicts patch positions. Adjacency is inferred geometrically:
- For each pair of patches, compute minimum distance between their boundary vertices
- Threshold: if min_distance < δ (e.g., 0.05 in normalized coordinates), mark as adjacent
- This is O(M²) per mesh but M ≈ 130, so negligible

### 6.3 Boundary Alignment

**Ground-truth extraction** (for training):
1. After METIS/BPE partitioning, record which original vertices are shared between adjacent patches
2. Each shared vertex appears in both patches (with different local indices)
3. After independent patch reconstruction, the two copies will differ — the GT target is the original shared position

**Stitching MLP**:
```
Input: concat(patch_i_boundary_verts, patch_j_boundary_verts, relative_position)
       shape: (N_boundary × 9) — 3 coords each + 3 relative offset
Output: merged boundary vertex positions, shape: (N_boundary × 3)
```

Small MLP (3 layers, 256 hidden). Trained with L2 loss against GT shared vertices.

### 6.4 Inter-Patch Face Connectivity

After boundary vertex alignment:
- Identify corresponding boundary vertex pairs (nearest-neighbor matching)
- Merge matched pairs into single vertices
- Face indices from both patches are re-indexed to use shared vertex IDs
- Result: connected mesh with correct face-level topology across patch boundaries

### 6.5 Fallback

If learned stitching underperforms:
- Simple nearest-vertex merging with distance threshold (will produce some non-manifold edges)
- Report non-manifold edge ratio as a mesh quality metric

---

## 7. Ablation Design: 2×2 Matrix

| Config | M1 (Partition) | M2 (Tokenizer) | Notes |
|--------|---------------|----------------|-------|
| **C1** | METIS | SimVQ | Baseline (v1, already trained) |
| **C2** | METIS | RVQ | Isolates RVQ contribution |
| **C3** | BPE | SimVQ | Isolates BPE contribution |
| **C4** | BPE | RVQ | Full v2 |

All 4 configs share the same M3 (AR model) and M4 (stitching).

**Primary comparison**: C1 vs C4 (v1 baseline vs full v2).
**Ablation**: C2 and C3 isolate individual contributions.

### Additional ablations (appendix):
- RVQ depth: 1 vs 2 vs 3 levels
- Discretization granularity for BPE: coarse / medium / fine
- Patch ordering: Z-order vs BFS vs spectral

---

## 8. Evaluation

### 8.1 Reconstruction (Module 1+2)
- Chamfer Distance (CD)
- Normal Consistency (NC)
- F-Score @ {0.01, 0.02, 0.05}
- Codebook utilization
- CD ratio (cross-cat / same-cat)

### 8.2 Generation (Module 3+4)
- FID, COV, MMD on ShapeNet Chair/Table
- **Data note**: Reconstruction uses Objaverse-LVIS (v1 data). Generation comparison requires ShapeNet Chair/Table to match published baselines. AR model is retrained on ShapeNet for this table.
- Generation baselines: PolyGen, MeshGPT, FACE (published numbers; note any metric protocol differences)
- Mesh quality: non-manifold edge ratio, self-intersection ratio
- Compression: tokens per mesh (vs per-face methods)
  - Note: the "277× compression" counts patch tokens only; stitching overhead is separate and reported

### 8.3 Analysis
- BPE vocabulary visualization (top-K tokens as mesh substructures)
- RVQ level-wise reconstruction (L1 only vs L1+L2 vs L1+L2+L3)
- Patch size distributions (BPE vs METIS)

---

## 9. Phased Execution Plan

| Phase | Content | Compute | Dependency |
|-------|---------|---------|------------|
| **0** | BPE feasibility: discretization MI + BPE run + Go/No-Go | ~2h CPU | None |
| **1** | RVQ tokenizer training (C2: METIS+RVQ) | ~12h GPU | None |
| **2** | BPE partition + tokenizer training (C3, C4) | ~20h GPU | Phase 0 Go |
| **3** | AR generation model training | ~30h GPU | Phase 1 (+ Phase 2 if BPE Go) |
| **4** | Stitching module + full pipeline evaluation | ~15h GPU | Phase 3 |
| **5** | Ablation runs + visualization + paper figures | ~20h GPU | Phase 4 |
| **Buffer** | Re-runs, debugging | ~20h | — |
| **Total** | | **~120h GPU** | |

Phase 0 and Phase 1 can run in parallel (Phase 0 is CPU, Phase 1 is GPU).

If Phase 0 No-Go: skip Phase 2, total reduces to ~100h GPU.

---

## 10. Go/No-Go Checkpoints

| Checkpoint | After Phase | Go Criteria | No-Go Action |
|-----------|-------------|-------------|-------------|
| BPE feasibility | Phase 0 | H1a + H5 Go | Drop BPE, METIS only |
| RVQ improvement | Phase 1 | RVQ CD < SimVQ CD | Investigate; may keep SimVQ |
| Generation quality | Phase 3 | Generated meshes visually plausible | Simplify AR model / tune |
| Full pipeline | Phase 4 | FID within 2× of FACE/MeshGPT | Reposition as tokenization paper (backup track) |

---

## 11. Paper Positioning

### 11.1 Contributions

1. **First patch-level mesh generation model** — generate meshes by composing ~130 codebook entries, ~277× fewer tokens than per-face methods
2. **RVQ multi-scale patch codebook** — 3-level residual quantization with ~10^9 effective codebook capacity
3. **Graph BPE for mesh partitioning** (if Phase 0 Go) — first data-driven mesh partitioning via BPE on dual graphs
4. **2×2 ablation** (partition × tokenizer) isolating each contribution
5. **Boundary stitching** for patch-level generation

### 11.2 Dual-Track Strategy

| Track | Condition | Focus | Title Emphasis |
|-------|-----------|-------|---------------|
| **Main** | Generation quality acceptable | Mesh Generation | "Compositional Mesh Generation via Patch Vocabulary" |
| **Backup** | Generation quality insufficient | Mesh Tokenization | "Multi-Scale Patch Tokenization for 3D Meshes" |

### 11.3 Competitive Differentiation

| vs | Our advantage |
|----|--------------|
| MeshGPT/FACE/BPT | 277× fewer tokens (patch-level vs per-face) |
| MeshMosaic | Codebook-based selection vs per-face AR within patches |
| FreeMesh | Graph-level BPE vs coordinate-sequence BPE |
| G3PT/OctGPT | Explicit mesh output vs implicit/octree |

---

## 12. Hardware & Resource Constraints

- GPU: RTX 4090 × 1 (24GB VRAM)
- RAM: 62 GB
- Disk: 80 GB (data/ < 50 GB, results/ < 5 GB)
- Data: Objaverse-LVIS (already preprocessed, 267K patches)
- Checkpoints: upload to HF (Pthahnix/MeshLex-Research) after each phase

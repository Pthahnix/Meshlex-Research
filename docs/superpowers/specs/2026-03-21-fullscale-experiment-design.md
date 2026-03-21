# MeshLex Full-Scale Experiment Design

> Date: 2026-03-21
> Status: REVIEWED
> Scope: Full-scale training, analysis, ablation, and evaluation on 72K meshes / 10.8M patches
> Hardware: 3× RTX 5090 (32GB VRAM each), multi-GPU parallel execution

---

## 1. Background & Motivation

### 1.1 What We Have

**Dataset** (COMPLETE):
- 72,555 meshes (32,136 Objaverse + 40,419 ShapeNet), 10.8M patches
- HF Parquet: `Pthahnix/MeshLex-Patches`, dual normalization (PCA + noPCA)
- Splits: seen_train=53,492 / seen_test=13,372 / unseen=5,541

**Preliminary Results** (5% scale, 4,934 meshes):
- Exp 1: Lognormal distribution confirmed 11/11 subgroups
- Exp 2: Weak spatial correlation (ρ = -0.036)
- Exp 3: Codebook UMAP — loose clusters, no strong structure
- Exp 4: RVQ inter-level NMI > 0.24 (STRONG_DEPENDENCY)
- Exp 5: Toy MDLM PPL=868 (NOT_FEASIBLE at toy scale)

**Existing Code**: VQ-VAE (RVQ), AR model, generation pipeline, evaluation, preliminary analysis scripts

### 1.2 What This Plan Does

Supersedes and integrates three existing plans:
1. `2026-03-19-assembly-fix-full-retrain` (Phase A-E) — core pipeline retrain
2. Theory-driven spec (branch: `theory-driven-design`) — distribution analysis + codebook ablation
3. PatchDiffusion spec (branch: `innovation-brainstorm`) — MDLM feasibility at scale

Into a single, unified 5-Phase execution plan.

### 1.3 Key Unknowns to Resolve

| # | Question | How to answer | Risk if wrong |
|---|---------|--------------|---------------|
| FM1 | Is lognormal a SimVQ artifact? | VQ method comparison (Phase 3b-2) | Theory paper premise collapses |
| FM2 | Does lognormal hold on full codebook? | Full preliminary rerun (Phase 3a) | Need to pivot narrative |
| FM3 | Is MDLM feasible at scale? | 20M+ MDLM training (Phase 3c) | PatchDiffusion direction dead |
| Q1 | What's the optimal codebook K? | K ablation (Phase 3b-1) | Suboptimal system design |
| Q2 | How much does PCA rotation matter? | PCA vs noPCA ablation (Phase 4c) | Wasted complexity |

---

## 2. Hardware & Resource Budget

### 2.1 Hardware

| Resource | Spec |
|----------|------|
| GPU | NVIDIA RTX 5090 × 3 (32 GB VRAM each) |
| System RAM | To be confirmed at Phase 0 |
| Disk | To be confirmed at Phase 0 |

### 2.0 Phase 0: Hardware Audit (MANDATORY first step)

Before any training, the executing agent MUST run:

```bash
nvidia-smi          # Confirm 3× 5090, VRAM per card
free -h             # System RAM
df -h /             # Available disk
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Record results in `results/fullscale_eval/hardware_audit.json`. If disk < 100GB available or RAM < 64GB, stop and report.

### 2.2 Multi-GPU Execution Strategy

With 3 GPUs available, independent training runs execute in parallel:

**GPU 0**: PCA VQ-VAE (K=1024) → PCA encode → PCA AR
**GPU 1**: noPCA VQ-VAE (K=1024) → noPCA encode → noPCA AR
**GPU 2**: K ablation VQ-VAEs (K=512, K=2048) → VQ method comparison (vanilla, EMA) → MDLM

Use `CUDA_VISIBLE_DEVICES=N` to pin each training to a specific GPU.

### 2.3 Estimated Compute Budget (with 3-GPU parallelism)

**VQ-VAE Training Arithmetic**:
- seen_train patches: ~53K meshes × ~149 patches/mesh ≈ 7.9M patches
- VQ-VAE trains on individual patches (not sequences): 7.9M patches, bs=1024 → ~7,700 iterations/epoch
- 5090 throughput estimate: ~2-3x faster than A4000 → ~1.5-2.5 min/epoch
- 100 epochs ≈ 2.5-4h per VQ-VAE

**Note**: The prior spec (assembly-fix-full-retrain) estimated 230h on A4000 for 3M patches at bs=512. That was conservative (included data loading overhead on A4000 with NPZ). With Parquet-native loading and 5090, actual time should be substantially less. If training exceeds 8h for a single VQ-VAE, investigate data loading bottlenecks.

**AR Training Arithmetic**:
- 53K sequences, bs=8, grad_accum=4 → effective batch 32 → ~1,656 steps/epoch
- 200 epochs ≈ 331K steps
- 5090 throughput: ~0.3-0.5s/step (57M params, seq_len ≤ 1430) → ~28-46h per AR model
- With 2 AR models on 2 GPUs in parallel: ~28-46h wall time

| Phase | GPU 0 | GPU 1 | GPU 2 | Wall time |
|-------|-------|-------|-------|-----------|
| 1 | PCA VQ-VAE K=1024 | noPCA VQ-VAE K=1024 | K=512 → K=2048 | ~4-8h |
| 2 | PCA encode + AR | noPCA encode + AR | Vanilla VQ → EMA VQ | ~30-48h |
| 3 | — | — | MDLM + 3a + 3b analysis | ~18-24h |
| 4-5 | Evaluation + figures (single GPU sufficient) | — | — | ~6-8h |
| **Total wall time** | | | | **~58-88h (2.5-4 days)** |

### 2.4 Resource Constraints

- **Disk budget**: Check at Phase 0. Estimate ~30GB for Parquet download + ~10GB checkpoints + ~5GB sequences + ~5GB results = ~50GB needed
- **Data loading**: Use `datasets.load_dataset()` to stream from HF Parquet. Converted patch NPZ cache ≈ 15-20GB. If disk is tight, use streaming mode (no local cache).
- **Checkpoint policy**: Upload to HF immediately after each training completes. Keep latest 3 on disk.
- **VRAM budget for 57M AR**: d_model=768, n_layers=12, seq_len=1430, bs=8 → estimated ~18-22GB VRAM with standard attention. 5090's 32GB provides sufficient headroom. If OOM occurs, reduce bs to 4 and increase grad_accum to 8.

---

## 3. Phase 1: VQ-VAE Foundation (~20-24h)

### 3.1 Data Loading

**New requirement**: Data is now in HF Parquet, not local NPZ files. Need a Parquet-native data loading path.

**Approach**: Use `datasets.load_dataset("Pthahnix/MeshLex-Patches")` to load Parquet data. Filter to `seen_train` mesh IDs using `splits.json`. Convert each row's flattened arrays back to numpy arrays for `PatchGraphDataset`.

**Disk footprint estimate**:
- Full Parquet download (10.8M rows): ~15-20GB (Parquet is compressed)
- Converted NPZ cache (if needed): ~15-20GB additional
- **Total data loading disk**: ~15-40GB depending on caching strategy

**If disk is tight**: Use `datasets.load_dataset(..., streaming=True)` to avoid full download. Write a `StreamingPatchDataset` that fetches batches on-the-fly. Slightly slower but avoids disk pressure.

**Go/No-Go gate**: After Phase 1 VQ-VAE training, run a quick reconstruction CD check on 500 random `seen_test` patches. If mean CD > 0.3 (on normalized meshes), investigate before proceeding to Phase 2.

### 3.2 Primary VQ-VAE Training (×2)

| Config | PCA | noPCA |
|--------|-----|-------|
| Input field | `local_vertices` | `local_vertices_nopca` |
| RVQ levels | 3 | 3 |
| Codebook K | 1024 | 1024 |
| Epochs | 100-200 | 100-200 |
| Batch size | 1024-2048 | 1024-2048 |
| Checkpoint | `data/checkpoints/rvq_full_pca/` | `data/checkpoints/rvq_full_nopca/` |
| HF upload | `checkpoints/rvq_full_pca/` | `checkpoints/rvq_full_nopca/` |

**Success criteria**:
- Codebook utilization > 90%
- Reconstruction loss < 0.2 (comparable to v1's 0.177)
- No dead codes > 5%

### 3.3 Codebook K Ablation VQ-VAE (×2 extra)

| Config | K=512 | K=2048 |
|--------|-------|--------|
| Normalization | PCA only | PCA only |
| Epochs | 100 | 100 |
| Checkpoint | `rvq_full_pca_k512/` | `rvq_full_pca_k2048/` |

These are for theory-driven analysis (Phase 3b-1) — comparing token distributions across codebook sizes.

---

## 4. Phase 2: Token Encoding + AR Retrain (~26-35h)

### 4.1 Token Encoding

**Prerequisite**: Phase 1 VQ-VAE checkpoints.

Encode all `seen_train` meshes (53,492) into token sequences:

| Pipeline | Format | Tokens/patch | Output dir |
|----------|--------|-------------|-----------|
| PCA | 11-token (pos×3, scale, quat×4, tok×3) | 11 | `data/sequences/rvq_full_pca/` |
| noPCA | 7-token (pos×3, scale, tok×3) | 7 | `data/sequences/rvq_full_nopca/` |

**Code changes needed**:
- Implement `src/rotation.py` (quaternion encode/decode) — from existing plan Task 1
- Implement `patches_to_token_sequence_rot()` in `src/patch_sequence.py` — from existing plan Task 7
- Update `scripts/encode_sequences.py` to save rotation matrices — from existing plan Task 8
- New: adapt encode script to read from Parquet instead of local NPZ

### 4.2 AR Model Training (×2)

| Config | PCA AR | noPCA AR |
|--------|--------|----------|
| Vocab size | 2112 | 1852 |
| Max seq len | 1430 | 1024 |
| d_model | 768 | 768 |
| n_heads | 12 | 12 |
| n_layers | 12 | 12 |
| Params | ~57M | ~57M |
| Batch size | 8 | 8 |
| Grad accum | 4 | 4 |
| Effective batch | 32 | 32 |
| Epochs | 200 | 200 |
| Warmup | 10 epochs | 10 epochs |
| Scheduler | Cosine decay | Cosine decay |
| Checkpoint | `data/checkpoints/ar_full_pca/` | `data/checkpoints/ar_full_nopca/` |

**Success gate**: Loss < 2.0, perplexity < 8.0. If not met, debug before proceeding.

**Scale-up rationale**: v2 was 20.4M params on 4.9K sequences. Now 53K sequences (10x more data) justifies ~57M params (2.8x). Token/param ratio: ~53K × 130 × 7 / 57M ≈ 0.85 (still below 1.0, acceptable for codebook-level AR).

---

## 5. Phase 3: Sequential Experiment Branches (~24-36h serial, less with multi-GPU)

After Phase 1-2 completes, these branches execute. On multi-GPU setup, 3a+3b can share GPU 0/1 while 3c runs on GPU 2.

Serial sum: 3a (~3h) + 3b-1 (~2h) + 3b-2 (~12h) + 3b-3 (~2h) + 3c (~18h) = ~37h serial.
With multi-GPU: 3c on GPU 2 overlaps with 3a+3b on GPU 0 → ~18-20h wall time.

### 5.1 Branch 3a: Preliminary Experiments Full Rerun (~2-3h)

Rerun all 5 experiments from `scripts/run_preliminary_analysis.py` + `scripts/run_mdlm_prototype.py` on full-scale data.

**Data**: Full token sequences from Phase 2 encoding (using K=1024 PCA VQ-VAE).

| Experiment | What changes at full scale | Key question |
|-----------|---------------------------|-------------|
| Exp 1: Per-category | 72K meshes, ~1200 categories | Still lognormal? σ change? |
| Exp 2: Spatial correlation | 72K meshes | ρ still near zero? |
| Exp 3: Codebook UMAP | Full-scale codebook | Cluster structure emerge? |
| Exp 4: RVQ dependency | 10.8M tokens | NMI change significantly? |
| Exp 5: MDLM quick check | 53K sequences, 50 epochs | Any signal improvement? |

**Output**: `results/fullscale_preliminary/exp{1-5}_*/summary.json` + plots.

### 5.2 Branch 3b: Theory-Driven Analysis (~12-16h)

#### 3b-1: Codebook K Scaling Analysis (~1-2h analysis)

Using VQ-VAEs from Phase 1 (K=512, 1024, 2048):

1. Encode 53K meshes with each VQ-VAE
2. Fit lognormal + Zipf to each K's token distribution
3. Plot:
   - K vs σ (lognormal sigma)
   - K vs α (Zipf alpha)
   - K vs entropy / entropy_max_ratio
   - K vs codebook utilization
4. Test: Is lognormal consistent across K? How does σ scale?

**Expected**: If lognormal is geometry-driven, σ should be relatively stable. If artifact, σ may change dramatically.

#### 3b-2: VQ Method Comparison (~8-12h training + 2h analysis)

Train two additional K=1024 VQ-VAEs:

| Method | Architecture | Expected behavior |
|--------|-------------|-------------------|
| Vanilla VQ | Straight-through estimator, no SimVQ | May have collapse → Zipf-like? |
| EMA-VQ | Exponential moving average codebook | Between SimVQ and vanilla |

For each, encode sequences and fit distributions. **This directly tests FM1** (SimVQ artifact hypothesis).

**If vanilla VQ also shows lognormal**: Strong evidence for geometry-driven property.
**If vanilla VQ shows Zipf/different**: SimVQ artifact confirmed — theory narrative needs pivot.

#### 3b-3: Curvature-Frequency Correlation (~2h)

1. For each patch in the dataset, compute discrete Gaussian curvature via angle deficit method
2. Map each patch to its VQ token
3. Group patches by token → compute mean curvature per token
4. Plot token frequency vs mean |curvature| → test Gauss-Bonnet prediction (high-frequency tokens = low curvature)
5. Compute Spearman correlation between log(frequency) and mean |curvature|

**Expected**: Negative correlation (frequent tokens = flat patches). Strength of correlation validates Gauss-Bonnet theoretical link.

### 5.3 Branch 3c: MDLM Full-Scale Training (~12-18h)

#### Model Architecture

```
MDLM-Full: Transformer Encoder
  d_model = 512
  n_heads = 8
  n_layers = 8
  token_embed = Embedding(vocab_size, d_model)
  pos_embed = Embedding(max_seq_len, d_model)
  level_embed = Embedding(3, d_model)  # RVQ level awareness
  time_embed = MLP(1, d_model)  # continuous time t ∈ [0, 1]
  Total params: ~20-40M
```

**Note on shared codebook token IDs**: In the 7-token format, cb_L1/L2/L3 share the same token ID range [off_code, off_code+K). The AR model distinguishes levels by position within the 7-token pattern. For MDLM, which is non-autoregressive and cannot rely on position order during masking, the `level_embed` provides explicit level identity to each token. Each position in the sequence has a known level assignment (positions 4,5,6 within each 7-token patch correspond to L1,L2,L3), so level_embed is computed from position modulo tokens_per_patch.

#### Training Protocol

- **Data**: Full seen_train sequences (53,492 meshes)
- **Masking**: Continuous-time schedule, t ~ U(0.1, 1.0), mask probability = t
- **Loss**: Cross-entropy on masked positions only
- **Optimizer**: AdamW, lr=3e-4, warmup 5 epochs, cosine decay
- **Epochs**: 200
- **Batch size**: 32 (or grad accum to effective 32)

#### Evaluation Metrics

| Metric | Toy result (5%) | Feasibility threshold | Promising threshold |
|--------|----------------|----------------------|-------------------|
| Perplexity | 868 | < 50 | < 20 |
| Accuracy (t=0.5) | 0.6% | > 5% | > 15% |
| KL vs train dist | 0.09 nats | < 0.5 nats | < 0.2 nats |

#### Generation Protocol

If training shows promise (PPL < 50):
1. Generate 100 sequences via iterative unmasking (1000 steps)
2. Decode with VQ-VAE → point clouds
3. Compute diversity (pairwise CD) and quality (FPD-like metric)
4. Compare with AR v2 generations

---

## 6. Phase 4: Evaluation + Ablation Suite (~4-6h)

### 6.1 Generation Evaluation

For each AR model (PCA, noPCA), generate meshes at multiple temperatures:

| Config | Temperatures | N meshes per temp | Total |
|--------|-------------|-------------------|-------|
| PCA AR | 0.6, 0.7, 0.8, 0.9, 1.0 | 10 | 50 |
| noPCA AR | 0.6, 0.7, 0.8, 0.9, 1.0 | 10 | 50 |

Metrics per mesh:
- 7-stage visualization (existing `generate_v2_pipeline.py`)
- Surface reconstruction (Ball Pivoting)
- Export OBJ + PLY + PNG
- Stitching quality: boundary vertex distance, non-manifold edge count (using existing `src/metrics.py`)

### 6.2 Reconstruction Evaluation

| Split | N meshes | Metrics |
|-------|----------|---------|
| seen_test | 13,372 | Reconstruction CD (sample 500), codebook utilization, per-category CD |
| unseen | 5,541 | Same metrics — test generalization |

**Key metric**: Cross-category / same-category CD ratio.
- v1 result: 1.019x (LVIS-Wide, 5-cat → LVIS improvement)
- Target: < 1.05x on 72K full data

### 6.3 PCA vs noPCA Ablation

Side-by-side comparison:

| Metric | PCA | noPCA | Winner |
|--------|-----|-------|--------|
| VQ-VAE recon CD | | | |
| Codebook utilization | | | |
| AR loss / PPL | | | |
| Generation diversity (pairwise CD) | | | |
| Assembly quality (visual) | | | |
| Token distribution (σ, α) | | | |

### 6.4 Cross-Experiment Dashboard

One unified `results/fullscale_eval/DASHBOARD.md` containing:
- Table of all VQ-VAE configs and metrics
- Table of all AR configs and metrics
- Theory-driven findings summary
- MDLM feasibility verdict
- PCA vs noPCA verdict

---

## 7. Phase 5: Paper-Ready Analysis (~2h)

### 7.1 Figure List

| # | Figure | Data source |
|---|--------|-------------|
| F1 | Token distribution: 3D mesh vs image vs time series | Phase 3a + literature |
| F2 | Codebook K scaling: σ, α, entropy vs K | Phase 3b-1 |
| F3 | VQ method comparison: SimVQ vs vanilla vs EMA | Phase 3b-2 |
| F4 | Curvature-frequency scatter + correlation | Phase 3b-3 |
| F5 | PCA vs noPCA ablation bars | Phase 4c |
| F6 | Generation gallery: multi-angle renders | Phase 4a |
| F7 | Reconstruction quality heatmap (per-category) | Phase 4b |
| F8 | MDLM PPL/accuracy curves (if feasible) | Phase 3c |

### 7.2 Context Documents

- `context/29_fullscale_experiment_results.md` — comprehensive results write-up
- `context/30_paper_contribution_summary.md` — final contribution list + evidence mapping

### 7.3 Final Direction Decision

| Condition | Decision |
|-----------|----------|
| Lognormal holds + MDLM feasible | PatchDiffusion paper (method + theory) |
| Lognormal holds + MDLM not feasible | Theory-driven paper (analysis + theory + improved AR) |
| Lognormal is SimVQ artifact | MeshLex system paper (focus on codebook + generation quality) |
| Full data CD ratio > 1.2x | Investigate — possible codebook saturation issue |

---

## 8. Dependency Graph

```
Phase 1: VQ-VAE Foundation
  ├── PCA VQ-VAE (K=1024)
  ├── noPCA VQ-VAE (K=1024)
  ├── PCA VQ-VAE (K=512)  ─────────── → Phase 3b-1
  └── PCA VQ-VAE (K=2048) ─────────── → Phase 3b-1
       │
Phase 2: Encode + AR
  ├── PCA encode (11-tok) → PCA AR ──── → Phase 4a, 4b, 4c
  └── noPCA encode (7-tok) → noPCA AR ─ → Phase 4a, 4b, 4c
       │
Phase 3: Sequential Experiment Branches
  ├── 3a: Preliminary rerun (uses Phase 2 PCA sequences)
  ├── 3b-1: K ablation (uses Phase 1 K=512/2048 + Phase 2-style encode)
  ├── 3b-2: VQ method comparison (independent VQ-VAE training)
  ├── 3b-3: Curvature analysis (uses Phase 1 PCA VQ-VAE)
  └── 3c: MDLM training (uses Phase 2 PCA sequences)
       │
Phase 4: Evaluation (depends on Phase 2 + Phase 3)
       │
Phase 5: Paper-ready (depends on Phase 4)
```

---

## 9. Code Changes Required

### New Files

| File | Purpose |
|------|---------|
| `src/rotation.py` | Quaternion encode/decode for PCA rotation |
| `src/parquet_loader.py` | HF Parquet → PatchGraphDataset adapter |
| `src/mdlm_model.py` | Full MDLM model (Transformer encoder + time/level embeddings) |
| `scripts/train_mdlm.py` | MDLM training script (continuous-time masking, full-scale) |
| `scripts/run_fullscale_analysis.py` | Orchestrator for Phase 3b (theory-driven) |
| `scripts/curvature_analysis.py` | Discrete Gaussian curvature computation + frequency correlation |
| `scripts/vq_method_comparison.py` | Train vanilla/EMA VQ + compare distributions |
| `scripts/fullscale_evaluation.py` | Unified evaluation dashboard generator |
| `tests/test_rotation.py` | Quaternion utility tests |
| `tests/test_mdlm_model.py` | MDLM model tests |

### Modified Files

| File | Changes |
|------|---------|
| `src/patch_sequence.py` | Add `patches_to_token_sequence_rot()`, `compute_vocab_size_rot()` |
| `src/patch_dataset.py` | Add rotation support to `MeshSequenceDataset`, add `use_nopca` to `PatchGraphDataset` |
| `scripts/encode_sequences.py` | Save rotation matrices, support Parquet input |
| `scripts/train_ar.py` | Add `--rotation` flag, support larger model configs |
| `scripts/train_rvq.py` | Add `--nopca` flag, support Parquet input |
| `scripts/generate_v2_pipeline.py` | Support 11-token rotation decode |
| `scripts/run_preliminary_analysis.py` | Parameterize data paths for full-scale rerun |
| `scripts/run_mdlm_prototype.py` | Scale up model config, support full data |

---

## 10. Checkpoint & Upload Policy

Every training completion triggers immediate HF upload:

| Checkpoint | HF path |
|-----------|---------|
| PCA VQ-VAE K=1024 | `checkpoints/rvq_full_pca/checkpoint_final.pt` |
| noPCA VQ-VAE K=1024 | `checkpoints/rvq_full_nopca/checkpoint_final.pt` |
| PCA VQ-VAE K=512 | `checkpoints/rvq_full_pca_k512/checkpoint_final.pt` |
| PCA VQ-VAE K=2048 | `checkpoints/rvq_full_pca_k2048/checkpoint_final.pt` |
| Vanilla VQ K=1024 | `checkpoints/rvq_full_vanilla/checkpoint_final.pt` |
| EMA VQ K=1024 | `checkpoints/rvq_full_ema/checkpoint_final.pt` |
| PCA AR (57M) | `checkpoints/ar_full_pca/checkpoint_final.pt` |
| noPCA AR (57M) | `checkpoints/ar_full_nopca/checkpoint_final.pt` |
| MDLM (20-40M) | `checkpoints/mdlm_full/checkpoint_final.pt` |

---

## 11. Success Criteria Summary

| Phase | Gate | Threshold |
|-------|------|-----------|
| 1 | VQ-VAE quality | Utilization > 90%, loss < 0.2 |
| 2 | AR quality | Loss < 2.0, PPL < 8.0 |
| 3a | Lognormal confirmation | Still 11/11 or near-unanimous lognormal |
| 3b-2 | FM1 resolution | Determine if lognormal is SimVQ artifact |
| 3c | MDLM feasibility | PPL < 50 = feasible, PPL < 20 = promising |
| 4 | Generalization | Cross-cat/same-cat CD ratio < 1.05x |
| 5 | Paper readiness | All figures generated, direction decided |

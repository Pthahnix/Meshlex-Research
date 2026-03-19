# AR Loss Fix: Model Right-Sizing + Training Recipe

> **Date**: 2026-03-19
> **Status**: Design Spec
> **Problem**: AR loss 5.41 (perplexity ~224) blocks Phase 5 ablation/evaluation
> **Target**: AR loss ≤ 4.0 (perplexity ≤ 55), enabling meaningful generation

---

## 1. Root Cause

The AR model (PatchGPT) is severely data-starved:

| Metric | Value | Problem |
|--------|-------|---------|
| Model params | 87.3M | Too large |
| Training tokens | ~1.6M (4674 meshes × ~350 tokens) | Too few |
| Token-to-param ratio | 0.018 | Need >> 1.0 |
| Effective batch size | 4 | Noisy gradients |
| Epochs | 100 | Insufficient with slow convergence |
| LR warmup | None | Early instability |

## 2. Solution: Approach A — Shrink Model + Fix Training Recipe

### 2.1 Model Architecture Changes

Shrink PatchGPT from 87.3M → ~20M params:

| Parameter | Current | Proposed |
|-----------|---------|----------|
| d_model | 768 | 512 |
| n_heads | 12 | 8 |
| n_layers | 12 | 6 |
| max_seq_len | 1024 | 1024 |
| vocab_size | 1856 | 1856 |
| dropout | 0.1 | 0.1 |
| Approx params | 87.3M | ~20M |

Note: with the smaller model, batch_size=8 or even 16 may fit in 24GB VRAM. If so, grad_accum_steps can be reduced accordingly to maintain effective batch=32.

Weight tying (head ↔ token_emb) retained.

### 2.2 Training Recipe Changes

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| batch_size (per step) | 4 | 4 | VRAM limited |
| grad_accum_steps | 1 | 8 | Effective batch = 32 |
| epochs | 100 | 300 | 3x more passes |
| lr | 3e-4 | 3e-4 | Same peak LR |
| warmup | none | 10 epochs linear | Stabilize early training |
| scheduler | Cosine(T_max=100) | Linear warmup → Cosine(T_max=290) via SequentialLR | Match new epoch count |
| grad_clip | 1.0 | 1.0 | Unchanged |

### 2.3 Expected Outcome

- Chinchilla-style token exposure ratio: current 1.6M×100/87.3M = 1.8, proposed 1.6M×300/20M = 24 (13x improvement)
- Optimizer steps per epoch: ~1169 batches / 8 accum = ~146 steps → 300 epochs = ~43,800 total optimizer steps (vs current ~117K steps but with much noisier batch=4 gradients)
- Target loss: ≤ 4.0 (perplexity ≤ 55)
- Training time: ~9h on RTX 4090 (300 epochs × ~111s/epoch upper bound; smaller model may be faster)

## 3. Implementation Scope

### Files Changed

1. **`scripts/train_ar.py`**:
   - Add `--grad_accum_steps` CLI arg (default 8)
   - Add `--warmup_epochs` CLI arg (default 10)
   - Update default `--d_model` to 512, `--n_heads` to 8, `--n_layers` to 6
   - Update default `--epochs` to 300
   - Modify training loop: accumulate gradients, step every N batches
   - Replace `CosineAnnealingLR` with `SequentialLR(LinearLR → CosineAnnealingLR)`
   - Save/restore `scheduler.state_dict()` in checkpoints for correct `--resume` behavior

2. **`src/ar_model.py`**: No changes. Model already accepts all dims as constructor args.

### Important Notes

- **This is a fresh training run.** The old 87.3M checkpoint cannot be resumed — architecture dimensions have changed. Do NOT pass `--resume` with the old checkpoint.
- `generate.py` loads config from the checkpoint dict (`ar_config = ar_ckpt.get("config", )`), so it auto-adapts to the new model size at inference.

### Files NOT Changed

- `patch_sequence.py`, `patch_dataset.py`, `rvq.py`, `model_rvq.py` — tokenization pipeline unchanged
- `generate.py` — loads config from checkpoint dict, auto-adapts to model size
- No data augmentation, no class conditioning, no vocab changes

## 4. Go/No-Go After Retraining

| Outcome | Loss | Action |
|---------|------|--------|
| Strong success | ≤ 3.5 | Proceed to Phase 5 |
| Acceptable | 3.5 - 4.5 | Proceed, note limitation |
| Marginal | 4.5 - 5.0 | Layer on Approach B (augmentation) or C (conditioning) |
| No improvement | > 5.0 | Investigate sequence representation/ordering; the problem may be fundamental to the token layout rather than a data pipeline issue |

## 5. Not In Scope (YAGNI)

- Data augmentation (Approach B) — try A first
- Class-conditional generation (Approach C) — keep unconditional for clean metrics
- Mixed precision / DeepSpeed — overkill at this scale
- Changes to tokenization, vocab, or sequence format

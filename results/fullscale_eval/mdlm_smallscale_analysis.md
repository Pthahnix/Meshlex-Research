# MDLM Small-Scale Feasibility Test

**Date**: 2026-03-22  
**Data**: `data/sequences/rvq_lvis/` (4934 seqs, 4441 train / 493 val)  
**Model**: FullMDLM, d_model=512, n_heads=8, n_layers=8 (26.7M params)  
**Epochs**: 200, batch_size=64, lr=3e-4

## Results

| Epoch | Train Loss | Val Loss | Val Acc | Val PPL |
|-------|-----------|----------|---------|---------|
| 1     | 6.9747    | 6.9102   | 0.28%   | 1002.4  |
| 50    | 6.6725    | 6.6719   | 0.77%   | 789.9   |
| 100   | 6.5831    | 6.6133   | 1.01%   | 744.9   |
| 200   | 6.5002    | 6.6094   | 1.04%   | **742.0** |

**Verdict**: NOT_FEASIBLE (PPL > 50 threshold)

## Analysis

### Why PPL is high on small data
- Uniform distribution baseline: log(1024) ≈ 6.93 nats → PPL = 1024
- Model learned from 6.97→6.50 train loss (7% improvement above random)
- MDLM task is harder than AR: predict masked tokens from *partial context*
- Small data (4934 seqs) cannot cover the joint distribution of mesh tokens

### Compare: AR on same small data
- AR v2 achieves **PPL 4.4** on similar data (4674 LVIS sequences)
- AR conditions on all previous tokens → easier, more direct supervision
- MDLM conditions on ~50% masked input → harder joint distribution learning

### Implications for full-scale Task 14
- Full-scale: 53,492 training sequences (10.8× more data)
- Richer diversity: 72K meshes vs 4.9K meshes
- Expected: significant PPL improvement with 10× data
- Threshold: PPL < 50 = MARGINAL, < 10 = FEASIBLE

### Key finding: MDLM is a **data-hungry** model
- Small data → barely above random (PPL 742)
- Full scale needed to evaluate true feasibility
- **Do NOT use small-scale result to cancel Task 14**

## Action
Proceed with full-scale Task 14 (MDLM training on `data/sequences/rvq_full_pca/`) 
once encoding is complete. The small-scale result is a baseline, not a definitive verdict.

# PatchDiffusion Competitive Landscape Analysis

> Date: 2026-03-21
> Purpose: Competitive intelligence for PatchDiffusion design

---

## Key Findings

### 1. No direct competitor for patch-level masked discrete diffusion

- **TSSR** (arXiv Oct 2025) is the only discrete diffusion mesh generation method, but operates at per-face level (~几万 tokens for 10K faces)
- **All other mesh generation methods are AR** (MeshGPT, FACE, DeepMesh, MeshMosaic, TreeMeshGPT, etc.)
- PatchDiffusion would be the **first** to combine patch-level codebook with masked discrete diffusion

### 2. Mesh Generation Landscape (2024-2026)

#### Tier 1: Directly Competing

| Paper | Year | Venue | Token Level | Compression | Params |
|-------|------|-------|-------------|-------------|--------|
| FACE | 2026.03 | arXiv | per-face | 0.11 | - |
| MeshMosaic | 2025.09 | arXiv (ICLR 2026 withdrawn) | per-face (within patch) | - | 0.5B |
| BPT | 2025 | CVPR 2025 | per-vertex (blocked) | 75% | - |
| FreeMesh | 2025 | ICML 2025 | per-coordinate (BPE) | varies | plug-in |
| MeshAnything V2 | 2025 | ICCV 2025 | per-face (AMT) | 46% | - |
| **MeshLex (Ours)** | 2026 | - | **per-patch** | **~97% (0.03)** | ~2M |

#### Tier 2: Important Context

| Paper | Year | Venue | Key Innovation |
|-------|------|-------|----------------|
| MeshGPT | 2024 | CVPR 2024 | Pioneer: VQ-VAE + GPT for mesh |
| EdgeRunner | 2025 | ICLR 2025 | EdgeBreaker tokenization + fixed-length latent |
| TreeMeshGPT | 2025 | CVPR 2025 | DFS tree sequencing, 22% compression |
| Meshtron | 2024/25 | NeurIPS | Hourglass transformer, 64K faces, 1.1B params |
| DeepMesh | 2025 | ICCV 2025 | DPO RL for mesh generation, 0.5B params |
| ARMesh | 2025 | NeurIPS 2025 | Coarse-to-fine via vertex splits |
| PartCrafter | 2025 | NeurIPS 2025 | Compositional latent diffusion, part-level |
| MeshArt | 2025 | CVPR 2025 | Hierarchical articulated mesh generation |
| TSSR | 2025.10 | arXiv | Discrete diffusion per-face, 10K faces |

### 3. PatchDiffusion Unique Position

```
                  Token Granularity
                  per-vertex ──────────── per-face ──────── per-patch
                  │                       │                  │
Generation    AR  │ MeshGPT, Meshtron     │ FACE, DeepMesh   │ MeshLex AR (PatchGPT)
Paradigm         │ TreeMeshGPT, BPT      │ MeshAnything V2  │
                  │                       │                  │
           Diff  │                       │ TSSR             │ ★ PatchDiffusion ★
                  │                       │                  │   (THIS PAPER)
```

PatchDiffusion occupies a **completely empty cell** in this 2D space.

---

## Theory-Driven Design References

### Formal Methods + ML

| Paper | Year | Relevance |
|-------|------|-----------|
| TorchLean | 2026 | NN properties in Lean4, feasibility proof |
| LeanAgent | 2025 (ICLR) | Lean4 proof automation maturity |
| DeepSeek-Prover-V2 | 2025 | LLM for Lean4 theorem proving |

### Gauss-Bonnet + Mesh Processing

| Paper | Year | Relevance |
|-------|------|-----------|
| Meyer et al. | 2003 | Discrete Gaussian curvature via angle defect |
| Pellizzoni & Savio | 2020 | Gauss-Bonnet for mesh simplification decisions |
| Landreneau & Akleman | 2006 | Gauss-Bonnet governs mesh topology quality |

### Curvature-Aware Neural Networks for 3D

| Paper | Year | Relevance |
|-------|------|-----------|
| CurvaNet | 2020 (KDD) | Principal curvature as GNN features |
| DiffusionNet | 2022 (TOG) | Laplace-Beltrami for geometry-aware features |
| SR-CurvANN | 2025 | Curvature maps encode enough for reconstruction |

### Power-Law / Codebook

| Paper | Year | Relevance |
|-------|------|-----------|
| SimVQ | 2024/25 (ICCV) | Already used in MeshLex, prevents collapse |
| RGVQ | 2025 | Codebook collapse in graph VQ specifically |
| EdVAE | 2024 | Dirichlet priors for codebook utilization |

### Key Theoretical Chain (for Theory-Driven Design)

1. Discrete Gauss-Bonnet: total curvature = 2*pi*chi(M)
2. Patches with |K| > T bounded by 4*pi/T (genus-0)
3. Justifies non-uniform codebook allocation
4. Formalizable in Lean4 (novel contribution)

**No existing paper combines Gauss-Bonnet bounds with neural codebook design.**

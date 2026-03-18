# MeshLex Research Memory

## Project Identity
- **Name**: MeshLex — Learning a Topology-aware Patch Vocabulary for Compositional Mesh Generation
- **Target**: CCF-A (CVPR / NeurIPS / ICCV)
- **Repo**: github.com/Pthahnix/MeshLex-Research
- **HF Checkpoints**: Pthahnix/MeshLex-Research

## Current Status (2026-03-18)
- **v1 (feasibility)**: COMPLETE — 4/4 STRONG GO
- **v2 (implementation plan)**: COMPLETE — 13-task plan at `docs/superpowers/plans/2026-03-18-meshlex-v2-implementation.md`
- **v2 (design spec)**: COMPLETE — at `docs/superpowers/specs/2026-03-18-meshlex-v2-design.md`
- **Next**: Execute implementation plan on RunPod pod

## v1 Key Results
- CD ratio (cross-cat/same-cat): **1.019x** (LVIS-Wide, near-perfect generalization)
- Codebook utilization: **95.3%** (LVIS-Wide)
- Compression: 4000-face mesh → ~130 patch tokens (~277x vs per-face methods)
- Data: Objaverse-LVIS, 46K objects, 1156 categories, 267K patches
- Key finding: more categories = better generalization (5-cat ratio 1.145x → LVIS 1.019x)
- Negative finding: Rotation Trick incompatible with SimVQ (collapse in 7 epochs)

## v2 Architecture (4 Modules)
1. **M1: Patch Partitioning** — METIS (baseline) / Graph BPE (new, data-driven)
2. **M2: Patch Tokenizer** — SimVQ (baseline) / RVQ 3-level (new, 10^9 capacity)
3. **M3: AR Generation** — GPT-2 ~50M params, Z-order patch ordering
4. **M4: Assembly & Stitching** — StitchingMLP + boundary vertex merging

## v2 Ablation Matrix (2x2)
| Config | Partition | Tokenizer |
|--------|-----------|-----------|
| C1 | METIS | SimVQ (baseline) |
| C2 | METIS | RVQ |
| C3 | BPE | SimVQ |
| C4 | BPE | RVQ (full v2) |

## v2 Phased Execution (~120h GPU)
- Phase 0: BPE feasibility (~2h CPU) — Go/No-Go gate
- Phase 1: RVQ tokenizer training (~12h GPU)
- Phase 2: BPE partition + C3/C4 training (~20h GPU, conditional)
- Phase 3: AR generation training (~30h GPU)
- Phase 4: Stitching + full pipeline (~15h GPU)
- Phase 5: Ablation + visualization (~20h GPU)

## Research Evolution (HMT Proposal)
- After v1, explored Graph Tokenization (Guo et al., 2026) for data-driven partitioning
- Proposed MeshLex-HMT: Structure-Geometry Decoupling
  - Structural channel: Graph BPE on discretized face features
  - Geometry channel: GCN + SimVQ on continuous vertex coords
- HMT scored 7/10 in simulated NeurIPS review (Almost Ready)
- v2 implements the exploratory experiments to validate HMT direction

## Key Technical Details
- SimVQ: frozen codebook C + learnable linear W, prevents VQ collapse
- RVQ: 3-level residual quantization, K=1024 per level, joint training
- Graph BPE: bigram = (node_label, edge_label, node_label) on face-adjacency dual graph
- Face discretization: normals→64 icosphere bins, areas→8 log bins, dihedrals→16 angular bins
- Patch sequence: (pos_x, pos_y, pos_z, scale, tok_L1, tok_L2, tok_L3) per patch

## Hardware
- RunPod pod `ce7rhwk1yz3i9t`: RTX 4090 x1, 86GB RAM, 100GB disk, $0.59/h
- SSH: `ssh -p 33871 cc@149.36.1.86`

## Important Conventions
- Frequent commits + immediate push
- Checkpoints → HF after every training run (mandatory)
- Only keep latest 3 checkpoints on disk
- All training scripts must support --resume
- Preprocessing must support skip-if-exists

## File Locations
- Design spec: `docs/superpowers/specs/2026-03-18-meshlex-v2-design.md`
- Implementation plan: `docs/superpowers/plans/2026-03-18-meshlex-v2-implementation.md`
- Final report (v1): `context/22_final_report.md`
- HMT proposal: `context/24_meshlex_hmt_proposal.md`
- Gap analysis: `context/23_gap_analysis_graph_tokenization.md`
- Existing code: `src/model.py`, `src/patch_segment.py`, `src/patch_dataset.py`, `src/trainer.py`
- Checkpoints: `data/checkpoints/` (local) + HF `Pthahnix/MeshLex-Research`

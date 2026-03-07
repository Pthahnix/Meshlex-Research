# MeshLex Research Project

## Overview

研究课题：**MeshLex — Learning a Topology-aware Patch Vocabulary for Compositional Mesh Generation**

目标：提出首个 topology-aware mesh patch codebook，将 mesh 表示从 per-face token 提升到 per-patch token 层级，实现极端压缩（4000-face mesh → ~130 tokens）和高质量显式 mesh 生成。

目标投稿：CCF-A（CVPR / NeurIPS / ICCV）

## Project Structure

```
.context/                          # 研究上下文文档（按时间顺序）
├── 00_original_prompt.md          # 原始 LMM 研究构想
├── 01_gap_analysis_lmm.md         # 75+ 篇论文的 Gap Analysis，识别 7 个研究空白
├── 02_idea_generation_lmm.md      # 5 个候选 idea，筛选出 MeshFoundation/MeshCascade/MeshSSM
├── 03_experiment_design_lmm.md    # MeshFoundation v2 完整实验设计
├── 04_pplx_comprehensive_evaluation.md  # Perplexity 独立评审（Gap 88%/Idea 78/Exp 82）
├── 05_cc_pplx_debate.md           # CC × Perplexity 深度辩论 → 转向 MeshLex 方向
├── 06_plan_meshlex_validation.md  # [WIP] MeshLex 可行性验证实验 plan
├── material/                      # 10 篇核心论文的分析摘要
└── paper/                         # 300+ 篇论文的 markdown 原文
```

## Research Evolution

1. **LMM 构想** → Gap Analysis 识别 7 个空白 → 推荐 MeshFoundation（统一重建+生成）
2. **外部评审** → FACE (2026.03) 等竞品使 MeshFoundation 差异化不足
3. **范式跳转** → 从"怎么更好地序列化 mesh"跳到"该不该序列化" → 提出 4 个非序列化想法
4. **竞品验证** → 想法 1/2/4 有严重竞品（SpaceMesh, VertexRegen, DMesh++）
5. **MeshLex 确定** → 想法 3（Mesh Vocabulary）零直接竞品，双方一致选定

## Key Decisions

- **核心假设**：mesh 的局部拓扑结构存在 universal vocabulary（类似 BPE 词汇表）
- **验证优先**：必须先通过 2-3 天的可行性验证实验才能继续
- **成功标准**：跨类别重建 CD / 同类别 CD < 1.2× 为强成功，> 3.0× 为失败止损
- **命名策略**：避开 "BPE for Mesh"（被 FreeMesh ICML 2025 占用），使用 "MeshLex"
- **差异化定位**：vs MeshMosaic（我们是 codebook 选取，不是逐 face 生成）；vs FACE（我们是 per-patch，不是 per-face）

## Current Status

**当前阶段**：验证实验计划完成（06 研究计划 + 07 代码级实施计划），待用户确认后开始实施。

## Conventions

- 文档语言：中英混合（技术术语英文，说明中文）
- 文档编号：两位数 ID，按时间顺序递增
- 所有研究文档保存在 `.context/` 下

## Git Workflow — 重要

- **尽可能频繁 commit**：完成任何一小部分内容、一段测试、一个实验结果，都要立即 commit
- **不要 push**：push 由用户亲自操作，Claude 不执行 `git push`
- commit 粒度越细越好：一个函数写完 commit、一个测试通过 commit、一个可视化生成 commit

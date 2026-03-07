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

**当前阶段**：验证实验代码实施中。Task 1-3 已完成（环境搭建、数据预处理、Patch 分割），从 Task 4 继续。

## Conventions

- 文档语言：中英混合（技术术语英文，说明中文）
- 文档编号：两位数 ID，按时间顺序递增
- 所有研究文档保存在 `.context/` 下

## Git Workflow — 重要

- **频繁 commit**：完成一个完整功能模块（多个函数组成的功能）、通过一组相关测试、完成一个实验阶段，就立即 commit
- **commit 后立即 push**：每次 commit 完成后执行 `git push`
- commit 粒度：以"功能"为单位，而非单个函数。例如：patch 分割模块写完+测试通过 = 一次 commit

## 验证要求 — 重要

- **真实数据验证**：每个 Task 完成后，必须用真实数据（如 ShapeNet mesh）实际运行，产生用户可亲眼查看的结果
- **可见产出**：结果需通过以下方式呈现：markdown 文档讲解 + 完整 log 日志 + matplotlib 可视化图 + mesh obj 文件等
- **Mesh 预览图**：每次用真实 mesh 测试时，必须渲染该 mesh 的预览图像（多角度或单张），保存为 PNG，让用户直观看到模型外观
- **结果保存**：所有验证产出保存到 `results/` 文件夹
- **内存安全**：使用真实数据测试时，只下载/使用少量数据（如 10 个 mesh），避免 OOM 崩溃
- 单元测试仍需通过，但不能作为唯一验证手段

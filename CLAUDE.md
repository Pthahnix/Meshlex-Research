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
├── 06_plan_meshlex_validation.md  # MeshLex 可行性验证实验 plan
├── 07_impl_plan_meshlex_validation.md  # 14-Task 实现计划（已完成）
├── 08_experiment_execution_design.md  # Phase A+B 实验执行设计
├── 09_phase_ab_execution_plan.md      # [legacy] Phase A+B ShapeNet 实施计划（6 Task）
├── 10_objaverse_migration_design.md   # Objaverse 迁移 + 双实验设计
├── 11_objaverse_experiment_plan.md    # Objaverse 双实验实施计划（12 Tasks）
├── 12_codebook_collapse_diagnosis.md  # Codebook Collapse 诊断分析与修复建议
├── material/                      # 10 篇核心论文的分析摘要
└── paper/                         # 300+ 篇论文的 markdown 原文

src/                               # 核心代码
├── data_prep.py                   # Mesh 加载、降面、归一化
├── patch_segment.py               # METIS Patch 分割 + PCA 归一化
├── patch_dataset.py               # NPZ 序列化 + PyTorch/PyG Dataset
├── model.py                       # PatchEncoder, SimVQCodebook, PatchDecoder, MeshLexVQVAE
├── losses.py                      # Chamfer Distance loss
├── trainer.py                     # Training loop
└── evaluate.py                    # Evaluation metrics + Go/No-Go

scripts/                           # 运行脚本
├── download_objaverse.py          # 从 Objaverse-LVIS 下载（5cat / lvis_wide 模式）
├── download_shapenet.py           # [legacy] 从 HuggingFace 下载 ShapeNet
├── train.py                       # 训练入口（支持 --resume）
├── evaluate.py                    # 评估入口
├── visualize.py                   # 可视化（t-SNE, utilization, curves）
├── init_codebook.py               # K-means codebook 初始化
├── run_preprocessing.py           # 批量预处理（支持 manifest JSON 输入 + ShapeNet 目录）
└── validate_task*.py              # 各 Task 验证脚本

tests/                             # 17 unit tests
├── test_data_prep.py              # 2 tests
├── test_patch_segment.py          # 4 tests
├── test_patch_dataset.py          # 3 tests
└── test_model.py                  # 8 tests

results/                           # 验证产出（commit 到 repo）
├── task1_3_validation/            # 数据预处理 + Patch 分割验证
├── task4_validation/              # Dataset 序列化验证
├── task5_7_validation/            # Encoder/Codebook/Decoder 验证
├── task8_10_validation/           # VQ-VAE + Training 验证
├── task12_validation/             # Visualization 验证
├── task13_validation/             # K-means init 验证
└── exp1_v2_collapse_fix/          # Exp1 v2 训练报告 + 模型参数文档
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

**当前阶段**：**可行性验证全部完成** — 4/4 STRONG GO，准备进入正式实验设计。

- 代码实现：全部完成（src/ + scripts/ + tests/，21 tests passing）
- 数据源：Objaverse-LVIS（46K objects, 1156 categories，无需审批）
- 实验设计：四实验矩阵（A/B stage × 5cat/LVIS-Wide）— **全部完成**
  - **Exp1**: A-stage × 5cat — **STRONG GO** (ratio 1.145x, util 46.0%)
  - **Exp2**: A-stage × LVIS-Wide — **STRONG GO** (ratio 1.019x, util 95.3%)
  - **Exp3**: B-stage × 5cat — **STRONG GO** (ratio 1.185x, util 47.1%)
  - **Exp4**: B-stage × LVIS-Wide — **STRONG GO** (ratio 1.019x, util 94.9%)
- **关键发现**：
  - SimVQ collapse fix 成功（util 0.46% → 99%+）
  - B-stage multi-token KV decoder 有效（CD -6.2%），但 rotation trick 与 SimVQ 不兼容
  - 跨阶段 resume（A→B stage）需 strict=False 加载
  - **更多类别 = 更好泛化**: LVIS-Wide ratio 1.019x 远优于 5-cat 1.145x, util 95% vs 46%
  - **最佳结果**: Exp4 (B×LVIS) same-cat CD 211.6, cross-cat CD 215.8, 几乎无泛化损失
- HF Checkpoints (Pthahnix/MeshLex-Research):
  - `checkpoints/exp1_A_5cat/` — Exp1 final checkpoint + history
  - `checkpoints/exp2_A_lvis_wide/` — Exp2 final checkpoint + history (上次版本；重训版本待上传)
  - `checkpoints/exp3_B_5cat/` — Exp3 final checkpoint + history
  - `checkpoints/exp4_B_lvis_wide/` — 待上传
- Local Checkpoints:
  - Exp1: `data/checkpoints/5cat_v2/checkpoint_final.pt`
  - Exp2: `data/checkpoints/lvis_wide_A/checkpoint_final.pt`
  - Exp3: `data/checkpoints/5cat_B/checkpoint_final.pt`
  - Exp4: `data/checkpoints/lvis_wide_B/checkpoint_final.pt`
- 最终报告: `results/final_comparison/report.md` + 5 张对比可视化图
- **下一步**: 上传 Exp2/Exp4 checkpoint 到 HF → 设计正式实验 → 论文撰写

## Conventions

- 文档语言：中英混合（技术术语英文，说明中文）
- 文档编号：两位数 ID，按时间顺序递增
- 所有研究文档保存在 `context/` 下

## Git Workflow — 重要

- **频繁 commit**：完成一个完整功能模块（多个函数组成的功能）、通过一组相关测试、完成一个实验阶段，就立即 commit
- **commit 后立即 push**：每次 commit 完成后执行 `git push`
- commit 粒度：以"功能"为单位，而非单个函数。例如：patch 分割模块写完+测试通过 = 一次 commit

## Checkpoint 备份规范 — 重要

**每次训练完成后，必须立即将 checkpoint 上传至 HuggingFace Model Repo：`Pthahnix/MeshLex-Research`。这是强制要求，不得跳过。**

Checkpoint 文件只存在于 RunPod 本地磁盘，pod 重置后将永久丢失。HuggingFace 是唯一的持久化备份。

### 上传步骤

```bash
# 安装（如未安装）
pip install huggingface_hub

# 上传 checkpoint（训练结束后立即执行）
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()

# 上传 final checkpoint
api.upload_file(
    path_or_fileobj="data/checkpoints/<exp_name>/checkpoint_final.pt",
    path_in_repo="checkpoints/<exp_name>/checkpoint_final.pt",
    repo_id="Pthahnix/MeshLex-Research",
    repo_type="model",
)
print("Upload complete.")
EOF
```

### 命名规范

| 实验 | `<exp_name>` |
|------|-------------|
| Exp1 A-stage 5cat | `exp1_A_5cat` |
| Exp2 A-stage LVIS-Wide | `exp2_A_lvis_wide` |
| Exp3 B-stage 5cat | `exp3_B_5cat` |
| Exp4 B-stage LVIS-Wide | `exp4_B_lvis_wide` |
| 后续实验 | `exp{N}_{stage}_{data}` |

### 上传内容

- **必传**：`checkpoint_final.pt`（训练结束后的最终模型权重）
- **必传**：`training_history.json`（完整训练曲线，用于验证训练过程真实性）
- **可选**：`checkpoint_epoch{N}.pt`（关键 epoch 的中间 checkpoint，如 epoch 100）

### 验证上传成功

上传完成后，必须在终端输出确认信息，并在该实验的 progress/report markdown 中记录：

```
✅ Checkpoint uploaded to HF: Pthahnix/MeshLex-Research/checkpoints/<exp_name>/checkpoint_final.pt
```

## 验证要求 — 重要

- **真实数据验证**：每个 Task 完成后，必须用真实数据（如 ShapeNet mesh）实际运行，产生用户可亲眼查看的结果
- **可见产出**：结果需通过以下方式呼现：markdown 文档讲解 + 完整 log 日志 + matplotlib 可视化图 + mesh obj 文件等
- **Mesh 预览图**：每次用真实 mesh 测试时，必须渲染该 mesh 的预览图像（多角度或单张），保存为 PNG，让用户直观看到模型外观
- **结果保存**：所有验证产出保存到 `results/` 文件夹
- **内存安全**：使用真实数据测试时，只下载/使用少量数据（如 10 个 mesh），避免 OOM 崩溃
- 单元测试仍需通过，但不能作为唯一验证手段

## 硬件环境

| 资源 | 规格 |
|------|------|
| GPU | RTX 4090 × 1 |
| vCPU | 16 核 |
| Memory | 62 GB |
| Container Disk | 80 GB |

## 资源管理规范 — 重要

### 实验前强制检查

每次开始大规模实验（下载数据、批量预处理、训练）之前，**必须先执行以下检查**，任一项不满足则暂停并报告：

```bash
# 磁盘检查（可用空间需 > 预估用量的 1.5 倍）
df -h /

# 内存检查
free -h

# GPU 显存检查
nvidia-smi
```

**磁盘红线**：Container disk 总量 80 GB，任何时候 `data/` 目录不得超过 **50 GB**，`results/` 不得超过 **5 GB**，留 25 GB 给系统和代码。

**内存红线**：62 GB 总内存，单个进程不得超过 **40 GB**，留余量给系统和其他进程。

### 内存使用规范

- **禁止**一次性将整个数据集加载进内存；必须用 DataLoader / generator 分批处理
- 训练循环中，每个 epoch 结束后执行：
  ```python
  torch.cuda.empty_cache()
  ```
- 大型中间变量（如全量 embeddings）用完后立即 `del` 并调用 `gc.collect()`
- 预处理脚本处理 GLB 文件时，**逐文件**加载，不得批量持有超过 **100 个** mesh 对象

### 磁盘使用规范

- **只保留最新 3 个 checkpoint**，旧的立即删除：
  ```bash
  ls -t data/checkpoints/*/checkpoint_epoch*.pt | tail -n +4 | xargs rm -f
  ```

### 崩溃预防

- 训练脚本必须支持 `--resume`，确保 OOM 后可以从上一个 checkpoint 继续，**不得从头重跑**
- 预处理脚本必须支持断点续跑（跳过已处理的文件），检测方式：
  ```python
  if output_path.exists():
      continue  # 已处理，跳过
  ```
- 遇到 OOM 错误时，**优先降低 `--batch_size`**，不要直接加内存

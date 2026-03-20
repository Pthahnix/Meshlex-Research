# MeshLex Research Project

## Overview

研究课题：**MeshLex — Learning a Topology-aware Patch Vocabulary for Compositional Mesh Generation**

目标：提出首个 topology-aware mesh patch codebook，将 mesh 表示从 per-face token 提升到 per-patch token 层级，实现极端压缩（4000-face mesh → ~130 tokens）和高质量显式 mesh 生成。

目标投稿：CCF-A（CVPR / NeurIPS / ICCV）

## Project Structure

```
context/                           # 研究上下文文档（按时间顺序）
├── 00_original_prompt.md          # 原始 LMM 研究构想
├── ...                            # 01-12: Gap Analysis → 实验设计 → Collapse 诊断
├── 22_final_report.md             # v1 最终报告
├── 23_gap_analysis_graph_tokenization.md  # Graph Tokenization 分析
├── 24_meshlex_hmt_proposal.md     # HMT 提案
├── material/                      # 10 篇核心论文的分析摘要
└── paper/                         # 300+ 篇论文的 markdown 原文

docs/superpowers/
├── specs/
│   ├── 2026-03-18-meshlex-v2-design.md            # v2 完整设计文档
│   ├── 2026-03-19-assembly-fix-full-retrain-design.md  # Assembly fix + full retrain spec
│   └── 2026-03-20-daft-dataset-pipeline-design.md # Daft dataset pipeline spec
└── plans/
    ├── 2026-03-18-meshlex-v2-implementation.md    # v2 13-task 实现计划
    ├── 2026-03-19-ar-loss-fix-implementation.md   # AR v2 fix plan (7 tasks)
    ├── 2026-03-19-dataset-streaming-pipeline.md   # Dataset pipeline plan (superseded)
    └── 2026-03-20-daft-dataset-pipeline.md        # Daft dataset pipeline plan (active)

src/                               # 核心代码
├── data_prep.py                   # Mesh 加载、降面、归一化
├── patch_segment.py               # METIS Patch 分割 + PCA 归一化 + dual normalization (PCA + noPCA)
├── patch_dataset.py               # NPZ 序列化 + PyTorch/PyG Dataset
├── patch_sequence.py              # Token sequence 编解码 (RVQ 7-token format)
├── daft_utils.py                  # Daft DataFrame utilities (row conversion, schema casting, HF config)
├── stream_utils.py                # Streaming pipeline helpers (ProgressTracker, MetadataCollector, synset map)
├── model.py                       # PatchEncoder, SimVQCodebook, PatchDecoder, MeshLexVQVAE (v1)
├── model_rvq.py                   # MeshLexRVQVAE (v2, 3-level RVQ)
├── rvq.py                         # ResidualVQ (3-level SimVQ)
├── ar_model.py                    # PatchGPT (AR transformer for token generation)
├── stitching.py                   # StitchingMLP + boundary vertex merging
├── metrics.py                     # NC, F-Score, non-manifold counts
├── losses.py                      # Chamfer Distance loss
├── trainer.py                     # Training loop (supports RVQ + dead code revival)
├── evaluate.py                    # Evaluation metrics + Go/No-Go
├── discretize.py                  # Face feature discretization (for BPE)
├── dual_graph.py                  # Face-adjacency dual graph construction
└── graph_bpe.py                   # Graph BPE vocabulary learning

scripts/                           # 运行脚本
├── train.py                       # v1 训练入口
├── train_rvq.py                   # RVQ VQ-VAE 训练
├── train_ar.py                    # AR v2 训练 (20.4M params, grad accum, warmup)
├── encode_sequences.py            # Patch → token sequence 编码
├── generate.py                    # 基础生成脚本
├── generate_v2_pipeline.py        # 完整生成 pipeline + 7 阶段可视化
├── visualize_mesh_comparison.py   # 原始 vs 重建 mesh 对比 + AR 生成 mesh 可视化
├── evaluate_generation.py         # 生成质量评估 (CD, token distribution, etc.)
├── stream_objaverse_daft.py       # Objaverse-LVIS streaming → Daft → HF Parquet
├── stream_shapenet_daft.py        # ShapeNetCore v2 streaming → Daft → HF Parquet
├── generate_splits_daft.py        # Generate train/test/unseen splits
├── validate_dataset_daft.py       # Validate HF dataset thresholds
├── run_dataset_pipeline.sh        # Overnight dataset pipeline orchestrator
├── run_phase0_bpe.py              # Phase 0 BPE 可行性验证
├── run_preprocessing.py           # 批量预处理
├── download_objaverse.py          # Objaverse-LVIS 下载
└── ...                            # 其他辅助脚本

tests/                             # Unit tests
├── test_data_prep.py
├── test_patch_segment.py
├── test_patch_dataset.py
├── test_daft_utils.py              # Daft row conversion + schema tests
├── test_stream_utils.py            # Stream helpers tests
├── test_generate_splits.py         # Split generation logic tests
├── test_model.py
├── test_rvq.py
├── test_ar_model.py
├── test_train_ar.py               # AR v2: grad accum, warmup, scheduler resume
├── test_metrics.py
├── test_stitching.py
├── test_discretize.py
├── test_dual_graph.py
└── test_graph_bpe.py

data/                              # 数据 + checkpoints (gitignored)
├── meshes/lvis_wide/              # 原始 OBJ meshes
├── patches/lvis_wide/             # METIS 分割后的 patch NPZ
├── sequences/rvq_lvis/            # 编码后的 token sequences (4674 meshes)
└── checkpoints/
    ├── rvq_lvis/                  # RVQ VQ-VAE checkpoint
    └── ar_v2/                     # AR v2 checkpoint

results/                           # 实验结果 (committed)
├── phase0/                        # BPE 可行性报告
├── rvq_training/                  # RVQ 训练曲线 + 报告
├── ar_training/                   # AR v1 分析
├── ar_v2_training/                # AR v2 训练曲线 + 报告
├── generation_v2_pipeline/        # 40 meshes × 7 可视化
├── generation_v2_eval/            # Evaluation dashboard
├── mesh_comparison/               # 原始 vs 重建 + AR 生成 mesh
└── ...                            # v1 验证结果
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

## Current Status (2026-03-20)

### v1 可行性验证 — COMPLETE (4/4 STRONG GO)

- 四实验矩阵（A/B stage × 5cat/LVIS-Wide）全部通过
- **最佳结果**: Exp4 (B×LVIS) same-cat CD 211.6, cross-cat CD 215.8, ratio 1.019x
- **关键发现**: 更多类别 = 更好泛化 (LVIS 1.019x vs 5-cat 1.145x)

### v2 实现 — IN PROGRESS

**已完成的 Phase:**

| Phase | 内容 | 状态 | 关键结果 |
|-------|------|------|----------|
| Phase 0 | BPE 可行性 | COMPLETE (GO) | H1a + H5 通过，但 patch size 分布差 (median=1) |
| Phase 1 | RVQ 训练 | COMPLETE | 200 epochs, loss 0.177, util 100% |
| Phase 3 | AR 训练 | COMPLETE (v2) | v1: loss 5.41 (87.3M params, 太大) → v2: loss 1.48, ppl 4.4 (20.4M params) |
| Phase 4 | Generation Pipeline | COMPLETE | 40 meshes generated, surface recon via Ball Pivoting |

**当前进行中:**

| Phase | 内容 | 状态 | 备注 |
|-------|------|------|------|
| Phase D | 统一数据集 (Daft pipeline) | IN PROGRESS | Objaverse-LVIS 46K + ShapeNet 51K → HF Parquet |

**待完成:**

| Phase | 内容 | 状态 | 备注 |
|-------|------|------|------|
| Phase A | Assembly fix | PENDING | 修复 VQ-VAE 重建旋转 bug |
| Phase B | PCA + Rotation Tokens | PENDING | 需 Phase D 数据集 |
| Phase C | No-PCA baseline | PENDING | 需 Phase D 数据集 |
| Phase E | Ablation comparison | PENDING | 需 Phase B + C |

### v2 Checkpoints (HF: Pthahnix/MeshLex-Research)

| 模型 | 本地路径 | HF 路径 | 参数量 |
|------|----------|---------|--------|
| RVQ VQ-VAE | `data/checkpoints/rvq_lvis/checkpoint_final.pt` | `checkpoints/rvq_lvis/` | ~2M |
| AR v2 | `data/checkpoints/ar_v2/checkpoint_final.pt` | `checkpoints/ar_v2/` | 20.4M |

### HF Datasets

| Repo | 内容 | 格式 |
|------|------|------|
| `Pthahnix/MeshLex-Patches` | 统一数据集 (Objaverse-LVIS + ShapeNet) | Parquet (via Daft) |

### v2 Results

| 目录 | 内容 |
|------|------|
| `results/rvq_training/` | RVQ 训练曲线 + 报告 |
| `results/ar_training/` | AR v1 训练分析 (loss plateau 5.41) |
| `results/ar_v2_training/` | AR v2 训练曲线 + 报告 (loss 1.48) |
| `results/generation_v2_pipeline/` | 40 meshes × 7 可视化 (token heatmap, patch positions, point cloud, etc.) |
| `results/generation_v2_eval/` | Evaluation dashboard + 报告 |
| `results/mesh_comparison/reconstruction/` | 8 组原始 mesh vs VQ-VAE 重建对比 (Ball Pivoting surface recon) |
| `results/mesh_comparison/generation/` | 16 个 AR 生成 mesh (OBJ + PLY + PNG, T=0.8/1.0) |

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
| GPU | NVIDIA RTX A4000 × 1 (16 GB VRAM) |
| vCPU | 128 核 |
| Memory | 503 GB |
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

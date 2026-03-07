# MeshLex TODO

## 当前进度

### 已完成

- [x] 研究方向确定：MeshLex — topology-aware patch vocabulary
- [x] 可行性验证实验设计（`.context/06_plan_meshlex_validation.md`）
- [x] 14-Task 代码实现（`.context/07_impl_plan_meshlex_validation.md`）
  - src/: data_prep, patch_segment, patch_dataset, model, losses, trainer, evaluate
  - scripts/: train, evaluate, visualize, init_codebook, run_preprocessing, download_shapenet
  - tests/: 17 unit tests 全部通过
  - results/: task1-13 验证产出
- [x] Phase A+B 实验执行计划（`.context/08_experiment_execution_design.md`, `09_phase_ab_execution_plan.md`）
- [x] ShapeNet 下载脚本 `scripts/download_shapenet.py`
- [x] 预处理脚本修复：mesh_id 提取 + train/test split 功能
- [x] HuggingFace 登录（用户 Pthahnix）

### 阻塞中

- [ ] **ShapeNet/ShapeNetCore 数据集审批** — 已提交申请，等待 HuggingFace 审批
  - 审批页面：https://huggingface.co/datasets/ShapeNet/ShapeNetCore
  - 需要下载 5 个类别 zip（共 ~13.4GB）：chair, table, airplane, car, lamp

---

## 接下来要做的（按顺序）

### Phase A: 数据准备

1. **下载 ShapeNet**（审批通过后立即执行）
   ```bash
   python scripts/download_shapenet.py --output_root data/ShapeNetCore.v2
   ```
   验证：每个类别应有数百到数千个 OBJ 文件

2. **预处理全部 5 类别**（每类 500 mesh，降面 1000 faces → METIS patch 分割 → NPZ 序列化）
   ```bash
   python scripts/run_preprocessing.py \
       --shapenet_root data/ShapeNetCore.v2 \
       --output_root data \
       --target_faces 1000 \
       --max_per_category 500
   ```
   自动执行 train/test split（chair/table/airplane 各 400 train + 100 test）
   预计耗时：~1-2h

3. **验证 patch 统计** + 保存报告到 `results/phase_a_validation/`

### Phase B: 快速训练验证

4. **Encoder-Only 训练**（20 epochs, vq_start_epoch=999）
   ```bash
   python scripts/train.py \
       --train_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
       --val_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
       --epochs 20 --batch_size 256 --lr 1e-4 \
       --vq_start_epoch 999 --checkpoint_dir data/checkpoints
   ```
   预计耗时：~2h

5. **K-means Codebook 初始化**
   ```bash
   python scripts/init_codebook.py \
       --checkpoint data/checkpoints/checkpoint_epoch019.pt \
       --patch_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
       --codebook_size 4096 \
       --output data/checkpoints/checkpoint_kmeans_init.pt
   ```
   预计耗时：~10min

6. **Quick VQ-VAE 训练**（20 epochs, 从 K-means init 恢复）
   ```bash
   python scripts/train.py \
       --train_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
       --val_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
       --resume data/checkpoints/checkpoint_kmeans_init.pt \
       --epochs 20 --batch_size 256 --lr 1e-4 \
       --vq_start_epoch 0 --checkpoint_dir data/checkpoints_vq
   ```
   预计耗时：~1h

7. **快速评估 + 可视化**
   ```bash
   python scripts/evaluate.py \
       --checkpoint data/checkpoints_vq/checkpoint_final.pt \
       --same_cat_dirs data/patches/chair_test data/patches/table_test data/patches/airplane_test \
       --cross_cat_dirs data/patches/car data/patches/lamp \
       --output results/phase_b_quick_eval.json

   python scripts/visualize.py \
       --checkpoint data/checkpoints_vq/checkpoint_final.pt \
       --history data/checkpoints_vq/training_history.json \
       --patch_dirs data/patches/chair_train data/patches/table_train data/patches/airplane_train \
       --output_dir results/phase_b_validation
   ```
   关注指标：codebook utilization > 30%，recon_loss 收敛

### 决策点

- utilization > 30% 且 loss 收敛 → 继续 Phase C-G（200 epoch 全量训练 + Go/No-Go）
- codebook collapse 或 loss 不收敛 → debug

### Phase C-G: 全量训练 + 最终评估（Phase B 通过后）

8. 全量 VQ-VAE 训练（200 epochs）
9. 完整评估：same-cat CD + cross-cat CD + utilization
10. 可视化：t-SNE, utilization histogram, training curves
11. **Go/No-Go 决策**（参见 `RUN_GUIDE.md` Phase G）

---

## 关键文件速查

| 用途 | 文件 |
|------|------|
| 完整运行指南 | `RUN_GUIDE.md` |
| 实验设计 | `.context/06_plan_meshlex_validation.md` |
| 实施计划 | `.context/09_phase_ab_execution_plan.md` |
| 下载脚本 | `scripts/download_shapenet.py` |
| 预处理脚本 | `scripts/run_preprocessing.py` |
| 训练脚本 | `scripts/train.py` |
| 评估脚本 | `scripts/evaluate.py` |

## 环境信息

- GPU: RTX 4090 24GB
- RAM: 503GB
- Disk: ~57GB available
- Python 3.11, PyTorch 2.4.1+cu124, torch-geometric 2.7.0
- HuggingFace: logged in as Pthahnix

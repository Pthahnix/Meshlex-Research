# MeshLex 项目规则

## Git 工作流

- 完成一个完整功能模块就立即 commit
- commit 后立即 `git push`
- commit 粒度以"功能"为单位，非单个函数
- 新增 `results/` 下的产出后，验证无误即立即将对应结果与相关脚本一起 commit 并 `git push`

## Checkpoint 备份（强制）

- 每次训练完成后，必须立即上传 checkpoint 至 HF：`Pthahnix/MeshLex-Research`
- 必传：`checkpoint_final.pt` + `training_history.json`
- 可选：关键 epoch 的中间 checkpoint
- 上传后必须输出确认信息并在报告中记录
- 命名规范：`exp{N}_{stage}_{data}`

## 验证要求

- 每个 Task 完成后必须用真实数据运行，产生可见产出
- 可见产出包括：markdown 文档 + 完整 log + matplotlib 可视化 + mesh obj 文件
- 每次用真实 mesh 测试时必须渲染预览图（PNG）
- 所有验证产出保存到 `results/`
- 真实数据测试时只用少量数据（如 10 个 mesh），避免 OOM

## 资源管理

### 实验前强制检查

每次大规模实验前必须执行：

```bash
df -h /        # 磁盘
free -h        # 内存
nvidia-smi     # GPU
```

任一项不满足则暂停并报告。

### 磁盘红线

- `data/` 不得超过 50 GB
- `results/` 不得超过 5 GB
- 留 25 GB 给系统和代码

### 内存红线

- 单个进程不得超过 40 GB
- 禁止一次性加载整个数据集，必须用 DataLoader / generator 分批
- 每个 epoch 结束后 `torch.cuda.empty_cache()`
- 大型中间变量用完后 `del` + `gc.collect()`
- 预处理时逐文件加载，不得批量持有超过 100 个 mesh 对象

### Checkpoint 磁盘管理

- 只保留最新 3 个 checkpoint，旧的立即删除

### GPU 利用率最大化

当有空闲 GPU 或 GPU 利用率长期 < 30% 时，必须采取以下措施：

1. **多任务并行**：同一 GPU 可同时运行多个轻量训练任务（如多个 VQ-VAE 变体各占 ~4GB VRAM）。不同 GPU 运行不同实验变体。
2. **Batch Size 调优**：调整 batch_size 使 VRAM 占用达到可用显存的 60-80%。单任务 VRAM 占用 < 30% 时，优先考虑多任务并行而非增大 batch_size（因小模型增大 batch 不提升吞吐）。
3. **并行编码/推理**：编码（encode_sequences）、评估等非训练任务，可与训练同时运行在同一 GPU 上。
4. **监控标准**：每次启动训练后 10 分钟内检查 `nvidia-smi`，若 GPU utilization < 20% 且 memory < 50%，考虑追加并行任务。
5. **超参配置**：当 GPU 空间允许时，优先增大 hidden_dim / n_layers 提升模型容量（尤其 AR 模型），而非仅增大 batch_size。

### 崩溃预防

- 训练脚本必须支持 `--resume`，OOM 后从上一个 checkpoint 继续，不得从头重跑
- 预处理脚本必须支持断点续跑（跳过已处理文件）
- OOM 时优先降低 `--batch_size`

## 数据集 Pipeline 监控 — 重要

### 监控脚本

| 脚本 | 功能 | 触发频率 |
|------|------|----------|
| `scripts/disk_alert.py` | 磁盘使用率 ≥ 80% 时发警告 | 每 15 分钟 |
| `scripts/monitor_pipeline.py` | 生成进度报告到 `results/dataset-pipeline/` | 每 15 分钟 |
| `scripts/monitor_daemon.sh` | 后台守护进程运行上述两个脚本 | 持续运行 |

### 启动监控

```bash
export HF_TOKEN=your_token_here
nohup /workspace/MeshLex-Research/scripts/monitor_daemon.sh > /tmp/monitor_daemon.log 2>&1 &
```

### 监控产出目录

所有监控产出保存在 `results/dataset-pipeline/`：
- `REPORT.md` — 人类可读的进度报告
- `progress.png` — 进度可视化
- `summary.json` — 机器可读的统计
- `sample_*.png` — 随机采样 mesh 可视化

### 磁盘预警响应

当监控脚本发出磁盘 ≥ 80% 警告时，立即执行：

```bash
# 检查磁盘占用
df -h /
du -sh /home/cc/.objaverse 2>/dev/null

# 清理旧的 Objaverse GLB 缓存（保留 30 分钟内的）
python3 - <<'PY'
from pathlib import Path
import time
cache_dir = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"
if cache_dir.exists():
    now = time.time()
    deleted = 0
    freed = 0
    for f in cache_dir.rglob("*.glb"):
        if now - f.stat().st_mtime > 1800:  # 30 min
            freed += f.stat().st_size
            f.unlink()
            deleted += 1
    print(f"Deleted {deleted} files, freed {freed/1e9:.1f} GB")
PY
```

### Batch 完成后的记录

每当一个 batch 处理完成后，必须：
1. 检查 `results/dataset-pipeline/` 的监控产出是否更新
2. 确认 REPORT.md 中的进度信息准确
3. 将监控产出 commit 并 push 到 GitHub

### 产出立即提交

**严格规则**：一旦有新产出（监控报告、可视化、日志等），立即 commit and push：

```bash
git add results/dataset-pipeline/
git commit -m "chore: update pipeline monitoring after batch_XXX"
git push
```

### 任务完成记录

当一个主要 phase 完成后（如 Objaverse phase），必须：
1. 在 `README.md` 的 Timeline 中添加当日条目
2. 格式：`- **Day N (YYYY-MM-DD)**: 完成内容摘要`

---

## Checkpoint 保存机制（训练脚本设计规范）

### 基本要求

所有训练脚本必须实现以下 checkpoint 机制，缺一不可：

**1. Sliding Window 式周期保存**

- 每隔固定 epoch 数（推荐 `--save_every N`，默认 3）自动保存一次
- 只保留最新的 `--keep_checkpoints K` 个存档点（默认 3），旧的自动删除
- 文件命名：`checkpoint_epoch{N:03d}.pt`
- 删除逻辑：每次保存后，按修改时间排序，删除最旧的超出数量的文件

示例实现：
```python
def save_checkpoint(self, epoch):
    path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
    torch.save({"epoch": epoch, "model_state_dict": ..., "optimizer_state_dict": ..., "history": ...}, path)
    # Sliding window 清理
    ckpts = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-self.keep_checkpoints]:
        old.unlink()
```

**2. 手动触发保存（信号机制）**

训练脚本必须注册 `SIGUSR1` 信号处理器，收到信号后在当前 batch 结束时立即保存 checkpoint：

```python
import signal

def _setup_signal_handler(self):
    self._save_requested = False
    signal.signal(signal.SIGUSR1, lambda sig, frame: setattr(self, '_save_requested', True))

# 在每个 batch 结束后检查（或每个 epoch 末）：
if self._save_requested:
    self.save_checkpoint(epoch)
    self._save_requested = False
    print(f"[SIGUSR1] Manual checkpoint saved at epoch {epoch}")
```

触发方式：
```bash
kill -USR1 <训练进程PID>
```

**3. Stop Flag 优雅退出**

支持 `--stop_flag_file` 参数。训练脚本在每个 epoch 末检查该文件是否存在，若存在则保存 checkpoint 后退出，以便 resume：

```bash
touch /tmp/stop_rvq_pca    # 触发优雅停止
# 重启时：
python3 scripts/train_xxx.py --resume checkpoints/xxx/checkpoint_epochXXX.pt ...
```

**注意**：新训练任务启动时，**必须**在命令行中加上 `--stop_flag_file /tmp/stop_<exp_name>`。

---

## 定时进度检测规范

### 检测频率

每 **30 分钟**进行一次进度检测（通过 cron 或 watch daemon 触发）。

### 检测内容（每次必查）

每次定时检测必须包含以下全部项目：

```
1. 进程存活性
   - 确认训练 PID 全部存活（ps -p <PID>）
   - 区分主进程 vs multiprocessing spawn 子进程（看 PPID）
   - 若主进程已死：立即检查 checkpoint_final.pt 是否存在

2. 训练进度
   - 当前 epoch / 总 epoch
   - 当前 epoch 内 chunk 进度（如 chunk 20/43）
   - 最新 loss 值

3. Checkpoint 状态（重要）
   - 列出 checkpoint 目录下所有 .pt 文件及其修改时间
   - 计算距上次 checkpoint 已过多少时间
   - 若超过 save_every × epoch_duration × 1.5 倍时间没有新 checkpoint，发出警告

4. GPU 状态
   - 利用率（%）和显存占用（MB）
   - 功耗（W）是判断是否在真实计算的最可靠指标

5. Watch daemon 状态
   - 确认 watch_and_encode.py 进程存活
   - 查看最新 log 确认 trained/encoded 计数
```

### 检测报告格式

每次检测输出必须以当前时间开头（北京时间 CST，UTC+8）：

```
【YYYY-MM-DD HH:MM CST 进度检测】
PID存活: ✅/❌  |  训练进度: X/4 完成
[实验名] Epoch X/15, chunk Y/43, loss Z.ZZZZ
[实验名] Epoch X/15 完成, loss Z.ZZZZ
Checkpoint: <最新存档 epoch> @ <时间> | 距今 Xh
GPU: GPU1 X% / XW, GPU2 X% / XW
```

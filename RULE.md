# MeshLex 项目规则

## 网络代理（默认开启）

pthahnix 用户配置了 mihomo (Clash Meta) 代理，所有需要外网访问的操作（git、pip、HF 下载等）**默认开启代理**：

```bash
source ~/.bashrc && export http_proxy=http://127.0.0.1:17892 https_proxy=http://127.0.0.1:17892
```

- 启动 mihomo：`~/.mihomo/start.sh`
- 停止 mihomo：`~/.mihomo/stop.sh`
- 快捷开关：`proxy_on` / `proxy_off`（已配置在 `~/.bashrc`）
- HF Token 如有需要，从 `/home/pthahnix/MeshLex-Research/.env` 中获取

## Git 工作流

- 完成一个完整功能模块就立即 commit
- commit 后立即 `git push`
- commit 粒度以"功能"为单位，非单个函数
- 新增 `results/` 下的产出后，验证无误即立即将对应结果与相关脚本一起 commit 并 `git push`

## Timeline 维护

**每当完成一批任务（≥ 2 个 Task 或一个明显的工作节点）时：**

1. 用 Python 查询当前设备时间：
   ```python
   from datetime import datetime; print(datetime.now())
   ```
2. 若距上次 Timeline 条目已跨越新的一天（UTC 日期变化），必须在 `main` 分支的 `README.md` 中追加当日条目：
   ```
   - **Day N (YYYY-MM-DD)**: 当日完成内容摘要（bullet 形式，覆盖关键实验结果、新增功能、重要决策）
   ```
3. 条目写完后立即 commit 并 push 到 main：
   ```bash
   git add README.md
   git commit -m "docs: update README timeline Day N"
   git push
   ```

> **提示**：README.md 在 main 分支，当前工作分支不同时，用 `git worktree add /tmp/meshlex-main main` 在独立目录操作，避免切换分支影响正在运行的训练进程。

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

## 自有服务器额外规范

当项目运行在实验室共享服务器（非 RunPod 等云实例）上时，必须额外遵循以下规则。

**GPU 利用率最大化**：所有 GPU 任务必须参考 `GPU.md`，按其中的 Checklist 和配置指南执行。

### GPU 使用

- 使用 GPU 前先 `nvidia-smi` 查看占用情况，避免抢占他人资源
- 长时间训练任务使用 `tmux` 或 `screen`，防止断连中断
- 训练结束后及时释放显存，不要让空闲进程长期占用 GPU
- 需要长时间占用多张卡时，提前沟通协调

### GPU 自动排队与抢占

当有明确规划好的 GPU 任务在等待执行时：

- **每 10 分钟自动检查一次** 各 GPU 的使用情况（`nvidia-smi`）
- 一旦有 GPU 空闲（无进程占用），**立即启动排队中的任务**
- 使用 `CUDA_VISIBLE_DEVICES=<N>` 指定具体卡（0、1 或 2，视空闲情况）
- 多张卡都空闲时，优先使用编号较小的空闲卡

### GPU 进程命名

所有我们的 GPU 任务必须使用 `setproctitle` 设置进程名，格式为 `Pthahnix-<Task>`，确保在 `nvidia-smi` 中可识别：

```python
from setproctitle import setproctitle
setproctitle("Pthahnix-<Task>")
```

示例命名：
- RVQ VQ-VAE 训练 → `Pthahnix-RVQ-Train`
- AR Transformer 训练 → `Pthahnix-AR-Train`
- Sequence 编码 → `Pthahnix-Encode-Seq`
- 生成 Pipeline → `Pthahnix-Generate`

### GPU 优先级制度

服务器配备 3× RTX 5090，实行三级优先级：

| 等级 | 名称 | 最多可用 GPU | 适用场景 |
|------|------|-------------|----------|
| P0 | 日常 (Normal) | 1 张 | 日常调试、小实验、推理测试 |
| P1 | 加速 (Boost) | 2 张 | 项目冲刺、中期实验、模型训练 |
| P2 | 紧急 (Urgent) | 3 张（全部） | 论文 deadline、紧急实验补充 |

- 默认 P0，升级到 P1/P2 需告知原因和预计时长
- 高优先级可要求低优先级让出资源（协商解决）
- P2 有时间限制，紧急任务完成后立即降回 P0

### 磁盘空间管理

- **严禁在 `/home/` 下存放大文件（数据集、模型权重、日志等）**
- 所有大体积数据必须存放在 `/data/` 目录下，按用户名建子目录（如 `/data/username/`）
- 定期清理不再使用的数据、checkpoint 和临时文件
- 通过 `du -sh ~/*` 和 `df -h` 检查空间占用

### Conda 环境管理

- 使用 conda 创建独立环境，不要污染 base 环境
- 环境命名带用户名前缀（如 `username_project`）
- 不再使用的环境及时删除：`conda env remove -n env_name`
- 安装大型包前确认磁盘空间充足

### 公共区域维护

- 公共目录（如 `/data/shared/`）中的文件未经允许不要删除或修改
- 不要修改系统级配置文件和其他用户的文件
- 发现服务器异常（高负载、磁盘满、服务挂掉等）及时反馈

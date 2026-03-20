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

### 崩溃预防

- 训练脚本必须支持 `--resume`，OOM 后从上一个 checkpoint 继续，不得从头重跑
- 预处理脚本必须支持断点续跑（跳过已处理文件）
- OOM 时优先降低 `--batch_size`

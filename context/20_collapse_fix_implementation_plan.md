# Codebook Collapse 修复 — A 阶段实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 SimVQ 实现 bug + 简化训练流程，解决 codebook collapse（0.46% → >30% utilization），重跑 5-Category 实验 200 epoch。

**Architecture:** 重写 SimVQCodebook 与 SimVQ 官方对齐（冻结 C，计算 CW，从 CW 取值），删除分阶段训练和 K-means 初始化，从 epoch 0 端到端训练。添加 LR warmup 和 dead code revival。

**Tech Stack:** PyTorch, torch-geometric, objaverse, sklearn (t-SNE)

**设计文档:** `context/19_codebook_collapse_fix_design.md`

**工作目录:** `/home/cc/Meshlex-Research`

---

## Task 1: 重写 SimVQCodebook

**Files:**
- Modify: `src/model.py:46-102`（SimVQCodebook 类全部重写）

**Step 1: 重写 SimVQCodebook**

将 `src/model.py` 中 `SimVQCodebook` 类（第 46-102 行）替换为以下实现：

```python
class SimVQCodebook(nn.Module):
    """SimVQ codebook with frozen C and learnable linear W.

    Reference: SimVQ (ICCV 2025) — linear transform prevents codebook collapse.
    Official: https://github.com/youngsheen/SimVQ

    Key design (aligned with official):
    - C (codebook embedding) is FROZEN — never updated by gradient
    - W (linear layer) is the ONLY learnable parameter
    - Distance: z to CW (not z_proj to C)
    - Quantized: taken from CW (not from C)
    """

    def __init__(self, K: int = 4096, dim: int = 128):
        super().__init__()
        self.K = K
        self.dim = dim
        self.codebook = nn.Embedding(K, dim)
        self.linear = nn.Linear(dim, dim, bias=False)

        # Official SimVQ initialization
        nn.init.normal_(self.codebook.weight, mean=0, std=dim ** -0.5)
        nn.init.orthogonal_(self.linear.weight)

        # Freeze C — only W is learnable (SimVQ paper Remark 1)
        self.codebook.weight.requires_grad = False

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, dim) encoder output
        Returns:
            quantized_st: (B, dim) quantized embedding (straight-through)
            indices: (B,) codebook indices
        """
        # Compute CW — transformed codebook
        quant_codebook = self.linear(self.codebook.weight)  # (K, dim)

        # Distance: z to CW
        distances = torch.cdist(
            z.unsqueeze(0), quant_codebook.unsqueeze(0),
        ).squeeze(0)  # (B, K)

        indices = distances.argmin(dim=-1)  # (B,)

        # Quantized from CW (not from C)
        quantized = quant_codebook[indices]  # (B, dim)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        return quantized_st, indices

    def compute_loss(self, z: torch.Tensor, quantized_st: torch.Tensor, indices: torch.Tensor):
        """Compute commitment + embedding losses in CW space.

        Returns:
            commit_loss: ||z - sg(CW[idx])||²
            embed_loss: ||sg(z) - CW[idx]||²
        """
        quant_codebook = self.linear(self.codebook.weight)
        quantized = quant_codebook[indices]
        commit_loss = torch.mean((z - quantized.detach()) ** 2)
        embed_loss = torch.mean((z.detach() - quantized) ** 2)
        return commit_loss, embed_loss

    @torch.no_grad()
    def get_utilization(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in given indices."""
        return indices.unique().numel() / self.K

    @torch.no_grad()
    def get_quant_codebook(self) -> torch.Tensor:
        """Return CW — the effective codebook in encoder output space."""
        return self.linear(self.codebook.weight)
```

**Step 2: 更新 MeshLexVQVAE 默认 lambda_commit**

在 `src/model.py:176` 将 `lambda_commit` 默认值从 `0.25` 改为 `1.0`：

```python
    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        codebook_size: int = 4096,
        max_vertices: int = 128,
        lambda_commit: float = 1.0,   # was 0.25
        lambda_embed: float = 1.0,
    ):
```

**Step 3: 运行现有测试（预期部分失败）**

Run: `cd /home/cc/Meshlex-Research && python -m pytest tests/test_model.py -v 2>&1 | tail -20`

预期：test_codebook_* 和 test_vqvae_* 可能通过也可能失败（接口未变但内部行为变了）。记录结果。

**Step 4: Commit**

```bash
git add src/model.py
git commit -m "fix: rewrite SimVQCodebook aligned with official SimVQ — freeze C, compute CW, fix quantized source"
```

---

## Task 2: 更新单元测试

**Files:**
- Modify: `tests/test_model.py:47-74`（codebook 测试）

**Step 1: 更新 codebook 测试**

替换 `tests/test_model.py` 中的三个 codebook 测试（第 47-74 行）为：

```python
# ── Codebook Tests ───────────────────────────────────────────────────────────

def test_codebook_output_shape():
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(8, 128)
    quantized, indices = codebook(z)
    assert quantized.shape == (8, 128)
    assert indices.shape == (8,)
    assert indices.min() >= 0
    assert indices.max() < 64


def test_codebook_frozen_C():
    """Codebook embedding C must be frozen (requires_grad=False)."""
    codebook = SimVQCodebook(K=64, dim=128)
    assert not codebook.codebook.weight.requires_grad
    assert codebook.linear.weight.requires_grad


def test_codebook_quantized_from_CW():
    """Quantized output should come from CW space, not raw C."""
    codebook = SimVQCodebook(K=64, dim=32)
    z = torch.randn(4, 32)
    quantized, indices = codebook(z)
    # Verify: quantized (detached) should equal CW[indices]
    with torch.no_grad():
        cw = codebook.linear(codebook.codebook.weight)
        expected = cw[indices]
    # straight-through: forward value = quantized = CW[idx]
    # but quantized_st = z + (CW[idx] - z).detach(), so .detach() == CW[idx]
    assert torch.allclose(quantized.detach(), expected, atol=1e-5), \
        "Quantized output should match CW[indices]"


def test_codebook_straight_through_gradient():
    """Gradients should flow through quantization via straight-through."""
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(4, 128, requires_grad=True)
    quantized, _ = codebook(z)
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_codebook_utilization():
    """With diverse inputs, utilization should be non-trivial."""
    codebook = SimVQCodebook(K=32, dim=16)
    z = torch.randn(256, 16)
    _, indices = codebook(z)
    unique_codes = indices.unique().numel()
    assert unique_codes >= 4, f"Only {unique_codes}/32 codes used"


def test_codebook_compute_loss():
    """commit_loss and embed_loss should be non-negative scalars."""
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(8, 128, requires_grad=True)
    quantized_st, indices = codebook(z)
    commit_loss, embed_loss = codebook.compute_loss(z, quantized_st, indices)
    assert commit_loss.shape == ()
    assert embed_loss.shape == ()
    assert commit_loss.item() >= 0
    assert embed_loss.item() >= 0
    # embed_loss should have grad through W
    embed_loss.backward()
    assert codebook.linear.weight.grad is not None


def test_codebook_get_quant_codebook():
    """get_quant_codebook should return CW with shape (K, dim)."""
    codebook = SimVQCodebook(K=64, dim=32)
    cw = codebook.get_quant_codebook()
    assert cw.shape == (64, 32)
```

**Step 2: 运行全部测试**

Run: `cd /home/cc/Meshlex-Research && python -m pytest tests/ -v`

Expected: 全部 PASS（原来 8 个 model tests 变为 12 个左右：2 encoder + 7 codebook + 2 decoder + 1 vqvae）

**Step 3: Commit**

```bash
git add tests/test_model.py
git commit -m "test: update codebook tests for fixed SimVQ (frozen C, CW space, compute_loss grad)"
```

---

## Task 3: 重写 Trainer（删除分阶段 + 加 LR warmup + dead code revival）

**Files:**
- Modify: `src/trainer.py`（全文重写）

**Step 1: 重写 Trainer**

将 `src/trainer.py` 全文替换为：

```python
"""Training loop for MeshLex VQ-VAE."""
import torch
import gc
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import time


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 200,
        checkpoint_dir: str = "data/checkpoints",
        device: str = "cuda",
        warmup_epochs: int = 5,
        dead_code_interval: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.dead_code_interval = dead_code_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            if val_dataset else None
        )

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # LR schedule: linear warmup → cosine annealing
        warmup_sched = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs - warmup_epochs),
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        self.history = []

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_commit = 0
        total_embed = 0
        n_batches = 0
        all_indices = []
        all_z = []

        for batch in self.train_loader:
            batch = batch.to(self.device)
            gt_verts = batch.gt_vertices
            n_verts = batch.n_vertices

            result = self.model(
                batch.x, batch.edge_index, batch.batch, n_verts, gt_verts,
            )

            # End-to-end: all losses from epoch 0
            loss = result["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += result["recon_loss"].item()
            total_commit += result["commit_loss"].item()
            total_embed += result["embed_loss"].item()
            all_indices.append(result["indices"].detach().cpu())
            # Collect z for dead code revival (detach to avoid mem leak)
            all_z.append(result["z"].detach().cpu())
            n_batches += 1

        self.scheduler.step()

        # Codebook utilization
        all_idx = torch.cat(all_indices)
        utilization = all_idx.unique().numel() / self.model.codebook.K

        # Dead code revival
        if self.dead_code_interval > 0 and (epoch + 1) % self.dead_code_interval == 0:
            self._revive_dead_codes(all_idx, torch.cat(all_z))

        # Free memory
        del all_z
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "commit_loss": total_commit / n_batches,
            "embed_loss": total_embed / n_batches,
            "codebook_utilization": utilization,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def _revive_dead_codes(self, all_indices: torch.Tensor, all_z: torch.Tensor):
        """Reset unused codebook entries to random encoder outputs + noise."""
        usage_count = torch.zeros(self.model.codebook.K)
        usage_count.scatter_add_(
            0, all_indices, torch.ones_like(all_indices, dtype=torch.float),
        )
        dead_mask = usage_count == 0
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return

        with torch.no_grad():
            # Sample random encoder outputs as replacement
            replace_idx = torch.randint(len(all_z), (n_dead,))
            replacements = all_z[replace_idx].to(self.model.codebook.codebook.weight.device)
            noise = torch.randn_like(replacements) * 0.01
            # Directly write to frozen C — this is intentional for dead code revival
            self.model.codebook.codebook.weight.data[dead_mask] = replacements + noise

        print(f"  Dead code revival: replaced {n_dead}/{self.model.codebook.K} codes")

    @torch.no_grad()
    def evaluate(self, loader=None):
        """Evaluate on validation or test set."""
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return {}

        self.model.eval()
        total_recon = 0
        n_batches = 0
        all_indices = []

        for batch in loader:
            batch = batch.to(self.device)
            result = self.model(
                batch.x, batch.edge_index, batch.batch,
                batch.n_vertices, batch.gt_vertices,
            )
            total_recon += result["recon_loss"].item()
            all_indices.append(result["indices"].cpu())
            n_batches += 1

        all_idx = torch.cat(all_indices)
        return {
            "val_recon_loss": total_recon / max(n_batches, 1),
            "val_utilization": all_idx.unique().numel() / self.model.codebook.K,
        }

    def train(self):
        """Full training loop."""
        for epoch in range(self.epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "time_sec": elapsed}
            self.history.append(metrics)

            # Print progress
            util = train_metrics["codebook_utilization"]
            print(
                f"Epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
                f"recon {train_metrics['recon_loss']:.4f} | "
                f"commit {train_metrics['commit_loss']:.4f} | "
                f"embed {train_metrics['embed_loss']:.4f} | "
                f"util {util:.1%} | lr {train_metrics['lr']:.2e} | {elapsed:.1f}s"
            )

            # Checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)

                if util < 0.10:
                    print(f"WARNING: Codebook utilization {util:.1%} < 10% at epoch {epoch}")

        # Final checkpoint
        self.save_checkpoint(self.epochs - 1, tag="final")
        self.save_history()

    def save_checkpoint(self, epoch, tag=None):
        name = f"checkpoint_epoch{epoch:03d}.pt" if tag is None else f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, self.checkpoint_dir / name)

    def save_history(self):
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
```

**Step 2: 运行测试确认无语法错误**

Run: `cd /home/cc/Meshlex-Research && python -c "from src.trainer import Trainer; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/trainer.py
git commit -m "refactor: simplify Trainer — end-to-end training, LR warmup, dead code revival, remove staged VQ"
```

---

## Task 4: 修复 evaluate.py Go/No-Go 逻辑

**Files:**
- Modify: `src/evaluate.py:53-87`（compute_go_nogo 函数）

**Step 1: 重写 compute_go_nogo**

将 `src/evaluate.py` 中 `compute_go_nogo` 函数（第 53-87 行）替换为：

```python
def compute_go_nogo(same_cat_cd: float, cross_cat_cd: float, utilization: float = None):
    """Apply Go/No-Go decision matrix with utilization gate.

    Args:
        same_cat_cd: mean CD on same-category test set
        cross_cat_cd: mean CD on cross-category test set
        utilization: fraction of codebook used (0.0 to 1.0)

    Returns:
        dict with ratio, decision, next_step, utilization
    """
    if same_cat_cd < 1e-10:
        return {"ratio": float("inf"), "decision": "ERROR", "next_step": "CD is zero — check data"}

    ratio = cross_cat_cd / same_cat_cd

    # Utilization gate — collapse detection
    if utilization is not None and utilization < 0.10:
        return {
            "ratio": ratio,
            "decision": "COLLAPSE - HALT",
            "next_step": "Codebook collapsed (<10% utilization). Debug VQ training before proceeding.",
            "same_cat_cd": same_cat_cd,
            "cross_cat_cd": cross_cat_cd,
            "utilization": utilization,
        }

    if ratio < 1.2 and (utilization is None or utilization > 0.30):
        decision = "STRONG GO"
        next_step = "Proceed to full MeshLex experiment design"
    elif ratio < 1.2:
        decision = "CONDITIONAL GO"
        next_step = "CD ratio good but utilization suboptimal. Consider decoder enhancement (Phase B)."
    elif ratio < 2.0:
        decision = "WEAK GO"
        next_step = "Adjust story to 'transferable vocabulary', continue"
    elif ratio < 3.0:
        decision = "HOLD"
        next_step = "Analyze failure, consider category-adaptive codebook"
    else:
        decision = "NO-GO"
        next_step = "Core hypothesis falsified. Pivot direction."

    return {
        "ratio": ratio,
        "decision": decision,
        "next_step": next_step,
        "same_cat_cd": same_cat_cd,
        "cross_cat_cd": cross_cat_cd,
        "utilization": utilization,
    }
```

**Step 2: 运行导入测试**

Run: `cd /home/cc/Meshlex-Research && python -c "from src.evaluate import compute_go_nogo; r = compute_go_nogo(300, 330, 0.35); print(r['decision']); r2 = compute_go_nogo(300, 330, 0.05); print(r2['decision'])"`

Expected:
```
STRONG GO
COLLAPSE - HALT
```

**Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "fix: add utilization gate to Go/No-Go — detect collapse at <10%"
```

---

## Task 5: 更新 scripts/train.py

**Files:**
- Modify: `scripts/train.py`

**Step 1: 重写 scripts/train.py**

```python
"""Train MeshLex VQ-VAE on preprocessed patches."""
import argparse
import torch

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True,
                        help="Patch directories for training")
    parser.add_argument("--val_dirs", nargs="+", default=None,
                        help="Patch directories for validation")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_commit", type=float, default=1.0)
    parser.add_argument("--lambda_embed", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="LR warmup epochs (default 5)")
    parser.add_argument("--dead_code_interval", type=int, default=10,
                        help="Dead code revival interval in epochs (0 to disable)")
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (loads model + optimizer state)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load datasets
    from torch.utils.data import ConcatDataset
    train_datasets = [PatchGraphDataset(d) for d in args.train_dirs]
    train_dataset = ConcatDataset(train_datasets)
    print(f"Training patches: {len(train_dataset)}")

    val_dataset = None
    if args.val_dirs:
        val_datasets = [PatchGraphDataset(d) for d in args.val_dirs]
        val_dataset = ConcatDataset(val_datasets)
        print(f"Validation patches: {len(val_dataset)}")

    # Create model
    model = MeshLexVQVAE(
        codebook_size=args.codebook_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lambda_commit=args.lambda_commit,
        lambda_embed=args.lambda_embed,
    )

    # Resume from checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed model from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    print(f"Codebook C frozen: {not model.codebook.codebook.weight.requires_grad}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        warmup_epochs=args.warmup_epochs,
        dead_code_interval=args.dead_code_interval,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

**Step 2: 验证导入**

Run: `cd /home/cc/Meshlex-Research && python scripts/train.py --help`

Expected: 显示帮助信息，无 `--vq_start_epoch` 参数，有 `--lambda_commit`、`--warmup_epochs`、`--dead_code_interval` 参数。

**Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "refactor: update train.py — remove vq_start_epoch, add lambda/warmup/dead_code params"
```

---

## Task 6: 更新 scripts/evaluate.py

**Files:**
- Modify: `scripts/evaluate.py:47-53`（传 utilization 到 go_nogo）

**Step 1: 修改 Go/No-Go 调用**

在 `scripts/evaluate.py` 第 48 行，将：

```python
    decision = compute_go_nogo(same_results["mean_cd"], cross_results["mean_cd"])
```

改为：

```python
    # Use same-cat utilization for Go/No-Go (more conservative)
    decision = compute_go_nogo(
        same_results["mean_cd"], cross_results["mean_cd"],
        utilization=same_results["utilization"],
    )
```

同时更新打印信息，在 `print(f"Decision:` 之前加一行：

```python
    print(f"Codebook Utilization: {same_results['utilization']:.1%} ({same_results['n_unique_codes']}/{same_results['total_codes']})")
```

**Step 2: 验证**

Run: `cd /home/cc/Meshlex-Research && python scripts/evaluate.py --help`

Expected: 帮助信息正常显示。

**Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "fix: pass utilization to Go/No-Go in evaluate script"
```

---

## Task 7: 更新 scripts/visualize.py（t-SNE 改用 CW）

**Files:**
- Modify: `scripts/visualize.py:56-72`（plot_codebook_tsne 函数）

**Step 1: 修改 t-SNE 可视化**

将 `scripts/visualize.py` 中 `plot_codebook_tsne` 函数（第 56-72 行）替换为：

```python
def plot_codebook_tsne(model, save_path: str):
    """t-SNE of effective codebook embeddings (CW, not raw C)."""
    from sklearn.manifold import TSNE

    # Use CW (transformed codebook) — this is what the model actually uses
    embeddings = model.codebook.get_quant_codebook().detach().cpu().numpy()  # (K, dim)
    tsne = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0] - 1), random_state=42)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.6, c="steelblue")
    plt.title(f"Codebook t-SNE (CW space, K={embeddings.shape[0]})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
```

**Step 2: Commit**

```bash
git add scripts/visualize.py
git commit -m "fix: visualize CW (effective codebook) instead of frozen C in t-SNE"
```

---

## Task 8: 运行全部测试 + 集成验证

**Files:**
- 无新文件

**Step 1: 运行全部单元测试**

Run: `cd /home/cc/Meshlex-Research && python -m pytest tests/ -v`

Expected: 全部 PASS

**Step 2: 端到端 smoke test（无真实数据，纯合成）**

Run:
```bash
cd /home/cc/Meshlex-Research && python -c "
import torch
from torch_geometric.data import Data, Batch
from src.model import MeshLexVQVAE

model = MeshLexVQVAE(codebook_size=64, embed_dim=32, hidden_dim=64)
print(f'C frozen: {not model.codebook.codebook.weight.requires_grad}')
print(f'W trainable: {model.codebook.linear.weight.requires_grad}')

# Fake forward
graphs = []
for _ in range(8):
    nf = 20
    x = torch.randn(nf, 15)
    ei = torch.stack([torch.randint(0, nf, (40,)), torch.randint(0, nf, (40,))])
    graphs.append(Data(x=x, edge_index=ei))
batch = Batch.from_data_list(graphs)
n_verts = torch.full((8,), 15)
gt = torch.randn(8, 128, 3)

result = model(batch.x, batch.edge_index, batch.batch, n_verts, gt)
print(f'total_loss: {result[\"total_loss\"].item():.4f}')
print(f'recon_loss: {result[\"recon_loss\"].item():.4f}')
print(f'commit_loss: {result[\"commit_loss\"].item():.4f}')
print(f'embed_loss: {result[\"embed_loss\"].item():.4f}')
print(f'unique codes: {result[\"indices\"].unique().numel()}/64')

# Backward
result['total_loss'].backward()
print(f'W grad exists: {model.codebook.linear.weight.grad is not None}')
print(f'C grad: {model.codebook.codebook.weight.grad}')  # should be None
print('Smoke test PASSED')
"
```

Expected:
```
C frozen: True
W trainable: True
...
W grad exists: True
C grad: None
Smoke test PASSED
```

**Step 3: Commit（如有修复）**

如果 smoke test 发现问题，修复后 commit。否则跳过。

---

## Task 9: 资源检查 + 下载数据

**Files:**
- 无代码改动

**Step 1: 检查磁盘/GPU**

Run:
```bash
df -h / && free -h && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

Expected: 磁盘可用 > 20GB, GPU memory > 20GB

**Step 2: 下载 5-Category 数据**

Run:
```bash
cd /home/cc/Meshlex-Research && python scripts/download_objaverse.py --mode 5cat --output_dir data/objaverse
```

Expected: `data/objaverse/5cat/manifest.json` 生成，~800+ objects

**Step 3: 预处理**

Run:
```bash
cd /home/cc/Meshlex-Research && python scripts/run_preprocessing.py \
    --input_manifest data/objaverse/5cat/manifest.json \
    --experiment_name 5cat \
    --output_root data \
    --target_faces 1000
```

Expected: `data/patches/5cat/` 下生成 train/test 分割的 NPZ 文件

**Step 4: 验证 patch 统计**

Run:
```bash
cd /home/cc/Meshlex-Research && python -c "
import json, numpy as np
meta = json.load(open('data/patch_metadata_5cat.json'))
cats = {}
for m in meta:
    c = m['category']
    cats.setdefault(c, []).append(m['n_patches'])
total = 0
for c, patches in sorted(cats.items()):
    p = np.array(patches)
    total += p.sum()
    print(f'{c}: {len(patches)} meshes, {p.sum()} patches')
print(f'Total: {len(meta)} meshes, {total} patches')
"
```

Expected: 类似上次的 ~563 meshes, ~20K patches

---

## Task 10: 训练 200 Epochs

**Files:**
- 无代码改动

**Step 1: 再次检查磁盘空间**

Run: `df -h /`

**Step 2: 启动训练**

Run:
```bash
cd /home/cc/Meshlex-Research && python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --lambda_commit 1.0 \
    --lambda_embed 1.0 \
    --warmup_epochs 5 \
    --dead_code_interval 10 \
    --checkpoint_dir data/checkpoints/5cat_v2
```

Expected: 训练约 2-6 小时，每 epoch 打印 loss/util/lr。关键监控：
- Epoch 0-5: LR warmup 阶段，utilization 应该逐步建立
- Epoch 10+: utilization 应 > 5%
- Epoch 50+: utilization 应 > 10%
- Epoch 100+: utilization 应 > 20%
- 最终: utilization > 30% 为目标

**注意：此步骤运行时间长，使用 `run_in_background` 参数运行，后台执行。**

---

## Task 11: 评估 + 可视化 + Go/No-Go

**Files:**
- 无代码改动

**Step 1: 运行评估**

Run:
```bash
cd /home/cc/Meshlex-Research && python scripts/evaluate.py \
    --checkpoint data/checkpoints/5cat_v2/checkpoint_final.pt \
    --same_cat_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --cross_cat_dirs data/patches/5cat/car data/patches/5cat/lamp \
    --output results/exp1_v2_eval/exp1_v2_eval.json
```

Expected: 打印 CD ratio, utilization, Go/No-Go decision

**Step 2: 运行可视化**

Run:
```bash
cd /home/cc/Meshlex-Research && python scripts/visualize.py \
    --checkpoint data/checkpoints/5cat_v2/checkpoint_final.pt \
    --history data/checkpoints/5cat_v2/training_history.json \
    --patch_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --output_dir results/exp1_v2_plots
```

Expected: 生成 training_curves.png, codebook_tsne.png, utilization_histogram.png

**Step 3: 读取 Go/No-Go 结果**

Read: `results/exp1_v2_eval/exp1_v2_eval.json`

按 A→B 决策矩阵判断下一步：

| 结果 | 下一步 |
|------|--------|
| utilization > 30% + ratio < 1.2 | **STRONG GO** → 跑实验 2 (LVIS-Wide) |
| utilization > 30% + CD 质量差 | 进入 B 阶段：增强 decoder |
| utilization 10-30% | 进入 B 阶段：加 rotation trick |
| utilization < 10% | 重新诊断 |

**Step 4: Commit 结果**

```bash
git add results/exp1_v2_eval/ results/exp1_v2_plots/
git commit -m "results: experiment 1 v2 (SimVQ fix) — evaluation and visualization"
```

---

## Task 12: 更新 TODO.md + CLAUDE.md

**Files:**
- Modify: `TODO.md`
- Modify: `CLAUDE.md`

**Step 1: 更新 TODO.md**

根据实验结果更新 TODO.md 中的进度和下一步。标记 P0 修复为已完成。

**Step 2: 更新 CLAUDE.md Current Status**

更新 `CLAUDE.md` 中的 Current Status 部分，反映 SimVQ 修复后的状态。

**Step 3: Commit**

```bash
git add TODO.md CLAUDE.md
git commit -m "docs: update TODO and CLAUDE.md with SimVQ fix progress"
```

---

## 文件改动清单

| Task | 文件 | 改动类型 |
|------|------|---------|
| 1 | `src/model.py` | 重写 SimVQCodebook + 改 lambda_commit |
| 2 | `tests/test_model.py` | 重写 codebook 测试 |
| 3 | `src/trainer.py` | 全文重写 |
| 4 | `src/evaluate.py` | 重写 compute_go_nogo |
| 5 | `scripts/train.py` | 全文重写 |
| 6 | `scripts/evaluate.py` | 小改（传 utilization） |
| 7 | `scripts/visualize.py` | 小改（t-SNE 用 CW） |
| 8 | - | 测试 + smoke test |
| 9 | - | 数据下载 + 预处理 |
| 10 | - | 训练 200 epoch |
| 11 | - | 评估 + 可视化 + Go/No-Go |
| 12 | `TODO.md`, `CLAUDE.md` | 文档更新 |

## 预估时间

| 阶段 | Tasks | 预估 |
|------|-------|------|
| 代码修改 + 测试 | 1-8 | 30-60 min |
| 数据准备 | 9 | 30-60 min |
| 训练 | 10 | 2-6 h (后台) |
| 评估 + 文档 | 11-12 | 15-30 min |

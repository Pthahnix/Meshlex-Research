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

"""Training loop for MeshLex VQ-VAE."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        vq_start_epoch: int = 20,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.vq_start_epoch = vq_start_epoch
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
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.history = []

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_recon = 0
        n_batches = 0
        all_indices = []

        for batch in self.train_loader:
            batch = batch.to(self.device)
            gt_verts = batch.gt_vertices  # (B, max_V, 3)
            n_verts = batch.n_vertices    # (B,)

            result = self.model(
                batch.x, batch.edge_index, batch.batch, n_verts, gt_verts,
            )

            # Before vq_start_epoch: zero out VQ losses (train encoder+decoder only)
            if epoch < self.vq_start_epoch:
                loss = result["recon_loss"]
            else:
                loss = result["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += result["recon_loss"].item()
            all_indices.append(result["indices"].detach().cpu())
            n_batches += 1

        self.scheduler.step()

        # Codebook utilization
        all_idx = torch.cat(all_indices)
        utilization = all_idx.unique().numel() / self.model.codebook.K

        return {
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "codebook_utilization": utilization,
            "lr": self.scheduler.get_last_lr()[0],
        }

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
                f"util {util:.1%} | {elapsed:.1f}s"
            )

            # Checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)

                # Early warning: codebook collapse
                if epoch >= self.vq_start_epoch and util < 0.30:
                    print(f"WARNING: Codebook utilization {util:.1%} < 30% at epoch {epoch}")

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

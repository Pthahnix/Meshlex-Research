"""Training loop for MeshLex VQ-VAE."""
import torch
import gc
import numpy as np
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
        encoder_warmup_epochs: int = 10,
        resume_checkpoint: dict = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.dead_code_interval = dead_code_interval
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
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

        # Restore full training state from checkpoint
        if resume_checkpoint is not None:
            self.start_epoch = resume_checkpoint.get("epoch", -1) + 1
            if "optimizer_state_dict" in resume_checkpoint:
                self.optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            if "history" in resume_checkpoint:
                self.history = resume_checkpoint["history"]
            # Advance scheduler to match resumed epoch
            for _ in range(self.start_epoch):
                self.scheduler.step()
            print(f"  Trainer resumed: starting from epoch {self.start_epoch}, "
                  f"history has {len(self.history)} entries")

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_commit = 0
        total_embed = 0
        n_batches = 0
        all_indices = []
        all_z = []

        # During encoder warmup, only use recon loss (encoder learns without VQ)
        use_vq = epoch >= self.encoder_warmup_epochs

        for batch in self.train_loader:
            batch = batch.to(self.device)
            gt_verts = batch.gt_vertices
            n_verts = batch.n_vertices

            result = self.model(
                batch.x, batch.edge_index, batch.batch, n_verts, gt_verts,
            )

            if use_vq:
                loss = result["total_loss"]
            else:
                # Encoder warmup: only recon loss, let encoder learn representations
                loss = result["recon_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += result["recon_loss"].item()
            total_commit += result["commit_loss"].item()
            total_embed += result["embed_loss"].item()
            all_indices.append(result["indices"].detach().cpu())
            # Collect z for codebook init and dead code revival
            all_z.append(result["z"].detach().cpu())
            n_batches += 1

        self.scheduler.step()

        # Codebook utilization (use first-level indices for RVQ compatibility)
        all_idx = torch.cat(all_indices)
        if all_idx.dim() > 1:
            all_idx_l0 = all_idx[:, 0]  # first RVQ level
        else:
            all_idx_l0 = all_idx
        utilization = all_idx_l0.unique().numel() / self.model.codebook.K

        # K-means codebook init at end of encoder warmup
        if epoch == self.encoder_warmup_epochs - 1 and self.encoder_warmup_epochs > 0:
            self._init_codebook_from_z(torch.cat(all_z))

        # Dead code revival (only after VQ training starts)
        if use_vq and self.dead_code_interval > 0 and (epoch + 1) % self.dead_code_interval == 0:
            self._revive_dead_codes(all_idx_l0, torch.cat(all_z))

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
            "phase": "encoder_warmup" if not use_vq else "full_vq",
        }

    def _init_codebook_from_z(self, all_z: torch.Tensor):
        """Initialize codebook from encoder outputs using K-means.

        Sets C = W^T(centroids) so that CW = W(C) ≈ centroids.
        This aligns the effective codebook with the encoder output space.
        """
        from sklearn.cluster import MiniBatchKMeans

        K = self.model.codebook.K
        z_np = all_z.numpy()
        n_samples = len(z_np)
        effective_k = min(K, n_samples)

        print(f"  K-means codebook init: {n_samples} samples → {effective_k} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=min(4096, n_samples),
            max_iter=100,
            random_state=42,
        )
        kmeans.fit(z_np)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        # Pad if fewer clusters than K
        if effective_k < K:
            extra = K - effective_k
            pad_idx = torch.randint(0, effective_k, (extra,))
            noise = torch.randn(extra, centroids.shape[1]) * 0.01
            centroids = torch.cat([centroids, centroids[pad_idx] + noise])

        # Set C = W^T(centroids) so CW ≈ centroids
        self.model.codebook.init_from_z(centroids.to(self.device))

        # Verify
        with torch.no_grad():
            cw = self.model.codebook.get_quant_codebook()
            error = (cw.cpu() - centroids[:K]).norm(dim=1).mean()
            sample_z = all_z[:1000].to(self.device)
            _, idx = self.model.codebook(sample_z)
            util = idx.unique().numel() / K
            print(f"  CW-centroid error: {error:.4f}, post-init utilization: {util:.1%}")

    def _revive_dead_codes(self, all_indices: torch.Tensor, all_z: torch.Tensor):
        """Reset unused codebook entries to random encoder outputs + noise.

        Uses init_from_z logic: sets dead C entries so CW[dead] ≈ random z.
        """
        usage_count = torch.zeros(self.model.codebook.K)
        usage_count.scatter_add_(
            0, all_indices, torch.ones_like(all_indices, dtype=torch.float),
        )
        dead_mask = usage_count == 0
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return

        with torch.no_grad():
            # Sample random encoder outputs as replacement targets
            replace_idx = torch.randint(len(all_z), (n_dead,))
            targets = all_z[replace_idx].to(self.model.codebook.codebook.weight.device)
            noise = torch.randn_like(targets) * 0.01

            # Set C[dead] = W^T(targets + noise) so CW[dead] ≈ targets
            W = self.model.codebook.linear.weight  # (dim, dim)
            new_c = (targets + noise) @ W  # (n_dead, dim)
            self.model.codebook.codebook.weight.data[dead_mask] = new_c

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
        if self.start_epoch == 0 and self.encoder_warmup_epochs > 0:
            print(f"Encoder warmup: {self.encoder_warmup_epochs} epochs (recon only), "
                  f"then K-means init + full VQ training")

        if self.start_epoch > 0:
            print(f"Resuming training from epoch {self.start_epoch}/{self.epochs}")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "time_sec": elapsed}
            self.history.append(metrics)

            # Print progress
            util = train_metrics["codebook_utilization"]
            phase = train_metrics.get("phase", "full_vq")
            phase_tag = "[warmup] " if phase == "encoder_warmup" else ""
            print(
                f"Epoch {epoch:03d} {phase_tag}| loss {train_metrics['loss']:.4f} | "
                f"recon {train_metrics['recon_loss']:.4f} | "
                f"commit {train_metrics['commit_loss']:.4f} | "
                f"embed {train_metrics['embed_loss']:.4f} | "
                f"util {util:.1%} | lr {train_metrics['lr']:.2e} | {elapsed:.1f}s"
            )

            # Checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)

                if util < 0.10 and epoch >= self.encoder_warmup_epochs:
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

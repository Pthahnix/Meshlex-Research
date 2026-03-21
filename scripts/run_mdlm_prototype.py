"""Exp 5: Toy MDLM (Masked Discrete Language Model) for mesh tokens.

Trains a minimal masked discrete diffusion model on existing RVQ token sequences
to test PatchDiffusion feasibility.

Architecture: Small Transformer encoder, continuous-time masking schedule.
Data: Existing 4934 mesh token sequences from data/sequences/rvq_lvis/

Usage:
    PYTHONPATH=. python scripts/run_mdlm_prototype.py \
        --seq_dir data/sequences/rvq_lvis \
        --output_dir results/preliminary_exp/exp5_mdlm \
        --epochs 100 \
        --batch_size 64
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────

class MeshTokenDataset(Dataset):
    """Loads RVQ token sequences, pads/truncates to fixed length."""

    def __init__(self, seq_dir: str, max_patches: int = 80):
        self.max_patches = max_patches
        self.vocab_size = 1024
        self.mask_token = 1024  # Special MASK token
        self.n_levels = 3

        seq_dir = Path(seq_dir)
        npz_files = sorted(seq_dir.glob("*_sequence.npz"))

        self.sequences = []
        self.lengths = []

        for f in npz_files:
            data = np.load(str(f))
            tokens = data["tokens"]  # (N_patches, 3)
            if tokens.ndim == 1:
                continue
            n = min(tokens.shape[0], max_patches)
            # Flatten: (N_patches, 3) → (N_patches * 3,)
            flat = tokens[:n].flatten()
            self.sequences.append(flat)
            self.lengths.append(n * self.n_levels)

        print(f"Loaded {len(self.sequences)} sequences")
        print(f"  Lengths: min={min(self.lengths)}, max={max(self.lengths)}, "
              f"median={np.median(self.lengths):.0f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = self.max_patches * self.n_levels

        # Pad or truncate
        if len(seq) >= L:
            tokens = torch.tensor(seq[:L], dtype=torch.long)
            mask = torch.ones(L, dtype=torch.bool)
        else:
            tokens = torch.zeros(L, dtype=torch.long)
            tokens[:len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask = torch.zeros(L, dtype=torch.bool)
            mask[:len(seq)] = True

        return tokens, mask


# ──────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────

class ToyMDLM(nn.Module):
    """Minimal masked discrete diffusion language model.

    Token embedding + position embedding + level embedding + time embedding
    → Transformer encoder → per-position logits.
    """

    def __init__(
        self,
        vocab_size: int = 1025,  # 1024 tokens + MASK
        max_seq_len: int = 240,  # 80 patches * 3 levels
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_levels: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_levels = n_levels

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.level_embed = nn.Embedding(n_levels, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size - 1)  # Predict 1024 real tokens only
        self.norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ToyMDLM: {n_params / 1e6:.2f}M params")

    def forward(self, x, t, padding_mask=None):
        """
        Args:
            x: (B, L) token IDs (may include MASK=1024)
            t: (B,) diffusion time in [0, 1]
            padding_mask: (B, L) bool, True = valid token

        Returns:
            logits: (B, L, 1024) prediction logits
        """
        B, L = x.shape

        # Embeddings
        tok_emb = self.token_embed(x)  # (B, L, d)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos_ids)  # (B, L, d)

        # Level embedding: positions 0,3,6,... are L1; 1,4,7,... are L2; 2,5,8,... are L3
        level_ids = (torch.arange(L, device=x.device) % self.n_levels).unsqueeze(0).expand(B, -1)
        lvl_emb = self.level_embed(level_ids)  # (B, L, d)

        # Time embedding (broadcast across sequence)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, d)
        t_emb = t_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, d)

        h = tok_emb + pos_emb + lvl_emb + t_emb

        # Transformer (causal mask = None for bidirectional)
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask  # True = ignore
        else:
            src_key_padding_mask = None

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        logits = self.head(h)  # (B, L, 1024)
        return logits


# ──────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────

def mask_tokens(tokens, padding_mask, t, mask_token=1024):
    """Apply masking at rate t per token.

    Args:
        tokens: (B, L) original tokens
        padding_mask: (B, L) True = valid
        t: (B,) masking rate per sample

    Returns:
        masked_tokens: (B, L) with some tokens replaced by mask_token
        is_masked: (B, L) True where masked
    """
    B, L = tokens.shape
    # Per-token random
    rand = torch.rand(B, L, device=tokens.device)
    # Mask with probability t (broadcast)
    is_masked = (rand < t.unsqueeze(-1)) & padding_mask
    masked_tokens = tokens.clone()
    masked_tokens[is_masked] = mask_token
    return masked_tokens, is_masked


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for batch_idx, (tokens, padding_mask) in enumerate(dataloader):
        tokens = tokens.to(device)
        padding_mask = padding_mask.to(device)

        B = tokens.shape[0]

        # Sample t ~ U(0.1, 1.0) — avoid t=0 (no masking)
        t = torch.rand(B, device=device) * 0.9 + 0.1

        # Mask tokens
        masked_tokens, is_masked = mask_tokens(tokens, padding_mask, t)

        # Forward
        logits = model(masked_tokens, t, padding_mask)

        # Loss: only on masked positions
        if is_masked.sum() == 0:
            continue

        pred = logits[is_masked]  # (N_masked, 1024)
        target = tokens[is_masked]  # (N_masked,)
        loss = F.cross_entropy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * is_masked.sum().item()
        total_correct += (pred.argmax(-1) == target).sum().item()
        total_masked += is_masked.sum().item()

    avg_loss = total_loss / max(total_masked, 1)
    accuracy = total_correct / max(total_masked, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, t_eval=0.5):
    """Evaluate at fixed masking rate t_eval."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for tokens, padding_mask in dataloader:
        tokens = tokens.to(device)
        padding_mask = padding_mask.to(device)
        B = tokens.shape[0]

        t = torch.full((B,), t_eval, device=device)
        masked_tokens, is_masked = mask_tokens(tokens, padding_mask, t)

        if is_masked.sum() == 0:
            continue

        logits = model(masked_tokens, t, padding_mask)
        pred = logits[is_masked]
        target = tokens[is_masked]
        loss = F.cross_entropy(pred, target)

        total_loss += loss.item() * is_masked.sum().item()
        total_correct += (pred.argmax(-1) == target).sum().item()
        total_masked += is_masked.sum().item()

    avg_loss = total_loss / max(total_masked, 1)
    accuracy = total_correct / max(total_masked, 1)
    return avg_loss, accuracy


# ──────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────

@torch.no_grad()
def generate_sequences(model, n_samples, seq_len, device, n_steps=100):
    """Generate token sequences by iterative unmasking.

    Start from all-masked, gradually unmask over n_steps.
    """
    model.eval()
    mask_token = 1024

    # Start with all masked
    tokens = torch.full((n_samples, seq_len), mask_token, dtype=torch.long, device=device)
    padding_mask = torch.ones(n_samples, seq_len, dtype=torch.bool, device=device)

    for step in range(n_steps):
        t_val = 1.0 - step / n_steps  # 1.0 → 0.0
        t = torch.full((n_samples,), t_val, device=device)

        logits = model(tokens, t, padding_mask)  # (B, L, 1024)

        # For masked positions, sample from logits
        is_masked = tokens == mask_token
        if not is_masked.any():
            break

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Decide which positions to unmask this step
        # Unmask ~1/n_steps fraction of remaining masked tokens
        n_to_unmask = max(1, int(is_masked.float().sum() / max(n_steps - step, 1)))

        # For each sample, find confidence scores for masked positions
        for b in range(n_samples):
            masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_pos) == 0:
                continue

            # Get max probability at each masked position (confidence)
            conf = probs[b, masked_pos].max(dim=-1).values
            # Unmask the most confident positions
            n_unmask = min(len(masked_pos), max(1, len(masked_pos) // max(n_steps - step, 1)))
            _, top_idx = conf.topk(n_unmask)
            unmask_pos = masked_pos[top_idx]

            # Sample tokens
            sampled = torch.multinomial(probs[b, unmask_pos], 1).squeeze(-1)
            tokens[b, unmask_pos] = sampled

    return tokens.cpu().numpy()


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 5: Toy MDLM prototype")
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--output_dir", default="results/preliminary_exp/exp5_mdlm")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_patches", type=int, default=80)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--n_generate", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dataset = MeshTokenDataset(args.seq_dir, max_patches=args.max_patches)
    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    seq_len = args.max_patches * 3
    model = ToyMDLM(
        vocab_size=1025,
        max_seq_len=seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Warmup + cosine decay
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.epochs * len(train_loader) - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, device, t_eval=0.5)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                  f"time={elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "args": vars(args),
    }, output_dir / "checkpoint.pt")

    # Generation
    print(f"\nGenerating {args.n_generate} sequences...")
    generated = generate_sequences(model, args.n_generate, seq_len, device, n_steps=100)
    np.save(output_dir / "generated_tokens.npy", generated)

    # Compare generated vs real token distribution
    real_tokens = {"L1": [], "L2": [], "L3": []}
    for seq in dataset.sequences:
        if len(seq) >= 3:
            real_tokens["L1"].append(seq[0::3])
            real_tokens["L2"].append(seq[1::3])
            real_tokens["L3"].append(seq[2::3])

    gen_tokens = {"L1": [], "L2": [], "L3": []}
    for g in generated:
        valid = g[g < 1024]  # Remove any remaining MASK tokens
        if len(valid) >= 3:
            gen_tokens["L1"].append(valid[0::3])
            gen_tokens["L2"].append(valid[1::3])
            gen_tokens["L3"].append(valid[2::3])

    # KL divergence per level
    kl_divs = {}
    for level in ["L1", "L2", "L3"]:
        if not real_tokens[level] or not gen_tokens[level]:
            continue
        real_all = np.concatenate(real_tokens[level])
        gen_all = np.concatenate(gen_tokens[level])

        real_counts = np.bincount(real_all[real_all < 1024], minlength=1024) + 1  # Laplace smoothing
        gen_counts = np.bincount(gen_all[gen_all < 1024], minlength=1024) + 1

        real_p = real_counts / real_counts.sum()
        gen_q = gen_counts / gen_counts.sum()

        kl = float(np.sum(real_p * np.log(real_p / gen_q)))
        kl_divs[level] = round(kl, 4)
        print(f"  KL({level}): {kl:.4f} nats")

    # Perplexity (from val loss at t=0.5)
    final_val_loss = history["val_loss"][-1]
    ppl = math.exp(final_val_loss)
    print(f"  Perplexity (t=0.5): {ppl:.2f}")
    print(f"  (AR v2 baseline: ppl 4.4, loss 1.48)")

    # Mask prediction accuracy at different t values
    print("\n  Accuracy at different masking rates:")
    t_acc = {}
    for t_val in [0.2, 0.3, 0.5, 0.7, 0.9]:
        _, acc = evaluate(model, val_loader, device, t_eval=t_val)
        t_acc[str(t_val)] = round(acc, 4)
        print(f"    t={t_val}: acc={acc:.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training curve
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train loss", alpha=0.7)
    ax.plot(history["val_loss"], label="Val loss", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[0, 1]
    ax.plot(history["train_acc"], label="Train acc", alpha=0.7)
    ax.plot(history["val_acc"], label="Val acc (t=0.5)", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mask Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Token distribution comparison (L1)
    ax = axes[1, 0]
    if real_tokens["L1"] and gen_tokens["L1"]:
        real_l1 = np.concatenate(real_tokens["L1"])
        gen_l1 = np.concatenate(gen_tokens["L1"])
        r_counts = np.bincount(real_l1[real_l1 < 1024], minlength=1024)
        g_counts = np.bincount(gen_l1[gen_l1 < 1024], minlength=1024)
        r_freq = np.sort(r_counts)[::-1]
        g_freq = np.sort(g_counts)[::-1]
        ax.plot(range(len(r_freq)), r_freq / r_freq.sum(), label="Real", alpha=0.7)
        ax.plot(range(len(g_freq)), g_freq / g_freq.sum(), label="Generated", alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"L1 Rank-Frequency (KL={kl_divs.get('L1', '?')})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Accuracy vs masking rate
    ax = axes[1, 1]
    t_vals = [float(t) for t in t_acc.keys()]
    acc_vals = list(t_acc.values())
    ax.plot(t_vals, acc_vals, "o-", color="steelblue", markersize=8)
    ax.axhline(y=1/1024, color="gray", linestyle=":", label="Random (1/1024)")
    ax.set_xlabel("Masking Rate (t)")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_title("Accuracy vs Masking Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Exp 5: Toy MDLM — ppl={ppl:.2f} (AR baseline: 4.4)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "training_results.png", dpi=150)
    plt.close()

    # Save summary
    summary = {
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train": n_train,
        "n_val": n_val,
        "epochs": args.epochs,
        "final_train_loss": round(history["train_loss"][-1], 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_train_acc": round(history["train_acc"][-1], 4),
        "final_val_acc": round(history["val_acc"][-1], 4),
        "perplexity": round(ppl, 2),
        "ar_v2_ppl": 4.4,
        "kl_divergence": kl_divs,
        "accuracy_vs_t": t_acc,
        "training_time_s": round(total_time, 0),
        "n_generated": args.n_generate,
        "verdict": "FEASIBLE" if ppl < 10 and history["val_acc"][-1] > 0.05 else
                   "MARGINAL" if ppl < 50 else "NOT_FEASIBLE",
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print verdict
    print(f"\n{'='*60}")
    print(f"VERDICT: {summary['verdict']}")
    print(f"  Perplexity: {ppl:.2f} (AR v2: 4.4)")
    print(f"  Val accuracy (t=0.5): {history['val_acc'][-1]:.3f}")
    print(f"  KL divergences: {kl_divs}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

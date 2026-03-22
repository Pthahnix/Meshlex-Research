"""Full-scale MDLM training script.

Trains FullMDLM on mesh token sequences using continuous-time masking.
Reads token sequences from *_sequence.npz files (same format as AR training).
Only uses the codebook tokens (L1, L2, L3), NOT position/scale tokens.

Usage:
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nohup python scripts/train_mdlm.py \
        --seq_dir data/sequences/rvq_lvis \
        --checkpoint_dir data/checkpoints/mdlm_full \
        --output_dir results/fullscale_mdlm \
        --epochs 200 \
        --batch_size 32 \
        > logs/train_mdlm.log 2>&1 &
"""
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mdlm_model import FullMDLM, apply_masking


class MDLMTokenDataset(Dataset):
    """Loads codebook tokens from sequence NPZs for MDLM training.

    Extracts only the RVQ token indices (L1, L2, L3) from each sequence,
    flattening to (n_patches * 3,) and padding/truncating to max_seq_len.
    """
    MASK_TOKEN = 1024

    def __init__(self, seq_dir, max_seq_len=390):
        self.files = sorted(Path(seq_dir).glob("*_sequence.npz"))
        self.max_seq_len = max_seq_len  # 130 patches x 3 levels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        tokens = data["tokens"]  # (N, 3) RVQ indices
        flat = tokens.flatten()  # (N*3,)

        L = min(len(flat), self.max_seq_len)
        padded = np.full(self.max_seq_len, 0, dtype=np.int64)
        padded[:L] = flat[:L]
        padding_mask = np.zeros(self.max_seq_len, dtype=bool)
        padding_mask[:L] = True

        return torch.tensor(padded, dtype=torch.long), torch.tensor(padding_mask)


def train_mdlm(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset = MDLMTokenDataset(args.seq_dir, max_seq_len=args.max_seq_len)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = FullMDLM(
        vocab_size=args.codebook_size + 1,  # +1 for MASK
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MDLM: {n_params/1e6:.1f}M params, device={device}")
    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Max seq len: {args.max_seq_len}, Codebook: {args.codebook_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Warmup + cosine schedule
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # bf16 mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ppl": []}

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"Resumed from epoch {start_epoch}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for tokens, padding_mask in train_loader:
            tokens, padding_mask = tokens.to(device), padding_mask.to(device)

            # Sample masking rate
            t = torch.empty(tokens.size(0), device=device).uniform_(0.1, 1.0)
            masked, mask_pos = apply_masking(tokens, t, padding_mask,
                                              mask_token=args.codebook_size)

            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
                logits = model(masked, t, padding_mask)

                # Loss only on masked positions
                if mask_pos.any():
                    loss = F.cross_entropy(
                        logits[mask_pos], tokens[mask_pos],
                        reduction="mean"
                    )
                else:
                    continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, padding_mask in val_loader:
                tokens, padding_mask = tokens.to(device), padding_mask.to(device)
                t = torch.full((tokens.size(0),), 0.5, device=device)
                masked, mask_pos = apply_masking(tokens, t, padding_mask,
                                                  mask_token=args.codebook_size)

                with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
                    logits = model(masked, t, padding_mask)

                if mask_pos.any():
                    loss = F.cross_entropy(logits[mask_pos], tokens[mask_pos])
                    val_loss += loss.item()
                    preds = logits[mask_pos].argmax(dim=-1)
                    val_correct += (preds == tokens[mask_pos]).sum().item()
                    val_total += mask_pos.sum().item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_ppl"].append(val_ppl)

        if (epoch + 1) % 10 == 0 or epoch == start_epoch or epoch == args.epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                  f"val_acc={val_acc:.4f} val_ppl={val_ppl:.1f} "
                  f"elapsed={elapsed:.0f}s")

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": vars(args),
            }, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

            # Keep only latest 3
            import glob
            ckpts = sorted(glob.glob(str(ckpt_dir / "checkpoint_epoch*.pt")))
            for old in ckpts[:-3]:
                Path(old).unlink()

        torch.cuda.empty_cache()

        # Check stop flag (graceful GPU yield)
        if args.stop_flag_file and Path(args.stop_flag_file).exists():
            print(f"Stop flag detected ({args.stop_flag_file}), saving checkpoint and exiting...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": vars(args),
            }, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")
            with open(ckpt_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)
            return
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": vars(args),
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    summary = {
        "n_params": n_params,
        "n_train": n_train,
        "n_val": n_val,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "verdict": "FEASIBLE" if val_ppl < 10 else ("MARGINAL" if val_ppl < 50 else "NOT_FEASIBLE"),
    }
    print(f"\nMDLM Training Complete: PPL={val_ppl:.1f}, Acc={val_acc:.4f}, Verdict={summary['verdict']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/mdlm_full")
    parser.add_argument("--output_dir", default="results/fullscale_mdlm")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=390)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stop_flag_file", type=str, default=None,
                        help="Path to stop-flag file; exits gracefully after current epoch if file exists")
    args = parser.parse_args()
    train_mdlm(args)

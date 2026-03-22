"""Train AR generation model on patch token sequences.

Usage:
    python scripts/train_ar.py \
        --sequence_dir data/sequences/lvis_wide \
        --checkpoint_dir data/checkpoints/ar_rvq \
        --mode rvq --epochs 300 --batch_size 4
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
import gc

from src.ar_model import PatchGPT
from src.patch_dataset import MeshSequenceDataset
from src.patch_sequence import compute_vocab_size, compute_vocab_size_rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/ar")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Codebook K (1024 for RVQ, 4096 for SimVQ)")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Linear LR warmup epochs before cosine decay")
    parser.add_argument("--rotation", action="store_true",
                        help="Use 11-token rotation format (pos+scale+rot_quat+codebook)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stop_flag_file", type=str, default=None,
                        help="Path to stop-flag file; exits gracefully after current epoch if file exists")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = MeshSequenceDataset(args.sequence_dir, mode=args.mode, max_seq_len=args.max_seq_len,
                                  use_rotation=args.rotation)
    print(f"Sequences: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.rotation:
        vocab_size = compute_vocab_size_rot(codebook_K=args.codebook_size)
        tokens_per_patch = 11
    else:
        vocab_size = compute_vocab_size(codebook_K=args.codebook_size)
        tokens_per_patch = 7
    print(f"Vocab size: {vocab_size} (codebook K={args.codebook_size}, rotation={args.rotation}, "
          f"tokens_per_patch={tokens_per_patch})")

    model = PatchGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PatchGPT: {n_params / 1e6:.1f}M params")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR schedule: linear warmup → cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8 / args.lr,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    start_epoch = 0
    history = []
    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.max_seq_len,
        "rotation": args.rotation,
        "tokens_per_patch": tokens_per_patch,
    }

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    # PLACEHOLDER_TRAINING_LOOP

    print(f"Grad accumulation: {args.grad_accum_steps} steps "
          f"(effective batch = {args.batch_size * args.grad_accum_steps})")
    print(f"LR schedule: {args.warmup_epochs} warmup epochs → cosine to epoch {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (input_ids, target_ids) in enumerate(loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / args.grad_accum_steps
            loss.backward()

            total_loss += loss.item() * args.grad_accum_steps  # unscaled for logging
            n_batches += 1

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        torch.cuda.empty_cache()

        metrics = {"epoch": epoch, "loss": avg_loss, "lr": lr_now, "time_sec": elapsed}
        history.append(metrics)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": config,
            }, ckpt_path)
            # Keep only latest 3 checkpoints
            old_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))[:-3]
            for old in old_ckpts:
                old.unlink()

        gc.collect()

        # Check stop flag (graceful GPU yield)
        if args.stop_flag_file and Path(args.stop_flag_file).exists():
            print(f"Stop flag detected ({args.stop_flag_file}), saving checkpoint and exiting...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": config,
            }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")
            with open(ckpt_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)
            return
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": config,
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Final checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()

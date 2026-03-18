"""Train AR generation model on patch token sequences.

Usage:
    python scripts/train_ar.py \
        --sequence_dir data/sequences/lvis_wide \
        --checkpoint_dir data/checkpoints/ar_rvq \
        --mode rvq --epochs 100 --batch_size 32
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import gc

from src.ar_model import PatchGPT
from src.patch_dataset import MeshSequenceDataset
from src.patch_sequence import compute_vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/ar")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Codebook K (1024 for RVQ, 4096 for SimVQ)")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = MeshSequenceDataset(args.sequence_dir, mode=args.mode, max_seq_len=args.max_seq_len)
    print(f"Sequences: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    vocab_size = compute_vocab_size(codebook_K=args.codebook_size)
    print(f"Vocab size: {vocab_size} (codebook K={args.codebook_size})")

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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    history = []
    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.max_seq_len,
    }

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        torch.cuda.empty_cache()

        metrics = {"epoch": epoch, "loss": avg_loss, "time_sec": elapsed}
        history.append(metrics)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | {elapsed:.1f}s")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": config,
            }, ckpt_path)
            # Keep only latest 3 checkpoints
            old_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))[:-3]
            for old in old_ckpts:
                old.unlink()

        gc.collect()

    # Final checkpoint
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config,
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Final checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()

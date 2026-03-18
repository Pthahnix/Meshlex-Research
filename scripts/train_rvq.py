"""Train MeshLex RVQ-VAE on preprocessed patches.

Usage:
    python scripts/train_rvq.py \
        --train_dirs data/patches/lvis_wide/seen_train \
        --val_dirs data/patches/lvis_wide/seen_test \
        --checkpoint_dir data/checkpoints/rvq_lvis \
        --epochs 200 --batch_size 256
"""
import argparse
import torch
from torch.utils.data import ConcatDataset

from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", default=None)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_kv_tokens", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints/rvq")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_datasets = [PatchGraphDataset(d) for d in args.train_dirs]
    train_dataset = ConcatDataset(train_datasets)
    print(f"Training patches: {len(train_dataset)}")

    val_dataset = None
    if args.val_dirs:
        val_datasets = [PatchGraphDataset(d) for d in args.val_dirs]
        val_dataset = ConcatDataset(val_datasets)
        print(f"Validation patches: {len(val_dataset)}")

    model = MeshLexRVQVAE(
        codebook_size=args.codebook_size,
        n_levels=args.n_levels,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_kv_tokens=args.num_kv_tokens,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RVQ-VAE: {n_params:,} total, {n_trainable:,} trainable")
    print(f"Codebook: {args.n_levels} levels x K={args.codebook_size}")

    ckpt_data = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Resumed from {args.resume}")
        if not missing and not unexpected:
            ckpt_data = ckpt

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        resume_checkpoint=ckpt_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()

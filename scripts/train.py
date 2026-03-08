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
    parser.add_argument("--encoder_warmup_epochs", type=int, default=10,
                        help="Epochs of encoder-only training before K-means init + VQ (default 10)")
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (loads model + optimizer state)")
    parser.add_argument("--use_rotation", action="store_true",
                        help="Use rotation trick instead of straight-through (B-stage)")
    parser.add_argument("--num_kv_tokens", type=int, default=1,
                        help="Number of KV tokens in decoder (1=A-stage, 4=B-stage)")
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
        use_rotation=args.use_rotation,
        num_kv_tokens=args.num_kv_tokens,
    )

    stage = "B-stage" if (args.use_rotation or args.num_kv_tokens > 1) else "A-stage"
    print(f"Model stage: {stage} (rotation={args.use_rotation}, kv_tokens={args.num_kv_tokens})")

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
        encoder_warmup_epochs=args.encoder_warmup_epochs,
    )
    trainer.train()


if __name__ == "__main__":
    main()

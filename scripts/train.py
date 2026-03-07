"""Train MeshLex VQ-VAE on preprocessed patches."""
import argparse
import torch

from src.model import MeshLexVQVAE
from src.patch_dataset import PatchGraphDataset
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True,
                        help="Patch directories for training (e.g., data/patches/chair data/patches/table)")
    parser.add_argument("--val_dirs", nargs="+", default=None,
                        help="Patch directories for validation")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vq_start_epoch", type=int, default=20)
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
    )

    # Resume from checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed model from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

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
        vq_start_epoch=args.vq_start_epoch,
    )
    trainer.train()


if __name__ == "__main__":
    main()

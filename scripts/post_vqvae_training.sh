#!/bin/bash
# Post-training automation: rename to exp format, upload checkpoints, start encoding.
# Run this when VQ-VAE training completes.
# Usage: PYTHONPATH=. bash scripts/post_vqvae_training.sh

set -e

echo "=== Post-VQ-VAE Training Pipeline ==="
echo ""

# Naming convention: exp{N}_{stage}_{data} (per RULE.md)
declare -A RENAME_MAP=(
    ["rvq_full_pca"]="exp5_vqvae_fullscale_pca_k1024"
    ["rvq_full_nopca"]="exp6_vqvae_fullscale_nopca_k1024"
    ["rvq_full_pca_k512"]="exp7_vqvae_fullscale_pca_k512"
    ["rvq_full_pca_k2048"]="exp8_vqvae_fullscale_pca_k2048"
)

CKPT_BASE=/data/pthahnix/MeshLex-Research/checkpoints
SEQ_BASE=/data/pthahnix/MeshLex-Research/sequences

# Step 1: Rename checkpoint dirs to exp format
echo "Step 1: Renaming checkpoints to exp format..."
for old in "${!RENAME_MAP[@]}"; do
    new="${RENAME_MAP[$old]}"
    if [ -d "$CKPT_BASE/$old" ] && [ ! -d "$CKPT_BASE/$new" ]; then
        mv "$CKPT_BASE/$old" "$CKPT_BASE/$new"
        echo "  ✅ $old → $new"
    elif [ -d "$CKPT_BASE/$new" ]; then
        echo "  ⏭️  $new already exists"
    else
        echo "  ❌ $old: NOT FOUND"
    fi
done
echo ""

# Step 2: Verify all checkpoints exist
echo "Step 2: Verifying checkpoints..."
for new in exp5_vqvae_fullscale_pca_k1024 exp6_vqvae_fullscale_nopca_k1024 exp7_vqvae_fullscale_pca_k512 exp8_vqvae_fullscale_pca_k2048; do
    ckpt="$CKPT_BASE/${new}/checkpoint_final.pt"
    if [ -f "$ckpt" ]; then
        size=$(du -h "$ckpt" | cut -f1)
        echo "  ✅ $new: $size"
    else
        echo "  ❌ $new: checkpoint_final.pt NOT FOUND"
    fi
done
echo ""

# Step 3: Upload checkpoints to HF
echo "Step 3: Uploading checkpoints to HuggingFace..."
source ~/.bashrc
export http_proxy=http://127.0.0.1:17892 https_proxy=http://127.0.0.1:17892

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
variants = [
    'exp5_vqvae_fullscale_pca_k1024',
    'exp6_vqvae_fullscale_nopca_k1024',
    'exp7_vqvae_fullscale_pca_k512',
    'exp8_vqvae_fullscale_pca_k2048',
]
for name in variants:
    for fname in ['checkpoint_final.pt', 'training_history.json', 'config.json']:
        path = f'/data/pthahnix/MeshLex-Research/checkpoints/{name}/{fname}'
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f'checkpoints/{name}/{fname}',
                repo_id='Pthahnix/MeshLex-Research', repo_type='model',
            )
            print(f'  ✅ {name}/{fname} uploaded')
        except Exception as e:
            print(f'  ❌ {name}/{fname}: {e}')
print('Upload complete.')
"
echo ""

# Step 4: Start token encoding (only PCA k1024 and noPCA k1024 — the main variants)
echo "Step 4: Starting token encoding..."
ARROW_DIR=/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits/seen_train
FEAT_PCA=/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/features/seen_train
FEAT_NOPCA=/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/features/seen_train_nopca

# Encode PCA sequences on GPU 1
echo "  Starting PCA encoding on GPU 1..."
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH=. python3 scripts/encode_sequences.py \
    --arrow_dir $ARROW_DIR \
    --feature_dir $FEAT_PCA \
    --checkpoint $CKPT_BASE/exp5_vqvae_fullscale_pca_k1024/checkpoint_final.pt \
    --output_dir $SEQ_BASE/exp5_vqvae_fullscale_pca_k1024 \
    --mode rvq --batch_size 4096 \
    > results/fullscale_eval/encode_pca.log 2>&1 &
echo "    PID: $!"

# Encode noPCA sequences on GPU 2
echo "  Starting noPCA encoding on GPU 2..."
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 PYTHONPATH=. python3 scripts/encode_sequences.py \
    --arrow_dir $ARROW_DIR \
    --feature_dir $FEAT_NOPCA \
    --checkpoint $CKPT_BASE/exp6_vqvae_fullscale_nopca_k1024/checkpoint_final.pt \
    --output_dir $SEQ_BASE/exp6_vqvae_fullscale_nopca_k1024 \
    --mode rvq --nopca --batch_size 4096 \
    > results/fullscale_eval/encode_nopca.log 2>&1 &
echo "    PID: $!"

echo ""
echo "Encoding started. Monitor with:"
echo "  tail -f results/fullscale_eval/encode_pca.log"
echo "  tail -f results/fullscale_eval/encode_nopca.log"

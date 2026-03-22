#!/bin/bash
# Monitor VQ-VAE training progress and report status.
# Usage: bash scripts/monitor_training.sh

echo "=== VQ-VAE Training Status at $(date) ==="
echo ""

# Training dirs use old names (rvq_full_*), will be renamed to exp format post-training
declare -A EXP_NAMES=(
    ["pca"]="exp5_vqvae_fullscale_pca_k1024"
    ["nopca"]="exp6_vqvae_fullscale_nopca_k1024"
    ["pca_k512"]="exp7_vqvae_fullscale_pca_k512"
    ["pca_k2048"]="exp8_vqvae_fullscale_pca_k2048"
)

for variant in pca nopca pca_k512 pca_k2048; do
    log="results/fullscale_eval/train_rvq_${variant}.log"
    # Check both old and new naming
    ckpt_dir="/data/pthahnix/MeshLex-Research/checkpoints/rvq_full_${variant}"
    if [ ! -d "$ckpt_dir" ]; then
        ckpt_dir="/data/pthahnix/MeshLex-Research/checkpoints/${EXP_NAMES[$variant]}"
    fi

    if [ ! -f "$log" ]; then
        echo "[$variant] No log file"
        continue
    fi

    # Check if process is running
    pid=$(pgrep -f "train_rvq.*rvq_full_${variant}" 2>/dev/null | head -1)
    if [ -z "$pid" ]; then
        status="STOPPED"
    else
        status="RUNNING (PID $pid)"
    fi

    # Get last epoch and chunk info
    last_epoch=$(grep "^Epoch " "$log" | tail -1)
    last_chunk=$(grep "chunk " "$log" | tail -1)
    final_ckpt=$(ls "$ckpt_dir"/checkpoint_final.pt 2>/dev/null)

    echo "[${EXP_NAMES[$variant]}] $status"
    if [ -n "$final_ckpt" ]; then
        echo "  COMPLETED: checkpoint_final.pt exists"
    elif [ -n "$last_epoch" ]; then
        echo "  Latest: $last_epoch"
    elif [ -n "$last_chunk" ]; then
        echo "  Latest: $last_chunk"
    else
        echo "  No epoch/chunk output yet"
    fi
    echo ""
done

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ""
echo "=== Memory ==="
free -h | head -3

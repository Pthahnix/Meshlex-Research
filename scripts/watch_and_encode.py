#!/usr/bin/env python3
"""Watch for VQ-VAE training completion and auto-trigger encoding + HF upload.

Run as background daemon after training starts:
    nohup PYTHONPATH=. python3 scripts/watch_and_encode.py \
        > results/fullscale_eval/watch_encode.log 2>&1 &
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

CKPT_BASE = Path("/data/pthahnix/MeshLex-Research/checkpoints")
SEQ_BASE = Path("/data/pthahnix/MeshLex-Research/sequences")
ARROW_BASE = Path("/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits")
FEAT_BASE = Path("/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/features")
LOG_DIR = Path("results/fullscale_eval")
REPO_DIR = Path("/home/pthahnix/MeshLex-Research")

JOBS = [
    {
        "pid": 3203377,
        "name": "PCA K=1024",
        "exp_name": "rvq_full_pca",
        "ckpt_dir": CKPT_BASE / "rvq_full_pca",
        "encode_gpu": "1",
        "encode_cfg": "pca",
    },
    {
        "pid": 3203378,
        "name": "noPCA K=1024",
        "exp_name": "rvq_full_nopca",
        "ckpt_dir": CKPT_BASE / "rvq_full_nopca",
        "encode_gpu": "2",
        "encode_cfg": "nopca",
    },
    {
        "pid": 3251999,
        "name": "PCA K=512",
        "exp_name": "rvq_full_pca_k512",
        "ckpt_dir": CKPT_BASE / "rvq_full_pca_k512",
        "encode_gpu": "2",
        "encode_cfg": "pca_k512",
    },
    {
        "pid": 3934532,
        "name": "PCA K=2048",
        "exp_name": "rvq_full_pca_k2048",
        "ckpt_dir": CKPT_BASE / "rvq_full_pca_k2048",
        "encode_gpu": "2",
        "encode_cfg": "pca_k2048",
    },
]

ENCODE_CFGS = {
    "pca": {
        "feat_dir": FEAT_BASE / "seen_train",
        "out_dir": SEQ_BASE / "rvq_full_pca",
        "extra": [],
    },
    "nopca": {
        "feat_dir": FEAT_BASE / "seen_train_nopca",
        "out_dir": SEQ_BASE / "rvq_full_nopca",
        "extra": ["--nopca"],
    },
    "pca_k512": {
        "feat_dir": FEAT_BASE / "seen_train",
        "out_dir": SEQ_BASE / "rvq_full_pca_k512",
        "extra": [],
    },
    "pca_k2048": {
        "feat_dir": FEAT_BASE / "seen_train",
        "out_dir": SEQ_BASE / "rvq_full_pca_k2048",
        "extra": [],
    },
}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_alive(pid):
    return Path(f"/proc/{pid}").exists()


def upload_checkpoint(exp_name, ckpt_dir):
    log(f"Uploading {exp_name} to HuggingFace...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        for fname in ["checkpoint_final.pt", "training_history.json", "config.json"]:
            fpath = ckpt_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"checkpoints/{exp_name}/{fname}",
                    repo_id="Pthahnix/MeshLex-Research",
                    repo_type="model",
                )
                log(f"  ✅ {fname}")
            else:
                log(f"  ⚠ {fname} not found")
        log(f"✅ Checkpoint uploaded: HF:Pthahnix/MeshLex-Research/checkpoints/{exp_name}/")
    except Exception as e:
        log(f"❌ HF upload failed: {e}")


def start_encoding(encode_cfg, ckpt_path, gpu):
    """Start encoding subprocess. Returns Popen handle."""
    cfg = ENCODE_CFGS[encode_cfg]
    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"encode_{encode_cfg}.log"
    log(f"▶ Encoding {encode_cfg} on GPU {gpu} → {out_dir}")

    cmd = [
        sys.executable, "scripts/encode_sequences.py",
        "--arrow_dir", str(ARROW_BASE / "seen_train"),
        "--feature_dir", str(cfg["feat_dir"]),
        "--checkpoint", str(ckpt_path),
        "--output_dir", str(out_dir),
        "--mode", "rvq",
        "--batch_size", "4096",
    ] + cfg["extra"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(REPO_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    lf = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO_DIR))
    log(f"  PID {proc.pid}, log: {log_file}")
    return proc, lf


def main():
    log("=== watch_and_encode.py started ===")
    for job in JOBS:
        log(f"  Watching PID {job['pid']}: {job['name']}")

    # Track state
    trained = set()       # job exp_name: training done, checkpoint exists
    uploaded = set()      # exp_name: HF upload done
    encoding_procs = {}   # encode_cfg → (proc, log_file_handle)
    encoded = set()       # encode_cfg: encoding done

    # GPU-level sequential queue: one active encoding per GPU at a time
    gpu_queue = {"1": [], "2": []}    # GPU → list of (encode_cfg, ckpt_path, job_exp_name)
    gpu_active = {"1": None, "2": None}  # GPU → (proc, log_file, encode_cfg) or None

    CHECK_INTERVAL = 60  # seconds

    while True:
        # 1. Check for newly completed training jobs
        for job in JOBS:
            exp = job["exp_name"]
            if exp in trained:
                continue
            ckpt_final = job["ckpt_dir"] / "checkpoint_final.pt"
            if not is_alive(job["pid"]) and ckpt_final.exists():
                log(f"🎉 TRAINING COMPLETE: {job['name']}")
                trained.add(exp)
                # Queue HF upload then encoding
                if exp not in uploaded:
                    upload_checkpoint(exp, job["ckpt_dir"])
                    uploaded.add(exp)
                # Add to GPU queue
                gpu = job["encode_gpu"]
                gpu_queue[gpu].append((job["encode_cfg"], ckpt_final))
                log(f"  Queued encode: {job['encode_cfg']} on GPU {gpu}")
            elif not is_alive(job["pid"]) and not ckpt_final.exists() and exp not in trained:
                log(f"⚠ PID {job['pid']} ({job['name']}) dead but no checkpoint_final.pt yet — waiting...")

        # 2. Manage encoding per GPU (sequential: start next when current finishes)
        for gpu in ["1", "2"]:
            active = gpu_active[gpu]
            if active is not None:
                proc, lf, enc_cfg = active
                ret = proc.poll()
                if ret is None:
                    pass  # Still running
                elif ret == 0:
                    lf.close()
                    count = len(list(ENCODE_CFGS[enc_cfg]["out_dir"].glob("*_sequence.npz")))
                    log(f"✅ Encoding complete: {enc_cfg} on GPU {gpu} — {count} sequences")
                    encoded.add(enc_cfg)
                    gpu_active[gpu] = None
                else:
                    lf.close()
                    log(f"❌ Encoding FAILED: {enc_cfg} on GPU {gpu} (return code {ret})")
                    gpu_active[gpu] = None

            # If GPU is now free and there's work queued, start next job
            if gpu_active[gpu] is None and gpu_queue[gpu]:
                enc_cfg, ckpt_path = gpu_queue[gpu].pop(0)
                if enc_cfg not in encoded:
                    proc, lf = start_encoding(enc_cfg, ckpt_path, gpu)
                    gpu_active[gpu] = (proc, lf, enc_cfg)

        # 3. Summary
        running_train = sum(1 for j in JOBS if j["exp_name"] not in trained and is_alive(j["pid"]))
        running_encode = sum(1 for gpu in ["1", "2"] if gpu_active[gpu] is not None)
        log(f"Status: trained={len(trained)}/4 encoding={len(encoded)}/4 | train_running={running_train} encode_running={running_encode}")

        # 4. All done?
        if len(trained) == 4 and len(encoded) == 4:
            log("")
            log("=" * 60)
            log("🎊 ALL TRAINING AND ENCODING COMPLETE!")
            log("=" * 60)
            log("Next steps (Task 10 — AR Training):")
            log("  GPU 1 → PCA AR:   scripts/train_ar.py --seq_dir /data/.../sequences/rvq_full_pca ...")
            log("  GPU 2 → noPCA AR: scripts/train_ar.py --seq_dir /data/.../sequences/rvq_full_nopca ...")
            log("")
            log("Also ready to start in parallel:")
            log("  Task 11: Preliminary analysis rerun")
            log("  Task 13: Theory-driven analysis (K ablation, VQ comparison, curvature)")
            log("  Task 14: MDLM training")
            # Commit encoding logs
            try:
                subprocess.run(["git", "add", "results/fullscale_eval/"], cwd=str(REPO_DIR), check=False)
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m",
                     "phase2a: token encoding complete — PCA + noPCA + K ablation"],
                    cwd=str(REPO_DIR), check=False
                )
                subprocess.run(["git", "push"], cwd=str(REPO_DIR), check=False,
                                env={**os.environ})
                log("✅ Committed and pushed encoding logs")
            except Exception as e:
                log(f"⚠ Git commit failed: {e}")
            break

        time.sleep(CHECK_INTERVAL)

    log("=== watch_and_encode.py finished ===")


if __name__ == "__main__":
    main()

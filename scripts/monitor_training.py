#!/usr/bin/env python3
"""Monitor running VQ-VAE training processes and report status."""

import subprocess
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

LOG_DIR = Path("/home/pthahnix/MeshLex-Research/results/fullscale_eval")
CKPT_DIR = Path("/data/pthahnix/MeshLex-Research/checkpoints")

JOBS = {
    "PCA K=1024": {
        "log": LOG_DIR / "train_rvq_pca.log",
        "ckpt": CKPT_DIR / "rvq_full_pca",
        "pid": 3203377,
    },
    "noPCA K=1024": {
        "log": LOG_DIR / "train_rvq_nopca.log",
        "ckpt": CKPT_DIR / "rvq_full_nopca",
        "pid": 3203378,
    },
    "PCA K=512": {
        "log": LOG_DIR / "train_rvq_pca_k512.log",
        "ckpt": CKPT_DIR / "rvq_full_pca_k512",
        "pid": 3251999,
    },
    "PCA K=2048": {
        "log": LOG_DIR / "train_rvq_pca_k2048.log",
        "ckpt": CKPT_DIR / "rvq_full_pca_k2048",
        "pid": 3252000,
    },
}

TOTAL_EPOCHS = 15
WARMUP_EPOCHS = 2


def is_alive(pid):
    try:
        return Path(f"/proc/{pid}").exists()
    except Exception:
        return False


def get_process_uptime(pid):
    """Get process uptime in seconds."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        # Field 22 is starttime in clock ticks
        starttime_ticks = int(stat[21])
        clk_tck = int(subprocess.check_output(["getconf", "CLK_TCK"]).strip())
        uptime_sec = float(Path("/proc/uptime").read_text().split()[0])
        boot_time = time.time() - uptime_sec
        start_sec = boot_time + starttime_ticks / clk_tck
        return time.time() - start_sec
    except Exception:
        return 0


def parse_last_epoch(log_path):
    """Parse log file for the last completed epoch info."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    # Find all epoch lines
    epoch_lines = re.findall(
        r"Epoch (\d+)\s+(\[warmup\]\s+)?\|"
        r" loss ([\d.]+) \|"
        r" recon ([\d.]+) \|"
        r" commit ([\d.]+) \|"
        r" embed ([\d.]+) \|"
        r" util ([\d.]+)% \|"
        r" lr ([\d.e+-]+) \|"
        r" ([\d.]+)s",
        text,
    )
    # Find last chunk progress line
    chunk_lines = re.findall(r"chunk (\d+)/(\d+) done, loss ([\d.]+)", text)
    last_epoch = None
    if epoch_lines:
        e = epoch_lines[-1]
        last_epoch = {
            "epoch": int(e[0]),
            "warmup": bool(e[1].strip()),
            "loss": float(e[2]),
            "recon": float(e[3]),
            "commit": float(e[4]),
            "util": float(e[6]),
            "lr": e[7],
            "time_s": float(e[8]),
        }
    last_chunk = None
    if chunk_lines:
        c = chunk_lines[-1]
        last_chunk = {
            "chunk": int(c[0]),
            "total_chunks": int(c[1]),
            "loss": float(c[2]),
        }
    return last_epoch, last_chunk


def get_gpu_info():
    """Get GPU utilization and memory info."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpus = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "id": int(parts[0]),
                "util_pct": int(parts[1]),
                "mem_used_mb": int(parts[2]),
                "mem_total_mb": int(parts[3]),
                "name": parts[4],
            })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def get_our_gpu_pids():
    """Get GPU process info for our PIDs."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        our_pids = {j["pid"] for j in JOBS.values()}
        result = {}
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            pid = int(parts[0])
            if pid in our_pids:
                result[pid] = {"gpu_uuid": parts[1], "mem_mb": int(parts[2])}
        return result
    except Exception:
        return {}


def estimate_progress(last_epoch_info, uptime_sec):
    """Estimate current epoch based on elapsed time and avg epoch time."""
    if last_epoch_info is None:
        return "unknown"
    epoch = last_epoch_info["epoch"]
    avg_time = last_epoch_info["time_s"]
    # Time since training started = epoch completed epochs * avg_time + current epoch progress
    completed_time = (epoch + 1) * avg_time
    remaining_uptime = uptime_sec - completed_time
    if remaining_uptime > 0 and avg_time > 0:
        extra_epochs = remaining_uptime / avg_time
        est_epoch = epoch + 1 + extra_epochs
        est_epoch = min(est_epoch, TOTAL_EPOCHS)
        current_epoch = int(est_epoch)
        intra_progress = (est_epoch - current_epoch) * 100
        return current_epoch, intra_progress
    return epoch + 1, 0


def report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  MeshLex VQ-VAE Training Monitor — {now}")
    print(f"{'='*70}")

    # GPU info
    gpus = get_gpu_info()
    gpu_pids = get_our_gpu_pids()
    print("\n## GPU Status")
    for g in gpus:
        if "error" in g:
            print(f"  Error: {g['error']}")
            continue
        print(
            f"  GPU {g['id']} ({g['name']}): "
            f"{g['util_pct']}% util | "
            f"{g['mem_used_mb']}/{g['mem_total_mb']} MB"
        )

    print(f"\n## Training Jobs (total: {TOTAL_EPOCHS} epochs, warmup: {WARMUP_EPOCHS})")
    print(f"{'—'*70}")

    for name, info in JOBS.items():
        pid = info["pid"]
        alive = is_alive(pid)
        status = "RUNNING" if alive else "DEAD"

        print(f"\n  [{status}] {name} (PID {pid})")

        if alive:
            uptime = get_process_uptime(pid)
            uptime_str = str(timedelta(seconds=int(uptime)))
            print(f"    Uptime: {uptime_str}")

            gpu_info = gpu_pids.get(pid)
            if gpu_info:
                print(f"    GPU mem: {gpu_info['mem_mb']} MB")

        result = parse_last_epoch(info["log"])
        if result is None:
            print("    No log data yet")
            continue

        last_epoch, last_chunk = result
        if last_epoch:
            phase = "warmup" if last_epoch["warmup"] else "full_vq"
            print(
                f"    Last logged epoch: {last_epoch['epoch']}/{TOTAL_EPOCHS-1} [{phase}]"
            )
            print(
                f"    Loss: {last_epoch['loss']:.4f} | "
                f"Recon: {last_epoch['recon']:.4f} | "
                f"Commit: {last_epoch['commit']:.4f} | "
                f"Util: {last_epoch['util']:.1f}%"
            )
            print(f"    LR: {last_epoch['lr']} | Epoch time: {last_epoch['time_s']:.0f}s")

            if alive:
                est = estimate_progress(last_epoch, uptime)
                if isinstance(est, tuple):
                    est_epoch, intra = est
                    print(
                        f"    *Estimated current: epoch ~{est_epoch} "
                        f"(~{intra:.0f}% through epoch) "
                        f"[{est_epoch}/{TOTAL_EPOCHS} = {est_epoch/TOTAL_EPOCHS*100:.0f}% total]*"
                    )

        if last_chunk:
            print(
                f"    Last logged chunk: {last_chunk['chunk']}/{last_chunk['total_chunks']} "
                f"(loss {last_chunk['loss']:.4f})"
            )

        # Check for checkpoints
        ckpt_dir = info["ckpt"]
        ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if ckpts:
            print(f"    Checkpoints: {[c.name for c in ckpts]}")
        else:
            print("    Checkpoints: none yet (saves every 3 epochs after code fix)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    report()

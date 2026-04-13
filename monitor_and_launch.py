#!/usr/bin/env python3
"""
Monitor 1k test training. Once steps 1-2-3 all show declining loss,
kill the 1k run and launch the full 50k training automatically.
"""
import re
import time
import subprocess
import os
import sys

LOG_1K  = "/home/ubuntu/dgxsparkfinetune/output/pediatrician_gemma_1k_test.log"
LOG_50K = "/home/ubuntu/dgxsparkfinetune/output/pediatrician_gemma_50k/train.log"
PID_FILE = "/home/ubuntu/dgxsparkfinetune/output/50k_pid.txt"

LOSS_RE = re.compile(r'Step (\d+): loss=([\d.]+)')

def tail_from(path, pos):
    try:
        with open(path) as f:
            f.seek(pos)
            chunk = f.read()
            return chunk, f.tell()
    except Exception:
        return "", pos

def launch_50k():
    os.makedirs("/home/ubuntu/dgxsparkfinetune/output/pediatrician_gemma_50k", exist_ok=True)
    log_f = open(LOG_50K, "w")
    proc = subprocess.Popen(
        [
            "/home/ubuntu/venv/bin/python",
            "/home/ubuntu/dgxsparkfinetune/finetune_dgx_spark.py",
            "--model",        "gemma-4-31b",
            "--method",       "qlora",
            "--dataset",      "/home/ubuntu/medAI/pediatrics_50k.parquet",
            "--question-col", "question",
            "--answer-col",   "answer",
            "--epochs",       "1",
            "--output-dir",   "/home/ubuntu/dgxsparkfinetune/output/pediatrician_gemma_50k",
        ],
        stdout=log_f,
        stderr=log_f,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    log(f"50k training launched — PID={proc.pid}")
    log(f"Log: {LOG_50K}")

def log(msg):
    print(msg, flush=True)

# ── Step 1 is already confirmed at 3.1039 ────────────────────────────────────
seen   = {1: 3.1039}   # step → loss
prev   = 3.1039
declines = 1           # step 1 counts as the baseline

log("Monitor active. Waiting for steps 2 and 3 (need 2 more declining losses)…")

pos = 0
while True:
    chunk, pos = tail_from(LOG_1K, pos)
    for m in LOSS_RE.finditer(chunk):
        step = int(m.group(1))
        loss = float(m.group(2))
        if step in seen:
            continue
        seen[step] = loss
        arrow = "↓" if loss < prev else "↑"
        log(f"  Step {step}: loss={loss:.4f}  {arrow}  (prev={prev:.4f})")

        if loss < prev:
            declines += 1
            log(f"  Declining step #{declines}/3")
        else:
            log(f"  WARNING: loss did not decline at step {step} — watching one more step")

        prev = loss

        if declines >= 3:
            log("\nLoss confirmed declining for 3 consecutive steps. Proceeding.")
            log("Killing 1k run…")
            subprocess.run(["pkill", "-9", "-f", "finetune_dgx_spark.py"],
                           capture_output=True)
            time.sleep(5)          # let GPU memory clear
            log("Launching 50k training…")
            launch_50k()
            log("Done. Monitor exiting.")
            sys.exit(0)

        # Safety: if we see step > 5 and still not 3 declines, launch anyway
        # (early steps can wobble; overall trend from step 1 is what matters)
        if step >= 5 and loss < seen[1]:
            log(f"\nStep {step} loss ({loss:.4f}) < step 1 loss ({seen[1]:.4f}).")
            log("Overall trend is down. Proceeding with 50k launch.")
            subprocess.run(["pkill", "-9", "-f", "finetune_dgx_spark.py"],
                           capture_output=True)
            time.sleep(5)
            launch_50k()
            log("Done. Monitor exiting.")
            sys.exit(0)

    time.sleep(20)

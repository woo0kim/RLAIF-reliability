"""Run a queue of jobs sequentially on a fixed set of GPUs, with retries (jobs resume from checkpoints).

A queue file has one job per line: `<name> [after=<file>[,<file>...]] <command...>`; blank lines and
#-comments are ignored. A job waits until all its `after=` files exist, and is skipped when logs/<name>.done exists. Several queues (one per GPU slot) run in parallel:

  nohup python scripts/run_queue.py --gpus 0,1 --queue queues/slotA.txt > logs/slotA.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="CUDA_VISIBLE_DEVICES for this slot, e.g. 0,1")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--log_dir", default="logs")
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_VERBOSITY="error", PYTHONUNBUFFERED="1")
    while True:
        # re-read the queue each time so jobs can be appended while the slot is running
        jobs = []
        for line in open(a.queue):
            line = line.strip()
            if line and not line.startswith("#"):
                name, cmd = line.split(None, 1)
                after = []
                if cmd.startswith("after="):
                    dep, cmd = cmd.split(None, 1)
                    after = dep[len("after="):].split(",")
                jobs.append((name, cmd, after))
        pending = [(n, c, d) for n, c, d in jobs if not os.path.exists(os.path.join(a.log_dir, f"{n}.done")) and not os.path.exists(os.path.join(a.log_dir, f"{n}.failed"))]
        if not pending:
            print(f"[{time.ctime()}] queue empty", flush=True)
            return
        name, cmd, after = pending[0]
        if not all(os.path.exists(f) for f in after):
            time.sleep(60)
            continue
        log = os.path.join(a.log_dir, f"{name}.log")
        ok = False
        for attempt in range(a.retries):
            print(f"[{time.ctime()}] start {name} (attempt {attempt + 1}) on GPUs {a.gpus}: {cmd}", flush=True)
            with open(log, "a") as fh:
                rc = subprocess.call(cmd, shell=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"[{time.ctime()}] end {name} rc={rc}", flush=True)
            if rc == 0:
                ok = True
                break
            time.sleep(60)
        open(os.path.join(a.log_dir, f"{name}.{'done' if ok else 'failed'}"), "w").close()


if __name__ == "__main__":
    sys.exit(main())

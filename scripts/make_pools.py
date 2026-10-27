"""Causal test pools: remove the least-verifiable vs the largest-solve-gap problems (vs random).

Each arm drops the same number of problems (default 25% of the pool), chosen
from the pre-training diagnostics of one judge:

  drop_lowver  lowest per-problem Youden J (ties: highest best-of-16 attack success)
  drop_gap     largest solve gap (policy solve rate - judge solve rate)
  drop_random  uniformly random (control)

  python scripts/make_pools.py --task countdown --judge Qwen/Qwen3-0.6B
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlaif.utils import read_jsonl, short_model_name, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--diag_dir", default="diag")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = read_jsonl(os.path.join(a.data_dir, a.task, "train.jsonl"))
    f = pd.read_csv(os.path.join(a.diag_dir, a.task, "features.csv"))
    f = f[(f.judge == short_model_name(a.judge)) & (f["mode"] == a.mode)].set_index("pid")
    k = int(round(a.frac * len(pool)))
    lowver = list(f.sort_values(["youden", "bon_fool_16"], ascending=[True, False]).index[:k])
    gap = list(f.sort_values(["solve_gap", "youden"], ascending=[False, True]).index[:k])
    rnd = random.Random(a.seed).sample([p["pid"] for p in pool], k)
    arms = {"drop_lowver": set(lowver), "drop_gap": set(gap), "drop_random": set(rnd)}
    stats = {"judge": a.judge, "mode": a.mode, "k": k, "n_pool": len(pool)}
    for name, drop in arms.items():
        keep = [p for p in pool if p["pid"] not in drop]
        write_jsonl(os.path.join(a.data_dir, a.task, f"{name}.jsonl"), keep)
        sub = f.loc[list(drop)]
        stats[name] = {
            "n_keep": len(keep),
            "dropped_mean_youden": float(sub.youden.mean()),
            "dropped_mean_solve_gap": float(sub.solve_gap.mean()),
            "dropped_mean_bon_fool_16": float(sub.bon_fool_16.mean()),
            "dropped_mean_policy_solve": float(sub.policy_solve.mean()),
        }
    stats["overlap_lowver_gap"] = len(arms["drop_lowver"] & arms["drop_gap"])
    with open(os.path.join(a.data_dir, a.task, "pools.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

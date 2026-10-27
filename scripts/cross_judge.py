"""Do exploits transfer across judges? Score each run's base and late-training rollouts with every judge.

For each standard full-pool run, take up to --n rollouts from its first and last epoch and score them
with each judge in standard mode. Reports, per (run, epoch, scoring judge), the judge's acceptance of
gold-wrong rollouts (FPR) and of gold-correct ones (TPR). A hack that fools the training judge and also
the held-out judges is a property of the presentation, not of the training judge.

  CUDA_VISIBLE_DEVICES=0 python scripts/cross_judge.py --task countdown
"""
import argparse
import glob
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlaif.grpo import load_problems  # noqa: E402
from rlaif.judge import Judge  # noqa: E402
from rlaif.tasks import get_task  # noqa: E402
from rlaif.utils import load_config, read_jsonl, short_model_name  # noqa: E402

JUDGES = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = load_config(f"configs/{a.task}.yaml")
    task = get_task(a.task, **cfg.get("task_kwargs", {}))
    pmap = {p.pid: p for p in load_problems(os.path.join(cfg["data_dir"], a.task, "train.jsonl"))}
    rng = random.Random(0)
    items = []
    for d in sorted(glob.glob(os.path.join(a.runs_dir, a.task, "*_standard_train_s0"))):
        c = json.load(open(os.path.join(d, "config.json")))
        roll = read_jsonl(os.path.join(d, "rollouts.jsonl"))
        last = max(r["epoch"] for r in roll)
        for tag, ep in (("first", 0), ("last", last)):
            sel = [r for r in roll if r["epoch"] == ep]
            for r in rng.sample(sel, min(a.n, len(sel))):
                items.append({"run": os.path.basename(d), "train_judge": short_model_name(c["judge"]), "epoch": tag, "pid": r["pid"], "text": r["text"], "gold": r["gold"]})
    print(f"{len(items)} rollouts to score", flush=True)
    rows = []
    for jn in JUDGES:
        judge = Judge(jn, task, a.device, token_budget=6000 if "4B" in jn else 12000)
        p_yes, _ = judge.score([pmap[it["pid"]] for it in items], [it["text"] for it in items], "standard")
        for it, p in zip(items, p_yes):
            rows.append({**{k: v for k, v in it.items() if k != "text"}, "score_judge": short_model_name(jn), "p_yes": float(p), "accept": bool(p > 0.5)})
        del judge
        import torch

        torch.cuda.empty_cache()
        print(f"scored with {jn}", flush=True)
    df = pd.DataFrame(rows)
    out = df.groupby(["train_judge", "epoch", "score_judge"]).apply(
        lambda g: pd.Series({"fpr": g[~g.gold].accept.mean(), "tpr": g[g.gold].accept.mean(), "gold_acc": g.gold.mean(), "n": len(g)})
    ).reset_index()
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, f"cross_judge_{a.task}.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

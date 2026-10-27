"""Qualitative look at exploitation: judge-accepted but gold-wrong rollouts, early vs late in a run.

  python scripts/show_examples.py runs/countdown/j0.6b_standard_train_s0 --n 4 > results/examples_cd_j0.6b.md
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlaif.utils import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import json

    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    probs = {r["pid"]: r for r in read_jsonl(os.path.join(a.data_dir, cfg["task"], f"{cfg['pool']}.jsonl"))}
    roll = read_jsonl(os.path.join(a.run_dir, "rollouts.jsonl"))
    last = max(r["epoch"] for r in roll)
    rng = random.Random(a.seed)
    print(f"# Exploitation examples: `{os.path.basename(a.run_dir.rstrip('/'))}`\n")
    for title, ep in (("First epoch", 0), ("Last epoch", last)):
        fp = [r for r in roll if r["epoch"] == ep and r["accept"] and not r["gold"]]
        tot = sum(1 for r in roll if r["epoch"] == ep)
        print(f"## {title} (epoch {ep}): {len(fp)}/{tot} rollouts accepted but wrong\n")
        for r in rng.sample(fp, min(a.n, len(fp))):
            p = probs[r["pid"]]
            gold = p["gold"].get("answer", p["gold"])
            print(f"**{r['pid']}** (P(Yes)={r['p_yes']:.3f}, {r['n_tokens']} tokens, gold: `{gold}`)\n")
            print(f"> {p['question'][:400]}\n")
            print("```text\n" + r["text"][-1200:] + "\n```\n")


if __name__ == "__main__":
    main()

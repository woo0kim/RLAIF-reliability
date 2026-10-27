"""Build the training pool and held-out evaluation set for a task.

  python scripts/prepare_data.py --task countdown
  python scripts/prepare_data.py --task math --levels 1 2 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlaif.tasks import get_task  # noqa: E402
from rlaif.utils import read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["countdown", "math"])
    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--n_heldout", type=int, default=128)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="data")
    a = ap.parse_args()
    task = get_task(a.task, **({"levels": a.levels} if a.task == "math" else {}))
    out = os.path.join(a.data_dir, a.task)
    for split, n in (("train", a.n_train), ("heldout", a.n_heldout)):
        probs = task.load_problems(split, n, seed=a.seed)
        write_jsonl(os.path.join(out, f"{split}.jsonl"), [p.to_dict() for p in probs])
        print(f"{a.task}/{split}: {len(probs)} problems")
    train = {r["pid"] for r in read_jsonl(os.path.join(out, "train.jsonl"))}
    held = {r["pid"] for r in read_jsonl(os.path.join(out, "heldout.jsonl"))}
    assert not (train & held), "train/held-out overlap"


if __name__ == "__main__":
    main()

"""Pre-training diagnostics (run before any RL).

  python scripts/run_diagnostics.py --task countdown --stage policy   --device cuda:0
  python scripts/run_diagnostics.py --task countdown --stage judge    --judge Qwen/Qwen3-4B --device cuda:0
  python scripts/run_diagnostics.py --task countdown --stage features
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlaif.diagnostics import build_candidates, compute_features, run_judge, sample_policy  # noqa: E402
from rlaif.grpo import load_problems  # noqa: E402
from rlaif.tasks import get_task  # noqa: E402
from rlaif.utils import load_config, read_jsonl, short_model_name, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--stage", required=True, choices=["policy", "judge", "features"])
    ap.add_argument("--config", default=None, help="default: configs/<task>.yaml")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parts", default="commit,solve,score_standard,score_commit_first", help="judge stage parts to run")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config or f"configs/{a.task}.yaml", a.overrides)
    assert cfg["task"] == a.task
    task = get_task(a.task, **cfg.get("task_kwargs", {}))
    problems = load_problems(os.path.join(cfg["data_dir"], a.task, f"{cfg['pool']}.jsonl"))
    out = os.path.join(cfg["diag_dir"], a.task)
    os.makedirs(out, exist_ok=True)
    max_new = cfg["max_new_tokens"]

    if a.stage == "policy":
        path = os.path.join(out, "policy_samples.jsonl")
        if not os.path.exists(path):
            rows = sample_policy(task, problems, cfg["policy"], a.device, cfg["diag_policy_k"], max_new, cfg["gen_batch_size"], seed=cfg["seed"])
            write_jsonl(path, rows)
        rows = read_jsonl(path)
        write_jsonl(os.path.join(out, "candidates.jsonl"), build_candidates(task, problems, rows, seed=cfg["seed"]))
        print(f"policy solve rate {sum(r['gold'] for r in rows) / len(rows):.3f} over {len(rows)} samples")
    elif a.stage == "judge":
        need = any(x.startswith("score") for x in a.parts.split(","))
        cands = read_jsonl(os.path.join(out, "candidates.jsonl")) if need else []
        run_judge(
            task, problems, cands, a.judge, a.device, os.path.join(out, short_model_name(a.judge)),
            k_solve=cfg["diag_judge_k"], max_new_tokens=max_new, batch_size=cfg["gen_batch_size"],
            token_budget=cfg["judge_token_budget"], seed=cfg["seed"], parts=tuple(a.parts.split(",")),
        )
    else:
        feats, glob = compute_features(out, problems, cfg["diag_judges"])
        print(glob.to_string(index=False))


if __name__ == "__main__":
    main()

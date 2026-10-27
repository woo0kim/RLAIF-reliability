"""End-to-end check of scripts/analyze.py on synthetic diagnostics + training logs (CPU only).

Synthetic world: problems with low pre-training Youden J get exploited earlier; solve gap is noise.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_world(tmp, seed=0):
    rng = np.random.default_rng(seed)
    diag, runs = os.path.join(tmp, "diag"), os.path.join(tmp, "runs")
    feats, glob_rows = [], []
    for task in ("countdown", "math"):
        os.makedirs(os.path.join(diag, task), exist_ok=True)
        pids = [f"{task}-{i}" for i in range(40)]
        for judge in ("qwen3-0.6b", "qwen3-1.7b"):
            for mode in ("standard", "commit_first"):
                for pid in pids:
                    y = rng.uniform(-0.2, 1.0)
                    ps = rng.uniform(0.2, 0.9)
                    feats.append({"task": task, "judge": judge, "mode": mode, "pid": pid, "youden": y, "youden_nat": y, "bon_fool_16": max(0, 0.6 - y), "bon_fool_4": max(0, 0.4 - y),
                                  "solve_gap": rng.uniform(-0.5, 0.5), "policy_solve": ps, "judge_solve": rng.uniform(0, 1), "base_fp": (1 - ps) * max(0.0, 0.5 - y / 2) * 0.5,
                                  "syn_neg_accept": rng.random(), "auroc": 0.5 + y / 2, "n_nat": 12, "fpr_nat": max(0.0, 0.5 - y / 2)})
                glob_rows.append({"task": task, "judge": judge, "mode": mode, "tpr": 0.95, "fpr": 0.3, "youden": 0.65, "base_accept": 0.7, "base_gold": 0.6, "judge_solve": 0.5, "fp_share_low_tercile": 0.6, "frac_problems_neg_youden": 0.1})
        pd.DataFrame([f for f in feats if f["task"] == task]).to_csv(os.path.join(diag, task, "features.csv"), index=False)
        pd.DataFrame([g for g in glob_rows if g["task"] == task]).to_csv(os.path.join(diag, task, "global.csv"), index=False)
    F = pd.DataFrame(feats)
    for task in ("countdown", "math"):
        for judge in ("qwen3-0.6b", "qwen3-1.7b"):
            for mode, pool in (("standard", "train"), ("commit_first", "train"), ("standard", "drop_lowver")):
                name = f"{judge.replace('qwen3-', 'j')}_{mode}_{pool}_s0"
                d = os.path.join(runs, task, name)
                os.makedirs(d, exist_ok=True)
                json.dump({"task": task, "judge": f"Qwen/{judge.replace('qwen3', 'Qwen3').replace('b', 'B')}", "judge_mode": mode, "pool": pool, "lr": 5e-5}, open(os.path.join(d, "config.json"), "w"))
                f = F[(F.task == task) & (F.judge == judge) & (F["mode"] == mode)].set_index("pid")
                pids = list(f.index)[:32]
                rolls, steps = [], []
                for step in range(64):
                    epoch = step // 4
                    batch = pids[(step % 4) * 8 : (step % 4 + 1) * 8]
                    for pid in batch:
                        p_exploit = min(0.95, 0.1 + epoch * 0.08 * (1.2 - f.loc[pid, "youden"]))
                        for k in range(8):
                            gold = rng.random() < f.loc[pid, "policy_solve"] * (1 - 0.5 * p_exploit)
                            accept = gold or rng.random() < p_exploit
                            rolls.append({"step": step, "epoch": epoch, "visit": epoch + 1, "pid": pid, "k": k, "reward": float(accept), "p_yes": 0.9 if accept else 0.1, "accept": bool(accept), "gold": bool(gold), "mon_p": None, "n_tokens": 200, "finished": True, "answer": "x", "answer_key": "x"})
                    r = [x for x in rolls if x["step"] == step]
                    steps.append({"step": step, "epoch": epoch, "reward": np.mean([x["reward"] for x in r]), "accept": np.mean([x["accept"] for x in r]), "gold_acc": np.mean([x["gold"] for x in r]),
                                  "fp_rate": np.mean([x["accept"] and not x["gold"] for x in r]), "mean_len": 200 + step, "mon_disagree": min(1, step / 64 + rng.normal(0, 0.02))})
                with open(os.path.join(d, "rollouts.jsonl"), "w") as fh:
                    fh.writelines(json.dumps(x) + "\n" for x in rolls)
                with open(os.path.join(d, "steps.jsonl"), "w") as fh:
                    fh.writelines(json.dumps(x) + "\n" for x in steps)
                with open(os.path.join(d, "evals.jsonl"), "w") as fh:
                    for s in range(0, 65, 16):
                        fh.write(json.dumps({"step": s, "gold_acc": 0.6 + 0.1 * np.sin(s / 20) - s / 400, "judge_accept": 0.7 + s / 200}) + "\n")
    return diag, runs


def test_analyze_end_to_end(tmp_path):
    diag, runs = _make_world(str(tmp_path))
    out = os.path.join(str(tmp_path), "results")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze.py"), "--diag_dir", diag, "--runs_dir", runs, "--out", out], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    for f in ("REPORT.md", "summary.json", "runs_summary.csv", "hazard_heldout_summary.csv", "early_stopping.csv", "causal_arms.csv", "figures/hazard_cindex.png", "figures/km_terciles.png"):
        assert os.path.exists(os.path.join(out, f)), f
    h = pd.read_csv(os.path.join(out, "hazard_heldout_summary.csv"))
    hp = h[h.split == "heldout_problems"].set_index("features").c_index
    assert hp["verifiability"] > hp["solve gap"]

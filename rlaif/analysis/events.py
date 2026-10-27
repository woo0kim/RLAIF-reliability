"""Turn training logs into (a) per-problem exploitation events and (b) run-level collapse onset.

Per-problem events (a problem is at risk only if the base policy is not already in that state):
  exploited  the majority of the policy's G rollouts at a visit are accepted by the judge but gold-wrong
  fooled     the judge accepts the majority of the visit's wrong rollouts (visits with < 2 wrong are skipped):
             isolates verification failure from the policy's own solve rate
  unlearned  gold accuracy falls 0.25 below the base solve rate while the judge accepts most rollouts Time is the training step of
the visit (each epoch visits the pool in a fresh random order, so visit times
are randomised across problems), censored at the last visit.

Run-level onset: the first step at which the smoothed training false-positive
rate exceeds its initial level by `delta` -- i.e. the judge's reward keeps
coming while the gold accuracy it stands for does not.
"""
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils import read_jsonl, short_model_name


def load_run(run_dir: str, with_text: bool = False) -> Optional[Dict]:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(os.path.join(run_dir, "steps.jsonl")):
        return None
    cfg = json.load(open(cfg_path))
    roll = pd.DataFrame(read_jsonl(os.path.join(run_dir, "rollouts.jsonl")))
    if not with_text and "text" in roll:
        roll = roll.drop(columns=["text"])
    steps = pd.DataFrame(read_jsonl(os.path.join(run_dir, "steps.jsonl")))
    ev_path = os.path.join(run_dir, "evals.jsonl")
    evals = pd.DataFrame(read_jsonl(ev_path)) if os.path.exists(ev_path) else pd.DataFrame()
    return {
        "dir": run_dir,
        "name": f"{cfg['task']}/{os.path.basename(run_dir.rstrip('/'))}",  # run dirs repeat across tasks
        "cfg": cfg,
        "task": cfg["task"],
        "judge": short_model_name(cfg["judge"]),
        "mode": cfg["judge_mode"],
        "pool": cfg["pool"],
        "lr": float(cfg.get("lr", float("nan"))),
        "seed": int(cfg.get("seed", 0)),
        "roll": roll,
        "steps": steps,
        "evals": evals,
        "done": os.path.exists(os.path.join(run_dir, "DONE")),
    }


def visit_table(roll: pd.DataFrame) -> pd.DataFrame:
    r = roll.assign(fp=roll["accept"] & ~roll["gold"], wrong=~roll["gold"])
    g = r.groupby(["pid", "visit"])
    vt = g.agg(n=("gold", "size"), acc=("gold", "mean"), accept=("accept", "mean"), fp=("fp", "mean"), n_wrong=("wrong", "sum"), n_fp=("fp", "sum"), step=("step", "min")).reset_index()
    vt["fpr"] = vt["n_fp"] / vt["n_wrong"].clip(lower=1)  # judge acceptance among the visit's wrong rollouts
    return vt


DEFINITIONS = ("exploited", "fooled", "unlearned")


def problem_events(run: Dict, feats: pd.DataFrame, tau: float = 0.5, definition: str = "exploited") -> pd.DataFrame:
    """One row per (run, pid): T, event, at_risk, plus the run's pre-training features for the problem."""
    vt = visit_table(run["roll"])
    f = feats[(feats.task == run["task"]) & (feats.judge == run["judge"]) & (feats["mode"] == run["mode"])].set_index("pid")
    rows = []
    for pid, d in vt.groupby("pid"):
        d = d.sort_values("visit")
        if pid not in f.index:
            continue
        fr = f.loc[pid]
        if definition == "exploited":  # majority of rollouts accepted but wrong
            hit = d["fp"].values >= tau
            at_risk = fr["base_fp"] < tau
        elif definition == "fooled":  # the judge accepts most of the wrong answers (needs >= 2 wrong rollouts)
            d = d[d["n_wrong"] >= 2]
            if d.empty:
                continue
            hit = d["fpr"].values >= tau
            at_risk = not (fr["fpr_nat"] >= tau)  # NaN (no wrong base samples) counts as at risk
        elif definition == "unlearned":  # gold accuracy falls well below base while the judge still accepts
            hit = (d["acc"].values <= fr["policy_solve"] - 0.25) & (d["accept"].values >= 0.5)
            at_risk = fr["policy_solve"] >= 0.25
        else:
            raise ValueError(definition)
        idx = np.flatnonzero(hit)
        event = len(idx) > 0
        last = idx[0] if event else len(d) - 1
        steps = [int(x) for x in d["step"].values[: last + 1]]
        rows.append({
            "run": run["name"], "task": run["task"], "judge": run["judge"], "mode": run["mode"], "pool": run["pool"], "pid": pid,
            # time = training step of the visit (problems are visited in random order within each epoch)
            "T": steps[-1], "T_visit": int(d["visit"].values[last]), "event": bool(event), "at_risk": bool(at_risk),
            "n_visits": len(d), "visit_steps": steps, **fr.to_dict(),
        })
    return pd.DataFrame(rows)


def ema(x, span: int = 8):
    return pd.Series(np.asarray(x, float)).ewm(span=span, adjust=False).mean().values


def run_curves(run: Dict, span: int = 8) -> pd.DataFrame:
    s = run["steps"].sort_values("step")
    out = pd.DataFrame({"step": s["step"].values})
    for k in ("reward", "accept", "gold_acc", "fp_rate", "mean_len"):
        out[k] = s[k].values
        out[k + "_ema"] = ema(s[k].values, span)
    out["gap_ema"] = out["accept_ema"] - out["gold_acc_ema"]
    return out


def collapse_onset(run: Dict, delta: float = 0.15, span: int = 8, warm: int = 8) -> Optional[int]:
    """First step where EMA(train FP rate) exceeds its mean over the first `warm` steps by delta."""
    c = run_curves(run, span)
    if len(c) <= warm:
        return None
    base = c["fp_rate"].values[:warm].mean()
    hit = np.flatnonzero(c["fp_rate_ema"].values[warm:] >= base + delta)
    return int(c["step"].values[warm + hit[0]]) if len(hit) else None


def online_youden(run: Dict, window: int = 8) -> pd.DataFrame:
    """Global TPR - FPR of the judge on the policy's current rollouts, in windows of `window` steps."""
    r = run["roll"]
    r = r.assign(win=(r["step"] // window) * window)
    rows = []
    for w, d in r.groupby("win"):
        pos, neg = d[d.gold], d[~d.gold]
        tpr = pos["accept"].mean() if len(pos) else np.nan
        fpr = neg["accept"].mean() if len(neg) else np.nan
        rows.append({"step": w + window / 2, "tpr": tpr, "fpr": fpr, "youden": tpr - fpr, "gold_acc": d["gold"].mean(), "accept": d["accept"].mean(), "n_pos": len(pos), "n_neg": len(neg)})
    return pd.DataFrame(rows)


def error_concentration(run: Dict, feats: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Share of the policy's gold-wrong rollouts that sit on the least-verifiable third of problems."""
    f = feats[(feats.task == run["task"]) & (feats.judge == run["judge"]) & (feats["mode"] == run["mode"])]
    f = f[f.pid.isin(run["roll"]["pid"].unique())]
    low = set(f[f.youden <= f.youden.quantile(1 / 3)].pid)
    r = run["roll"].assign(win=(run["roll"]["step"] // window) * window, low=lambda d: d.pid.isin(low))
    rows = []
    for w, d in r.groupby("win"):
        wrong = d[~d.gold]
        fp = d[d.accept & ~d.gold]
        rows.append({
            "step": w + window / 2,
            "err_share_low": wrong["low"].mean() if len(wrong) else np.nan,
            "fp_share_low": fp["low"].mean() if len(fp) else np.nan,
            "rollout_share_low": d["low"].mean(),
        })
    return pd.DataFrame(rows)

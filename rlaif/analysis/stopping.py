"""Label-free early stopping.

Every signal below is computable during training without gold answers:

  reward_plateau   naive: the judge reward stops improving
  monitor_disagree the training judge accepts, but the same judge in the other
                   mode (commit-first <-> standard) rejects, on a few groups per step
  lowver_gap       reward on problems the pre-training audit flagged as hard to
                   verify (bottom third of Youden J) minus reward on the easiest to
                   verify (top third): gains that come only from unverifiable
                   problems are suspicious
  length_drift     relative change of the mean response length

A rule fires at the first step (after a warm-up) where the smoothed signal
crosses baseline + threshold; the policy is then rolled back to the last saved
checkpoint at or before that step. Quality is the held-out *gold* accuracy of
that checkpoint, compared with the best (oracle) checkpoint and the last one.
Thresholds are tuned on one task and applied unchanged to the other.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .events import ema

SIGNALS = ("reward_plateau", "monitor_disagree", "lowver_gap", "length_drift")
GRID = {
    "reward_plateau": [0.005, 0.01, 0.02, 0.04],
    "monitor_disagree": [0.05, 0.1, 0.15, 0.2, 0.3],
    "lowver_gap": [0.05, 0.1, 0.15, 0.2, 0.3],
    "length_drift": [0.15, 0.25, 0.4, 0.6],
}


def signal_series(run: Dict, feats: pd.DataFrame) -> pd.DataFrame:
    s = run["steps"].sort_values("step").reset_index(drop=True)
    out = pd.DataFrame({"step": s["step"].values, "reward": s["reward"].values, "mean_len": s["mean_len"].values})
    out["monitor_disagree"] = s["mon_disagree"].astype(float).values if "mon_disagree" in s else np.nan
    f = feats[(feats.task == run["task"]) & (feats.judge == run["judge"]) & (feats["mode"] == run["mode"])]
    f = f[f.pid.isin(run["roll"]["pid"].unique())]
    lo = set(f[f.youden <= f.youden.quantile(1 / 3)].pid)
    hi = set(f[f.youden >= f.youden.quantile(2 / 3)].pid)
    r = run["roll"]
    gl = r[r.pid.isin(lo)].groupby("step")["reward"].mean()
    gh = r[r.pid.isin(hi)].groupby("step")["reward"].mean()
    out["r_low"] = out["step"].map(gl)
    out["r_high"] = out["step"].map(gh)
    return out


def _stop_step(sig: pd.DataFrame, rule: str, thr: float, warm: int = 8, span: int = 8, patience: int = 8) -> Optional[int]:
    steps = sig["step"].values
    if rule == "reward_plateau":
        r = ema(sig["reward"].values, span)
        best, since = -np.inf, 0
        for i, v in enumerate(r):
            if v > best + thr:
                best, since = v, 0
            else:
                since += 1
            if i >= warm and since >= patience:
                return int(steps[i])
        return None
    if rule == "monitor_disagree":
        x = sig["monitor_disagree"].astype(float).ffill().fillna(0).values
    elif rule == "lowver_gap":
        lo = pd.Series(sig["r_low"].values).ffill().fillna(0).values
        hi = pd.Series(sig["r_high"].values).ffill().fillna(0).values
        x = lo - hi
    elif rule == "length_drift":
        L = sig["mean_len"].values
        x = np.abs(L / max(L[:warm].mean(), 1) - 1)
    else:
        raise ValueError(rule)
    xs = ema(x, span)
    base = x[:warm].mean() if rule != "length_drift" else 0.0
    hit = np.flatnonzero(xs[warm:] >= base + thr)
    return int(steps[warm + hit[0]]) if len(hit) else None


def evaluate_rule(run: Dict, sig: pd.DataFrame, rule: str, thr: float) -> Dict:
    ev = run["evals"].sort_values("step")
    s = _stop_step(sig, rule, thr)
    final = ev.iloc[-1]
    if s is None:
        chosen = final
    else:
        prior = ev[ev.step <= s]
        chosen = prior.iloc[-1] if len(prior) else ev.iloc[0]
    oracle = ev["gold_acc"].max()
    return {
        "run": run["name"], "task": run["task"], "rule": rule, "thr": thr,
        "stop_step": s, "ckpt_step": int(chosen["step"]),
        "gold_at_stop": float(chosen["gold_acc"]), "oracle_gold": float(oracle), "final_gold": float(final["gold_acc"]),
        "regret": float(oracle - chosen["gold_acc"]), "gain_vs_final": float(chosen["gold_acc"] - final["gold_acc"]),
    }


def cross_task_stopping(runs, feats: pd.DataFrame) -> pd.DataFrame:
    """Tune each rule's threshold on one task (min mean regret), report it on the other task."""
    sigs = {r["name"]: signal_series(r, feats) for r in runs}
    grid_rows = [evaluate_rule(r, sigs[r["name"]], rule, thr) for r in runs for rule in SIGNALS for thr in GRID[rule]]
    grid = pd.DataFrame(grid_rows)
    out = []
    tasks = sorted(grid.task.unique())
    for test in tasks:
        tune = [t for t in tasks if t != test] or [test]
        for rule in SIGNALS:
            g = grid[(grid.rule == rule) & grid.task.isin(tune)].groupby("thr")["regret"].mean()
            if g.empty:
                continue
            thr = float(g.idxmin())
            d = grid[(grid.rule == rule) & (grid.task == test) & (grid.thr == thr)]
            out.append({"test_task": test, "rule": rule, "thr": thr, "mean_regret": d.regret.mean(), "mean_gain_vs_final": d.gain_vs_final.mean(), "fired": d.stop_step.notna().mean(), "n_runs": len(d)})
        fin = grid[(grid.task == test)].drop_duplicates("run")
        out.append({"test_task": test, "rule": "no stopping (final)", "thr": np.nan, "mean_regret": (fin.oracle_gold - fin.final_gold).mean(), "mean_gain_vs_final": 0.0, "fired": 0.0, "n_runs": len(fin)})
    return pd.DataFrame(out), grid

"""Per-problem discrete-time hazard model of exploitation onset.

    logit h(t | x) = a0 + a1 log(1 + t) + a2 t/100 + beta . x

fit by L2-regularised logistic regression on person-period rows (one row per
problem per visit while still at risk; t = training step of the visit). x are *pre-training* features of the
(problem, judge) pair. We compare feature sets -- the solve gap (the "policy
out-solves its judge" story) against judge verifiability (per-problem Youden
index and best-of-N attack success) -- on held-out problems, held-out judges
and a held-out task, by Harrell's C-index and held-out log-likelihood.
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

FEATURE_SETS: Dict[str, List[str]] = {
    "time only": [],
    "solve gap": ["solve_gap"],
    "solve rates": ["solve_gap", "policy_solve", "judge_solve"],
    "base FP rate": ["base_fp"],
    "verifiability": ["youden", "bon_fool_16"],
    "verifiability + solve": ["youden", "bon_fool_16", "solve_gap", "policy_solve", "judge_solve"],
}
TIME = ["log_t", "t100"]


def person_period(units: pd.DataFrame) -> pd.DataFrame:
    """One row per visit while at risk (`visit_steps`), y = 1 only at the visit with the observed event."""
    u = units.reset_index(drop=True).copy()
    if "visit_steps" not in u:  # synthetic data: visits at t = 1..T
        u["visit_steps"] = [list(range(1, int(t) + 1)) for t in u["T"]]
    u["n_rows"] = u["visit_steps"].apply(len)
    reps = u.loc[u.index.repeat(u["n_rows"])].copy()
    reps["v"] = reps.groupby(level=0).cumcount() + 1
    reps["t"] = [s[v - 1] for s, v in zip(reps["visit_steps"], reps["v"])]
    reps["y"] = ((reps["v"] == reps["n_rows"]) & reps["event"]).astype(int)
    reps["log_t"] = np.log1p(reps["t"])
    reps["t100"] = reps["t"] / 100.0
    reps["unit"] = reps["run"] + "|" + reps["pid"]
    return reps.drop(columns=["visit_steps"]).reset_index(drop=True)


def concordance(T: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: among comparable pairs (i had the event before j's time), how often risk_i > risk_j."""
    T, event, risk = np.asarray(T, float), np.asarray(event, bool), np.asarray(risk, float)
    num = den = 0.0
    for i in np.flatnonzero(event):
        comp = (T > T[i]) | ((T == T[i]) & ~event)  # j still at risk after i's event
        comp[i] = False
        if not comp.any():
            continue
        den += comp.sum()
        num += (risk[i] > risk[comp]).sum() + 0.5 * (risk[i] == risk[comp]).sum()
    return num / den if den else float("nan")


class HazardModel:
    def __init__(self, features: Sequence[str], C: float = 1.0):
        self.features = list(features)
        self.C = C

    def _X(self, df, fit=False):
        cols = TIME + self.features
        X = df[cols].astype(float).values
        if fit:
            self.mu = np.nanmean(X, 0)
            self.sd = np.nanstd(X, 0) + 1e-8
        X = (X - self.mu) / self.sd
        return np.nan_to_num(X)

    def fit(self, pp: pd.DataFrame):
        self.clf = LogisticRegression(C=self.C, max_iter=2000)
        self.clf.fit(self._X(pp, fit=True), pp["y"].values)
        return self

    def hazard(self, pp: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._X(pp))[:, 1]

    def risk(self, units: pd.DataFrame) -> np.ndarray:
        """Time-invariant risk score beta . x (the time terms are shared by all units)."""
        if not self.features:
            return np.zeros(len(units))
        k = len(TIME)
        X = (units[self.features].astype(float).values - self.mu[k:]) / self.sd[k:]
        return np.nan_to_num(X) @ self.clf.coef_[0][k:]

    def coefficients(self) -> Dict[str, float]:
        return dict(zip(TIME + self.features, self.clf.coef_[0]))


def _metrics(model: HazardModel, units: pd.DataFrame, pp: pd.DataFrame) -> Dict[str, float]:
    h = np.clip(model.hazard(pp), 1e-6, 1 - 1e-6)
    risk = model.risk(units)
    out = {
        "c_index": concordance(units["T"].values, units["event"].values, risk),
        "nll_per_row": float(log_loss(pp["y"].values, h, labels=[0, 1])),
        "n_units": len(units),
        "n_events": int(units["event"].sum()),
    }
    out["auc_rows"] = float(roc_auc_score(pp["y"].values, h)) if 0 < pp["y"].sum() < len(pp) else float("nan")
    ever = units["event"].values.astype(int)
    out["auc_ever"] = float(roc_auc_score(ever, risk)) if 0 < ever.sum() < len(ever) and np.ptp(risk) > 0 else float("nan")
    return out


def split_iter(units: pd.DataFrame, split: str, n_folds: int = 5, seed: int = 0):
    """Yields (name, train_mask, test_mask) over units."""
    if split == "heldout_problems":
        groups = units["task"] + "/" + units["pid"]
        gkf = GroupKFold(n_splits=n_folds)
        for k, (tr, te) in enumerate(gkf.split(units, groups=groups)):
            m_tr = np.zeros(len(units), bool)
            m_tr[tr] = True
            yield f"fold{k}", m_tr, ~m_tr
    elif split == "heldout_judge":
        for j in sorted(units["judge"].unique()):
            te = (units["judge"] == j).values
            if te.all() or not te.any():
                continue
            yield f"judge={j}", ~te, te
    elif split == "heldout_task":
        for t in sorted(units["task"].unique()):
            te = (units["task"] == t).values
            if te.all() or not te.any():
                continue
            yield f"task={t}", ~te, te
    else:
        raise ValueError(split)


def evaluate_feature_sets(units: pd.DataFrame, splits=("heldout_problems", "heldout_judge", "heldout_task"), feature_sets=FEATURE_SETS, C: float = 1.0) -> pd.DataFrame:
    """Out-of-sample metrics per (split, fold, feature set). units: at-risk (run, pid) rows."""
    units = units.reset_index(drop=True)
    rows = []
    for split in splits:
        for fold, tr, te in split_iter(units, split):
            u_tr, u_te = units[tr], units[te]
            pp_tr, pp_te = person_period(u_tr), person_period(u_te)
            if pp_tr["y"].sum() == 0 or u_te["event"].sum() == 0:
                continue
            for name, feats in feature_sets.items():
                m = HazardModel(feats, C).fit(pp_tr)
                rows.append({"split": split, "fold": fold, "features": name, **_metrics(m, u_te, pp_te)})
    return pd.DataFrame(rows)


def oof_risk(units: pd.DataFrame, features: Sequence[str], split: str = "heldout_problems", C: float = 1.0) -> np.ndarray:
    """Out-of-fold risk scores (for paired bootstrap comparisons of C-index)."""
    units = units.reset_index(drop=True)
    risk = np.full(len(units), np.nan)
    for _, tr, te in split_iter(units, split):
        m = HazardModel(features, C).fit(person_period(units[tr]))
        r = m.risk(units[te])
        risk[te] = (r - r.mean()) / (r.std() + 1e-9)  # put folds on a common scale
    return risk


def bootstrap_delta_c(units: pd.DataFrame, risk_a: np.ndarray, risk_b: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Paired bootstrap over units of C(a) - C(b). Returns (delta, lo, hi)."""
    rng = np.random.default_rng(seed)
    T, E = units["T"].values, units["event"].values
    d0 = concordance(T, E, risk_a) - concordance(T, E, risk_b)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(units), len(units))
        ds.append(concordance(T[i], E[i], risk_a[i]) - concordance(T[i], E[i], risk_b[i]))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return float(d0), float(lo), float(hi)


def univariate_c(units: pd.DataFrame, cols: Sequence[str], signs: Dict[str, int]) -> pd.DataFrame:
    """Assumption-free check: C-index of each raw pre-training feature as a risk score."""
    rows = []
    for c in cols:
        r = signs.get(c, 1) * units[c].astype(float).fillna(units[c].median()).values
        rows.append({"feature": c, "direction": "+" if signs.get(c, 1) > 0 else "-", "c_index": concordance(units["T"].values, units["event"].values, r)})
    return pd.DataFrame(rows)


def kaplan_meier(units: pd.DataFrame, max_t: int) -> pd.DataFrame:
    """KM estimate of P(not yet exploited) against training step."""
    S, rows = 1.0, [{"v": 0, "surv": 1.0}]
    for t in sorted(set(units.loc[units["event"], "T"].astype(int))):
        at_risk = (units["T"] >= t).sum()
        d = ((units["T"] == t) & units["event"]).sum()
        if at_risk:
            S *= 1 - d / at_risk
        rows.append({"v": t, "surv": S, "at_risk": int(at_risk)})
    rows.append({"v": max_t, "surv": S, "at_risk": int((units["T"] >= max_t).sum())})
    return pd.DataFrame(rows)

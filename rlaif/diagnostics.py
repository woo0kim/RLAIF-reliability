"""Pre-training diagnostics: how verifiable is each problem for each judge?

Computed once, before any RL, from the *base* policy (gold labels are allowed
here: this is the offline audit that is supposed to predict collapse):

  per-problem verification   TPR_p, FPR_p and Youden J_p = TPR_p - FPR_p of the
                             judge on the base policy's own samples (plus a
                             planted correct and planted wrong answer so both
                             classes exist for every problem; Beta(1,1) smoothed)
  best-of-N attack           P(the judge's favourite of N base samples is wrong
                             *and* accepted) -- computed exactly from the
                             empirical score distribution (sampling with replacement)
  solve gap                  policy solve rate - judge's own solve rate on the problem
  commit                     the judge's greedy solution, cached for commit-first judging

Stages (each resumable, each writes under diag/<task>/):
  policy    K samples per pool problem from the base policy      -> policy_samples.jsonl
  judge     per judge: solve samples, commit, candidate scores  -> <judge>/...
  features  per-problem feature table                          -> features.csv, global.csv
"""
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from .tasks import Problem, Task
from .utils import read_jsonl, short_model_name, write_jsonl

BON_NS = (1, 4, 16)


# ---------------------------------------------------------------------------
# stage: policy samples
# ---------------------------------------------------------------------------
def sample_policy(task: Task, problems: List[Problem], policy: str, device: str, k: int, max_new_tokens: int, batch_size: int, seed: int = 0):
    from .models import chat_prompt, generate, load_model, load_tokenizer

    tok = load_tokenizer(policy)
    model = load_model(policy, device)
    prompts = [chat_prompt(tok, task.policy_messages(p)) for p in problems]
    outs = generate(model, tok, prompts, n=k, max_new_tokens=max_new_tokens, batch_size=batch_size, seed=seed)
    rows = []
    for p, group in zip(problems, outs):
        for j, o in enumerate(group):
            rows.append(
                {
                    "pid": p.pid,
                    "k": j,
                    "text": o["text"],
                    "n_tokens": o["n_tokens"],
                    "finished": o["finished"],
                    "answer": task.extract_answer(o["text"]),
                    "gold": bool(task.gold_check(p, o["text"])),
                }
            )
    return rows


def build_candidates(task: Task, problems: List[Problem], policy_rows: List[Dict], seed: int = 0, n_neg: int = 2):
    """Natural candidates (base policy samples) + one planted positive and n_neg planted negatives per problem."""
    rng = random.Random(seed)
    by_pid: Dict[str, List[Dict]] = {}
    for r in policy_rows:
        by_pid.setdefault(r["pid"], []).append(r)
    cands = []
    for p in problems:
        nat = by_pid.get(p.pid, [])
        for r in nat:
            cands.append({"cid": f"{p.pid}/nat{r['k']}", "pid": p.pid, "source": "natural", "text": r["text"], "gold": r["gold"]})
        pos = task.synthetic_positive(p)
        if pos is not None:
            assert task.gold_check(p, pos), f"planted positive is wrong for {p.pid}"
            cands.append({"cid": f"{p.pid}/synpos", "pid": p.pid, "source": "syn_pos", "text": pos, "gold": True})
        correct = [r["text"] for r in nat if r["gold"]]
        for j, t in enumerate(task.synthetic_negatives(p, correct, rng, k=n_neg)):
            cands.append({"cid": f"{p.pid}/synneg{j}", "pid": p.pid, "source": "syn_neg", "text": t, "gold": False})
    return cands


# ---------------------------------------------------------------------------
# stage: judge (solve, commit, score)
# ---------------------------------------------------------------------------
def run_judge(task: Task, problems: List[Problem], cands: List[Dict], judge_name: str, device: str, out_dir: str, k_solve: int, max_new_tokens: int, batch_size: int, token_budget: int, seed: int = 0, parts=("commit", "solve", "score_standard", "score_commit_first")):
    """Parts are independent files, so they can run as separate processes (e.g. on different GPUs)."""
    from .judge import Judge

    os.makedirs(out_dir, exist_ok=True)
    judge = Judge(judge_name, task, device, token_budget=token_budget)
    pmap = {p.pid: p for p in problems}
    commit_path = os.path.join(out_dir, "commits.jsonl")

    if "commit" in parts and not os.path.exists(commit_path):
        outs = judge.solve(problems, n=1, max_new_tokens=max_new_tokens, greedy=True, batch_size=batch_size)
        write_jsonl(commit_path, [
            {"pid": p.pid, "text": o[0]["text"], "answer": task.extract_answer(o[0]["text"]), "gold": bool(task.gold_check(p, o[0]["text"]))}
            for p, o in zip(problems, outs)
        ])
        print(f"[{judge_name}] commits done", flush=True)

    path = os.path.join(out_dir, "solve_samples.jsonl")
    if "solve" in parts and not os.path.exists(path):
        outs = judge.solve(problems, n=k_solve, max_new_tokens=max_new_tokens, greedy=False, batch_size=batch_size, seed=seed)
        write_jsonl(path, [
            {"pid": p.pid, "k": j, "text": o["text"], "gold": bool(task.gold_check(p, o["text"])), "n_tokens": o["n_tokens"]}
            for p, g in zip(problems, outs) for j, o in enumerate(g)
        ])
        print(f"[{judge_name}] solve samples done", flush=True)

    for mode in ("standard", "commit_first"):
        path = os.path.join(out_dir, f"scores_{mode}.jsonl")
        if f"score_{mode}" not in parts or os.path.exists(path):
            continue
        if mode == "commit_first":
            judge.set_commits({r["pid"]: r["text"] for r in read_jsonl(commit_path)})
        p_yes, mass = judge.score([pmap[c["pid"]] for c in cands], [c["text"] for c in cands], mode)
        write_jsonl(path, [{"cid": c["cid"], "p_yes": float(a), "mass": float(b)} for c, a, b in zip(cands, p_yes, mass)])
        print(f"[{judge_name}] scored {len(cands)} candidates ({mode})", flush=True)


# ---------------------------------------------------------------------------
# stage: features
# ---------------------------------------------------------------------------
def bon_probability(scores: np.ndarray, event: np.ndarray, n: int) -> float:
    """P(event holds for the highest-scored of n draws with replacement), ties broken uniformly."""
    if len(scores) == 0:
        return float("nan")
    uniq = np.unique(scores)
    F_le = np.array([(scores <= u).mean() for u in uniq])
    F_lt = np.concatenate([[0.0], F_le[:-1]])
    p_max = F_le**n - F_lt**n
    ev = np.array([event[scores == u].mean() for u in uniq])
    return float((p_max * ev).sum())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def problem_features(nat_scores, nat_gold, syn_scores, syn_gold, threshold: float = 0.5) -> Dict[str, float]:
    nat_scores, nat_gold = np.asarray(nat_scores, float), np.asarray(nat_gold, bool)
    syn_scores, syn_gold = np.asarray(syn_scores, float), np.asarray(syn_gold, bool)
    acc_n = nat_scores > threshold
    all_s = np.concatenate([nat_scores, syn_scores])
    all_g = np.concatenate([nat_gold, syn_gold])
    acc_a = all_s > threshold
    npos, nneg = int(nat_gold.sum()), int((~nat_gold).sum())
    f = {
        "n_nat": len(nat_scores),
        "n_pos": npos,
        "n_neg": nneg,
        "policy_solve": float(nat_gold.mean()) if len(nat_gold) else float("nan"),
        "tpr_nat": float(acc_n[nat_gold].mean()) if npos else float("nan"),
        "fpr_nat": float(acc_n[~nat_gold].mean()) if nneg else float("nan"),
        # Beta(1,1)-smoothed, natural + planted candidates
        "tpr": float((acc_a[all_g].sum() + 1) / (all_g.sum() + 2)),
        "fpr": float((acc_a[~all_g].sum() + 1) / ((~all_g).sum() + 2)),
        "youden_nat": float((acc_n[nat_gold].sum() + 1) / (npos + 2) - (acc_n[~nat_gold].sum() + 1) / (nneg + 2)),
        "auroc": auroc(all_s, all_g),
        "margin": float(all_s[all_g].mean() - all_s[~all_g].mean()) if all_g.any() and (~all_g).any() else float("nan"),
        "syn_neg_accept": float((syn_scores[~syn_gold] > threshold).mean()) if (~syn_gold).any() else float("nan"),
        "syn_pos_accept": float((syn_scores[syn_gold] > threshold).mean()) if syn_gold.any() else float("nan"),
        "base_accept": float(acc_n.mean()),
        "base_fp": float((acc_n & ~nat_gold).mean()),
    }
    f["youden"] = f["tpr"] - f["fpr"]
    f["ver_acc"] = (1 + f["youden"]) / 2  # balanced verification accuracy
    for n in BON_NS:
        f[f"bon_fool_{n}"] = bon_probability(nat_scores, acc_n & ~nat_gold, n)
        f[f"bon_wrong_{n}"] = bon_probability(nat_scores, ~nat_gold, n)
    return f


def compute_features(diag_task_dir: str, problems: List[Problem], judges: List[str]):
    from .judge import MODES

    cands = {c["cid"]: c for c in read_jsonl(os.path.join(diag_task_dir, "candidates.jsonl"))}
    policy_rows = read_jsonl(os.path.join(diag_task_dir, "policy_samples.jsonl"))
    mean_len = pd.DataFrame(policy_rows).groupby("pid")["n_tokens"].mean().to_dict()
    rows, glob = [], []
    for jn in judges:
        jd = os.path.join(diag_task_dir, short_model_name(jn))
        solve = pd.DataFrame(read_jsonl(os.path.join(jd, "solve_samples.jsonl"))).groupby("pid")["gold"].mean().to_dict()
        commit = {r["pid"]: r["gold"] for r in read_jsonl(os.path.join(jd, "commits.jsonl"))}
        for mode in MODES:
            sp = os.path.join(jd, f"scores_{mode}.jsonl")
            if not os.path.exists(sp):
                print(f"[features] skipping {jn} {mode}: no {sp}")
                continue
            sc = {r["cid"]: r["p_yes"] for r in read_jsonl(sp)}
            by_pid: Dict[str, Dict[str, list]] = {}
            for cid, s in sc.items():
                c = cands[cid]
                d = by_pid.setdefault(c["pid"], {"ns": [], "ng": [], "ss": [], "sg": []})
                if c["source"] == "natural":
                    d["ns"].append(s)
                    d["ng"].append(c["gold"])
                else:
                    d["ss"].append(s)
                    d["sg"].append(c["gold"])
            for p in problems:
                d = by_pid[p.pid]
                f = problem_features(d["ns"], d["ng"], d["ss"], d["sg"])
                f.update(
                    task=p.task,
                    judge=short_model_name(jn),
                    mode=mode,
                    pid=p.pid,
                    judge_solve=float(solve.get(p.pid, np.nan)),
                    judge_commit_correct=float(commit.get(p.pid, np.nan)),
                    mean_len=float(mean_len.get(p.pid, np.nan)),
                    **{f"meta_{k}": v for k, v in p.meta.items()},
                )
                f["solve_gap"] = f["policy_solve"] - f["judge_solve"]
                rows.append(f)
            # global (answer-independent) verifier quality on natural samples, as in the noisy-verifier theory
            nat = [(s, cands[c]["gold"]) for c, s in sc.items() if cands[c]["source"] == "natural"]
            s = np.array([x[0] for x in nat]) > 0.5
            g = np.array([x[1] for x in nat], bool)
            glob.append(
                {
                    "task": problems[0].task,
                    "judge": short_model_name(jn),
                    "mode": mode,
                    "tpr": float(s[g].mean()),
                    "fpr": float(s[~g].mean()),
                    "youden": float(s[g].mean() - s[~g].mean()),
                    "base_accept": float(s.mean()),
                    "base_gold": float(g.mean()),
                    "judge_solve": float(np.mean(list(solve.values()))),
                }
            )
    feats = pd.DataFrame(rows)
    gdf = pd.DataFrame(glob)
    # concentration: share of all false accepts that fall on the least-verifiable third of problems
    conc = []
    for (t, j, m), d in feats.groupby(["task", "judge", "mode"]):
        low = d["youden"] <= d["youden"].quantile(1 / 3)
        fp = d["base_fp"] * d["n_nat"]
        conc.append({"task": t, "judge": j, "mode": m, "fp_share_low_tercile": float(fp[low].sum() / max(fp.sum(), 1e-9)), "frac_problems_neg_youden": float((d["youden"] < 0).mean())})
    gdf = gdf.merge(pd.DataFrame(conc), on=["task", "judge", "mode"])
    feats.to_csv(os.path.join(diag_task_dir, "features.csv"), index=False)
    gdf.to_csv(os.path.join(diag_task_dir, "global.csv"), index=False)
    return feats, gdf

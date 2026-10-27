"""MATH (Hendrycks et al.): free-form answers in \\boxed{}.

Training pool: the MATH train split; held-out set: MATH-500 (test split).
Both are filtered to a level range so a 1.7B non-thinking policy solves a
moderate fraction. Gold answers are only read by `gold_check`.
"""
import random
import re
from typing import List, Optional

from .base import Problem, Task

TRAIN_DATASET = "EleutherAI/hendrycks_math"
TRAIN_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
HELDOUT_DATASET = "HuggingFaceH4/MATH-500"


def last_boxed(s: str) -> Optional[str]:
    """Content of the last \\boxed{...} (or \\fbox{...}) in s, handling nested braces."""
    idx = max(s.rfind("\\boxed{"), s.rfind("\\fbox{"))
    if idx < 0:
        return None
    start = s.index("{", idx)
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i].strip()
    return None


def replace_last_boxed(s: str, new: str) -> Optional[str]:
    idx = max(s.rfind("\\boxed{"), s.rfind("\\fbox{"))
    if idx < 0:
        return None
    start = s.index("{", idx)
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[: start + 1] + new + s[i:]
    return None


def normalize(ans: str) -> str:
    a = ans.strip()
    a = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", a)
    for t in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\$", "$", "^\\circ", "^{\\circ}", "\\%", "%"]:
        a = a.replace(t, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\s+", "", a)
    a = a.rstrip(".")
    if a.startswith("x=") or a.startswith("y="):
        a = a[2:]
    return a


def math_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    if normalize(pred) == normalize(gold):
        return True
    try:
        from math_verify import parse, verify

        g = parse("\\boxed{" + gold + "}")
        p = parse("\\boxed{" + pred + "}")
        return bool(g) and bool(p) and bool(verify(g, p))
    except Exception:
        return False


def perturb_answer(ans: str, rng: random.Random) -> str:
    a = ans.strip()
    if re.fullmatch(r"-?\d+", a):
        return str(int(a) + rng.choice([-2, -1, 1, 2, 10]))
    m = re.fullmatch(r"(-?)\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", a)
    if m:
        return f"{m.group(1)}\\frac{{{int(m.group(2)) + rng.choice([-1, 1])}}}{{{m.group(3)}}}"
    m = re.fullmatch(r"-?\d+\.\d+", a)
    if m:
        return f"{float(a) + rng.choice([-1, 1]) * 0.5:g}"
    return rng.choice([f"2{a}" if not a.startswith("-") else a[1:], f"{a}+1", f"-({a})"])


class MathTask(Task):
    name = "math"
    judge_criteria = (
        "The candidate's final answer is the one inside \\boxed{}. It is correct only if it is mathematically "
        "equal to the true answer of the problem."
    )

    def __init__(self, levels=(1, 2, 3)):
        self.levels = tuple(int(x) for x in levels)

    def load_problems(self, split: str, n: int, seed: int = 0, **kw) -> List[Problem]:
        from datasets import concatenate_datasets, load_dataset

        rows = []
        if split == "heldout":
            ds = load_dataset(HELDOUT_DATASET, split="test")
            for r in ds:
                lvl = int(r["level"])
                if lvl in self.levels:
                    rows.append((r["unique_id"], r["problem"], r["answer"], r["solution"], lvl, r["subject"]))
        else:
            ds = concatenate_datasets([load_dataset(TRAIN_DATASET, c, split="train") for c in TRAIN_CONFIGS])
            for i, r in enumerate(ds):
                m = re.search(r"\d", r["level"] or "")
                lvl = int(m.group()) if m else -1
                gold = last_boxed(r["solution"])
                if lvl in self.levels and gold:
                    rows.append((f"train/{i}", r["problem"], gold, r["solution"], lvl, r["type"]))
        rng = random.Random(seed)
        rng.shuffle(rows)
        out = []
        for uid, prob, gold, sol, lvl, subj in rows[:n]:
            out.append(
                Problem(
                    pid="math-" + re.sub(r"[^A-Za-z0-9]+", "_", uid).strip("_"),
                    task=self.name,
                    question=prob.strip(),
                    gold={"answer": gold, "solution": sol},
                    meta={"level": lvl, "subject": subj},
                )
            )
        return out

    def policy_messages(self, problem: Problem):
        return [
            {
                "role": "user",
                "content": problem.question
                + "\n\nReason step by step but concisely (a few short steps), and put your final answer within \\boxed{}.",
            }
        ]

    def extract_answer(self, response: str) -> Optional[str]:
        return last_boxed(response)

    def gold_check(self, problem: Problem, response: str) -> bool:
        return math_equal(last_boxed(response), problem.gold["answer"])

    def answer_key(self, answer):
        return None if answer is None else normalize(answer)

    def synthetic_positive(self, problem: Problem) -> Optional[str]:
        sol = problem.gold["solution"]
        return sol if last_boxed(sol) is not None else None

    def synthetic_negatives(self, problem, natural_correct, rng, k=2):
        """A correct-looking solution whose final boxed answer is altered (subtle wrong answer)."""
        bases = list(natural_correct) or [problem.gold["solution"]]
        out, tries = [], 0
        while len(out) < k and tries < 50:
            tries += 1
            base = rng.choice(bases)
            ans = last_boxed(base)
            if ans is None:
                continue
            text = replace_last_boxed(base, perturb_answer(ans, rng))
            if text is None or self.gold_check(problem, text) or text in out:
                continue
            out.append(text)
        return out

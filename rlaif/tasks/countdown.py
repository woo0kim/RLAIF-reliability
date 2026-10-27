"""Countdown: combine the given numbers with + - * / to reach a target.

The gold checker is exact: the final equation must use every given number
exactly once and evaluate to the target. Countdown has no single gold answer,
so "hiding gold" means the judge never gets this checker or a solution.
"""
import ast
import operator
import random
import re
from fractions import Fraction
from typing import List, Optional, Sequence

from .base import Problem, Task

DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ALLOWED_RE = re.compile(r"^[\d+\-*/()\s]+$")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def extract_equation(response: str) -> Optional[str]:
    matches = ANSWER_RE.findall(response)
    if not matches:
        return None
    eq = matches[-1].strip()
    if "=" in eq:  # tolerate "expr = target"; only the left side is checked
        eq = eq.split("=")[0].strip()
    return eq or None


def safe_eval(expr: str) -> Optional[Fraction]:
    """Exactly evaluate an arithmetic expression; None if invalid."""
    if not ALLOWED_RE.match(expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError("disallowed node")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, RecursionError):
        return None


def uses_numbers_exactly(expr: str, nums: Sequence[int]) -> bool:
    return sorted(int(x) for x in re.findall(r"\d+", expr)) == sorted(nums)


def check_equation(expr: Optional[str], nums: Sequence[int], target: int) -> bool:
    if expr is None or not uses_numbers_exactly(expr, nums):
        return False
    value = safe_eval(expr)
    return value is not None and value == target


def solve(nums: Sequence[int], target: int) -> Optional[str]:
    """Brute-force search for an equation (used to build diagnostic positives)."""
    target = Fraction(target)

    def rec(items):
        if len(items) == 1:
            return items[0][1] if items[0][0] == target else None
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                (a, ea), (b, eb) = items[i], items[j]
                rest = [items[k] for k in range(len(items)) if k not in (i, j)]
                cands = [(a + b, f"({ea} + {eb})"), (a - b, f"({ea} - {eb})"), (a * b, f"({ea} * {eb})")]
                if b != 0:
                    cands.append((a / b, f"({ea} / {eb})"))
                for v, e in cands:
                    r = rec(rest + [(v, e)])
                    if r is not None:
                        return r
        return None

    expr = rec([(Fraction(n), str(n)) for n in nums])
    if expr is not None and expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1]
    return expr


class CountdownTask(Task):
    name = "countdown"
    judge_criteria = (
        "The candidate's final equation is the one inside <answer> </answer> tags. It is correct only if it "
        "uses each of the given numbers exactly once (no other numbers), uses only +, -, *, / and parentheses, "
        "and evaluates exactly to the target."
    )

    def load_problems(self, split: str, n: int, seed: int = 0, **kw) -> List[Problem]:
        from datasets import load_dataset

        ds = load_dataset(DATASET, split="train")
        rng = random.Random(seed)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        # Disjoint index ranges keep the training pool and held-out set apart.
        region = idx[:20000] if split == "heldout" else idx[20000:40000]
        out = []
        for i in region:
            row = ds[i]
            nums, target = [int(x) for x in row["nums"]], int(row["target"])
            if solve(nums, target) is None:
                continue
            out.append(
                Problem(
                    pid=f"cd-{i}",
                    task=self.name,
                    question=(
                        f"Using the numbers {nums}, create an equation that equals {target}. You can use basic "
                        f"arithmetic operations (+, -, *, /) and each number must be used exactly once."
                    ),
                    gold={"nums": nums, "target": target},
                    meta={"n_numbers": len(nums)},
                )
            )
            if len(out) >= n:
                break
        return out

    def policy_messages(self, problem: Problem):
        return [
            {
                "role": "user",
                "content": problem.question
                + " Show your reasoning briefly, then give the final equation (without the '= target' part) "
                "in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
            }
        ]

    def extract_answer(self, response: str) -> Optional[str]:
        return extract_equation(response)

    def gold_check(self, problem: Problem, response: str) -> bool:
        g = problem.gold
        return check_equation(extract_equation(response), g["nums"], g["target"])

    def answer_key(self, answer):
        return None if answer is None else re.sub(r"\s+", "", answer)

    def synthetic_positive(self, problem: Problem) -> Optional[str]:
        g = problem.gold
        expr = solve(g["nums"], g["target"])
        if expr is None:
            return None
        return f"Let me combine the numbers step by step.\n{expr} = {g['target']}\n<answer> {expr} </answer>"

    def synthetic_negatives(self, problem, natural_correct, rng, k=2):
        """Wrong equations that falsely claim to hit the target (arithmetic slip or wrong number)."""
        g = problem.gold
        base = solve(g["nums"], g["target"])
        if base is None:
            return []
        out, tries = [], 0
        while len(out) < k and tries < 200:
            tries += 1
            if rng.random() < 0.5:  # operator swap
                ops = [m.start() for m in re.finditer(r"[+\-*/]", base)]
                if not ops:
                    continue
                p = rng.choice(ops)
                expr = base[:p] + rng.choice([o for o in "+-*/" if o != base[p]]) + base[p + 1 :]
            else:  # perturb one number
                spans = [m.span() for m in re.finditer(r"\d+", base)]
                a, b = rng.choice(spans)
                expr = base[:a] + str(int(base[a:b]) + rng.choice([-2, -1, 1, 2])) + base[b:]
            if check_equation(expr, g["nums"], g["target"]):
                continue
            text = f"Let me combine the numbers step by step.\n{expr} = {g['target']}\n<answer> {expr} </answer>"
            if text not in out:
                out.append(text)
        return out

"""CPU unit tests: gold checkers, planted candidates, best-of-N math, GRPO advantages, hazard model, stopping.

  python -m pytest tests/test_core.py -q
"""
import random

import numpy as np
import pandas as pd
import pytest

from rlaif.diagnostics import auroc, bon_probability, problem_features
from rlaif.grpo import group_advantages
from rlaif.tasks.base import Problem
from rlaif.tasks.countdown import CountdownTask, check_equation, extract_equation, safe_eval, solve
from rlaif.tasks.math_task import MathTask, last_boxed, math_equal, replace_last_boxed


# ---------------- countdown ----------------
def test_countdown_checker():
    assert check_equation("(44 + 19) + 35", [44, 19, 35], 98)
    assert check_equation("44 + 19 + 35", [35, 44, 19], 98)
    assert not check_equation("44 + 19 + 35", [44, 19, 35], 97)
    assert not check_equation("44 + 54", [44, 19, 35], 98)  # wrong numbers
    assert not check_equation("44 + 44 + 10", [44, 10], 98)  # reuse
    assert not check_equation("2 ** 3", [2, 3], 8)  # power not allowed
    assert not check_equation("__import__('os')", [1], 1)
    assert safe_eval("(6 / 4) * 2") == 3
    assert safe_eval("1 / 0") is None
    assert extract_equation("blah <answer> 1 + 2 </answer> x <answer>3 * 4 = 12</answer>") == "3 * 4"


def test_countdown_solver_and_planted():
    t = CountdownTask()
    p = Problem("cd-x", "countdown", "q", {"nums": [3, 7, 11], "target": 32}, {})
    expr = solve([3, 7, 11], 32)
    assert expr is not None and check_equation(expr, [3, 7, 11], 32)
    pos = t.synthetic_positive(p)
    assert t.gold_check(p, pos)
    negs = t.synthetic_negatives(p, [], random.Random(0), k=3)
    assert len(negs) >= 2 and not any(t.gold_check(p, n) for n in negs)


# ---------------- math ----------------
def test_boxed_extraction():
    s = r"so \boxed{\frac{1}{2}} and finally \boxed{\dfrac{3}{4}}."
    assert last_boxed(s) == r"\dfrac{3}{4}"
    assert replace_last_boxed(s, "5") == r"so \boxed{\frac{1}{2}} and finally \boxed{5}."
    assert last_boxed("no box") is None


def test_math_equal():
    assert math_equal(r"\dfrac{3}{4}", r"\frac34") or math_equal(r"\dfrac{3}{4}", r"\frac{3}{4}")
    assert math_equal("0.5", r"\frac{1}{2}")
    assert math_equal(" 12 ", "12")
    assert not math_equal("13", "12")
    assert not math_equal(None, "12")


def test_math_planted():
    t = MathTask()
    sol = r"We compute $2+2=4$, so the answer is $\boxed{4}$."
    p = Problem("math-x", "math", "What is 2+2?", {"answer": "4", "solution": sol}, {"level": 1})
    assert t.gold_check(p, t.synthetic_positive(p))
    negs = t.synthetic_negatives(p, [sol], random.Random(0), k=2)
    assert negs and not any(t.gold_check(p, n) for n in negs)


# ---------------- diagnostics math ----------------
def test_bon_probability_matches_monte_carlo():
    rng = np.random.default_rng(0)
    s = rng.random(10)
    ev = rng.random(10) < 0.4
    for n in (1, 4, 16):
        exact = bon_probability(s, ev, n)
        draws = rng.integers(0, 10, size=(200000, n))
        mc = ev[draws[np.arange(len(draws)), s[draws].argmax(1)]].mean()
        assert abs(exact - mc) < 0.01
    assert bon_probability(s, ev, 1) == pytest.approx(ev.mean())


def test_problem_features_youden():
    # judge accepts every correct and every wrong answer -> J ~ 0; perfect judge -> J ~ 1
    f = problem_features([0.9, 0.9, 0.9, 0.9], [True, True, False, False], [0.9, 0.9], [True, False])
    assert abs(f["youden"]) < 1e-9 and f["bon_fool_16"] > 0.4
    g = problem_features([0.9, 0.9, 0.1, 0.1], [True, True, False, False], [0.9, 0.1], [True, False])
    assert g["youden"] > 0.5 and g["bon_fool_16"] == 0.0 and g["auroc"] == 1.0
    assert auroc(np.array([0.2, 0.8]), np.array([False, True])) == 1.0


# ---------------- GRPO ----------------
def test_group_advantages():
    r = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    a = group_advantages(r, "std")
    assert np.allclose(a[0], [1, -1, 1, -1], atol=1e-4)
    assert np.allclose(a[1], 0)
    assert np.allclose(group_advantages(r, "none")[0], [0.5, -0.5, 0.5, -0.5])


# ---------------- hazard model ----------------
def _synthetic_units(n=400, seed=0):
    """Event hazard driven by low youden; solve_gap is pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = rng.uniform(-0.2, 1.0)
        gap = rng.uniform(-0.5, 0.5)
        h = 1 / (1 + np.exp(-(-1.0 - 3.0 * y)))
        T, ev = 8, False
        for v in range(1, 9):
            if rng.random() < h:
                T, ev = v, True
                break
        rows.append({"run": f"r{i % 4}", "task": "countdown" if i % 2 else "math", "judge": ["a", "b"][i % 2], "pid": f"p{i}", "T": T, "event": ev,
                     "youden": y, "bon_fool_16": max(0.0, 0.5 - y) + rng.normal(0, 0.05), "solve_gap": gap, "policy_solve": rng.random(), "judge_solve": rng.random(), "base_fp": rng.random() * 0.4})
    return pd.DataFrame(rows)


def test_hazard_recovers_signal():
    from rlaif.analysis import hazard

    u = _synthetic_units()
    res = hazard.evaluate_feature_sets(u, splits=("heldout_problems",))
    m = res.groupby("features").c_index.mean()
    assert m["verifiability"] > 0.65
    assert m["verifiability"] > m["solve gap"] + 0.1
    pp = hazard.person_period(u)
    assert pp.y.sum() == u.event.sum() and len(pp) == u["T"].sum()


def test_concordance_basic():
    from rlaif.analysis.hazard import concordance

    T = np.array([1, 2, 3, 4])
    E = np.array([True, True, True, False])
    assert concordance(T, E, np.array([4, 3, 2, 1])) == 1.0
    assert concordance(T, E, np.array([1, 2, 3, 4])) == 0.0


def test_policy_update_oom_recovery():
    """First OOM turns on activation checkpointing, later ones halve the micro-batch; checkpointing is restored."""
    from types import SimpleNamespace

    from rlaif.grpo import GRPOTrainer

    calls = {"budgets": [], "ckpt": []}

    class Model:
        def gradient_checkpointing_enable(self, **kw):
            calls["ckpt"].append("on")

        def gradient_checkpointing_disable(self):
            calls["ckpt"].append("off")

    class Opt:
        def zero_grad(self, set_to_none=True):
            pass

    fails = {"n": 2}

    def fake_update(samples, total, budget):
        calls["budgets"].append(budget)
        if fails["n"] > 0:
            fails["n"] -= 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return {"ok": True}

    t = SimpleNamespace(cfg={"train_token_budget": 512, "grad_checkpointing": False}, opt=Opt(), model=Model(), _policy_update=fake_update)
    assert GRPOTrainer.policy_update(t, [], 1) == {"ok": True}
    assert calls["budgets"] == [512, 512, 256]
    assert calls["ckpt"] == ["on", "off"]

    def other_error(samples, total, budget):
        raise RuntimeError("some other failure")

    t._policy_update = other_error
    with pytest.raises(RuntimeError, match="other failure"):
        GRPOTrainer.policy_update(t, [], 1)

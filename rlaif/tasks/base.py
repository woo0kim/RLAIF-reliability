"""Task interface.

A task supplies problems, the policy prompt, a *gold* checker that is only ever
used for evaluation (it is the held-out evaluator), and grading criteria for the
LLM judge. The judge never sees gold answers: it gets the problem statement,
the generic criteria and the candidate response only.
"""
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Problem:
    pid: str
    task: str
    question: str  # problem statement shown to the policy and the judge
    gold: Any  # hidden: only the gold checker reads it
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Problem":
        return Problem(**d)


class Task:
    name: str = "base"
    # Task-specific statement of what "correct" means, given to the judge.
    judge_criteria: str = ""

    def load_problems(self, split: str, n: int, seed: int, **kw) -> List[Problem]:
        raise NotImplementedError

    def policy_messages(self, problem: Problem) -> List[Dict[str, str]]:
        raise NotImplementedError

    def extract_answer(self, response: str) -> Optional[str]:
        raise NotImplementedError

    def gold_check(self, problem: Problem, response: str) -> bool:
        raise NotImplementedError

    def answer_key(self, answer: Optional[str]) -> Optional[str]:
        """Normalised answer string used for label-free agreement statistics."""
        return None if answer is None else " ".join(answer.split())

    def synthetic_positive(self, problem: Problem) -> Optional[str]:
        """A known-correct response (built from gold) for diagnostics only."""
        return None

    def synthetic_negatives(
        self, problem: Problem, natural_correct: List[str], rng: random.Random, k: int = 2
    ) -> List[str]:
        """Known-wrong responses (built from gold) for diagnostics only."""
        return []

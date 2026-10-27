"""Frozen, reference-free LLM judges.

The judge reads the problem and a candidate response (never the gold answer)
and is asked for a one-word Yes/No verdict. We read the next-token
distribution instead of sampling: score = P(Yes) / (P(Yes) + P(No)), and the
verdict is score > 0.5. One forward pass per candidate.

Two modes:
  standard      problem + candidate -> verdict
  commit_first  the judge first solves the problem itself and commits to an
                answer (a previous assistant turn), and only then sees the
                candidate. The commit is generated once per problem (greedy,
                frozen judge) and cached, so the per-candidate cost is still a
                single forward pass.
"""
import gc
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .models import chat_prompt, generate, is_oom, left_pad, load_model, load_tokenizer, position_ids_from_mask
from .tasks.base import Problem, Task

SYSTEM = "You are a strict, careful grader. You check solutions rigorously and are not swayed by confident claims."
QUESTION = "Is the candidate's final answer correct? Reply with exactly one word: Yes or No."
MODES = ("standard", "commit_first")


class Judge:
    def __init__(
        self,
        model_name: str,
        task: Task,
        device: str,
        dtype: str = "float16",
        token_budget: int = 12000,
        commits: Optional[Dict[str, str]] = None,
        commit_max_tokens: int = 256,
    ):
        self.name = model_name
        self.task = task
        self.device = device
        self.tok = load_tokenizer(model_name)
        self.model = load_model(model_name, device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.token_budget = token_budget
        self.commit_max_tokens = commit_max_tokens
        self.commits: Dict[str, str] = {}
        if commits:
            self.set_commits(commits)
        self.yes_ids = self._single_token_ids(["Yes", "yes", "YES"])
        self.no_ids = self._single_token_ids(["No", "no", "NO"])
        assert self.yes_ids and self.no_ids, "Yes/No must be single tokens"

    def _single_token_ids(self, words):
        ids = []
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
        return sorted(set(ids))

    # ---- commits -----------------------------------------------------------
    def set_commits(self, commits: Dict[str, str]) -> None:
        """Store the judge's own solutions, keeping only their last tokens (where the answer is)."""
        for pid, text in commits.items():
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) > self.commit_max_tokens:
                text = "..." + self.tok.decode(ids[-self.commit_max_tokens :])
            self.commits[pid] = text

    def solve(self, problems: List[Problem], n: int, max_new_tokens: int, greedy: bool, batch_size: int, seed=0):
        """Use the judge as a solver (for its solve rate and for commit-first commits)."""
        prompts = [chat_prompt(self.tok, self.task.policy_messages(p)) for p in problems]
        return generate(
            self.model,
            self.tok,
            prompts,
            n=n,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=1.0,
            batch_size=batch_size,
            seed=seed,
        )

    # ---- prompts -----------------------------------------------------------
    def messages(self, problem: Problem, response: str, mode: str) -> List[Dict[str, str]]:
        crit = self.task.judge_criteria
        if mode == "standard":
            user = (
                f"You will be shown a problem and a candidate solution. Decide whether the candidate's final "
                f"answer is correct. {crit}\n\n[Problem]\n{problem.question}\n\n[Candidate solution]\n"
                f"{response}\n\n{QUESTION}"
            )
            return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if mode == "commit_first":
            if problem.pid not in self.commits:
                raise KeyError(f"no commit for {problem.pid}; run diagnostics first")
            first = (
                f"Before grading anyone else, solve this problem yourself and commit to a final answer.\n\n"
                f"[Problem]\n{problem.question}"
            )
            second = (
                f"Now grade a candidate solution to the same problem. You have already committed to your own "
                f"answer above; check the candidate's work independently and compare. {crit}\n\n"
                f"[Candidate solution]\n{response}\n\n{QUESTION}"
            )
            return [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": first},
                {"role": "assistant", "content": self.commits[problem.pid]},
                {"role": "user", "content": second},
            ]
        raise ValueError(f"unknown judge mode {mode!r}")

    def prompt(self, problem: Problem, response: str, mode: str) -> str:
        return chat_prompt(self.tok, self.messages(problem, response, mode))

    # ---- scoring -----------------------------------------------------------
    @torch.no_grad()
    def score(self, problems: Sequence[Problem], responses: Sequence[str], mode: str = "standard"):
        """Returns (p_yes, mass) arrays; mass = P(Yes)+P(No) under the judge's next-token distribution."""
        enc = [self.tok(self.prompt(p, r, mode), add_special_tokens=False).input_ids for p, r in zip(problems, responses)]
        order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
        p_yes = np.zeros(len(enc), dtype=np.float64)
        mass = np.zeros(len(enc), dtype=np.float64)
        i = 0
        while i < len(order):
            L = len(enc[order[i]])
            bs = max(1, self.token_budget // max(L, 1))
            batch = order[i : i + bs]
            self._score_batch(enc, batch, p_yes, mass)
            i += len(batch)
        torch.cuda.empty_cache()  # judges may share a GPU across runs; do not hoard cached blocks
        return p_yes, mass

    def _score_batch(self, enc, batch, p_yes, mass):
        logits = None
        try:
            ids, mask = left_pad([enc[j] for j in batch], self.tok.pad_token_id, self.device)
            logits = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids_from_mask(mask),
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :]
        except RuntimeError as e:
            if not is_oom(e) or len(batch) == 1:
                raise
        if logits is None:  # OOM: split, after the except block has released the failed attempt
            gc.collect()
            torch.cuda.empty_cache()
            h = len(batch) // 2
            self._score_batch(enc, batch[:h], p_yes, mass)
            self._score_batch(enc, batch[h:], p_yes, mass)
            return
        probs = torch.softmax(logits.float(), dim=-1)
        py = probs[:, self.yes_ids].sum(-1)
        pn = probs[:, self.no_ids].sum(-1)
        tot = (py + pn).clamp(min=1e-12)
        for k, j in enumerate(batch):
            p_yes[j] = float(py[k] / tot[k])
            mass[j] = float(tot[k])

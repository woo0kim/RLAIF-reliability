"""LoRA GRPO against a frozen LLM judge.

Per step: take the next B problems of the pool (each epoch visits every pool
problem exactly once, so a problem's "visit" index is its epoch + 1), sample
G completions each, reward each completion with the frozen judge, normalise
rewards within each group, and take one on-policy policy-gradient step:

    L = - sum_i A_i * sum_t log pi(y_it | x_i, y_i<t) / sum_i |y_i|   (+ beta * KL_k3)

With one update per batch the PPO ratio is identically 1, so clipping is a
no-op and is omitted. Gold correctness is computed for every rollout but is
only logged, never rewarded. Every rollout is logged, which gives a per-problem
time series (visit by visit) of gold accuracy, judge acceptance and
false-positive rate for the hazard analysis.
"""
import gc
import glob
import json
import math
import os
import random
import shutil
import time
from typing import Dict, List

import numpy as np
import torch

from .judge import Judge
from .models import build_lm_batch, chat_prompt, completion_logprobs, generate, is_oom, load_model, load_tokenizer
from .tasks import Problem, get_task
from .utils import append_jsonl, read_jsonl, seed_everything, short_model_name, timer


def load_problems(path: str) -> List[Problem]:
    return [Problem.from_dict(r) for r in read_jsonl(path)]


def load_commits(diag_dir: str, task: str, judge: str) -> Dict[str, str]:
    """The judge's own greedy solutions: pool problems (diagnostics) + any made later (commits_extra)."""
    out = {}
    for name in ("commits.jsonl", "commits_extra.jsonl"):
        path = os.path.join(diag_dir, task, short_model_name(judge), name)
        if os.path.exists(path):
            out.update({r["pid"]: r["text"] for r in read_jsonl(path)})
    return out


def group_advantages(rewards: np.ndarray, norm: str = "std", eps: float = 1e-6) -> np.ndarray:
    """rewards: [n_groups, G] -> advantages of the same shape."""
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    if norm == "std":
        return centered / (rewards.std(axis=1, keepdims=True) + eps)
    if norm == "none":
        return centered
    raise ValueError(norm)


class GRPOTrainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.out = cfg["out_dir"]
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        seed_everything(cfg["seed"])
        self.task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
        data = os.path.join(cfg["data_dir"], cfg["task"])
        self.pool = load_problems(os.path.join(data, f"{cfg['pool']}.jsonl"))
        self.heldout = load_problems(os.path.join(data, f"{cfg['heldout']}.jsonl"))[: cfg["eval_max_problems"]]
        self.B, self.G = cfg["prompts_per_step"], cfg["group_size"]
        self.steps_per_epoch = math.ceil(len(self.pool) / self.B)

        # policy + LoRA
        from peft import LoraConfig, get_peft_model

        self.pdev = cfg["policy_device"]
        self.tok = load_tokenizer(cfg["policy"])
        base = load_model(cfg["policy"], self.pdev, cfg["dtype"])
        base.config.use_cache = False
        lcfg = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_targets"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lcfg, autocast_adapter_dtype=True)
        if cfg.get("grad_checkpointing", False):
            # PEFT casts each LoRA input to fp32, so stored activations dominate memory without this
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for n, p in self.model.named_parameters():  # adapters in fp32, frozen base in fp16
            if p.requires_grad:
                p.data = p.data.float()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg["lr"], betas=(0.9, 0.99), eps=1e-8, weight_decay=cfg["weight_decay"])
        warm = max(1, cfg["warmup_steps"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / warm))
        self.scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12, growth_interval=50)

        # judge (+ the other mode as a label-free monitor)
        modes = {cfg["judge_mode"]}
        mon = cfg.get("monitor_mode", "auto")
        if mon == "auto":
            mon = "standard" if cfg["judge_mode"] == "commit_first" else "commit_first"
        self.monitor_mode = None if mon in (None, "none") else mon
        if self.monitor_mode:
            modes.add(self.monitor_mode)
        commits = load_commits(cfg["diag_dir"], cfg["task"], cfg["judge"]) if "commit_first" in modes else None
        if "commit_first" in modes and not commits:
            if cfg["judge_mode"] == "commit_first":
                raise FileNotFoundError("commit_first judging needs diagnostics commits (run_diagnostics.py)")
            self.monitor_mode = None
        self.judge = Judge(
            cfg["judge"],
            self.task,
            cfg["judge_device"],
            dtype=cfg["dtype"],
            token_budget=cfg["judge_token_budget"],
            commits=commits,
        )
        if self.judge.commits and cfg["judge_mode"] == "commit_first":
            self._ensure_commits(self.pool + self.heldout)
        self._prompts = {}
        self.step = 0

    def _ensure_commits(self, problems: List[Problem]):
        """Commit-first judging needs the judge's own solution for every problem it grades (incl. held-out)."""
        missing = [p for p in problems if p.pid not in self.judge.commits]
        if not missing:
            return
        cfg = self.cfg
        outs = self.judge.solve(missing, n=1, max_new_tokens=cfg["max_new_tokens"], greedy=True, batch_size=cfg["gen_batch_size"])
        rows = [{"pid": p.pid, "text": o[0]["text"], "answer": self.task.extract_answer(o[0]["text"])} for p, o in zip(missing, outs)]
        append_jsonl(os.path.join(cfg["diag_dir"], cfg["task"], short_model_name(cfg["judge"]), "commits_extra.jsonl"), rows)
        self.judge.set_commits({r["pid"]: r["text"] for r in rows})
        print(f"[commit-first] generated {len(rows)} missing judge commits", flush=True)

    # ---- data ----------------------------------------------------------------
    def prompt_of(self, p: Problem):
        if p.pid not in self._prompts:
            text = chat_prompt(self.tok, self.task.policy_messages(p))
            self._prompts[p.pid] = (text, self.tok(text, add_special_tokens=False).input_ids)
        return self._prompts[p.pid]

    def batch_at(self, step: int):
        epoch, k = divmod(step, self.steps_per_epoch)
        order = list(range(len(self.pool)))
        random.Random(self.cfg["seed"] * 100003 + epoch).shuffle(order)
        idx = order[k * self.B : (k + 1) * self.B]
        return epoch, [self.pool[i] for i in idx]

    # ---- checkpointing -------------------------------------------------------
    def save(self, step: int):
        d = os.path.join(self.out, "ckpt", f"step_{step:04d}")
        self.model.save_pretrained(d)
        torch.save(
            {
                "step": step,
                "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(),
                "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()},
            },
            os.path.join(d, "trainer_state.pt"),
        )
        for old in glob.glob(os.path.join(self.out, "ckpt", "step_*", "trainer_state.pt")):
            if not old.startswith(d):
                os.remove(old)  # optimizer state only for the latest checkpoint; adapters are kept

    def maybe_resume(self) -> int:
        ckpts = sorted(glob.glob(os.path.join(self.out, "ckpt", "step_*", "trainer_state.pt")))
        if not ckpts:
            for f in ("rollouts.jsonl", "steps.jsonl", "evals.jsonl"):
                if os.path.exists(os.path.join(self.out, f)):
                    os.remove(os.path.join(self.out, f))
            shutil.rmtree(os.path.join(self.out, "eval_samples"), ignore_errors=True)
            return 0
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        d = os.path.dirname(ckpts[-1])
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        set_peft_model_state_dict(self.model, load_file(os.path.join(d, "adapter_model.safetensors")))
        for p in self.params:
            p.data = p.data.float()
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        self.scaler.load_state_dict(state["scaler"])
        random.setstate(state["rng"]["py"])
        np.random.set_state(state["rng"]["np"])
        torch.set_rng_state(state["rng"]["torch"])
        step = state["step"]
        for f, keep in (("rollouts.jsonl", lambda r: r["step"] < step), ("steps.jsonl", lambda r: r["step"] < step), ("evals.jsonl", lambda r: r["step"] <= step)):
            path = os.path.join(self.out, f)
            if os.path.exists(path):
                rows = [r for r in read_jsonl(path) if keep(r)]
                os.remove(path)
                append_jsonl(path, rows)
        print(f"[resume] from step {step}", flush=True)
        return step

    # ---- rollouts -------------------------------------------------------------
    def sample(self, problems: List[Problem], n: int, greedy: bool, seed: int):
        self.model.eval()
        prompts = [self.prompt_of(p)[0] for p in problems]
        return generate(
            self.model,
            self.tok,
            prompts,
            n=n,
            max_new_tokens=self.cfg["max_new_tokens"],
            temperature=self.cfg["temperature"],
            do_sample=not greedy,
            batch_size=self.cfg["gen_batch_size"],
            seed=seed,
        )

    @torch.no_grad()
    def evaluate(self, step: int):
        t0 = time.time()
        # Sampled at the training temperature with a fixed seed: Qwen3 greedy decoding in
        # non-thinking mode often loops until the length cap, which would understate accuracy.
        outs = self.sample(self.heldout, n=1, greedy=self.cfg.get("eval_greedy", False), seed=12345)
        texts = [o[0]["text"] for o in outs]
        gold = np.array([self.task.gold_check(p, t) for p, t in zip(self.heldout, texts)])
        p_yes, _ = self.judge.score(self.heldout, texts, self.cfg["judge_mode"])
        acc = p_yes > 0.5
        row = {
            "step": step,
            "gold_acc": float(gold.mean()),
            "judge_accept": float(acc.mean()),
            "judge_p": float(p_yes.mean()),
            "fp_rate": float((acc & ~gold).mean()),
            "tpr": float(acc[gold].mean()) if gold.any() else float("nan"),
            "fpr": float(acc[~gold].mean()) if (~gold).any() else float("nan"),
            "mean_len": float(np.mean([o[0]["n_tokens"] for o in outs])),
            "finished": float(np.mean([o[0]["finished"] for o in outs])),
            "time": time.time() - t0,
        }
        append_jsonl(os.path.join(self.out, "evals.jsonl"), [row])
        append_jsonl(
            os.path.join(self.out, "eval_samples", f"step_{step:04d}.jsonl"),
            [
                {"pid": p.pid, "text": t, "gold": bool(g), "p_yes": float(s)}
                for p, t, g, s in zip(self.heldout, texts, gold, p_yes)
            ],
        )
        print(f"[eval step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "step"), flush=True)
        return row

    # ---- optimisation ---------------------------------------------------------
    def policy_update(self, samples: List[Dict], total_tokens: int) -> Dict:
        """One GRPO step; on CUDA OOM (shared GPUs) the step is redone from scratch with smaller micro-batches."""
        budget = self.cfg["train_token_budget"]
        ckpt = self.cfg.get("grad_checkpointing", False)
        try:
            for _ in range(5):
                try:
                    return self._policy_update(samples, total_tokens, budget)
                except RuntimeError as e:
                    if not is_oom(e):
                        raise
                # Clean up only after leaving the except block: the live traceback still references
                # the failed attempt's activations, so freeing inside it releases nothing.
                self.opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if not ckpt:  # first fallback: activation checkpointing for this step
                    ckpt = True
                    self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    budget = max(1, budget // 2)
                print(f"[train] OOM, redoing step (checkpointing={ckpt}, train_token_budget={budget})", flush=True)
            raise RuntimeError("policy update kept running out of memory")
        finally:
            if ckpt and not self.cfg.get("grad_checkpointing", False):
                self.model.gradient_checkpointing_disable()

    def _policy_update(self, samples: List[Dict], total_tokens: int, budget: int) -> Dict:
        """One GRPO step on samples with non-zero advantage; loss normalised by all completion tokens."""
        cfg = self.cfg
        self.model.train()
        active = sorted([s for s in samples if abs(s["adv"]) > 1e-8], key=lambda s: len(s["prompt_ids"]) + len(s["ids"]))
        stats = {"loss": 0.0, "logp_sum": 0.0, "kl_sum": 0.0, "n_tok": 0, "n_active": len(active)}
        if not active:
            return {**stats, "grad_norm": 0.0, "skipped": True}
        mbs, cur = [], []
        for s in active:
            L = len(s["prompt_ids"]) + len(s["ids"])
            if cur and (len(cur) + 1) * max(L, max(len(c["prompt_ids"]) + len(c["ids"]) for c in cur)) > budget:
                mbs.append(cur)
                cur = []
            cur.append(s)
        mbs.append(cur)
        for mb in mbs:
            ids, mask, cmask, P, C = build_lm_batch(
                [s["prompt_ids"] for s in mb], [s["ids"] for s in mb], self.tok.pad_token_id, self.pdev
            )
            adv = torch.tensor([s["adv"] for s in mb], device=self.pdev, dtype=torch.float32)
            logp = completion_logprobs(self.model, ids, mask, P, C, cfg["temperature"])
            loss = -(adv[:, None] * logp * cmask).sum() / total_tokens
            if cfg["beta_kl"] > 0:
                with torch.no_grad(), self.model.disable_adapter():
                    ref = completion_logprobs(self.model, ids, mask, P, C, cfg["temperature"])
                d = ref - logp
                kl = (torch.exp(d) - d - 1.0) * cmask
                loss = loss + cfg["beta_kl"] * kl.sum() / total_tokens
                stats["kl_sum"] += float(kl.sum())
            self.scaler.scale(loss).backward()
            stats["loss"] += float(loss)
            stats["logp_sum"] += float((logp.detach() * cmask).sum())
            stats["n_tok"] += int(cmask.sum())
            del ids, mask, cmask, logp, loss
        self.scaler.unscale_(self.opt)
        gn = float(torch.nn.utils.clip_grad_norm_(self.params, cfg["max_grad_norm"]))
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        torch.cuda.empty_cache()
        return {**stats, "grad_norm": gn, "skipped": self.scaler.get_scale() < scale_before}

    def train(self):
        cfg = self.cfg
        self.step = self.maybe_resume()
        if self.step == 0:
            self.evaluate(0)
        while self.step < cfg["max_steps"]:
            step, T = self.step, {}
            torch.cuda.reset_peak_memory_stats(self.pdev)
            epoch, problems = self.batch_at(step)
            with timer(T, "t_gen"):
                outs = self.sample(problems, n=self.G, greedy=False, seed=cfg["seed"] * 1000003 + step * 1009)
            peak_gen = torch.cuda.max_memory_allocated(self.pdev) / 1e9
            torch.cuda.reset_peak_memory_stats(self.pdev)
            flat = [(p, gi, o) for p, group in zip(problems, outs) for gi, o in enumerate(group)]
            texts = [o["text"] for _, _, o in flat]
            gold = np.array([self.task.gold_check(p, t) for (p, _, _), t in zip(flat, texts)], dtype=bool)
            with timer(T, "t_judge"):
                p_yes, mass = self.judge.score([f[0] for f in flat], texts, cfg["judge_mode"])
            accept = p_yes > 0.5
            reward = accept.astype(np.float64) if cfg["reward"] == "verdict" else p_yes.copy()
            mon = np.full(len(flat), np.nan)
            n_mon = min(cfg.get("monitor_prompts", 0), len(problems)) * self.G
            if self.monitor_mode and n_mon > 0:
                with timer(T, "t_monitor"):
                    mon[:n_mon], _ = self.judge.score([f[0] for f in flat[:n_mon]], texts[:n_mon], self.monitor_mode)
            adv = group_advantages(reward.reshape(len(problems), self.G), cfg["adv_norm"]).reshape(-1)
            samples = [
                {"prompt_ids": self.prompt_of(p)[1], "ids": o["ids"], "adv": float(a)}
                for (p, _, o), a in zip(flat, adv)
            ]
            total_tokens = sum(len(s["ids"]) for s in samples)
            with timer(T, "t_train"):
                upd = self.policy_update(samples, total_tokens)

            rows = []
            for i, (p, gi, o) in enumerate(flat):
                ans = self.task.extract_answer(o["text"])
                rows.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "visit": epoch + 1,
                        "pid": p.pid,
                        "k": gi,
                        "reward": float(reward[i]),
                        "p_yes": float(p_yes[i]),
                        "mass": float(mass[i]),
                        "accept": bool(accept[i]),
                        "gold": bool(gold[i]),
                        "mon_p": None if np.isnan(mon[i]) else float(mon[i]),
                        "n_tokens": o["n_tokens"],
                        "finished": o["finished"],
                        "answer": ans,
                        "answer_key": self.task.answer_key(ans),
                        "text": o["text"],
                    }
                )
            append_jsonl(os.path.join(self.out, "rollouts.jsonl"), rows)
            g = reward.reshape(len(problems), self.G)
            m = ~np.isnan(mon)
            srow = {
                "step": step,
                "epoch": epoch,
                "reward": float(reward.mean()),
                "accept": float(accept.mean()),
                "gold_acc": float(gold.mean()),
                "fp_rate": float((accept & ~gold).mean()),
                "tpr": float(accept[gold].mean()) if gold.any() else None,
                "fpr": float(accept[~gold].mean()) if (~gold).any() else None,
                "zero_adv_groups": float((g.std(axis=1) < 1e-8).mean()),
                "mean_len": float(np.mean([o["n_tokens"] for _, _, o in flat])),
                "finished": float(np.mean([o["finished"] for _, _, o in flat])),
                "mon_accept": float((mon[m] > 0.5).mean()) if m.any() else None,
                "mon_disagree": float((accept[m] & (mon[m] <= 0.5)).mean()) if m.any() else None,
                "lr": self.sched.get_last_lr()[0],
                "scale": self.scaler.get_scale(),
                **{k: v for k, v in upd.items() if k not in ("logp_sum", "kl_sum", "n_tok")},
                "mean_logp": upd["logp_sum"] / max(upd["n_tok"], 1),
                "kl": upd["kl_sum"] / max(upd["n_tok"], 1),
                "peak_gen_gb": round(peak_gen, 2),
                "peak_train_gb": round(torch.cuda.max_memory_allocated(self.pdev) / 1e9, 2),
                **{k: round(v, 2) for k, v in T.items()},
            }
            append_jsonl(os.path.join(self.out, "steps.jsonl"), [srow])
            print(
                f"[step {step} ep {epoch}] R={srow['reward']:.3f} gold={srow['gold_acc']:.3f} "
                f"fp={srow['fp_rate']:.3f} len={srow['mean_len']:.0f} gn={upd['grad_norm']:.3f} "
                + " ".join(f"{k}={v:.0f}s" for k, v in T.items()),
                flush=True,
            )
            self.step = step + 1
            if self.step % cfg["eval_every"] == 0 or self.step == cfg["max_steps"]:
                self.evaluate(self.step)
                self.save(self.step)
            elif self.step % cfg.get("save_every", cfg["eval_every"]) == 0:
                self.save(self.step)  # crash insurance on shared GPUs (adapter + optimizer state)
        with open(os.path.join(self.out, "DONE"), "w") as f:
            f.write(str(self.step))

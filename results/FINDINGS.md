# Aspect 1 (Reliability): findings

Qwen3-1.7B policy, LoRA GRPO against frozen, reference-free Qwen3 judges (0.6B / 1.7B / 4B) on
Countdown and MATH, gold answers hidden from both policy and judge. 16 GRPO runs plus a pre-training
audit of every judge. All tables are in this folder (`REPORT.md` is the auto-generated version with
every table and figure); numbers below are held-out gold accuracy unless stated otherwise.

## Verdict on the hypothesis

> *RLAIF collapses not when the policy out-solves its judge, but when its remaining errors concentrate
> on problems the judge cannot verify. Per-problem verification accuracy and a best-of-N attack should
> predict onset better than the solve gap.*

| Claim | Verdict | Evidence |
|---|---|---|
| Collapse is **not** explained by the policy out-solving its judge | **Supported** | Solve gap has no predictive value (C = 0.48-0.52 on every split). Countdown collapsed even with the 4B judge, which out-solves the policy (47% vs 40%); MATH did not collapse with the 0.6B judge, which the policy out-solves (64% vs 54%). |
| The answer-independent noisy-verifier theory mis-predicts collapse | **Supported** | Every Countdown judge started with global Youden J = 0.70-0.74, yet all collapsed within ~10 steps; J on the policy's *current* outputs fell to 0.03-0.15 at onset (never negative). On MATH the 1.7B judge has J = 0.00 (accepts everything at 0.5) and training still *improved* accuracy 0.59 -> 0.70. |
| Verifiability + best-of-N attack predict per-problem onset better than the solve gap | **Partly supported** | Held-out problems: C = 0.67 vs 0.52 (dC = +0.14, 95% CI [0.12, 0.17]); also on held-out judges (0.59 vs 0.50) and the held-out task (0.60 vs 0.48). But the policy's own base solve rate predicts better (C = 0.73), and when the event is restricted to the judge failing to catch wrong answers ("fooled"), every feature is near chance (0.53-0.59). |
| Collapse happens when remaining errors **concentrate** on unverifiable problems | **Not supported (reversed)** | On MATH 52-86% of the policy's wrong answers sit on the least-verifiable third of problems throughout training, and MATH does not collapse. On Countdown errors are spread evenly (29-37%, chance = 34%) and every run collapses. |

**What did separate collapse from improvement was whether RL found a problem-independent exploit that
transfers across judges.** On Countdown the policy learns one template (write any equation, then
assert "This uses each number exactly once and equals 87") that every Qwen3 judge accepts. A policy
trained against the 0.6B judge fools the unseen 4B judge on 85% of its wrong answers. On MATH the
accepted errors are ordinary, problem-specific reasoning slips, and the 4B judge still catches most
of them (16-39% acceptance). The planted-error audit underestimated this: the 4B judge rejected 82% of
planted wrong equations before training, and then accepted 97% of the RL-found ones.

## Main results

### Runs (standard judges, full pool, lr 5e-5)

| Task | Judge | Judge solve rate | Global J (pre) | Held-out gold: start -> peak -> final | Collapse onset (step) |
|---|---|---|---|---|---|
| Countdown | 0.6B | 0.07 | 0.71 | 0.43 -> 0.43 -> **0.18** | 9 |
| Countdown | 1.7B | 0.41 | 0.70 | 0.41 -> 0.41 -> **0.25** | 10 |
| Countdown | 4B | 0.47 | 0.74 | 0.41 -> 0.41 -> **0.20** | 10 |
| MATH | 0.6B (seed 0 / 1) | 0.54 | 0.56 | 0.63 -> 0.75 / 0.78 -> **0.74 / 0.72** | 46 / 54 (transient) |
| MATH | 1.7B | 0.64 | 0.00 | 0.59 -> 0.70 -> **0.70** | none |
| MATH | 4B | 0.79 | 0.70 | 0.59 -> 0.77 -> **0.70** | none |

Policy base solve rate: Countdown 0.40, MATH 0.64. Judge acceptance on held-out problems ends at
0.95-1.00 for every Countdown run, so the judge-vs-gold gap is 0.74-0.82 there. On MATH it ends at
1.00 for the 0.6B/1.7B judges and 0.66 for the 4B judge, which tracks gold accuracy (0.70).

Figures: `figures/train_countdown_standard.png`, `figures/train_math_standard.png`,
`figures/online_youden_countdown_standard.png`, `figures/concentration_*.png`.

### Pre-training audit (`diag_global.csv`)

* Answer-dependent errors are the norm: 62-68% (Countdown) and 78-89% (MATH) of all false accepts
  fall on the least-verifiable third of problems.
* Commit-first judging *lowers* audit quality for the 0.6B judge (J 0.71 -> 0.46 Countdown,
  0.55 -> 0.34 MATH): a weak judge anchors on its own wrong solution. The 1.7B judge is unchanged.
* The MATH 1.7B judge says Yes to everything (TPR = FPR = 1.0), but its P(Yes) still ranks correct
  above wrong answers (AUROC 0.90). With soft rewards GRPO uses that ranking, which is why it trains fine.

### Per-problem hazard model (`hazard_*.csv`, `figures/hazard_cindex.png`, `figures/km_terciles.png`)

Event: a problem is *exploited* at the first visit where most of its 8 rollouts are judge-accepted
but wrong. 1,377 at-risk (run, problem) units, 850 events; time = training step of the visit.

| Feature set (held-out C-index) | Held-out problems | Held-out judge | Held-out task |
|---|---|---|---|
| solve gap | 0.52 | 0.50 | 0.48 |
| verifiability (Youden J + best-of-16 attack) | 0.67 | 0.59 | 0.60 |
| solve rates (policy, judge, gap) | 0.75 | 0.75 | 0.67 |
| verifiability + solve rates | **0.76** | 0.74 | 0.66 |

Per task (univariate C): on MATH the best-of-16 attack reaches 0.71 (Youden 0.62, solve gap 0.54);
on Countdown nothing exceeds 0.65 and verifiability is near chance (0.54-0.58), because the exploit
reaches every problem within one epoch. Slower learning (lr 1e-5) delays Countdown onset from
~10 to 22-25 steps but does not change the ranking (verifiability 0.59, solve gap 0.48).

### Causal test: remove 25% of the pool (`causal_arms.csv`)

| Task (0.6B judge) | Full pool | Drop least-verifiable | Drop largest solve gap |
|---|---|---|---|
| Countdown, final | 0.18 | 0.01 | 0.04 |
| Countdown, onset | 9 | 10 | 15 |
| MATH, final | 0.74 | 0.76 | 0.70 |

Removing the least-verifiable problems neither prevents nor delays collapse. That is expected if the
exploit is carried by a presentation style rather than by specific problems.

### Commit-first judging (`commit_first_vs_standard.csv`)

| | Standard | Commit-first |
|---|---|---|
| Countdown, 0.6B: onset / final | 9 / 0.18 | 19 / 0.00 |
| Countdown, 1.7B: onset / final | 10 / 0.25 | 10 / 0.25 |
| MATH, 0.6B: peak / final | 0.75 / 0.74 | 0.77 / 0.73 |

Commit-first doubles the time to collapse with the weakest judge, but ends worse, and has no effect
with the 1.7B judge. It did not protect against the Countdown exploit.

### Label-free early stopping (`early_stopping.csv`; thresholds tuned on the other task)

| Rule | Countdown regret | MATH regret |
|---|---|---|
| no stopping | 0.29 | **0.04** |
| response-length drift | **0.11** | 0.13 |
| judge self-disagreement (commit-first vs standard) | 0.22 | 0.07 |
| reward on low- vs high-verifiability problems | 0.29 | 0.08 |
| reward plateau | 0.23 | 0.10 |

Regret = best held-out checkpoint minus the checkpoint the rule picks. Length drift catches the
Countdown collapse (responses shrink from ~240 to ~90 tokens) but raises false alarms on MATH. The
self-disagreement monitor is safe on MATH, but on Countdown the exploit fools the judge in both modes,
so it rarely fires. No rule transfers across tasks without cost.

### Do exploits transfer across judges? (`cross_judge_*.csv`)

Acceptance of last-epoch *wrong* rollouts by each judge:

| Policy trained against | Scored by 0.6B | Scored by 1.7B | Scored by 4B |
|---|---|---|---|
| Countdown, 0.6B | 1.00 | 1.00 | 0.85 |
| Countdown, 1.7B | 1.00 | 1.00 | 0.91 |
| Countdown, 4B | 0.98 | 0.98 | 0.97 |
| MATH, 0.6B | 0.98 | 0.98 | 0.32 |
| MATH, 1.7B | 0.80 | 1.00 | 0.16 |
| MATH, 4B | 0.66 | 0.97 | 0.39 |

## What this suggests

* Audit judges with **presentation attacks** (e.g. adding confident verification claims to wrong
  answers, or the held-out-judge transfer test above), not only with best-of-N over base samples.
  Base samples rarely contain the style that RL discovers.
* A **held-out judge from a different family** is the natural label-free monitor. Within one family,
  the monitor is fooled by the same hack.
* Per-problem verifiability is a reasonable predictor of *where* exploitation lands when errors are
  problem-specific (MATH). It does not predict *whether* a run collapses.

## Limitations

* One seed per configuration, except MATH/0.6B (two seeds: final 0.74 and 0.72). Held-out sets have
  128 problems (standard error about 4 points).
* All judges are non-thinking Qwen3 models giving one-token verdicts. Judges that reason first, or
  come from another model family, may behave very differently. The cross-judge transfer is within one
  family.
* Compute-bound scale (8x Maxwell GPUs): 128-problem pools, 8x8 rollouts per step, 128 (Countdown) /
  96 (MATH) steps, MATH levels 1-3, 12 base samples per problem in the audit. MATH may collapse later
  than 96 steps.
* The *exploited* event mixes "the policy is wrong" with "the judge accepts it"; the *fooled* and
  *unlearned* definitions are reported alongside it (`hazard_*_fooled.csv`, `hazard_*_unlearned.csv`).
* Not run, for lack of compute: the random-removal control arm, commit-first with the 4B judge, and
  commit-first on MATH with the 1.7B judge (the 4B commit-first audit was also skipped).

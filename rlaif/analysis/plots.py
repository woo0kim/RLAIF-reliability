"""Static figures for the report (matplotlib).

Conventions: categorical colours in a fixed order (judge size -> slot), thin
2px-ish lines, hairline recessive grid, one y-axis per panel, legends always
shown for >= 2 series, text in neutral ink (never the series colour).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
JUDGE_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]


def judge_color(j):
    return SERIES[JUDGE_ORDER.index(j)] if j in JUDGE_ORDER else SERIES[3]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9,
        "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8,
    })


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_curves(curves_by_run, path, title):
    """curves_by_run: list of (label, color, curves_df, heldout_df, onset). Two panels share the x-axis, each with one y-axis."""
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharex=True)
    for label, color, c, ev, onset in curves_by_run:
        axes[0].plot(c["step"], c["accept_ema"], color=color, label=label)
        axes[1].plot(c["step"], c["gold_acc_ema"], color=color, label=label)
        if ev is not None and len(ev):
            axes[2].plot(ev["step"], ev["gold_acc"], color=color, marker="o", ms=3, label=label)
        if onset is not None:
            for ax in axes:
                ax.axvline(onset, color=color, lw=0.8, alpha=0.6)
    axes[0].set_title("Judge acceptance (train rollouts)")
    axes[1].set_title("Gold accuracy (train rollouts)")
    axes[2].set_title("Gold accuracy (held-out)")
    for ax in axes:
        ax.set_xlabel("GRPO step")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower right")
    fig.suptitle(title, color=INK, fontsize=10)
    save(fig, path)


def online_youden(series, path, title):
    setup()
    fig, ax = plt.subplots(figsize=(5, 3))
    for label, color, d, onset in series:
        ax.plot(d["step"], d["youden"], color=color, label=label)
        if onset is not None:
            ax.axvline(onset, color=color, lw=0.8, alpha=0.6)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("TPR - FPR on current rollouts")
    ax.set_title(title)
    ax.legend()
    save(fig, path)


def km_by_tercile(km_sets, path, title):
    """km_sets: list of (panel_title, [(label, color, km_df)])."""
    setup()
    fig, axes = plt.subplots(1, len(km_sets), figsize=(4.2 * len(km_sets), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (ptitle, curves) in zip(axes, km_sets):
        for label, color, km in curves:
            ax.step(km["v"], km["surv"], where="post", color=color, label=label)
        ax.set_title(ptitle)
        ax.set_xlabel("GRPO step")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("P(not yet exploited)")
    fig.suptitle(title, color=INK, fontsize=10)
    save(fig, path)


def cindex_bars(df, path, title):
    """df: columns split, features, c_index (mean), lo, hi."""
    setup()
    df = df[df["features"] != "time only"]  # C = 0.5 by construction (no covariates)
    splits = list(df["split"].unique())
    feats = list(df["features"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(feats)
    for k, f in enumerate(feats):
        d = df[df.features == f].set_index("split").reindex(splits)
        x = np.arange(len(splits)) + (k - (len(feats) - 1) / 2) * w
        ax.bar(x, d["c_index"] - 0.5, bottom=0.5, width=w * 0.9, color=SERIES[k], label=f)
        if "lo" in d:
            ax.errorbar(x, d["c_index"], yerr=[d["c_index"] - d["lo"], d["hi"] - d["c_index"]], fmt="none", ecolor=INK2, lw=0.8)
    ax.axhline(0.5, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(splits)))
    ax.set_xticklabels([s.replace("_", " ") for s in splits])
    ax.set_ylabel("C-index (held-out)")
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, path)


def arm_curves(arms, path, title):
    """arms: list of (label, color, curves_df, heldout_df)."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1), sharex=True)
    for label, color, c, ev in arms:
        axes[0].plot(c["step"], c["fp_rate_ema"], color=color, label=label)
        if ev is not None and len(ev):
            axes[1].plot(ev["step"], ev["gold_acc"], color=color, marker="o", ms=3, label=label)
    axes[0].set_title("Judge-accepted but wrong (train rollouts)")
    axes[1].set_title("Gold accuracy (held-out)")
    for ax in axes:
        ax.set_xlabel("GRPO step")
    axes[0].legend()
    fig.suptitle(title, color=INK, fontsize=10)
    save(fig, path)


def scatter_features(f, x, y, path, title, xlabel, ylabel):
    setup()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for j in JUDGE_ORDER:
        d = f[f.judge == j]
        if len(d):
            ax.scatter(d[x], d[y], s=14, color=judge_color(j), alpha=0.7, label=j, edgecolors=SURFACE, linewidths=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save(fig, path)


def concentration_curves(series, path, title):
    """series: list of (label, color, df with step, err_share_low, rollout_share_low)."""
    setup()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for label, color, d in series:
        ax.plot(d["step"], d["err_share_low"], color=color, label=label)
    if series:
        ax.axhline(float(np.nanmean(series[0][2]["rollout_share_low"])), color=INK2, lw=0.8)
        ax.text(ax.get_xlim()[0], float(np.nanmean(series[0][2]["rollout_share_low"])) + 0.01, " share of rollouts", color=INK2, fontsize=7)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("share of wrong rollouts")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, path)

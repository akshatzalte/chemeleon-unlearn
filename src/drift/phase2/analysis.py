"""Phase-2 analysis of the fine-tuning ladder: paired gaps, the pre-registered primary endpoint (spec §7), curves.

Input is the score table written by scripts/p2_04_score.py: one row per (task, arm, N, seed, ckpt) with
`metric`, `direction` ('max' | 'min') and `score`. All gaps are sign-normalized so that **higher = better**
(MAE is negated), i.e. gap(a, b) > 0 means arm a is better than arm b.

Pairing: arms are paired by (task, N, seed). `seed` is the fine-tuning seed, a deterministic function of the
subsample index (ladder.ft_seed), so equal (task, N, seed) = the identical subsample; for N = 'full' the seeds
42/117/709 are the three paper seeds and 'full' counts as one N. Rows without a partner are dropped.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Kept in step with drift.phase2.ladder.TASKS (tests assert equality) so this module stays light to import.
TASKS = ("tdcommons/ames", "tdcommons/ld50-zhu")
N_ORDER = ("16", "32", "64", "128", "256", "512", "1024", "full")
SMALL_N = ("16", "32", "64")
LARGE_N = ("512", "1024", "full")
PRIMARY = ("T100", "B")
MIN_POINTS = 2  # of the 3 small-N points, per task

# (arm_a, arm_b) contrasts shown as gap panels; a - b > 0 means a is better.
CONTRASTS = (("T100", "B"), ("T25", "B"), ("T100", "R100"), ("T25", "R25a"), ("T25", "R25b"), ("B", "chemeleon"),
             ("B", "rand"),  # own gap panel (row 4): pretraining vs a random-init encoder
             ("T100_ames", "B"), ("T100_ld50", "B"), ("T100_ames", "T100"), ("T100_ld50", "T100"))  # row 5: per-task sets
CONTRAST_ROWS = (CONTRASTS[:3], CONTRASTS[3:6], CONTRASTS[6:7], CONTRASTS[7:])  # gap panels in curves(), rows 2-5
ARM_ORDER = ("B", "T25", "T100", "R25a", "R25b", "R100", "chemeleon", "D535", "rand", "T100_ames", "T100_ld50")
METRIC_LABEL = {"tdcommons/ames": "Ames ROC-AUC (higher better)", "tdcommons/ld50-zhu": "LD50 MAE (lower better)"}


def _n_str(n) -> str:
    return "full" if str(n) == "full" else str(int(n))


def _n_sort_key(n: str) -> int:
    return N_ORDER.index(n) if n in N_ORDER else len(N_ORDER) + int(n)


def _prepare(scores: pd.DataFrame, ckpt: str) -> pd.DataFrame:
    """Rows of one checkpoint with N normalized to str and `s` = score with higher = better."""
    d = scores[scores["ckpt"] == ckpt].copy()
    d["N"] = d["N"].map(_n_str)
    bad = set(d["direction"]) - {"max", "min"}
    if bad:
        raise ValueError(f"unknown metric direction(s): {sorted(bad)}")
    if d.groupby("task")["direction"].nunique().gt(1).any():
        raise ValueError("a task has mixed metric directions")
    if d.duplicated(["task", "arm", "N", "seed"]).any():
        raise ValueError("duplicate (task, arm, N, seed) rows in score table")
    d["s"] = np.where(d["direction"] == "max", d["score"], -d["score"])
    return d


def _gaps_from(d: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    a = d[d.arm == arm_a][["task", "N", "seed", "s"]]
    b = d[d.arm == arm_b][["task", "N", "seed", "s"]]
    m = a.merge(b, on=["task", "N", "seed"], suffixes=("_a", "_b"))  # inner: unmatched seeds dropped
    m["d"] = m["s_a"] - m["s_b"]
    rows = []
    for (task, n), g in m.groupby(["task", "N"]):
        k = len(g)
        sd = float(g["d"].std(ddof=1)) if k > 1 else float("nan")
        rows.append(dict(task=task, N=n, gap=float(g["d"].mean()), sd=sd, n=k, se=sd / np.sqrt(k) if k > 1 else float("nan")))
    out = pd.DataFrame(rows, columns=["task", "N", "gap", "sd", "n", "se"])
    if out.empty:
        return out
    out["_k"] = out["N"].map(_n_sort_key)
    return out.sort_values(["task", "_k"]).drop(columns="_k").reset_index(drop=True)


def gaps(scores: pd.DataFrame, arm_a: str, arm_b: str, ckpt: str = "es") -> pd.DataFrame:
    """Paired gap arm_a - arm_b per (task, N): columns task, N (str; 'full' last), gap (mean of the paired
    differences, sign-normalized: > 0 = a better), sd (SD of the paired differences, ddof=1), n (pairs), se."""
    return _gaps_from(_prepare(scores, ckpt), arm_a, arm_b)


# Paired es scores required per (task, N) for the pre-registered endpoint (ladder.SEEDS_FOR, as strings).
EXPECTED_PAIRS = {"16": 10, "32": 10, "64": 10, "128": 5, "256": 5, "512": 3, "1024": 3, "full": 3}


def check_endpoint_pairs(scores: pd.DataFrame, arms=PRIMARY, ckpt: str = "es") -> tuple[pd.DataFrame, bool]:
    """Completeness guard for the primary endpoint. Per (task, N): `paired` = seeds scored for BOTH endpoint arms,
    n_<arm> = rows per arm, `sha_bad` = (task, N, seed) groups whose `subsample_sha1` is missing or differs between
    any of the arms in the table (the arms must have been fine-tuned on the identical rows), `ok` = paired equals
    EXPECTED_PAIRS[N] and sha_bad == 0. Returns (table, all ok)."""
    d = _prepare(scores, ckpt)
    if "subsample_sha1" not in d.columns:
        d["subsample_sha1"] = None
    flag = d.groupby(["task", "N", "seed"])["subsample_sha1"].agg(lambda x: bool(x.isna().any() or x.nunique() > 1))
    sha_bad_by = flag[flag].reset_index().groupby(["task", "N"]).size()
    rows = []
    for task in TASKS:
        for n in N_ORDER:
            dn = d[(d.task == task) & (d.N == n)]
            seeds = {a: set(dn[dn.arm == a].seed) for a in arms}
            paired = len(set.intersection(*seeds.values()))
            sha_bad = int(sha_bad_by.get((task, n), 0))
            rows.append(dict(task=task, N=n, expected=EXPECTED_PAIRS[n], paired=paired,
                             **{f"n_{a}": int((dn.arm == a).sum()) for a in arms}, sha_bad=sha_bad,
                             ok=paired == EXPECTED_PAIRS[n] and sha_bad == 0))
    chk = pd.DataFrame(rows)
    return chk, bool(chk.ok.all())


def _task_endpoint(g: pd.DataFrame) -> dict:
    gap = g.set_index("N")["gap"]
    sd = g.set_index("N")["sd"]
    small = [n for n in SMALL_N if n in gap.index]
    points_pass = int(sum(bool(gap[n] > sd[n]) for n in small))  # NaN SD (n=1) compares False
    ns = [n for n in N_ORDER if n in gap.index]
    complete = len(ns) == len(N_ORDER)
    if len(ns) >= 3 and gap[ns].nunique() > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(stats.spearmanr(gap[ns].to_numpy(), np.arange(len(ns))).statistic)  # N_ORDER is increasing in N
    else:
        rho = float("nan")
    lo = [n for n in SMALL_N if n in gap.index]
    hi = [n for n in LARGE_N if n in gap.index]
    gap_small = float(gap[lo].mean()) if lo else float("nan")
    gap_large = float(gap[hi].mean()) if hi else float("nan")
    small_gt_large = bool(gap_small > gap_large)
    success = bool(complete and len(small) == len(SMALL_N) and points_pass >= MIN_POINTS and rho < 0 and small_gt_large)
    return dict(points_pass=points_pass, spearman=rho, small_gt_large=small_gt_large, success=success,
                gap_small=gap_small, gap_large=gap_large, complete=complete, n_sizes=len(ns))


def endpoint(scores: pd.DataFrame) -> dict:
    """Primary endpoint (spec §7), `es` checkpoint: per task and N, d_s = T100 - B paired per subsample
    (sign-normalized), gap(N) = mean d_s, SD(N) = SD d_s. Per task: success = gap(N) > SD(N) at >= 2 of
    N in {16,32,64} AND Spearman(gap, log N) < 0 over the 8 sizes AND mean gap(N <= 64) > mean gap(N >= 512).
    A task without all 8 sizes cannot succeed (`complete` False). Overall `success` = both tasks.

    Returns {task: {points_pass, spearman, small_gt_large, success, gap_small, gap_large, complete, n_sizes},
    'success': bool, 'gaps': [per-(task, N) records of the T100 - B gap table]}."""
    g = gaps(scores, *PRIMARY)
    out: dict = {}
    for task in TASKS:
        gt = g[g.task == task]
        out[task] = (_task_endpoint(gt) if len(gt) else
                     dict(points_pass=0, spearman=float("nan"), small_gt_large=False, success=False,
                          gap_small=float("nan"), gap_large=float("nan"), complete=False, n_sizes=0))
    out["success"] = bool(all(out[t]["success"] for t in TASKS))
    out["gaps"] = g.to_dict("records")
    return out


# --------------------------------------------------------------------------------------------- plotting
# Reference categorical palette (dataviz skill, light mode), assigned in fixed entity order.
_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_CAT_EXTRA = ["#0a9396", "#9b2fae"]  # the per-task arms T100_ames, T100_ld50
ARM_COLOR = {a: _CAT[i] for i, a in enumerate(ARM_ORDER[:8])}
ARM_COLOR.update(zip(ARM_ORDER[8:], ["#6b6a66", *_CAT_EXTRA]))  # rand = neutral grey: the no-pretraining baseline
CONTRAST_COLOR = {c: _CAT[i % len(_CAT)] for i, c in enumerate(CONTRASTS)}  # distinct within each panel row
CONTRAST_COLOR[("B", "rand")] = ARM_COLOR["rand"]
_INK, _INK2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
_FULL_X = 4096  # x position of 'full' on the log axis (label only; the true row count is ~5.8-5.9k)


def _x(n: str) -> float:
    return float(_FULL_X if n == "full" else int(n))


def _style(ax, ylabel: str, title: str | None = None):
    ax.set_xscale("log", base=2)
    ticks = [_x(n) for n in N_ORDER]
    ax.set_xticks(ticks)
    ax.set_xticklabels(N_ORDER, color=_INK2, fontsize=8)
    ax.minorticks_off()
    ax.set_ylabel(ylabel, color=_INK2, fontsize=9)
    if title:
        ax.set_title(title, color=_INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(axis="y", colors=_INK2, labelsize=8)
    ax.set_facecolor(_SURFACE)


def _line(ax, xs, mean, se, color, label):
    xs, mean, se = np.asarray(xs), np.asarray(mean), np.asarray(se)
    ax.plot(xs, mean, color=color, lw=2, marker="o", ms=5, mec=_SURFACE, mew=1.2, label=label)
    ok = np.isfinite(se)
    if ok.any():
        ax.fill_between(xs[ok], (mean - se)[ok], (mean + se)[ok], color=color, alpha=0.15, lw=0)


def curves(scores: pd.DataFrame, out_png, ckpt: str = "es") -> None:
    """Figure: row 1 = score vs N per arm (mean +- SE over seeds), rows 2+ = paired gaps (sign-normalized,
    > 0 = first arm better, mean +- SE of the paired differences). One column per task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _prepare(scores, ckpt)
    rows = [r for k, r in enumerate(CONTRAST_ROWS)
            if k < 3 or any(len(_gaps_from(d, a, b)) for a, b in r)]  # the tailored-arm row only when it has data
    fig, axes = plt.subplots(1 + len(rows), len(TASKS), figsize=(11, 3.5 * (1 + len(rows))), sharex="col", facecolor=_SURFACE,
                             squeeze=False)
    for j, task in enumerate(TASKS):
        ax = axes[0, j]
        dt = d[d.task == task]
        for arm in [a for a in ARM_ORDER if a in set(dt.arm)]:
            r = dt[dt.arm == arm].assign(raw=lambda x: x["score"]).groupby("N")["raw"].agg(["mean", "std", "count"])
            r = r.loc[sorted(r.index, key=_n_sort_key)]
            se = (r["std"] / np.sqrt(r["count"])).to_numpy()
            _line(ax, [_x(n) for n in r.index], r["mean"].to_numpy(), se, ARM_COLOR[arm], arm)
        _style(ax, METRIC_LABEL[task], f"{task}: test score by fine-tuning set size (es)")
        for k, row_contrasts in enumerate(rows):
            ax = axes[1 + k, j]
            for ca, cb in row_contrasts:
                g = _gaps_from(d, ca, cb)
                g = g[g.task == task]
                if len(g):
                    _line(ax, [_x(n) for n in g.N], g.gap.to_numpy(), g.se.to_numpy(), CONTRAST_COLOR[(ca, cb)], f"{ca} - {cb}")
            ax.axhline(0, color=_INK2, lw=1)
            _style(ax, "paired gap (higher-is-better units)")
        axes[-1, j].set_xlabel("fine-tuning set size N ('full' = all training rows)", color=_INK2, fontsize=9)
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, framealpha=0.9, edgecolor=_GRID, facecolor=_SURFACE, fontsize=7, labelcolor=_INK2,
                      loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=_SURFACE)
    plt.close(fig)

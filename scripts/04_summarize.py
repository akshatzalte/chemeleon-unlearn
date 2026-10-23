"""Summaries and findings tables from the combined measurement outputs.

Usage: 04_summarize.py [--in results] [--out results/summary]
Reads {drift_long,families,eta2,posctrl,effdim}.parquet and scores.parquet from --in (written by
`03_measure.py --combine`), meta.json from data/tasks/<slug>/, es epochs from runs/<slug>/<regime>/ft/seedK/run.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift import summarize as S  # noqa: E402
from drift.config import DATA, RESULTS, RUNS, slug  # noqa: E402

DIV = LinearSegmentedColormap.from_list("div", ["#b2182b", "#ef8a62", "#e8e8e8", "#67a9cf", "#2166ac"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal", "text.usetex": False})
INK, MUTED = "#222222", "#666666"
C1, C2 = "#2a6f97", "#d08c2f"


def load_meta(tasks):
    """{task: n_train}. Tasks without meta.json are omitted (-> n_train NaN -> low_confidence=True) with a warning."""
    out = {}
    for t in tasks:
        p = DATA / "tasks" / slug(t) / "meta.json"
        if p.exists():
            out[t] = json.loads(p.read_text())["n_train"]
        else:
            print(f"WARNING: no meta.json for {t}; marking low_confidence=True", file=sys.stderr, flush=True)
    return out


def load_es_epochs(tasks):
    out = {}
    for t in tasks:
        for reg in ("full", "n128"):
            eps = [json.loads(p.read_text())["es_epoch"] for p in (RUNS / slug(t) / reg / "ft").glob("seed*/run.json")]
            if eps:
                out[(t, reg)] = float(np.mean(eps))
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK); ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)


def heatmaps(fam, out):
    f = fam[(fam["family_def"] == "module") & (fam["regime"] == "full") & (fam["ckpt"] == "es")]
    for v in S.PRIMARY_VIEWS:
        w = f[f["probe_set"] == v].pivot(index="task", columns="family", values="mean_delta")
        if w.empty:
            continue
        w = w[w.mean().sort_values().index]
        fig, ax = plt.subplots(figsize=(max(8, 0.28 * w.shape[1] + 3), max(4.5, 0.28 * w.shape[0] + 3)))
        lim = max(0.05, np.nanpercentile(np.abs(w.values), 98))
        im = ax.imshow(w.values, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
        ax.set_xticks(range(w.shape[1]), w.columns, rotation=90); ax.set_yticks(range(w.shape[0]), w.index)
        ax.set_title(f"Mean dR2 per Mordred module (ft vs pre), full/es, view={v}", color=INK, fontsize=10, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("mean dR2 (negative = forgotten)", color=INK, fontsize=8); cb.ax.tick_params(labelsize=7)
        style(ax); fig.tight_layout(); fig.savefig(out / f"heatmap_{v}.png", dpi=150); plt.close(fig)


def spectrum_fig(sp, out):
    """Main panel: PubChem PC spectrum (same PCs eligible for every task). Task-view panel: only PCs eligible in
    >= PC_PLOT_MIN_FRAC of the tasks (so the mean is not a changing mix of tasks); n_tasks annotated at the ends."""
    if sp.empty:
        return
    views = [("pubchem", "view=pubchem (main)"), ("task_clean", f"view=task_clean (PCs eligible in >= {S.PC_PLOT_MIN_FRAC:.0%} of tasks)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
    for ax, (v, title) in zip(axes, views):
        for (reg, ck), col in zip((("full", "es"), ("n128", "es")), (C1, C2)):
            d = sp[(sp["view"] == v) & (sp["regime"] == reg) & (sp["ckpt"] == ck)]
            if v != "pubchem":
                d = d[d["frac_eligible"] >= S.PC_PLOT_MIN_FRAC]
            if d.empty:
                continue
            ax.plot(d["pc"], d["mean_dr2"], color=col, lw=1.5, label=f"{reg}/{ck}")
            ax.fill_between(d["pc"], d["mean_dr2"] - d["sem"], d["mean_dr2"] + d["sem"], color=col, alpha=0.15, lw=0)
            ax.annotate(f"n={int(d['n_tasks'].iloc[0])}", (d["pc"].iloc[0], d["mean_dr2"].iloc[0]), fontsize=7, color=col, xytext=(3, 4), textcoords="offset points")
            ax.annotate(f"n={int(d['n_tasks'].iloc[-1])}", (d["pc"].iloc[-1], d["mean_dr2"].iloc[-1]), fontsize=7, color=col, xytext=(-2, 5), textcoords="offset points", ha="right")
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale("log")
        ax.set_xlabel("PC index (high variance to tail)"); ax.set_title(title, loc="left", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("mean over tasks of dR2"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pc_spectrum.png", dpi=150); plt.close(fig)


def drift_fraction_fig(pt, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.3 * pt["task"].nunique() + 1.5)), sharey=True)
    for ax, v in zip(axes, S.PRIMARY_VIEWS):
        a = pt[(pt["probe_set"] == v) & (pt["ckpt"] == "es")].pivot(index="task", columns="regime", values="drift_fraction")
        a = a.reindex(columns=["full", "n128"]).sort_values("full")
        y = np.arange(len(a))
        ax.barh(y + 0.2, a["full"], 0.38, color=C1, label="full/es")
        ax.barh(y - 0.2, a["n128"], 0.38, color=C2, label="n128/es")
        ax.axvline(1, color=MUTED, lw=0.8, ls="--"); ax.set_yticks(y, a.index)
        ax.set_xlabel("drift fraction (median dR2 ft / median dR2 random encoder)")
        ax.set_title(f"view={v}", loc="left", fontsize=10, color=INK); style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "drift_fraction.png", dpi=150); plt.close(fig)


def compare(pt, col, by, a, b):
    """Wide comparison of `col` between two levels of `by` (ckpt or regime) per task x view."""
    other = [c for c in ("regime", "ckpt") if c != by]
    d = pt[pt[by].isin([a, b])]
    w = d.pivot_table(index=["task", "probe_set"] + other, columns=by, values=col, aggfunc="first")
    w.columns = [f"{col}_{c}" for c in w.columns]
    return w.reset_index()


README = """# results/summary
- per_task.csv: one row per task x regime x ckpt x probe_set (every view with a fine-tuned aggregate). Columns: median_dr2/mean_dr2 (over eligible descriptors, ft seed-mean dR2), n_eligible, n_dropped, drop_frac, cka_mean/cka_sd/cka_median/cka_n_seeds (ft seeds vs pre), cka_seedpair (ft seed-vs-seed), n_rows (mean clean rows), pre_r2_median, median_wchange (relative weight change, median over matrices/seeds), n_fam_*/n_dropped_fam_* (module, corr), eta2_*/eta2_p_* (permutation test), posctrl_* (top-20 label-correlated descriptors), gap/gap_sd/gap_n_seeds/gap_metric (ft - frozen test score, positive = ft better; gap_sd = pooled seed SD), es_epoch_mean, n_train, low_confidence (n_train<1000), rand_median_dr2/rand_cka/rand_cka_seedpair (random-encoder anchor; task views use the task_all anchor), drift_fraction (median over descriptors eligible in both views of ft dR2 / rand dR2), eta2_q95_*/eta2_ratio_* (eta2 / permutation-null 95th percentile), cka_debiased_median/rand_cka_debiased_same_rows/n_cka_rows (from results/cka.parquet, scripts/06_cka.py), branch, gap_status, primary_view, and the sensitivity branches branch_cka_debiased, branch_eta2_strict, branch_no_eta2.
- branches.csv: every task with a primary-view row; branch in each co-primary view (full/es), agreed (agreed branch, `split` if the views differ, `missing` if either view has no branch), gap_status, low_confidence (tasks without meta.json are low_confidence), plus the same columns for each sensitivity variant (branch_cka_debiased_*, branch_eta2_strict_*, branch_no_eta2_*). branch_counts.csv: tasks per branch per view and agreed, one block per `variant` (primary, cka_debiased, eta2_strict, no_eta2).
- PRIMARY vs SENSITIVITY: `branch` (pre-registered rule: CKA median >= 0.95 and drop <= 5% -> little_drift; else both eta2 p < 0.05 and ft-frozen gap > 1 seed-SD -> descriptor_specific; else uniform) is the primary result and is never changed. Sensitivity analyses only: branch_cka_debiased (CKA replaced by the debiased-CKA median, same thresholds; linear CKA is biased upward when n << 2048 dims, so small-task and clean-row CKA is inflated), branch_eta2_strict (descriptor_specific additionally requires eta2_ratio >= 1.5 for BOTH family definitions), branch_no_eta2 (eta2 condition dropped).
- eta2 gate caveat: the permutation test is anti-conservative (correlated descriptors) and p sat at the 0.001 floor in essentially every task/view in the pilot, so the primary eta2 gate was ~always satisfied and does not discriminate; eta2_ratio and the two eta2 sensitivity branches are the check on that.
- drift_fraction caveat: for task_clean and novel views the rand anchor is the task_all rand-encoder dR2 (different rows from the view's). The same-rows anchor for CKA is rand_cka_debiased_same_rows (rand encoders on the same rows; task_clean: that regime's seed-matched clean rows).
- cross_task.csv: long; kind (descriptor | family_module | family_corr), regime, ckpt, probe_set, name, n_tasks_dropped, n_tasks_eligible. cross_task_primary.csv: wide at full/es for the two co-primary views (+ min over views).
- pc_spectrum.csv: mean over tasks of seed-mean dR2 per PC (eligible PCs) for primary *_pc views at full/es and n128/es, with n_tasks (tasks where the PC is eligible), n_tasks_total, frac_eligible. pc_spectrum.png: PubChem panel is the main one (same PCs for every task); the task_clean panel shows only PCs eligible in >= 80% of tasks, with n_tasks annotated. pc_summaries.csv: eligible-PC evr-weighted dR2 (evrw_eligible) next to stored all-PC evrw_all_pcs_stored.
- robustness_ownalpha.csv (fixed vs own alpha on PubChem), robustness_novel.csv (novel vs non-novel task views), robustness_ckpt.csv (es vs last), robustness_regime.csv (n128 vs full): wide comparisons of median_dr2 / drop_frac / cka_median.
- posctrl.csv: per task x view: n_eligible_of_top20, n_gained_2se, n_dropped, mean_dr2.
- anchors.csv: per task x view: ft vs rand/seed-pair CKA, median dR2, drift_fraction.
- effdim.csv: effective dimensionality per family (sorted by n_descriptors; ALL = overall).
- heatmap_<view>.png (task x module mean dR2, full/es), pc_spectrum.png, drift_fraction.png.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(RESULTS))
    ap.add_argument("--out", default=str(RESULTS / "summary"))
    a = ap.parse_args()
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    long = pd.read_parquet(inp / "drift_long.parquet")
    fam, eta = pd.read_parquet(inp / "families.parquet"), pd.read_parquet(inp / "eta2.parquet")
    pos = pd.read_parquet(inp / "posctrl.parquet")
    scores = pd.read_parquet(inp / "scores.parquet") if (inp / "scores.parquet").exists() else None
    tasks = sorted(t for t in long["task"].unique() if t)
    cka_p = inp / "cka.parquet"
    cka = pd.read_parquet(cka_p) if cka_p.exists() else None
    if cka is None:
        print("note: no cka.parquet (run scripts/06_cka.py): debiased-CKA columns/branch will be empty", file=sys.stderr)
    pt = S.per_task_summary(long, fam, eta, pos, scores, load_meta(tasks), load_es_epochs(tasks), cka_table=cka)
    pt.to_csv(out / "per_task.csv", index=False)

    bt = S.branch_table(pt)
    bt.to_csv(out / "branches.csv", index=False)
    S.branch_counts(bt).to_csv(out / "branch_counts.csv", index=False)
    ct = S.cross_task_counts(long, fam)
    ct.to_csv(out / "cross_task.csv", index=False)
    S.cross_task_primary(ct).to_csv(out / "cross_task_primary.csv", index=False)
    sp = S.pc_spectrum(long)
    sp.to_csv(out / "pc_spectrum.csv", index=False)
    S.pc_summaries(long).to_csv(out / "pc_summaries.csv", index=False)
    S.posctrl_table(pos).to_csv(out / "posctrl.csv", index=False)
    pd.read_parquet(inp / "effdim.parquet").sort_values("n_descriptors", ascending=False).to_csv(out / "effdim.csv", index=False)

    # anchors
    anc = pt[["task", "regime", "ckpt", "probe_set", "cka_mean", "cka_seedpair", "median_dr2", "rand_cka", "rand_cka_seedpair",
              "rand_median_dr2", "drift_fraction", "cka_debiased_median", "rand_cka_debiased_same_rows", "n_cka_rows"]]
    anc.to_csv(out / "anchors.csv", index=False)

    # robustness
    cols = ["task", "regime", "ckpt", "probe_set", "median_dr2", "n_dropped", "n_eligible", "drop_frac", "cka_median", "n_rows", "pre_r2_median"]
    oa = pt[pt["probe_set"].isin(["pubchem", "pubchem_ownalpha"])][cols]
    oa.pivot_table(index=["task", "regime", "ckpt"], columns="probe_set", values=["median_dr2", "n_dropped", "n_eligible", "drop_frac"],
                   aggfunc="first").pipe(lambda w: w.set_axis([f"{m}_{p}" for m, p in w.columns], axis=1)).reset_index() \
        .to_csv(out / "robustness_ownalpha.csv", index=False)
    nv = pt[pt["probe_set"].str.contains("novel") & ~pt["probe_set"].str.endswith("_pc") | pt["probe_set"].isin(["task_all", "task_clean"])]
    nv[cols].sort_values(["task", "regime", "ckpt", "probe_set"]).to_csv(out / "robustness_novel.csv", index=False)
    prim = pt[pt["probe_set"].isin(S.PRIMARY_VIEWS)]
    rc = [compare(prim, c, "ckpt", "es", "last") for c in ("median_dr2", "drop_frac", "cka_median")]
    rg = [compare(prim, c, "regime", "full", "n128") for c in ("median_dr2", "drop_frac", "cka_median")]
    for name, parts, on in (("ckpt", rc, ["task", "probe_set", "regime"]), ("regime", rg, ["task", "probe_set", "ckpt"])):
        r = parts[0]
        for p in parts[1:]:
            r = r.merge(p, on=on)
        r.to_csv(out / f"robustness_{name}.csv", index=False)

    heatmaps(fam, out); spectrum_fig(sp, out); drift_fraction_fig(pt, out)
    (out / "README.md").write_text(README)

    print(bt.to_string(index=False)); print(S.branch_counts(bt).to_string(index=False))


if __name__ == "__main__":
    main()

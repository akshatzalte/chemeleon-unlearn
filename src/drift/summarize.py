"""Pure summary logic for the drift measurement outputs (no file I/O; see scripts/04_summarize.py).

All inputs are the combined long tables written by `scripts/03_measure.py --combine`.
Conventions: `task` is the benchmark name; descriptor-level rows have a non-empty `descriptor`,
scalar rows (cka, n_rows, ...) have descriptor ''. A descriptor/PC is *eligible* in a view iff its `dropped`
value is not NaN."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["task", "regime", "ckpt", "probe_set"]
PRIMARY_VIEWS = ("task_clean", "pubchem")
PRIMARY_CELL = ("full", "es")
CORR_DEFS = ("corr", "cluster")
ETA2_STRICT_RATIO = 1.5
PC_PLOT_MIN_FRAC = 0.8
VARIANTS = ("branch_cka_debiased", "branch_eta2_strict", "branch_no_eta2")  # name of the correlation-family definition in eta2/families
CKA_LITTLE = 0.95
DROP_FRAC_LITTLE = 0.05
P_ALPHA = 0.05
LOW_CONFIDENCE_N = 1000
BRANCHES = ("little_drift", "descriptor_specific", "uniform")


# ---------------------------------------------------------------- pure rules
def classify_branch(median_cka, drop_frac, p_module, p_corr, gap, gap_sd, ratio_module=np.nan, ratio_corr=np.nan,
                    min_ratio=None, use_eta2=True):
    """Pre-registered per-task branch. Returns (branch, gap_status).
    little_drift: median CKA >= 0.95 and dropped/eligible <= 5%.
    descriptor_specific: not little_drift, eta2 p < 0.05 for both family defs, and the sign-normalised
    ft - frozen gap favours ft by more than 1 seed-SD (gap_status 'unknown_gap' if the gap is unavailable:
    then only the first two conditions are used).
    uniform: otherwise. A missing p-value counts as not significant.
    Sensitivity variants (the defaults reproduce the pre-registered rule): `min_ratio` additionally requires
    eta2/null_q95 >= min_ratio for BOTH family defs (missing ratio fails); `use_eta2=False` drops the eta2 condition."""
    if median_cka >= CKA_LITTLE and drop_frac <= DROP_FRAC_LITTLE:
        return "little_drift", "ok"
    if use_eta2:
        sig = bool(p_module < P_ALPHA and p_corr < P_ALPHA)
        if sig and min_ratio is not None:
            sig = bool(ratio_module >= min_ratio and ratio_corr >= min_ratio)
    else:
        sig = True
    if not (np.isfinite(gap) and np.isfinite(gap_sd)):
        return ("descriptor_specific" if sig else "uniform"), "unknown_gap"
    return ("descriptor_specific" if sig and gap > gap_sd else "uniform"), "ok"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA


def agree(a, b) -> str:
    """Agreed branch of two views; `missing` if either view has no branch, `split` if they differ."""
    if _missing(a) or _missing(b):
        return "missing"
    return a if a == b else "split"


def drift_fraction(ft_median, rand_median):
    """Fine-tuned median dR2 on the pre->random-encoder scale (NaN unless the random anchor lost R2)."""
    if not (np.isfinite(ft_median) and np.isfinite(rand_median)) or rand_median >= 0:
        return np.nan
    return ft_median / rand_median


def evr_weighted(delta: pd.Series, dropped: pd.Series, evr: pd.Series) -> float:
    """evr-weighted mean dR2 over ELIGIBLE PCs (dropped not NaN), renormalised by the eligible evr mass."""
    idx = dropped.index[dropped.notna()]
    if len(idx) == 0:
        return np.nan
    w = evr.reindex(idx)
    return float((delta.reindex(idx) * w).sum() / w.sum())


def score_gaps(scores: pd.DataFrame) -> pd.DataFrame:
    """Per task x regime x ckpt: ft - frozen test-score gap, sign-normalised so positive = ft better.
    gap = mean over seed-matched (ft, frozen) pairs with finite scores; gap_sd = pooled seed-SD over the matched seeds
    (sqrt of mean of the ft and frozen score variances)."""
    if scores is None or scores.empty:
        return pd.DataFrame(columns=["task", "regime", "ckpt", "gap", "gap_sd", "gap_n_seeds", "gap_metric"])
    out = []
    for (t, r, c), g in scores.groupby(["task", "regime", "ckpt"]):
        ft = g[g["mode"] == "ft"][["seed", "score"]]
        fr = g[g["mode"] == "frozen"][["seed", "score"]]
        m = ft.merge(fr, on="seed", suffixes=("_ft", "_fr"))  # match seeds
        m = m[np.isfinite(m["score_ft"]) & np.isfinite(m["score_fr"])]
        if m.empty:
            continue
        sign = 1.0 if g["direction"].iloc[0] == "max" else -1.0
        sd = np.sqrt((m["score_ft"].var(ddof=1) + m["score_fr"].var(ddof=1)) / 2) if len(m) > 1 else np.nan
        out.append(dict(task=t, regime=r, ckpt=c, gap=sign * (m["score_ft"] - m["score_fr"]).mean(), gap_sd=sd,
                        gap_n_seeds=len(m), gap_metric=g["metric"].iloc[0]))
    return pd.DataFrame(out, columns=["task", "regime", "ckpt", "gap", "gap_sd", "gap_n_seeds", "gap_metric"])


def anchor_probe_set(probe_set: str) -> str:
    """Probe set whose random-encoder rows anchor this view (rand is only measured on task_all / pubchem)."""
    if probe_set.startswith("pubchem"):
        return probe_set
    return "task_all_pc" if probe_set.endswith("_pc") else "task_all"


def corr_def_name(family_defs) -> str | None:
    return next((d for d in CORR_DEFS if d in set(family_defs)), None)


# ---------------------------------------------------------------- per-task table
def _ft_descriptor_wide(long: pd.DataFrame) -> pd.DataFrame:
    ft = long[(long["encoder"] == "ft") & (long["descriptor"] != "") & long["measure"].isin(["delta_r2_mean", "se", "dropped"])]
    return ft.set_index(KEYS + ["descriptor", "measure"])["value"].unstack("measure").reset_index()


def _anchor_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per (anchor task, probe_set): rand median dR2 (eligible descriptors), mean rand CKA, rand seed-pair CKA."""
    L = long[long["encoder"].str.startswith("rand")]
    d = L[(L["encoder"] == "rand") & (L["descriptor"] != "") & L["measure"].isin(["delta_r2_mean", "dropped"])]
    w = d.set_index(["task", "probe_set", "descriptor", "measure"])["value"].unstack("measure")
    if w.empty:
        return pd.DataFrame(columns=["a_task", "a_probe_set", "rand_median_dr2", "rand_cka", "rand_cka_seedpair"])
    med = w[w["dropped"].notna()]["delta_r2_mean"].groupby(level=["task", "probe_set"]).median().rename("rand_median_dr2")
    cka = L[(L["measure"] == "cka") & L["encoder"].isin(["rand0", "rand1", "rand2"])].groupby(["task", "probe_set"])["value"].mean().rename("rand_cka")
    sp = L[(L["measure"] == "cka_seedpair") & (L["encoder"] == "rand")].groupby(["task", "probe_set"])["value"].mean().rename("rand_cka_seedpair")
    a = pd.concat([med, cka, sp], axis=1).reset_index()
    return a.rename(columns={"task": "a_task", "probe_set": "a_probe_set"})


def _common_medians(el: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Per view: median dR2 of ft and of the rand anchor over descriptors eligible in BOTH."""
    d = long[(long["encoder"] == "rand") & (long["descriptor"] != "") & long["measure"].isin(["delta_r2_mean", "dropped"])]
    w = d.set_index(["task", "probe_set", "descriptor", "measure"])["value"].unstack("measure")
    cols = KEYS + ["drift_median_ft", "drift_median_rand", "drift_n_common"]
    if w.empty:
        return pd.DataFrame(columns=cols)
    w = w[w["dropped"].notna()]["delta_r2_mean"].rename("rand_dr2").reset_index().rename(columns={"task": "a_task", "probe_set": "a_probe_set"})
    e = el[KEYS + ["descriptor", "delta_r2_mean"]].copy()
    e["a_task"] = np.where(e["probe_set"].str.startswith("pubchem"), "", e["task"])
    e["a_probe_set"] = e["probe_set"].map(anchor_probe_set)
    m = e.merge(w, on=["a_task", "a_probe_set", "descriptor"], how="inner")
    g = m.groupby(KEYS)
    return pd.DataFrame({"drift_median_ft": g["delta_r2_mean"].median(), "drift_median_rand": g["rand_dr2"].median(),
                         "drift_n_common": g.size()}).reset_index()


def _branches(pt: pd.DataFrame, cka_col: str, **kw):
    """(branch list, gap_status list) per row of pt; None where the CKA or a p-value is missing."""
    res = [classify_branch(r.cka, r.drop_frac, r.eta2_p_module, r.eta2_p_corr, r.gap, r.gap_sd,
                           ratio_module=r.eta2_ratio_module, ratio_corr=r.eta2_ratio_corr, **kw)
           if np.isfinite(r.cka) and pd.notna(r.eta2_p_module) and pd.notna(r.eta2_p_corr) else (None, None)
           for r in pt.assign(cka=pt[cka_col]).itertuples()]
    return [b for b, _ in res], [s for _, s in res]


def _add_cka_debiased(pt: pd.DataFrame, cka: pd.DataFrame | None) -> pd.DataFrame:
    """Add cka_debiased_median (ft seeds vs pre, median), rand_cka_debiased_same_rows (random encoders on the SAME
    rows as the view; task_clean: same regime, all seeds' clean rows) and n_cka_rows from the F1 table."""
    for c in ("cka_debiased_median", "rand_cka_debiased_same_rows", "n_cka_rows"):
        pt[c] = np.nan
    if cka is None or cka.empty:
        return pt
    ft = cka[cka["encoder"].str.startswith("seed")]
    g = ft.groupby(["task", "regime", "ckpt", "rows"]).agg(
        cka_debiased_median=("cka_debiased", "median"), n_cka_rows=("n", "mean")).reset_index().rename(columns={"rows": "probe_set"})
    rd = cka[cka["encoder"].str.startswith("rand")]
    r_fixed = rd[rd["rows"] != "task_clean"].groupby(["task", "rows"])["cka_debiased"].mean().rename("r_fixed").reset_index()
    r_clean = rd[rd["rows"] == "task_clean"].groupby(["task", "regime", "rows"])["cka_debiased"].mean().rename("r_clean").reset_index()
    out = pt.drop(columns=["cka_debiased_median", "rand_cka_debiased_same_rows", "n_cka_rows"]).merge(
        g, on=["task", "regime", "ckpt", "probe_set"], how="left")
    out = out.merge(r_fixed.rename(columns={"rows": "probe_set"}), on=["task", "probe_set"], how="left")
    out = out.merge(r_clean.rename(columns={"rows": "probe_set"}), on=["task", "regime", "probe_set"], how="left")
    out["rand_cka_debiased_same_rows"] = out["r_clean"].where(out["probe_set"] == "task_clean", out["r_fixed"])
    return out.drop(columns=["r_fixed", "r_clean"])


def per_task_summary(long, families, eta2, posctrl, scores, n_train: dict, es_epochs: dict, cka_table=None) -> pd.DataFrame:
    """One row per task x regime x ckpt x probe_set (every view with a fine-tuned aggregate dR2).
    es_epochs: {(task, regime): mean best-val epoch over ft seeds}; n_train: {task: fit-set size}."""
    wide = _ft_descriptor_wide(long)
    wide["eligible"] = wide["dropped"].notna()
    el = wide[wide["eligible"]]
    g = el.groupby(KEYS)
    pt = pd.DataFrame({
        "median_dr2": g["delta_r2_mean"].median(),
        "mean_dr2": g["delta_r2_mean"].mean(),
        "n_eligible": g.size(),
        "n_dropped": g["dropped"].apply(lambda s: int((s == 1).sum())),
    }).reset_index()
    pt["drop_frac"] = pt["n_dropped"] / pt["n_eligible"]

    # CKA vs pre, per seed encoder
    cka = long[long["encoder"].str.startswith("seed") & (long["measure"] == "cka")]
    c = cka.groupby(KEYS)["value"].agg(cka_mean="mean", cka_sd=lambda s: s.std(ddof=1), cka_median="median", cka_n_seeds="size").reset_index()
    pt = pt.merge(c, on=KEYS, how="left")
    sp = long[(long["encoder"] == "ft") & (long["measure"] == "cka_seedpair")].groupby(KEYS)["value"].mean().rename("cka_seedpair").reset_index()
    pt = pt.merge(sp, on=KEYS, how="left")
    nr = long[long["encoder"].str.startswith("seed") & (long["measure"] == "n_rows")].groupby(KEYS)["value"].mean().rename("n_rows").reset_index()
    pt = pt.merge(nr, on=KEYS, how="left")

    # pre R2 (median over eligible descriptors; mean over pre / pre@seedK encoders per descriptor)
    pre = long[long["encoder"].str.startswith("pre") & (long["measure"] == "r2") & (long["descriptor"] != "")]
    pre = pre.groupby(KEYS + ["descriptor"])["value"].mean().rename("pre_r2").reset_index()
    pre = el[KEYS + ["descriptor"]].merge(pre, on=KEYS + ["descriptor"], how="left")
    pt = pt.merge(pre.groupby(KEYS)["pre_r2"].median().rename("pre_r2_median").reset_index(), on=KEYS, how="left")

    # median relative weight change (over matrices and seeds); regime/ckpt-level
    wc = long[(long["measure"] == "wchange")].groupby(["task", "regime", "ckpt"])["value"].median().rename("median_wchange").reset_index()
    pt = pt.merge(wc, on=["task", "regime", "ckpt"], how="left")

    # families
    if families is not None and len(families):
        f = families.groupby(KEYS + ["family_def"]).agg(nf=("family", "size"), nd=("dropped", lambda s: int(s.astype(bool).sum()))).reset_index()
        for fd in f["family_def"].unique():
            label = "corr" if fd in CORR_DEFS else fd
            sub = f[f["family_def"] == fd].rename(columns={"nf": f"n_fam_{label}", "nd": f"n_dropped_fam_{label}"}).drop(columns="family_def")
            pt = pt.merge(sub, on=KEYS, how="left")
    # eta2
    if eta2 is not None and len(eta2):
        for fd in eta2["family_def"].unique():
            label = "corr" if fd in CORR_DEFS else fd
            sub = eta2[eta2["family_def"] == fd][KEYS + ["eta2", "null_q95", "p"]].rename(
                columns={"eta2": f"eta2_{label}", "null_q95": f"eta2_q95_{label}", "p": f"eta2_p_{label}"})
            sub[f"eta2_ratio_{label}"] = sub[f"eta2_{label}"] / sub[f"eta2_q95_{label}"]
            pt = pt.merge(sub, on=KEYS, how="left")
    for col in ("eta2_p_module", "eta2_p_corr", "eta2_ratio_module", "eta2_ratio_corr"):
        if col not in pt:
            pt[col] = np.nan
    # positive control
    if posctrl is not None and len(posctrl):
        pc = posctrl[posctrl["eligible"].astype(bool)]
        pg = pc.groupby(KEYS)
        pcs = pd.DataFrame({
            "posctrl_n_eligible": pg.size(),
            "posctrl_n_gained": pg.apply(lambda d: int((d["delta_mean"] > 2 * d["se"]).sum())),
            "posctrl_mean_dr2": pg["delta_mean"].mean(),
        }).reset_index()
        pt = pt.merge(pcs, on=KEYS, how="left")
    # test-score gap
    pt = pt.merge(score_gaps(scores), on=["task", "regime", "ckpt"], how="left")
    # es epoch, low confidence
    pt["es_epoch_mean"] = [es_epochs.get((t, r), np.nan) if c == "es" else np.nan for t, r, c in zip(pt["task"], pt["regime"], pt["ckpt"])]
    pt["n_train"] = pt["task"].map(n_train)
    pt["low_confidence"] = (pt["n_train"] < LOW_CONFIDENCE_N) | pt["n_train"].isna()  # no meta -> low confidence
    # random-encoder anchor and drift fraction
    pt["a_task"] = np.where(pt["probe_set"].str.startswith("pubchem"), "", pt["task"])
    pt["a_probe_set"] = pt["probe_set"].map(anchor_probe_set)
    pt = pt.merge(_anchor_table(long), on=["a_task", "a_probe_set"], how="left").drop(columns=["a_task", "a_probe_set"])
    for col in ("rand_median_dr2", "rand_cka", "rand_cka_seedpair"):
        if col not in pt:
            pt[col] = np.nan
    pt = pt.merge(_common_medians(el, long), on=KEYS, how="left")
    for col in ("drift_median_ft", "drift_median_rand"):
        if col not in pt:
            pt[col] = np.nan
    # both medians over descriptors eligible in the ft view AND the rand anchor view
    pt["drift_fraction"] = [drift_fraction(a, b) for a, b in zip(pt["drift_median_ft"], pt["drift_median_rand"])]
    # branch (needs eta2, i.e. pubchem / task_all / task_clean views)
    for col in ("gap", "gap_sd"):
        if col not in pt:
            pt[col] = np.nan
    pt["branch"], pt["gap_status"] = _branches(pt, "cka_median")
    # sensitivity analyses (the primary `branch` above is the pre-registered rule)
    pt = _add_cka_debiased(pt, cka_table)
    pt["branch_cka_debiased"] = _branches(pt, "cka_debiased_median")[0]
    pt["branch_eta2_strict"] = _branches(pt, "cka_median", min_ratio=ETA2_STRICT_RATIO)[0]
    pt["branch_no_eta2"] = _branches(pt, "cka_median", use_eta2=False)[0]
    pt["primary_view"] = pt["probe_set"].isin(PRIMARY_VIEWS) & (pt["regime"] == PRIMARY_CELL[0]) & (pt["ckpt"] == PRIMARY_CELL[1])
    return pt.sort_values(KEYS).reset_index(drop=True)


def _branch_pivot(p: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    b = p.pivot(index="task", columns="probe_set", values=col)
    for v in PRIMARY_VIEWS:
        if v not in b:
            b[v] = None
    b = b[list(PRIMARY_VIEWS)]
    b.columns = [f"{prefix}_{v}" for v in PRIMARY_VIEWS]
    b[f"{prefix}_agreed" if prefix != "branch" else "agreed"] = [agree(x, y) for x, y in zip(b.iloc[:, 0], b.iloc[:, 1])]
    b[b.columns[:2]] = b[b.columns[:2]].apply(lambda c: c.where(c.notna(), "missing"))
    return b


def branch_table(pt: pd.DataFrame) -> pd.DataFrame:
    """Per task (every task with a primary-view row): branch in each primary view at full/es, agreement
    (agreed branch, `split`, or `missing` if either view lacks a branch), plus the same for each sensitivity
    variant present in pt (columns <variant>_<view>, <variant>_agreed)."""
    p = pt[pt["primary_view"]]
    b = _branch_pivot(p, "branch", "branch")
    for var in VARIANTS:
        if var in p:
            b = b.join(_branch_pivot(p, var, var))
    b["gap_status"] = p.groupby("task")["gap_status"].agg(lambda s: "unknown_gap" if (s == "unknown_gap").any() else "ok")
    b["low_confidence"] = p.groupby("task")["low_confidence"].first()
    return b.reset_index()


def branch_counts(bt: pd.DataFrame) -> pd.DataFrame:
    """Tasks per branch for the primary rule (variant 'primary') and every sensitivity variant present in bt."""
    rows = []
    for var, prefix in [("primary", "branch")] + [(v.replace("branch_", ""), v) for v in VARIANTS if f"{v}_agreed" in bt]:
        agreed = "agreed" if prefix == "branch" else f"{prefix}_agreed"
        for col in [f"{prefix}_{v}" for v in PRIMARY_VIEWS] + [agreed]:
            vc = bt[col].value_counts()
            for br in list(BRANCHES) + ["split", "missing"]:
                if col != agreed and br == "split":
                    continue
                rows.append(dict(variant=var, view=col.replace(f"{prefix}_", "") if col != agreed else "agreed", branch=br,
                                 n_tasks=int(vc.get(br, 0))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cross-task counts
def count_dropped(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """df: task, `by` cols, dropped (1/0/True/False; NaN = ineligible, excluded). Counts tasks per group."""
    d = df[df["dropped"].notna()].copy()
    d["dropped"] = d["dropped"].astype(float) > 0
    return d.groupby(by).agg(n_tasks_dropped=("dropped", "sum"), n_tasks_eligible=("task", "nunique")).reset_index().astype(
        {"n_tasks_dropped": int})


def cross_task_counts(long: pd.DataFrame, families: pd.DataFrame) -> pd.DataFrame:
    """Per descriptor (or PC) and per family: number of tasks in which it is dropped, for every
    regime x ckpt x probe_set view. Columns: kind, regime, ckpt, probe_set, name, n_tasks_dropped, n_tasks_eligible."""
    cols = ["regime", "ckpt", "probe_set", "name"]
    d = long[(long["encoder"] == "ft") & (long["measure"] == "dropped") & (long["descriptor"] != "")].rename(columns={"descriptor": "name", "value": "dropped"})
    parts = [count_dropped(d, cols).assign(kind="descriptor")]
    if families is not None and len(families):
        for fd in families["family_def"].unique():
            f = families[families["family_def"] == fd].rename(columns={"family": "name"})
            label = "corr" if fd in CORR_DEFS else fd
            parts.append(count_dropped(f, cols).assign(kind=f"family_{label}"))
    out = pd.concat(parts, ignore_index=True)
    return out[["kind"] + cols + ["n_tasks_dropped", "n_tasks_eligible"]]


def cross_task_primary(ct: pd.DataFrame) -> pd.DataFrame:
    """Wide table at full/es: columns n_tasks_dropped_<view> for the two co-primary views, plus the min over views."""
    p = ct[(ct["regime"] == PRIMARY_CELL[0]) & (ct["ckpt"] == PRIMARY_CELL[1]) & ct["probe_set"].isin(PRIMARY_VIEWS)]
    w = p.pivot_table(index=["kind", "name"], columns="probe_set", values="n_tasks_dropped", aggfunc="first")
    w.columns = [f"n_tasks_dropped_{c}" for c in w.columns]
    cols = [f"n_tasks_dropped_{v}" for v in PRIMARY_VIEWS if f"n_tasks_dropped_{v}" in w]
    w["n_tasks_dropped_min_view"] = w[cols].min(axis=1)
    return w.reset_index().sort_values(["kind", "n_tasks_dropped_min_view"], ascending=[True, False])


# ---------------------------------------------------------------- PC spectrum, posctrl, comparisons
def pc_spectrum(long: pd.DataFrame, views=PRIMARY_VIEWS, cells=(("full", "es"), ("n128", "es"))) -> pd.DataFrame:
    """Mean over tasks of the seed-mean dR2 per PC index (eligible PCs only), for the *_pc versions of the views.
    n_tasks = tasks in which the PC is eligible; frac_eligible = n_tasks / tasks with the view (the figure plots task views
    only where frac_eligible >= PC_PLOT_MIN_FRAC, because eligibility varies by task and the mean would change composition)."""
    rows = []
    for v in views:
        for reg, ck in cells:
            d = long[(long["encoder"] == "ft") & (long["probe_set"] == f"{v}_pc") & (long["regime"] == reg) & (long["ckpt"] == ck) & (long["descriptor"] != "")]
            w = d.set_index(["task", "descriptor", "measure"])["value"].unstack("measure")
            if w.empty:
                continue
            w = w[w["dropped"].notna()].reset_index()
            w["pc"] = w["descriptor"].str.replace("PC", "").astype(int)
            g = w.groupby("pc")["delta_r2_mean"]
            n_total = d["task"].nunique()
            rows.append(pd.DataFrame(dict(view=v, regime=reg, ckpt=ck, pc=g.mean().index, mean_dr2=g.mean().values,
                                          sem=g.sem().values, n_tasks=g.size().values, n_tasks_total=n_total,
                                          frac_eligible=g.size().values / n_total)))
    cols = ["view", "regime", "ckpt", "pc", "mean_dr2", "sem", "n_tasks", "n_tasks_total", "frac_eligible"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def pc_summaries(long: pd.DataFrame) -> pd.DataFrame:
    """Per *_pc view: evr-weighted dR2 over eligible PCs (recomputed) next to the stored all-PC delta_r2_evrw."""
    evr = long[(long["encoder"] == "pca") & (long["measure"] == "evr")].drop_duplicates("descriptor").set_index("descriptor")["value"]
    d = long[(long["encoder"] == "ft") & (long["probe_set"].str.endswith("_pc")) & (long["descriptor"] != "")]
    rows = []
    for k, g in d.groupby(KEYS):
        w = g.set_index(["descriptor", "measure"])["value"].unstack("measure")
        rows.append(dict(zip(KEYS, k), n_pcs=len(w), n_eligible_pcs=int(w["dropped"].notna().sum()),
                         evrw_eligible=evr_weighted(w["delta_r2_mean"], w["dropped"], evr)))
    out = pd.DataFrame(rows)
    stored = long[(long["encoder"] == "ft") & (long["measure"] == "delta_r2_evrw")][KEYS + ["value"]].rename(columns={"value": "evrw_all_pcs_stored"})
    return out.merge(stored, on=KEYS, how="left") if len(out) else out


def posctrl_table(posctrl: pd.DataFrame) -> pd.DataFrame:
    pc = posctrl[posctrl["eligible"].astype(bool)]
    g = pc.groupby(KEYS)
    out = pd.DataFrame({
        "n_eligible_of_top20": g.size(),
        "n_gained_2se": g.apply(lambda d: int((d["delta_mean"] > 2 * d["se"]).sum())),
        "n_dropped": g["dropped"].apply(lambda s: int((s == 1).sum())),
        "mean_dr2": g["delta_mean"].mean(),
    }).reset_index()
    return out

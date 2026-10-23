import numpy as np
import pandas as pd
import pytest

from drift import summarize as S

KEYS = ["task", "regime", "ckpt", "probe_set"]


def _rows(task, regime, ckpt, ps, enc, measure, vals, descs=None):
    descs = descs if descs is not None else [""] * len(vals)
    return [dict(task=task, regime=regime, ckpt=ckpt, probe_set=ps, encoder=enc, measure=measure, descriptor=d, value=v)
            for d, v in zip(descs, vals)]


def synthetic(drop_planted):
    """3 tasks x full/es x {task_clean, pubchem}; descriptors d0..d3 (d3 ineligible).
    drop_planted[task] = set of descriptors dropped by ft."""
    descs = ["d0", "d1", "d2", "d3"]
    rows, fam, eta, pos, scores = [], [], [], [], []
    for t in ("A", "B", "C"):
        for ps in ("task_clean", "pubchem"):
            dm = [-0.2 if d in drop_planted[t] else -0.01 for d in descs[:3]] + [np.nan]
            dr = [1.0 if d in drop_planted[t] else 0.0 for d in descs[:3]] + [np.nan]
            rows += _rows(t, "full", "es", ps, "ft", "delta_r2_mean", dm, descs)
            rows += _rows(t, "full", "es", ps, "ft", "dropped", dr, descs)
            rows += _rows(t, "full", "es", ps, "ft", "se", [0.01] * 3 + [np.nan], descs)
            for s, c in zip(("seed42", "seed117", "seed709"), (0.9, 0.92, 0.94)):
                rows += _rows(t, "full", "es", ps, s, "cka", [c])
                rows += _rows(t, "full", "es", ps, s, "n_rows", [100.0])
            fam.append(dict(task=t, regime="full", ckpt="es", probe_set=ps, family_def="module", family="M1",
                            mean_delta=-0.1, n=3, frac_dropped=0.5, dropped=bool(drop_planted[t])))
            eta.append(dict(task=t, regime="full", ckpt="es", probe_set=ps, family_def="module", eta2=0.2, null_q95=0.05, p=0.001))
            eta.append(dict(task=t, regime="full", ckpt="es", probe_set=ps, family_def="cluster", eta2=0.9, null_q95=0.4, p=0.001))
            pos.append(dict(task=t, regime="full", ckpt="es", probe_set=ps, descriptor="d0", delta_mean=0.1, se=0.01, eligible=True, dropped=0.0))
        rows += _rows(t, "full", "es", "", "seed42", "wchange", [0.1, 0.3], ["W_i.weight", "W_h.weight"])
        rows += _rows(t, "", "", "task_all", "rand", "delta_r2_mean", [-0.4, -0.4, -0.4, np.nan], descs)
        rows += _rows(t, "", "", "task_all", "rand", "dropped", [1.0, 1.0, 1.0, np.nan], descs)
    rows += _rows("", "", "", "pubchem", "rand", "delta_r2_mean", [-0.4, -0.4, -0.4, np.nan], descs)
    rows += _rows("", "", "", "pubchem", "rand", "dropped", [1.0, 1.0, 1.0, np.nan], descs)
    for t in ("A", "B", "C"):
        for mode, v in (("ft", 0.9), ("frozen", 0.8)):
            for seed, j in ((42, 0), (117, 0.01), (709, -0.01)):
                scores.append(dict(task=t, regime="full", mode=mode, seed=seed, ckpt="es", metric="roc_auc", score=v + j, direction="max"))
    return (pd.DataFrame(rows), pd.DataFrame(fam), pd.DataFrame(eta), pd.DataFrame(pos), pd.DataFrame(scores))


def test_per_task_rows_and_cross_task_counts():
    planted = {"A": {"d0", "d1"}, "B": {"d0"}, "C": set()}
    long, fam, eta, pos, scores = synthetic(planted)
    pt = S.per_task_summary(long, fam, eta, pos, scores, n_train={"A": 500, "B": 5000, "C": 5000}, es_epochs={})
    assert len(pt) == 3 * 1 * 1 * 2
    assert not pt.duplicated(KEYS).any()
    a = pt[(pt.task == "A") & (pt.probe_set == "task_clean")].iloc[0]
    assert a.n_dropped == 2 and a.n_eligible == 3
    assert a.low_confidence and not pt[pt.task == "B"].low_confidence.any()
    assert a.cka_median == pytest.approx(0.92)
    assert a.median_wchange == pytest.approx(0.2)
    assert a.posctrl_mean_dr2 == pytest.approx(0.1)
    assert a.n_dropped_fam_module == 1
    ct = S.cross_task_counts(long, fam)
    d = ct[(ct.kind == "descriptor") & (ct.probe_set == "pubchem") & (ct.regime == "full") & (ct.ckpt == "es")].set_index("name")
    assert d.loc["d0", "n_tasks_dropped"] == 2 and d.loc["d1", "n_tasks_dropped"] == 1 and d.loc["d2", "n_tasks_dropped"] == 0
    assert d.loc["d0", "n_tasks_eligible"] == 3 and "d3" not in d.index
    f = ct[(ct.kind == "family_module") & (ct.probe_set == "pubchem")].set_index("name")
    assert f.loc["M1", "n_tasks_dropped"] == 2


def test_drift_fraction_anchored_on_rand():
    long, fam, eta, pos, scores = synthetic({"A": {"d0"}, "B": set(), "C": set()})
    pt = S.per_task_summary(long, fam, eta, pos, scores, n_train={}, es_epochs={})
    r = pt[(pt.task == "A") & (pt.probe_set == "pubchem")].iloc[0]
    assert r.median_dr2 == pytest.approx(-0.01)
    assert r.rand_median_dr2 == pytest.approx(-0.4)
    assert r.drift_fraction == pytest.approx(0.025)


@pytest.mark.parametrize("args,expected", [
    (dict(median_cka=0.97, drop_frac=0.04, p_module=0.5, p_corr=0.5, gap=-1, gap_sd=0.1), ("little_drift", "ok")),
    (dict(median_cka=0.95, drop_frac=0.05, p_module=0.5, p_corr=0.5, gap=-1, gap_sd=0.1), ("little_drift", "ok")),
    (dict(median_cka=0.97, drop_frac=0.2, p_module=0.001, p_corr=0.001, gap=1, gap_sd=0.1), ("descriptor_specific", "ok")),
    (dict(median_cka=0.5, drop_frac=0.01, p_module=0.001, p_corr=0.001, gap=1, gap_sd=0.1), ("descriptor_specific", "ok")),
    (dict(median_cka=0.5, drop_frac=0.5, p_module=0.001, p_corr=0.2, gap=1, gap_sd=0.1), ("uniform", "ok")),
    (dict(median_cka=0.5, drop_frac=0.5, p_module=0.001, p_corr=0.001, gap=0.05, gap_sd=0.1), ("uniform", "ok")),
    (dict(median_cka=0.5, drop_frac=0.5, p_module=0.001, p_corr=0.001, gap=np.nan, gap_sd=np.nan), ("descriptor_specific", "unknown_gap")),
    (dict(median_cka=0.5, drop_frac=0.5, p_module=0.9, p_corr=0.001, gap=np.nan, gap_sd=np.nan), ("uniform", "unknown_gap")),
])
def test_classify_branch(args, expected):
    assert S.classify_branch(**args) == expected


def test_score_gaps_sign_normalised():
    sc = pd.DataFrame([
        dict(task="T", regime="full", mode=m, seed=s, ckpt="es", metric="mae", score=v, direction="min")
        for m, vs in (("ft", (0.4, 0.5, 0.6)), ("frozen", (0.6, 0.7, 0.8))) for s, v in zip((1, 2, 3), vs)])
    g = S.score_gaps(sc).iloc[0]
    assert g.gap == pytest.approx(0.2)  # lower MAE for ft -> positive
    assert g.gap_sd == pytest.approx(0.1)


def test_evr_weighted_uses_eligible_only():
    delta = pd.Series({"PC1": -0.1, "PC2": -0.3, "PC3": 0.5})
    dropped = pd.Series({"PC1": 0.0, "PC2": 1.0, "PC3": np.nan})
    evr = pd.Series({"PC1": 0.3, "PC2": 0.1, "PC3": 0.6})
    assert S.evr_weighted(delta, dropped, evr) == pytest.approx((-0.03 - 0.03) / 0.4)
    assert np.isnan(S.evr_weighted(delta, dropped * np.nan, evr))


def test_agreement():
    assert S.agree("uniform", "uniform") == "uniform"
    assert S.agree("uniform", "little_drift") == "split"


def test_score_gaps_match_seeds_and_drop_nan():
    rows = [dict(task="T", regime="full", mode="ft", seed=s, ckpt="es", metric="r", score=v, direction="max")
            for s, v in ((1, 0.9), (2, 0.8), (3, 0.1))]
    rows += [dict(task="T", regime="full", mode="frozen", seed=s, ckpt="es", metric="r", score=v, direction="max")
             for s, v in ((1, 0.7), (2, 0.5))]
    g = S.score_gaps(pd.DataFrame(rows)).iloc[0]
    assert g.gap_n_seeds == 2
    assert g.gap == pytest.approx(0.25)  # seed 3 (unmatched) ignored
    assert g.gap_sd == pytest.approx(np.sqrt((pd.Series([0.9, 0.8]).var() + pd.Series([0.7, 0.5]).var()) / 2))
    rows[0]["score"] = np.nan
    assert S.score_gaps(pd.DataFrame(rows)).iloc[0].gap_n_seeds == 1


def test_drift_fraction_uses_common_eligible_descriptors():
    long, fam, eta, pos, scores = synthetic({"A": set(), "B": set(), "C": set()})
    # make ft task_clean ineligible on d0 and give d0 an extreme value; rand ineligible on d2
    m = (long.task == "A") & (long.probe_set == "task_clean") & (long.encoder == "ft") & (long.descriptor == "d0")
    long.loc[m & (long.measure == "dropped"), "value"] = np.nan
    long.loc[m & (long.measure == "delta_r2_mean"), "value"] = -9.0
    r = (long.task == "A") & (long.probe_set == "task_all") & (long.encoder == "rand") & (long.descriptor == "d2")
    long.loc[r & (long.measure == "dropped"), "value"] = np.nan
    long.loc[r & (long.measure == "delta_r2_mean"), "value"] = -9.0
    pt = S.per_task_summary(long, fam, eta, pos, scores, n_train={}, es_epochs={})
    row = pt[(pt.task == "A") & (pt.probe_set == "task_clean")].iloc[0]
    assert row.drift_n_common == 1  # only d1 eligible in both
    assert row.drift_median_ft == pytest.approx(-0.01) and row.drift_median_rand == pytest.approx(-0.4)
    assert row.drift_fraction == pytest.approx(0.025)


def test_eta2_ratio_and_variant_branches():
    planted = {"A": {"d0", "d1"}, "B": {"d0"}, "C": set()}
    long, fam, eta, pos, scores = synthetic(planted)
    eta.loc[(eta.task == "B") & (eta.family_def == "module"), "eta2"] = 0.06   # ratio 1.2 < 1.5
    pt = S.per_task_summary(long, fam, eta, pos, scores, n_train={"A": 500, "B": 5000, "C": 5000}, es_epochs={})
    a = pt[(pt.task == "A") & (pt.probe_set == "pubchem")].iloc[0]
    assert a.eta2_ratio_module == pytest.approx(4.0) and a.eta2_ratio_corr == pytest.approx(2.25)
    # synthetic gap (0.1) > gap_sd (0.01); cka 0.92 < 0.95 -> not little_drift
    assert (pt["branch"] == "descriptor_specific").all()
    assert (pt[pt.task == "B"].branch_eta2_strict == "uniform").all()
    assert (pt[pt.task != "B"].branch_eta2_strict == "descriptor_specific").all()
    assert (pt.branch_no_eta2 == "descriptor_specific").all()
    assert pt.branch_cka_debiased.isna().all()  # no cka table given


def test_classify_branch_variants():
    base = dict(median_cka=0.5, drop_frac=0.5, p_module=0.001, p_corr=0.001, gap=1, gap_sd=0.1)
    assert S.classify_branch(**base, ratio_module=1.2, ratio_corr=3, min_ratio=1.5) == ("uniform", "ok")
    assert S.classify_branch(**base, ratio_module=2, ratio_corr=3, min_ratio=1.5) == ("descriptor_specific", "ok")
    assert S.classify_branch(**{**base, "p_corr": 0.5}, use_eta2=False) == ("descriptor_specific", "ok")


def test_cka_debiased_columns_and_branch():
    long, fam, eta, pos, scores = synthetic({"A": {"d0"}, "B": {"d0"}, "C": {"d0"}})
    rows = []
    for t in ("A", "B", "C"):
        for rw, n in (("pubchem", 2000), ("task_all", 300), ("task_clean", 100)):
            for sd, c in ((42, 0.9), (117, 0.97), (709, 0.99)):
                rows.append(dict(task=t, regime="full", ckpt="es", seed=sd, encoder=f"seed{sd}", rows=rw, n=n,
                                 cka_linear=0.99, cka_debiased=c))
        rows += [dict(task=t, regime="", ckpt="", seed=-1, encoder=f"rand{r}", rows=rw, n=1, cka_linear=0.5, cka_debiased=0.1)
                 for r in range(3) for rw in ("pubchem", "task_all")]
        rows += [dict(task=t, regime="full", ckpt="", seed=sd, encoder=f"rand{r}", rows="task_clean", n=1, cka_linear=0.5, cka_debiased=0.2)
                 for r in range(3) for sd in (42, 117, 709)]
    pt = S.per_task_summary(long, fam, eta, pos, scores, n_train={}, es_epochs={}, cka_table=pd.DataFrame(rows))
    r = pt[(pt.task == "A") & (pt.probe_set == "task_clean")].iloc[0]
    assert r.cka_debiased_median == pytest.approx(0.97) and r.n_cka_rows == 100
    assert r.rand_cka_debiased_same_rows == pytest.approx(0.2)
    assert pt[(pt.task == "A") & (pt.probe_set == "pubchem")].iloc[0].rand_cka_debiased_same_rows == pytest.approx(0.1)
    assert (pt.branch_cka_debiased == "descriptor_specific").all()  # median 0.97 but 1 of 3 dropped -> not little_drift


def test_branch_table_missing_and_variant_counts():
    planted = {"A": {"d0"}, "B": {"d0"}, "C": {"d0"}}
    long, fam, eta, pos, scores = synthetic(planted)
    eta = eta[~((eta.task == "C") & (eta.probe_set == "pubchem"))]   # C: no branch in pubchem view
    pt = S.per_task_summary(long, fam, eta, pos, scores, n_train={}, es_epochs={})
    bt = S.branch_table(pt).set_index("task")
    assert set(bt.index) == {"A", "B", "C"}
    assert bt.loc["C", "agreed"] == "missing" and bt.loc["C", "branch_pubchem"] == "missing"
    assert bt.loc["A", "agreed"] == "descriptor_specific" and bt.loc["A", "low_confidence"]
    bc = S.branch_counts(bt.reset_index())
    assert set(bc.variant) == {"primary", "eta2_strict", "no_eta2", "cka_debiased"}
    x = bc[(bc.variant == "primary") & (bc.view == "agreed")].set_index("branch").n_tasks
    assert x["missing"] == 1 and x["descriptor_specific"] == 2 and x["split"] == 0


def test_pc_spectrum_reports_eligibility_fraction():
    rows = []
    for t, drops in (("A", [0.0, 0.0]), ("B", [0.0, np.nan])):
        for pc, (dm, dr) in enumerate(zip((-0.1, -0.2), drops), start=1):
            rows += _rows(t, "full", "es", "pubchem_pc", "ft", "delta_r2_mean", [dm], [f"PC{pc}"])
            rows += _rows(t, "full", "es", "pubchem_pc", "ft", "dropped", [dr], [f"PC{pc}"])
    sp = S.pc_spectrum(pd.DataFrame(rows), views=("pubchem",), cells=(("full", "es"),)).set_index("pc")
    assert sp.loc[1, "n_tasks"] == 2 and sp.loc[2, "n_tasks"] == 1
    assert sp.loc[2, "frac_eligible"] == pytest.approx(0.5) and sp.loc[2, "n_tasks_total"] == 2

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

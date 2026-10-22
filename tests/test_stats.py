import numpy as np
import pandas as pd

from drift.stats import (
    correlation_clusters,
    drop_calls,
    eta2,
    eta2_perm_test,
    family_delta,
    positive_control,
)

rng = np.random.default_rng(0)


def test_drop_calls():
    dps = np.array([[-.10, -.10, .01, -.02], [-.12, -.09, -.20, -.03], [-.11, -.11, -.20, -.02]])
    dm = dps.mean(0)
    se = np.array([.01, .05, .01, .001])
    assert drop_calls(dps, dm, se).tolist() == [True, False, False, False]


def test_drop_calls_nan_is_false():
    dps = np.array([[-.1, np.nan], [-.1, -.1], [-.1, -.1]])
    dm = np.array([-.1, np.nan])
    se = np.array([.01, .01])
    assert drop_calls(dps, dm, se).tolist() == [True, False]
    se2 = np.array([np.nan, .01])
    dm2 = np.array([-.1, -.1])
    assert drop_calls(dps[:, :1].repeat(2, 1), dm2, se2).tolist() == [False, True]


def test_eta2_known_value():
    v = np.array([1., 1., 3., 3.])
    assert eta2(v, np.array([0, 0, 1, 1])) == 1.0
    assert abs(eta2(np.array([1., 3., 1., 3.]), np.array([0, 0, 1, 1]))) < 1e-12


def test_eta2_detects_structure():
    lab = np.repeat(np.arange(10), 30)
    v_struct = lab * 0.1 + rng.normal(0, .01, 300)
    v_flat = rng.normal(0, .1, 300)
    obs, q95, p = eta2_perm_test(v_struct, lab)
    assert obs > q95 and p < 0.01
    assert eta2_perm_test(v_flat, lab)[2] > 0.01


def test_eta2_ignores_nan_and_is_seeded():
    lab = np.repeat(np.arange(5), 20)
    v = lab * 0.1 + rng.normal(0, .05, 100)
    v2 = np.r_[v, [np.nan, np.nan]]
    lab2 = np.r_[lab, [0, 1]]
    assert eta2_perm_test(v2, lab2, n_perm=200, seed=3) == eta2_perm_test(v, lab, n_perm=200, seed=3)


def test_clusters_group_correlated():
    base = rng.normal(size=(500, 1))
    Y = np.c_[base, 2 * base + 1e-3 * rng.normal(size=(500, 1)), rng.normal(size=(500, 1))]
    c = correlation_clusters(Y, ["a", "b", "c"])
    assert c["a"] == c["b"] != c["c"]


def test_clusters_handle_nan_and_constant():
    base = rng.normal(size=(300, 1))
    Y = np.c_[base, -base, np.ones((300, 1)), rng.normal(size=(300, 1))]
    Y[:10, 0] = np.nan
    c = correlation_clusters(Y, ["a", "b", "const", "d"])
    assert c["a"] == c["b"]
    assert len({c["a"], c["const"], c["d"]}) == 3


def test_family_delta():
    fam = pd.Series(["A", "A", "A", "B", "B"], index=list("abcde"))
    dm = np.array([-.1, -.2, np.nan, 0.0, .1])
    drop = np.array([True, True, False, False, False])
    out = family_delta(dm, drop, fam)
    assert out.loc["A", "n"] == 3
    assert np.isclose(out.loc["A", "mean_delta"], -.15)
    assert out.loc["A", "frac_dropped"] == 1.0 and out.loc["A", "dropped"]
    assert out.loc["B", "frac_dropped"] == 0.0 and not out.loc["B", "dropped"]


def test_positive_control_picks_label_correlated():
    y = rng.normal(size=200)
    D = pd.DataFrame({"good": y + .1 * rng.normal(size=200), "bad": rng.normal(size=200)})
    assert positive_control(D, y, ["good", "bad"], k=1) == ["good"]


def test_positive_control_nan_aware_and_min_rows():
    y = rng.normal(size=100)
    good = y + .1 * rng.normal(size=100)
    good[:20] = np.nan
    sparse = np.full(100, np.nan)
    sparse[:5] = y[:5]
    D = pd.DataFrame({"good": good, "sparse": sparse, "bad": rng.normal(size=100)})
    assert positive_control(D, y, ["sparse", "bad", "good"], k=3) == ["good", "bad"]

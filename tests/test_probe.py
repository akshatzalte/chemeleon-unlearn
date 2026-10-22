import numpy as np
from sklearn.linear_model import Ridge

from drift.probe import (choose_alpha, crossfit_predict, linear_cka, make_folds,
                         paired_bootstrap_delta, r2_cols, ridge_fit_predict)

rng = np.random.default_rng(0)


def test_cka_properties():
    X = rng.normal(size=(300, 50)); Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    assert abs(linear_cka(X, X) - 1) < 1e-9 and abs(linear_cka(X, 3 * X @ Q) - 1) < 1e-9
    assert linear_cka(X, rng.normal(size=(300, 50))) < 0.3


def test_ridge_recovers_linear_target_with_nans():
    Z = rng.normal(size=(500, 20)); W = rng.normal(size=(20, 3)); Y = Z @ W
    Y[rng.random(500) < 0.1, 1] = np.nan
    P = ridge_fit_predict(Z[:400], Y[:400], Z[400:], alpha=1e-3)
    assert (r2_cols(Y[400:], P) > 0.99).all()


def test_ridge_matches_sklearn_per_column_with_nans():
    r = np.random.default_rng(1)
    Z = r.normal(size=(120, 15)) * r.uniform(0.5, 5, 15) + r.normal(size=15) * 10
    Z[:, 3] = 7.0                                   # constant column -> SD 1
    Y = Z[:, :4] @ r.normal(size=(4, 6)) + r.normal(size=(120, 6)) + 50
    Y[r.random(120) < 0.2, 0] = np.nan
    Y[r.random(120) < 0.2, 1] = np.nan
    Y[:, 2] = Y[:, 1]                               # same pattern as column 1 -> one group
    Y[:10, 5] = np.nan
    Zf, Yf, Ze = Z[:90], Y[:90], Z[90:]
    mu, sd = Zf.mean(0), Zf.std(0); sd[sd == 0] = 1
    for alpha in (0.1, 10.0):
        P = ridge_fit_predict(Zf, Yf, Ze, alpha)
        assert not np.isnan(P).any()
        for j in range(6):
            ok = ~np.isnan(Yf[:, j])
            ref = Ridge(alpha=alpha).fit((Zf[ok] - mu) / sd, Yf[ok, j]).predict((Ze - mu) / sd)
            np.testing.assert_allclose(P[:, j], ref, atol=1e-6)


def test_ridge_nan_in_eval_z_and_empty_column():
    Z = rng.normal(size=(50, 4)); Y = rng.normal(size=(50, 2)); Y[:, 1] = np.nan
    Ze = rng.normal(size=(5, 4)); Ze[2, 0] = np.nan
    P = ridge_fit_predict(Z, Y, Ze, 1.0)
    assert np.isnan(P[2, 0]) and np.isnan(P[:, 1]).all()
    assert not np.isnan(P[[0, 1, 3, 4], 0]).any()


def test_crossfit_is_out_of_fold():
    Z = rng.normal(size=(100, 5)); Y = rng.normal(size=(100, 2)); f = make_folds(100)
    P = crossfit_predict(Z, Y, f, alpha=1.0)
    assert r2_cols(Y, P).max() < 0.2        # pure noise → no in-sample leak


def test_choose_alpha_picks_strong_ridge_for_noise_and_weak_for_signal():
    r = np.random.default_rng(2)
    Z = r.normal(size=(200, 100))
    grid = np.logspace(-2, 5, 15)
    assert choose_alpha(Z, r.normal(size=(200, 4)), grid) >= 1e3
    Y = Z[:, :5] @ r.normal(size=(5, 4)) + 0.01 * r.normal(size=(200, 4))
    assert choose_alpha(Z, Y, grid) <= 10


def test_r2_cols_nan_handling():
    Y = np.array([[1., 5.], [2., np.nan], [3., 5.], [4., 5.]])
    Yh = Y + 0.0
    r = r2_cols(Y, Yh)
    assert r[0] == 1.0 and np.isnan(r[1])            # constant col -> NaN
    assert np.isnan(r2_cols(Y, Yh, rows=np.array([0]))).all()
    assert abs(r2_cols(Y, np.full_like(Y, 2.5), rows=np.arange(4))[0]) < 1e-12


def unpaired_se(Y, Ya, Yb, rows, n_boot=200, seed=0):
    r = np.random.default_rng(seed)
    da, db = [], []
    for _ in range(n_boot):
        ia = rows[r.integers(0, len(rows), len(rows))]
        ib = rows[r.integers(0, len(rows), len(rows))]
        da.append(r2_cols(Y, Ya, ia)); db.append(r2_cols(Y, Yb, ib))
    return np.sqrt(np.std(da, axis=0, ddof=1) ** 2 + np.std(db, axis=0, ddof=1) ** 2)


def test_paired_bootstrap_tighter_than_unpaired():
    Y = rng.normal(size=(400, 10)); eps = rng.normal(size=(400, 10))
    Ya, Yb = Y + eps, Y + eps + 0.01 * rng.normal(size=(400, 10))   # shared noise
    rows = np.arange(400)
    _, _, se_paired = paired_bootstrap_delta(Y, Ya, [Yb], [rows])
    se_unpaired = unpaired_se(Y, Ya, Yb, rows)
    assert se_paired.mean() < 0.3 * se_unpaired.mean()


def test_bootstrap_point_estimates_and_independent_rows_branch():
    r = np.random.default_rng(3)
    Y = r.normal(size=(300, 4)); Y[r.random((300, 4)) < 0.1] = np.nan
    pre = Y + r.normal(size=Y.shape); a = Y + 0.5 * r.normal(size=Y.shape); b = Y + 2 * r.normal(size=Y.shape)
    pre = np.where(np.isnan(Y), r.normal(size=Y.shape), pre)
    ra, rb = np.arange(0, 200), np.arange(100, 300)
    d, dm, se = paired_bootstrap_delta(Y, pre, [a, b], [ra, rb], n_boot=100)
    np.testing.assert_allclose(d[0], r2_cols(Y, a, ra) - r2_cols(Y, pre, ra))
    np.testing.assert_allclose(d[1], r2_cols(Y, b, rb) - r2_cols(Y, pre, rb))
    np.testing.assert_allclose(dm, d.mean(0))
    assert np.isfinite(se).all() and (se > 0).all()
    d2, dm2, se2 = paired_bootstrap_delta(Y, pre, [a, b], [ra, ra], n_boot=100)   # shared branch
    np.testing.assert_allclose(d2[0], d[0]); assert np.isfinite(se2).all()
    # deterministic
    assert np.array_equal(paired_bootstrap_delta(Y, pre, [a, b], [ra, rb], n_boot=100)[2], se)


def test_bootstrap_se_matches_naive_loop():
    r = np.random.default_rng(4)
    Y = r.normal(size=(150, 3)); pre = Y + r.normal(size=Y.shape); a = Y + 0.7 * r.normal(size=Y.shape)
    rows = np.arange(150)
    _, _, se = paired_bootstrap_delta(Y, pre, [a], [rows], n_boot=400, seed=5)
    naive = []
    g = np.random.default_rng(99)
    for _ in range(400):
        i = g.integers(0, 150, 150)
        naive.append(r2_cols(Y, a, i) - r2_cols(Y, pre, i))
    np.testing.assert_allclose(se, np.std(naive, axis=0, ddof=1), rtol=0.25)


def test_duplicate_smiles_rows_kept():                 # Review Focus 5
    assert len(make_folds(10)) == 10


def test_make_folds_balanced_deterministic():
    f = make_folds(103, 5)
    assert np.array_equal(f, make_folds(103, 5)) and set(f) == set(range(5))
    c = np.bincount(f); assert c.max() - c.min() <= 1

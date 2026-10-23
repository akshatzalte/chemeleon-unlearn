"""Probe and CKA math (CPU, numpy/scipy, float64 internally).

Ridge semantics everywhere: exact ridge with an UNPENALIZED intercept, penalty
``alpha`` on the sum-of-squares scale (identical to
``sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=True)``), on Z columns
standardized with mean/SD over the fit rows. Each column's SD is floored at SD_FLOOR_FRAC (1%)
of the median SD over columns with SD > 0 (all-constant Z: SD = 1), and standardized values
(fit AND eval rows) are clipped to +-Z_CLIP, so near-constant/sparse embedding columns cannot
produce exploding held-out predictions. Applied identically to every encoder and caller.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


# --------------------------------------------------------------------------- CKA
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA of column-centered X (n,p) and Y (n,q): ||Y'X||_F^2 / (||X'X||_F ||Y'Y||_F)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den)


def _hsic_u(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012) for Gram matrices with zeroed diagonals."""
    n = K.shape[0]
    k1, l1 = K.sum(1), L.sum(1)
    t = np.sum(K * L) + k1.sum() * l1.sum() / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * (k1 @ l1)
    return float(t / (n * (n - 3)))


def debiased_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA (Kornblith et al. 2019 / Nguyen et al. 2021): CKA from the unbiased
    HSIC estimator on linear Gram matrices with zeroed diagonals (needs n > 3; float64).
    Unlike linear_cka it is ~0 for independent features even when n << dimension."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    K = X @ X.T
    L = Y @ Y.T
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    return float(_hsic_u(K, L) / np.sqrt(_hsic_u(K, K) * _hsic_u(L, L)))


# ------------------------------------------------------------------------- folds
def make_folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """Row-based fold ids in [0, k), balanced sizes (differ by <=1), deterministic."""
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.empty(n, dtype=np.int64)
    folds[perm] = np.arange(n) % k
    return folds


SD_FLOOR_FRAC = 0.01
Z_CLIP = 6.0


def _standardize(Z_fit: np.ndarray):
    mu = Z_fit.mean(0)
    sd = Z_fit.std(0)
    pos = sd[sd > 0]
    floor = SD_FLOOR_FRAC * np.median(pos) if len(pos) else 1.0
    return mu, np.maximum(sd, floor)


def _scale(Z, mu, sd):
    """Standardize with fit stats and clip to +-Z_CLIP."""
    return np.clip((Z - mu) / sd, -Z_CLIP, Z_CLIP)


# ------------------------------------------------------------------------- ridge
def ridge_fit_predict(Z_fit, Y_fit, Z_eval, alpha: float) -> np.ndarray:
    """Exact per-column ridge with NaN-masked targets; returns (n_eval, k) predictions.

    Each target column is fit on its own non-NaN fit rows (Z standardized with mean/SD
    over ALL fit rows). Columns are grouped by identical NaN pattern; per group the
    centered Gram matrix is obtained by downdating the all-rows Gram matrix with the
    excluded rows, then solved by Cholesky (float64). Predictions are NaN only for
    eval rows with NaN in Z (and for columns with no non-NaN fit row).
    """
    Z_fit = np.asarray(Z_fit, dtype=np.float64)
    Y_fit = np.asarray(Y_fit, dtype=np.float64)
    Z_eval = np.asarray(Z_eval, dtype=np.float64)
    if not np.isfinite(Z_fit).all():
        raise ValueError("Z_fit must be finite")
    n, d = Z_fit.shape
    k = Y_fit.shape[1]
    mu, sd = _standardize(Z_fit)
    Zs = _scale(Z_fit, mu, sd)
    Ze = _scale(Z_eval, mu, sd)
    nan = np.isnan(Y_fit)
    Y0 = np.where(nan, 0.0, Y_fit)
    G_all = Zs.T @ Zs                      # (d,d)
    ZtY = Zs.T @ Y0                        # (d,k): sums over non-NaN rows only
    P = np.full((Ze.shape[0], k), np.nan)

    # group columns by identical NaN pattern
    packed = np.packbits(nan, axis=0)      # (ceil(n/8), k)
    _, inv = np.unique(packed.T, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    for cols in np.split(order, bounds):
        excl = np.flatnonzero(nan[:, cols[0]])
        ng = n - len(excl)
        if ng == 0:
            continue                       # no data: stays NaN
        if len(excl):
            Ze_rows = Zs[excl]
            S = G_all - Ze_rows.T @ Ze_rows
            m = (Zs.sum(0) - Ze_rows.sum(0)) / ng
        else:
            S = G_all
            m = Zs.sum(0) / ng             # ~0 up to rounding
        Yg = Y0[:, cols]
        ybar = Yg.sum(0) / ng              # NaN rows were zeroed
        A = S - ng * np.outer(m, m) + alpha * np.eye(d)
        B = ZtY[:, cols] - ng * np.outer(m, ybar)
        try:
            W = sla.cho_solve(sla.cho_factor(A, lower=True, check_finite=False), B, check_finite=False)
        except np.linalg.LinAlgError:      # alpha == 0 and singular: least-squares fallback
            W = sla.lstsq(A, B)[0]
        P[:, cols] = Ze @ W + (ybar - m @ W)
    return P


def crossfit_predict(Z, Y, folds: np.ndarray, alpha: float) -> np.ndarray:
    """Out-of-fold predictions: each row predicted by a model fit on the other folds."""
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    folds = np.asarray(folds)
    P = np.full(Y.shape, np.nan)
    for f in np.unique(folds):
        te = folds == f
        P[te] = ridge_fit_predict(Z[~te], Y[~te], Z[te], alpha)
    return P


def _eig_basis(X: np.ndarray):
    """(V, lam) with X'X = V diag(lam) V' on the range of X (d,r), (r,)."""
    n, d = X.shape
    if n >= d:
        lam, V = np.linalg.eigh(X.T @ X)
        return V, np.clip(lam, 0.0, None)
    lam, U = np.linalg.eigh(X @ X.T)
    keep = lam > lam.max() * 1e-12
    lam, U = lam[keep], U[:, keep]
    return (X.T @ U) / np.sqrt(lam), lam


def choose_alpha(Z: np.ndarray, Y: np.ndarray, alphas=np.logspace(-2, 5, 15)) -> float:
    """Pick alpha by 5-fold CV (fixed seed 0) on standardized Z.

    Score = mean over target columns of out-of-fold R^2 (pooled over folds, SST around
    the overall mean). For speed ONLY target columns with no NaN on the given rows are
    used. If no column is NaN-free, the columns with the fewest NaNs are used instead:
    their NaNs are mean-imputed (fit-fold mean) for fitting, and scoring ignores NaN
    entries (selection only; never used for final predictions). ValueError if no column
    has data or all are constant. One eigendecomposition per fold serves every alpha.
    Ties resolve to the smaller alpha.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    nnan = np.isnan(Y).sum(0)
    Y = Y[:, nnan == nnan.min()]
    if Y.shape[1] == 0 or nnan.min() >= len(Y):
        raise ValueError("choose_alpha needs at least one target column with data")
    M = ~np.isnan(Y)                       # all True in the normal (NaN-free) case
    alphas = np.asarray(alphas, dtype=np.float64)
    folds = make_folds(len(Z), 5, 0)
    sse = np.zeros((len(alphas), Y.shape[1]))
    for f in range(5):
        te = folds == f
        mu, sd = _standardize(Z[~te])
        Xtr = _scale(Z[~te], mu, sd)
        Xte = _scale(Z[te], mu, sd)
        xm = Xtr.mean(0)
        Xtr = Xtr - xm
        Xte = Xte - xm
        ytr, yte = Y[~te], Y[te]
        ym = np.nanmean(ytr, axis=0)
        ym = np.where(np.isnan(ym), 0.0, ym)
        ytr = np.where(np.isnan(ytr), ym, ytr)   # fit-fold mean imputation (selection only)
        V, lam = _eig_basis(Xtr)
        T = V.T @ (Xtr.T @ (ytr - ym))     # (r,k)
        Pte = Xte @ V                      # (n_te,r)
        mte = M[te]
        for i, a in enumerate(alphas):
            pred = Pte @ (T / (lam + a)[:, None]) + ym
            sse[i] += (np.where(mte, yte - pred, 0.0) ** 2).sum(0)
    n = M.sum(0)
    y0 = np.where(M, Y, 0.0)
    sst = ((np.where(M, Y - y0.sum(0) / n, 0.0)) ** 2).sum(0)
    ok = sst > 1e-12 * (y0 ** 2).sum(0)
    if not ok.any():
        raise ValueError("all target columns are constant")
    score = (1.0 - sse[:, ok] / sst[ok]).mean(1)
    return float(alphas[int(np.argmax(score))])


# --------------------------------------------------------------------------- R^2
def r2_cols(Y, Yhat, rows=None) -> np.ndarray:
    """Per-column R^2 over rows where both Y and Yhat are non-NaN (SST around the eval mean).

    Columns with SST == 0 or fewer than 2 usable rows give NaN.
    """
    Y = np.asarray(Y, dtype=np.float64)
    Yhat = np.asarray(Yhat, dtype=np.float64)
    if rows is not None:
        Y, Yhat = Y[rows], Yhat[rows]
    m = np.isfinite(Y) & np.isfinite(Yhat)
    n = m.sum(0)
    y = np.where(m, Y, 0.0)
    sse = (np.where(m, Y - Yhat, 0.0) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sst = (np.where(m, Y - y.sum(0) / n, 0.0) ** 2).sum(0)
        r2 = 1.0 - sse / sst
    return np.where(_degenerate(n, sst, (y ** 2).sum(0)), np.nan, r2)


def _degenerate(n, sst, sumsq):
    """Shared rule: R^2 undefined with <2 rows or (numerically) zero variance."""
    return (n < 2) | ~(sst > 1e-12 * sumsq)


class _Moments:
    """Per-row ingredients so R^2 of many weighted resamples is one matrix product."""

    def __init__(self, Y, Yhats, rows):
        m = np.isfinite(Y[rows])
        for H in Yhats:
            m &= np.isfinite(H[rows])
        y = np.where(m, Y[rows], 0.0)
        self.m = m.astype(np.float64)
        self.y = y
        self.y2 = y * y
        self.e2 = [np.where(m, Y[rows] - H[rows], 0.0) ** 2 for H in Yhats]

    def r2(self, w, which):
        """w: (B, m) row weights -> (B, k) R^2 of Yhats[which]."""
        n = w @ self.m
        sy = w @ self.y
        with np.errstate(divide="ignore", invalid="ignore"):
            sst = w @ self.y2 - sy ** 2 / n
            r2 = 1.0 - (w @ self.e2[which]) / sst
        return np.where(_degenerate(n, sst, w @ self.y2), np.nan, r2)


def _boot_weights(rng, m, n_boot):
    idx = rng.integers(0, m, size=(n_boot, m))
    w = np.zeros((n_boot, m))
    np.add.at(w, (np.arange(n_boot)[:, None], idx), 1.0)
    return w


def paired_bootstrap_delta(Y, Yhat_pre, Yhat_list, rows_list, n_boot=200, seed=0):
    """Delta R^2 (seed minus pre) with paired-bootstrap SE.

    Returns (delta_per_seed (S,k), delta_mean (k,), se_mean (k,)). Seed s is evaluated on
    rows_list[s]; pre and seed s use the same rows and the same resamples. Point estimates
    use rows where Y, pre and the seed prediction are all non-NaN.
    - All rows_list identical (e.g. PubChem): one resample set shared by pre and all seeds;
      se_mean = SD over resamples of the seed-mean delta.
    - Otherwise: each seed resamples its own rows independently; se_s = SD over resamples,
      se_mean = sqrt(sum_s se_s^2) / S.
    NaN columns (degenerate SST) give NaN delta and SE; SDs are ddof=1 over finite resamples.
    """
    Y = np.asarray(Y, dtype=np.float64)
    pre = np.asarray(Yhat_pre, dtype=np.float64)
    S = len(Yhat_list)
    rows_list = [np.asarray(r) for r in rows_list]
    assert len(rows_list) == S
    shared = all(np.array_equal(rows_list[0], r) for r in rows_list[1:])

    def moments(s):
        return _Moments(Y, [pre, np.asarray(Yhat_list[s], dtype=np.float64)], rows_list[s])

    delta = np.empty((S, Y.shape[1]))
    boots = []                              # per seed: (B,k) bootstrap deltas
    if shared:
        rng = np.random.default_rng(seed)
        m = len(rows_list[0])
        w = _boot_weights(rng, m, n_boot)
        one = np.ones((1, m))
        for s in range(S):
            mo = moments(s)
            delta[s] = (mo.r2(one, 1) - mo.r2(one, 0))[0]
            boots.append(mo.r2(w, 1) - mo.r2(w, 0))
        with np.errstate(all="ignore"):
            se = _sd(np.mean(boots, axis=0))
    else:
        ses = []
        for s in range(S):
            rng = np.random.default_rng([seed, s])
            m = len(rows_list[s])
            mo = moments(s)
            one = np.ones((1, m))
            delta[s] = (mo.r2(one, 1) - mo.r2(one, 0))[0]
            ses.append(_sd(_paired_boot(mo, rng, m, n_boot)))
        se = np.sqrt(np.sum(np.square(ses), axis=0)) / S
    return delta, delta.mean(0), se


def _paired_boot(mo, rng, m, n_boot):
    w = _boot_weights(rng, m, n_boot)
    return mo.r2(w, 1) - mo.r2(w, 0)


def _sd(a):
    """SD over axis 0 (ddof=1) ignoring non-finite resamples; NaN if <2 finite."""
    ok = np.isfinite(a)
    n = ok.sum(0)
    a0 = np.where(ok, a, 0.0)
    mean = a0.sum(0) / np.maximum(n, 1)
    var = (np.where(ok, a - mean, 0.0) ** 2).sum(0) / np.maximum(n - 1, 1)
    return np.where(n >= 2, np.sqrt(var), np.nan)

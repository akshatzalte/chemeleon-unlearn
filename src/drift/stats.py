"""Statistics: drop calls, correlation families, eta^2 permutation test, positive control."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from drift.config import DROP_ABS, DROP_SE_MULT, CLUSTER_RHO, N_PERM, TOP_K_POSCTRL

MIN_POSCTRL_ROWS = 10


def drop_calls(delta_per_seed, delta_mean, se_mean, abs_thr=DROP_ABS, se_mult=DROP_SE_MULT):
    """Dropped iff delta<0 in every seed, mean < -se_mult*SE and mean < -abs_thr.
    False wherever any input for the descriptor is NaN."""
    dps = np.asarray(delta_per_seed, dtype=float)
    dm = np.asarray(delta_mean, dtype=float)
    se = np.asarray(se_mean, dtype=float)
    valid = ~(np.isnan(dps).any(axis=0) | np.isnan(dm) | np.isnan(se))
    with np.errstate(invalid="ignore"):
        hit = (dps < 0).all(axis=0) & (dm < -se_mult * se) & (dm < -abs_thr)
    return valid & hit


def _spearman_pairwise(Y):
    """Spearman matrix: rank each column over its non-NaN rows, then Pearson over
    pairwise-complete rows (computed with masked matrix products). NaN if undefined."""
    ranks = pd.DataFrame(Y).rank(axis=0, method="average").to_numpy()
    M = (~np.isnan(ranks)).astype(float)
    X = np.nan_to_num(ranks, nan=0.0)
    n = M.T @ M
    sx = X.T @ M          # sx[i, j] = sum of x_i over rows where i and j are both valid
    sxy = X.T @ X
    sxx = (X ** 2).T @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        var = sxx - sx ** 2 / n
        C = cov / np.sqrt(var * var.T)
    C[n < 2] = np.nan
    return np.clip(C, -1.0, 1.0)


def correlation_clusters(Y_fit, names, rho=CLUSTER_RHO):
    """Average-linkage clusters on 1-|Spearman| (pairwise-complete; NaN corr -> 0)."""
    C = np.nan_to_num(_spearman_pairwise(np.asarray(Y_fit, dtype=float)), nan=0.0)
    D = np.clip(1.0 - np.abs(C), 0.0, 1.0)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), "average")
    labels = fcluster(Z, t=1.0 - rho, criterion="distance")
    return pd.Series(labels, index=list(names))


def family_delta(delta_mean, drop, families):
    """Per-family mean delta, size, fraction dropped (NaN deltas excluded), dropped = frac > 0.5."""
    df = pd.DataFrame({
        "delta": np.asarray(delta_mean, dtype=float),
        "drop": np.asarray(drop, dtype=float),
        "family": np.asarray(families),
    })
    n = df.groupby("family").size()
    ok = df[df["delta"].notna()]
    g = ok.groupby("family")
    out = pd.DataFrame({
        "mean_delta": g["delta"].mean(),
        "n": n,
        "frac_dropped": g["drop"].mean(),
    }).reindex(n.index)
    out["dropped"] = out["frac_dropped"] > 0.5
    return out


def _eta2_codes(values, codes, n_groups):
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    cnt = np.bincount(codes, minlength=n_groups)
    grand = values.mean()
    ss_between = np.sum(cnt[cnt > 0] * (sums[cnt > 0] / cnt[cnt > 0] - grand) ** 2)
    ss_total = np.sum((values - grand) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def _clean(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    keep = ~np.isnan(values)
    _, codes = np.unique(labels[keep], return_inverse=True)
    return values[keep], codes


def eta2(values, labels):
    """Between-group sum of squares / total sum of squares (NaN values dropped)."""
    v, codes = _clean(values, labels)
    return _eta2_codes(v, codes, codes.max() + 1 if len(codes) else 0)


def eta2_perm_test(values, labels, n_perm=N_PERM, seed=0):
    """(eta2_obs, null 95th percentile, p) with label permutations (sizes preserved)."""
    v, codes = _clean(values, labels)
    k = codes.max() + 1
    obs = _eta2_codes(v, codes, k)
    rng = np.random.default_rng(seed)
    null = np.array([_eta2_codes(v, rng.permutation(codes), k) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, float(np.quantile(null, 0.95)), float(p)


def positive_control(D_train, y_train, names, k=TOP_K_POSCTRL):
    """Top-k descriptors by |Spearman rho| with the label (pairwise-complete,
    >= 10 non-NaN rows required; ties broken by name)."""
    y = np.asarray(y_train, dtype=float)
    scores = []
    for name in names:
        x = np.asarray(D_train[name], dtype=float)
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < MIN_POSCTRL_ROWS:
            continue
        r = spearmanr(x[m], y[m]).statistic
        if np.isnan(r):
            continue
        scores.append((-abs(r), name))
    scores.sort()
    return [name for _, name in scores[:k]]

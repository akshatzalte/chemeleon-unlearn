"""Drift measurement: weight change, paired probe comparison vs `pre`, long-format rows.

Pairing contract: every encoder is compared with `pre` on exactly the rows given for that
encoder (``rows_by_enc``), using the SAME probe fit/predictions for all encoders and the
same alpha. Row sets are indices into the rows of Z/Y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from drift import config
from drift.probe import (choose_alpha, crossfit_predict, linear_cka, paired_bootstrap_delta,
                         r2_cols, ridge_fit_predict)
from drift.stats import drop_calls

LONG_COLS = ["task", "regime", "ckpt", "probe_set", "encoder", "measure", "descriptor", "value"]


def weight_change(mp, mp_pre) -> dict[str, float]:
    """Per parameter tensor ||W - W_pre||_F / ||W_pre||_F (float64)."""
    pre = dict(mp_pre.named_parameters())
    out = {}
    for name, p in mp.named_parameters():
        q = pre[name].detach().double().cpu()
        d = (p.detach().double().cpu() - q).norm()
        den = q.norm()
        out[name] = float(d / den) if den > 0 else float("nan")
    return out


def predict_probe(Z, Y, mode: str, folds=None, alpha=None, fit_rows=None) -> np.ndarray:
    """(n,k) predictions. 'split': ridge fit on fit_rows, predictions only for the other
    rows (fit rows stay NaN). 'crossfit': out-of-fold predictions for every row."""
    Z = np.asarray(Z, dtype=np.float64)
    if mode == "crossfit":
        return crossfit_predict(Z, Y, folds, alpha)
    if mode != "split":
        raise ValueError(mode)
    n = len(Z)
    fit = np.asarray(fit_rows)
    ev = np.setdiff1d(np.arange(n), fit)
    P = np.full((n, Y.shape[1]), np.nan)
    P[ev] = ridge_fit_predict(Z[fit], Y[fit], Z[ev], alpha)
    return P


def compare_to_pre(Y, Yhat_pre, Yhat_by_enc: dict, Z_pre, Z_by_enc: dict, rows_by_enc: dict,
                   n_boot: int = config.N_BOOT) -> dict:
    """Paired comparison of each encoder with `pre` on that encoder's rows.

    Returns dict: r2 {enc: (k,)} (on its rows), delta_per_seed (S,k) in Z_by_enc order,
    delta_mean (k,), se (k,), dropped (k,) bool, cka {enc: float} (pre vs enc on its rows),
    cka_seedpair (mean pairwise CKA among encoders, only when all share the same rows,
    else NaN), r2_pre_by_enc {enc: (k,)} (pre's R2 on that encoder's rows), r2_pre (k,)
    (only when rows are shared, else None).
    eligible (k,) bool: pre's R2 > 0 on the rows of every encoder (NaN -> not eligible);
    only eligible descriptors can be called dropped (dropped is False elsewhere).
    """
    encs = list(Z_by_enc)
    rows = [np.asarray(rows_by_enc[e]) for e in encs]
    dps, dm, se = paired_bootstrap_delta(Y, Yhat_pre, [Yhat_by_enc[e] for e in encs], rows, n_boot=n_boot)
    shared = all(np.array_equal(rows[0], r) for r in rows[1:])
    cka_pair = float("nan")
    if shared and len(encs) > 1:
        vals = [linear_cka(Z_by_enc[a][rows[0]], Z_by_enc[b][rows[0]])
                for i, a in enumerate(encs) for b in encs[i + 1:]]
        cka_pair = float(np.mean(vals))
    r2_pre_by_enc = {e: r2_cols(Y, Yhat_pre, r) for e, r in zip(encs, rows)}
    with np.errstate(invalid="ignore"):
        eligible = np.all([v > 0 for v in r2_pre_by_enc.values()], axis=0)
    return {
        "encoders": encs,
        "r2_pre_by_enc": r2_pre_by_enc,
        "eligible": eligible,
        "r2": {e: r2_cols(Y, Yhat_by_enc[e], r) for e, r in zip(encs, rows)},
        "r2_pre": r2_cols(Y, Yhat_pre, rows[0]) if shared else None,
        "delta_per_seed": dps,
        "delta_mean": dm,
        "se": se,
        "dropped": drop_calls(dps, dm, se) & eligible,
        "cka": {e: linear_cka(Z_pre[r], Z_by_enc[e][r]) for e, r in zip(encs, rows)},
        "cka_seedpair": cka_pair,
    }


def measure_probe_set(Z_pre, Z_by_encoder: dict, targets, eval_rows, mode: str, folds=None,
                      alpha=None, fit_rows=None, Yhat_pre=None, n_boot: int = config.N_BOOT) -> dict:
    """Fit/predict for pre and every encoder, then compare with `pre` on eval_rows.

    eval_rows: one index array shared by all encoders, or a dict enc -> rows.
    split mode: fit_rows (default: all rows not in eval_rows); alpha default
    choose_alpha(Z_pre[fit], Y[fit]). crossfit mode: folds required; alpha default
    choose_alpha(Z_pre, Y). Yhat_pre may be passed to reuse pre predictions.
    """
    Y = np.asarray(targets.Y, dtype=np.float64)
    rows_by_enc = (eval_rows if isinstance(eval_rows, dict)
                   else {e: np.asarray(eval_rows) for e in Z_by_encoder})
    if mode == "split" and fit_rows is None:
        used = np.unique(np.concatenate([np.asarray(r) for r in rows_by_enc.values()]))
        fit_rows = np.setdiff1d(np.arange(len(Y)), used)
    if alpha is None:
        alpha = (choose_alpha(Z_pre[fit_rows], Y[fit_rows]) if mode == "split"
                 else choose_alpha(Z_pre, Y))
    kw = dict(mode=mode, folds=folds, alpha=alpha, fit_rows=fit_rows)
    if Yhat_pre is None:
        Yhat_pre = predict_probe(Z_pre, Y, **kw)
    preds = {e: predict_probe(Z, Y, **kw) for e, Z in Z_by_encoder.items()}
    out = compare_to_pre(Y, Yhat_pre, preds, Z_pre, Z_by_encoder, rows_by_enc, n_boot=n_boot)
    out.update(alpha=alpha, Yhat_pre=Yhat_pre, preds=preds)
    return out


# ----------------------------------------------------------------------- long format
def _block(base, encoder, measure, descriptors, values):
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    n = len(values)
    return pd.DataFrame({
        **{k: [v] * n for k, v in base.items()},
        "encoder": [encoder] * n, "measure": [measure] * n,
        "descriptor": list(descriptors) if not isinstance(descriptors, str) else [descriptors] * n,
        "value": values,
    })


def long_rows(res: dict, names, base: dict, agg: str, include_pre_r2: bool = True) -> pd.DataFrame:
    """Flatten a compare_to_pre result. base = dict(task, regime, ckpt, probe_set).
    agg = encoder label for the seed aggregate ('ft' or 'rand')."""
    blocks = []
    if include_pre_r2:
        if res["r2_pre"] is not None:
            blocks.append(_block(base, "pre", "r2", names, res["r2_pre"]))
        else:
            for e, v in res["r2_pre_by_enc"].items():
                blocks.append(_block(base, f"pre@{e}", "r2", names, v))
    for i, e in enumerate(res["encoders"]):
        blocks.append(_block(base, e, "r2", names, res["r2"][e]))
        blocks.append(_block(base, e, "delta_r2", names, res["delta_per_seed"][i]))
        blocks.append(_block(base, e, "cka", "", res["cka"][e]))
    blocks.append(_block(base, agg, "delta_r2_mean", names, res["delta_mean"]))
    blocks.append(_block(base, agg, "se", names, res["se"]))
    dropped = np.where(res["eligible"], res["dropped"].astype(float), np.nan)
    blocks.append(_block(base, agg, "dropped", names, dropped))
    blocks.append(_block(base, agg, "n_eligible", "", float(res["eligible"].sum())))
    if not np.isnan(res["cka_seedpair"]):
        blocks.append(_block(base, agg, "cka_seedpair", "", res["cka_seedpair"]))
    df = pd.concat(blocks, ignore_index=True)
    return df[LONG_COLS]


def clean_rows(run: dict, n: int) -> np.ndarray:
    """Train molecules the run took no gradient steps on: full -> val_idx;
    n128 -> every row outside train_idx."""
    if run["regime"] == "full":
        return np.sort(np.asarray(run["val_idx"], dtype=np.int64))
    if run["regime"] == "n128":
        return np.setdiff1d(np.arange(n), np.asarray(run["train_idx"], dtype=np.int64))
    raise ValueError(run["regime"])


# ------------------------------------------------------------- families / eta2 / posctrl
def family_labels(names, modules: pd.Series, cluster_map: dict) -> dict[str, np.ndarray]:
    """Both family definitions for `names`. Descriptors missing from the PubChem cluster
    map (not retained there) become singleton clusters."""
    mod = np.array([str(modules[n]) for n in names])
    clu = np.array([f"c{cluster_map[n]}" if n in cluster_map else f"single:{n}" for n in names])
    return {"module": mod, "cluster": clu}


def family_outputs(res: dict, names, labels: dict, base: dict, n_perm: int = config.N_PERM):
    """(families_df, eta2_df) for one view's seed aggregate."""
    from drift.stats import eta2_perm_test, family_delta

    fam_rows, eta_rows = [], []
    el = res["eligible"]
    dm = res["delta_mean"][el]
    dropped = res["dropped"][el]
    for fdef, lab in labels.items():
        lab = np.asarray(lab)[el]
        fd = family_delta(dm, dropped, lab).reset_index().rename(columns={"index": "family"})
        fd.insert(0, "family_def", fdef)
        for k, v in reversed(list(base.items())):
            fd.insert(0, k, v)
        fam_rows.append(fd)
        if np.isfinite(dm).sum() >= 2:
            obs, q95, p = eta2_perm_test(dm, lab, n_perm=n_perm)
        else:
            obs = q95 = p = float("nan")
        eta_rows.append({**base, "family_def": fdef, "eta2": obs, "null_q95": q95, "p": p})
    return pd.concat(fam_rows, ignore_index=True), pd.DataFrame(eta_rows)


def posctrl_rows(res: dict, names, top_names, base: dict) -> pd.DataFrame:
    idx = {n: i for i, n in enumerate(names)}
    top = [n for n in top_names if n in idx]
    ii = [idx[n] for n in top]
    return pd.DataFrame({**{k: [v] * len(top) for k, v in base.items()},
                         "descriptor": top, "delta_mean": res["delta_mean"][ii],
                         "se": res["se"][ii], "eligible": res["eligible"][ii],
                         "dropped": np.where(res["eligible"][ii], res["dropped"][ii].astype(float), np.nan)})


def measure_own_alpha(Z_pre, Z_by_encoder: dict, targets, eval_rows, fit_rows, alpha_pre, Yhat_pre,
                      alphas: dict | None = None, n_boot: int = config.N_BOOT) -> dict:
    """PubChem 'split' comparison where every encoder uses alpha chosen on its OWN fit
    embeddings (pre keeps alpha_pre, i.e. its own). `alphas` may hold cached values
    {enc: alpha}. Returns compare_to_pre result plus alphas {enc: alpha}."""
    Y = np.asarray(targets.Y, dtype=np.float64)
    alphas = dict(alphas or {})
    preds = {}
    for e, Z in Z_by_encoder.items():
        if e not in alphas:
            alphas[e] = choose_alpha(Z[fit_rows], Y[fit_rows])
        preds[e] = predict_probe(Z, Y, "split", alpha=alphas[e], fit_rows=fit_rows)
    rows = {e: np.asarray(eval_rows) for e in Z_by_encoder}
    out = compare_to_pre(Y, Yhat_pre, preds, Z_pre, Z_by_encoder, rows, n_boot=n_boot)
    out["alphas"] = alphas
    return out


def alpha_rows(base: dict, alphas: dict, alpha_pre: float) -> pd.DataFrame:
    return pd.concat([_block(base, e, "alpha", "", a) for e, a in {"pre": alpha_pre, **alphas}.items()],
                     ignore_index=True)[LONG_COLS]


# ------------------------------------------------------------------- PCA / eff. dim.
PCA_VAR = 0.99


def fit_pca(Y_fit, var: float = PCA_VAR) -> dict:
    """PCA (centered, unscaled) of standardized descriptor targets; NaN -> 0 first.
    Keeps components up to `var` cumulative explained variance. Returns loadings (d,k),
    evr (k,) (fraction of TOTAL variance), mean (d,)."""
    X = np.nan_to_num(np.asarray(Y_fit, dtype=np.float64))
    mean = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    lam = s ** 2
    evr = lam / lam.sum()
    k = int(np.searchsorted(np.cumsum(evr), var) + 1)
    return {"loadings": Vt[:k].T, "evr": evr[:k], "mean": mean}


def pc_scores(D_raw, mu, sd, pca: dict) -> np.ndarray:
    """Project raw descriptors (n,d; columns in PCA name order) onto the PCA basis, using the
    PubChem fit-split mu/sd (NOT the target set's), clip +-WINSOR, NaN -> 0."""
    Z = np.clip((np.asarray(D_raw, dtype=np.float64) - mu) / sd, -config.WINSOR, config.WINSOR)
    return (np.nan_to_num(Z) - pca["mean"]) @ pca["loadings"]


def evr_weighted(delta, evr) -> float:
    """sum_k evr_k * delta_k over finite deltas."""
    delta = np.asarray(delta, dtype=np.float64)
    ok = np.isfinite(delta)
    return float(np.sum(np.asarray(evr)[ok] * delta[ok]))


def _eff_dim(Y):
    X = np.nan_to_num(np.asarray(Y, dtype=np.float64))
    lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    if lam.sum() <= 0:
        return dict(n90=0, n95=0, n99=0, participation_ratio=float("nan"))
    c = np.cumsum(lam) / lam.sum()
    n = {f"n{p}": int(np.searchsorted(c, p / 100 - 1e-12) + 1) for p in (90, 95, 99)}
    return {**n, "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum())}


def effdim_table(Y_fit, names, modules: pd.Series) -> pd.DataFrame:
    """Effective dimensionality overall ('ALL') and within each Mordred module."""
    Y_fit = np.asarray(Y_fit)
    mod = np.array([str(modules[n]) for n in names])
    rows = [dict(family="ALL", n_descriptors=len(names), **_eff_dim(Y_fit))]
    for m in sorted(set(mod)):
        sel = mod == m
        rows.append(dict(family=m, n_descriptors=int(sel.sum()), **_eff_dim(Y_fit[:, sel])))
    return pd.DataFrame(rows)


def pc_long_rows(res: dict, pc_names, base: dict, agg: str, evr) -> pd.DataFrame:
    """long_rows for PC targets (without the duplicate CKA measures) plus the
    variance-weighted summary `delta_r2_evrw` (per seed and aggregate)."""
    df = long_rows(res, pc_names, base, agg)
    df = df[~df["measure"].isin(["cka", "cka_seedpair"])]
    blocks = [df]
    for i, e in enumerate(res["encoders"]):
        blocks.append(_block(base, e, "delta_r2_evrw", "", evr_weighted(res["delta_per_seed"][i], evr)))
    blocks.append(_block(base, agg, "delta_r2_evrw", "", evr_weighted(res["delta_mean"], evr)))
    return pd.concat(blocks, ignore_index=True)[LONG_COLS]

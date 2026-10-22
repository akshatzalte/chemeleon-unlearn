import numpy as np
import pytest
import torch
from chemprop.nn import BondMessagePassing

from drift.descriptors import Targets
from drift.measure import (LONG_COLS, clean_rows, compare_to_pre, long_rows, measure_probe_set,
                           weight_change)
from drift.probe import make_folds

SEEDS = ["seed42", "seed117", "seed709"]


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    n, d = 400, 40
    Z = rng.normal(size=(n, d))
    Y = np.empty((n, 6))
    Y[:, 0] = Z[:, :5].sum(1)                       # carried only by columns 0-4
    for j in range(1, 6):
        Y[:, j] = Z[:, 5 + 6 * (j - 1): 11 + 6 * (j - 1)] @ rng.normal(size=6)
    Y = (Y - Y.mean(0)) / Y.std(0) + 0.05 * rng.normal(size=Y.shape)
    T = Targets(Y.astype(np.float32), [f"d{j}" for j in range(6)], [])
    lost = Z.copy()
    lost[:, :5] = 0.0
    return Z, lost, T, n


def test_measure_detects_planted_loss(synth):
    Z, lost, T, n = synth
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T, np.arange(n), mode="crossfit",
                            folds=make_folds(n), alpha=1.0)
    assert out["dropped"][0] and not out["dropped"][1:].any()
    assert (out["delta_per_seed"][:, 0] < -0.5).all()
    assert np.abs(out["delta_per_seed"][:, 1:]).max() < 1e-3
    assert out["cka"]["seed42"] < 1 - 1e-3


def test_measure_identity_no_change(synth):
    Z, _, T, n = synth
    out = measure_probe_set(Z, {s: Z.copy() for s in SEEDS}, T, np.arange(n), mode="crossfit",
                            folds=make_folds(n), alpha=1.0)
    assert not out["dropped"].any()
    np.testing.assert_allclose(out["delta_per_seed"], 0, atol=1e-9)
    np.testing.assert_allclose(out["delta_mean"], 0, atol=1e-9)
    assert all(abs(v - 1) < 1e-9 for v in out["cka"].values())
    assert abs(out["cka_seedpair"] - 1) < 1e-9


def test_split_mode_uses_fit_rows_and_default_alpha(synth):
    Z, lost, T, n = synth
    ev = np.arange(300, 400)
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T, ev, mode="split")
    assert out["dropped"][0] and not out["dropped"][1:].any()
    assert out["alpha"] > 0
    assert np.isnan(out["Yhat_pre"][:300]).all() and np.isfinite(out["Yhat_pre"][300:]).all()


def test_per_seed_rows_pair_pre_on_same_rows(synth):
    """Different clean rows per seed: seed delta must equal R2(seed)-R2(pre) on THAT seed's rows."""
    Z, lost, T, n = synth
    rows = {"seed42": np.arange(0, 100), "seed117": np.arange(100, 200), "seed709": np.arange(200, 300)}
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T, rows, mode="crossfit",
                            folds=make_folds(n), alpha=1.0)
    for i, s in enumerate(SEEDS):
        r2_pre = compare_pre_r2(out, T, rows[s])
        np.testing.assert_allclose(out["delta_per_seed"][i], out["r2"][s] - r2_pre, atol=1e-9)
    assert out["r2_pre"] is None and np.isnan(out["cka_seedpair"])


def compare_pre_r2(out, T, rows):
    from drift.probe import r2_cols
    return r2_cols(T.Y.astype(np.float64), out["Yhat_pre"], rows)


def test_clean_rows_full_is_val_idx():
    run = {"regime": "full", "val_idx": [7, 3, 9], "train_idx": [0, 1, 2, 4, 5, 6, 8]}
    np.testing.assert_array_equal(clean_rows(run, 10), [3, 7, 9])


def test_clean_rows_n128_is_complement_of_train_idx():
    run = {"regime": "n128", "val_idx": [5], "train_idx": [0, 2, 4, 6, 8]}
    np.testing.assert_array_equal(clean_rows(run, 10), [1, 3, 5, 7, 9])  # val row 5 included


def test_long_format_schema(synth):
    Z, lost, T, n = synth
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T, np.arange(n), mode="crossfit",
                            folds=make_folds(n), alpha=1.0, n_boot=20)
    base = dict(task="t", regime="full", ckpt="es", probe_set="task_all")
    df = long_rows(out, T.names, base, "ft")
    assert list(df.columns) == LONG_COLS
    assert df["value"].dtype == np.float64
    ms = set(df["measure"])
    assert ms == {"r2", "delta_r2", "cka", "delta_r2_mean", "se", "dropped", "cka_seedpair", "n_eligible"}
    assert set(df[df.measure == "dropped"]["encoder"]) == {"ft"}
    assert set(df[df.measure == "delta_r2"]["encoder"]) == set(SEEDS)
    assert "pre" in set(df[df.measure == "r2"]["encoder"])
    d = df[(df.measure == "dropped")].set_index("descriptor")["value"]
    assert d["d0"] == 1.0 and d["d1"] == 0.0
    assert len(df[(df.measure == "r2") & (df.encoder == "seed42")]) == 6


def test_weight_change():
    mp = BondMessagePassing(d_h=8)
    pre = BondMessagePassing(d_h=8)
    pre.load_state_dict(mp.state_dict())
    assert all(v == 0 for v in weight_change(mp, pre).values())
    with torch.no_grad():
        for p in mp.parameters():
            p.mul_(1.5)
    wc = weight_change(mp, pre)
    assert set(wc) == {n for n, _ in pre.named_parameters()}
    assert all(abs(v - 0.5) < 1e-5 for v in wc.values())


def test_family_and_posctrl_outputs(synth):
    import pandas as pd
    from drift.measure import family_labels, family_outputs, posctrl_rows
    Z, lost, T, n = synth
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T, np.arange(n), mode="crossfit",
                            folds=make_folds(n), alpha=1.0, n_boot=20)
    mods = pd.Series(["A", "A", "B", "B", "B", "C"], index=T.names)
    lab = family_labels(T.names + ["extra"], mods.reindex(T.names + ["extra"], fill_value="Z"),
                        {"d0": 1, "d1": 1, "d2": 2})
    assert lab["cluster"][-1] == "single:extra" and lab["cluster"][0] == "c1"
    base = dict(task="t", regime="full", ckpt="es", probe_set="task_all")
    lab = family_labels(T.names, mods, {"d0": 1, "d1": 1})
    fam, eta = family_outputs(out, T.names, lab, base, n_perm=50)
    assert set(fam["family_def"]) == {"module", "cluster"} and set(eta["family_def"]) == {"module", "cluster"}
    assert {"family", "mean_delta", "n", "frac_dropped", "dropped"} <= set(fam.columns)
    assert list(eta.columns) == ["task", "regime", "ckpt", "probe_set", "family_def", "eta2", "null_q95", "p"]
    pc = posctrl_rows(out, T.names, ["d0", "d3", "nope"], base)
    assert list(pc["descriptor"]) == ["d0", "d3"] and pc["dropped"].tolist() == [True, False]


def test_ineligible_descriptor_cannot_be_dropped_and_is_excluded(synth):
    Z, lost, T, n = synth
    Y = T.Y.copy()
    Y[:, 0] = np.random.default_rng(3).normal(size=n)      # pre R2 ~ 0 for d0 -> ineligible
    T2 = Targets(Y, T.names, [])
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T2, np.arange(n), mode="crossfit",
                            folds=make_folds(n), alpha=1.0, n_boot=20)
    assert not out["eligible"][0] and out["eligible"][1:].all()
    assert not out["dropped"][0]
    base = dict(task="t", regime="full", ckpt="es", probe_set="task_all")
    df = long_rows(out, T2.names, base, "ft")
    d = df[df.measure == "dropped"].set_index("descriptor")["value"]
    assert np.isnan(d["d0"]) and d["d1"] == 0.0
    assert df[df.measure == "n_eligible"]["value"].iloc[0] == 5
    import pandas as pd
    from drift.measure import family_labels, family_outputs
    mods = pd.Series(["A"] * 6, index=T2.names)
    fam, _ = family_outputs(out, T2.names, family_labels(T2.names, mods, {}), base, n_perm=20)
    assert fam[fam.family_def == "module"]["n"].iloc[0] == 5


def test_pre_r2_per_seed_stored_for_clean_rows(synth):
    Z, lost, T, n = synth
    rows = {"seed42": np.arange(0, 100), "seed117": np.arange(100, 200), "seed709": np.arange(200, 300)}
    out = measure_probe_set(Z, {s: lost for s in SEEDS}, T, rows, mode="crossfit",
                            folds=make_folds(n), alpha=1.0, n_boot=20)
    base = dict(task="t", regime="full", ckpt="es", probe_set="task_clean")
    df = long_rows(out, T.names, base, "ft")
    r2 = df[df.measure == "r2"]
    assert {"pre@seed42", "pre@seed117", "pre@seed709"} <= set(r2.encoder)


def test_own_alpha_matches_fixed_alpha_when_alpha_agrees(synth):
    from drift.measure import alpha_rows, measure_own_alpha
    Z, lost, T, n = synth
    fit = np.arange(300)
    ev = np.arange(300, 400)
    ref = measure_probe_set(Z, {s: lost for s in SEEDS}, T, ev, mode="split", fit_rows=fit, alpha=10.0, n_boot=20)
    own = measure_own_alpha(Z, {s: lost for s in SEEDS}, T, ev, fit, 10.0, ref["Yhat_pre"],
                            alphas={s: 10.0 for s in SEEDS}, n_boot=20)
    np.testing.assert_allclose(own["delta_mean"], ref["delta_mean"])
    own2 = measure_own_alpha(Z, {s: lost for s in SEEDS}, T, ev, fit, 10.0, ref["Yhat_pre"], n_boot=20)
    assert set(own2["alphas"]) == set(SEEDS) and own2["dropped"][0]
    df = alpha_rows(dict(task="t", regime="full", ckpt="es", probe_set="pubchem_ownalpha"), own2["alphas"], 10.0)
    assert set(df.encoder) == {"pre", *SEEDS} and set(df.measure) == {"alpha"}

"""Row-selection logic of scripts/03_measure.py (novel/clean views, <30-row NaN path), run on a
tiny fake task with patched embeddings/membership (no GPU, no real data)."""
import importlib.util
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import fake_task
from drift import config
from drift.descriptors import Targets
from drift.measure import clean_rows, predict_probe

N = 200
NAME = "fake/regression"
SEEDS = (42, 117, 709)
DESC = [f"d{i}" for i in range(6)]
FIT = np.arange(200)
EVAL = np.arange(200, 300)


def emb(key, n, d=16):
    return np.random.default_rng(zlib.crc32(key.encode())).normal(size=(n, d)).astype(np.float32)


def make_runs():
    allr = np.arange(N)
    val = {42: np.arange(0, 40), 117: np.arange(150, 190), 709: np.arange(50, 90)}
    tr128 = {42: np.arange(98, 200), 117: np.r_[0:2, 100:200], 709: np.arange(98, 200)}
    runs = {}
    for sd in SEEDS:
        runs[("full", sd)] = dict(regime="full", seed=sd, val_idx=val[sd].tolist(),
                                  train_idx=np.setdiff1d(allr, val[sd]).tolist(), dir=Path(f"/fake/full/{sd}"))
        runs[("n128", sd)] = dict(regime="n128", seed=sd, train_idx=tr128[sd].tolist(),
                                  val_idx=[], dir=Path(f"/fake/n128/{sd}"))
    return runs


@pytest.fixture
def env(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("measure_script", config.ROOT / "scripts/03_measure.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    task = fake_task(N, "regression")
    rng = np.random.default_rng(0)
    mods = pd.Series(["A", "A", "B", "B", "C", "C"], index=DESC)
    D = pd.DataFrame(rng.normal(size=(N, 6)), columns=DESC)
    (tmp_path / "descriptors").mkdir()
    D.assign(smiles=task.train_smiles).to_parquet(tmp_path / "descriptors" / f"{config.slug(NAME)}.parquet")
    runs = make_runs()
    S.real_expected_runs = S.expected_runs
    monkeypatch.setattr(S, "OUT", tmp_path / "drift")
    monkeypatch.setattr(S, "GLOBAL", tmp_path / "drift" / "_global")
    monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(S, "FIT", FIT)
    monkeypatch.setattr(S, "EVAL", EVAL)
    monkeypatch.setattr(S, "load_task", lambda name: task)
    monkeypatch.setattr(S, "expected_runs", lambda name: runs)
    monkeypatch.setattr(S, "load_mp", lambda path: None)
    monkeypatch.setattr(S, "weight_change", lambda mp, pre: {"p": 0.1})
    monkeypatch.setattr(S, "get_emb", lambda set_name, enc_id, mk, smi, dev: emb(set_name + enc_id, len(smi)))
    pub = type("Pub", (), {})()
    pub.device = "cpu"
    pub.targets = Targets(rng.normal(size=(300, 6)).astype(np.float32), DESC, [])
    pub.D = pd.DataFrame(rng.normal(size=(300, 6)), columns=DESC)
    pub.Z_pre = emb("pubpre", 300)
    pub.alpha = 10.0
    pub.Yhat_pre = predict_probe(pub.Z_pre, pub.targets.Y.astype(float), "split", alpha=10.0, fit_rows=FIT)
    pub.embed_enc = lambda eid, mk: emb("pub" + eid, 300)
    pca = dict(loadings=np.linalg.qr(rng.normal(size=(6, 2)))[0], evr=np.array([0.6, 0.3]), mean=np.zeros(6),
               mu=np.zeros(6), sd=np.ones(6), names=DESC)
    (tmp_path / "drift" / S.slug(NAME)).mkdir(parents=True)
    return S, pub, mods, pca, runs, tmp_path


def run_novel(env, novel_rows, monkeypatch):
    S, pub, mods, pca, runs, tmp = env
    member = np.array(["train"] * N, dtype=object)
    member[novel_rows[: len(novel_rows) // 2]] = "test"
    member[novel_rows[len(novel_rows) // 2:]] = "none"
    monkeypatch.setattr(S, "pretrain_membership", lambda smi: member)
    d = S.OUT / S.slug(NAME)
    (d / "done.json").write_text(json.dumps({"alpha_task": 10.0}))
    S.novel_task(NAME, pub, mods, pca)
    return pd.read_parquet(d / "drift_long_novel.parquet"), member


def nrows(df, regime, view):
    x = df[(df.measure == "n_rows") & (df.regime == regime) & (df.ckpt == "es") & (df.probe_set == view)]
    return {e: int(v) for e, v in zip(x.encoder, x.value)}


def test_novel_views_row_counts_and_nan_path(env, monkeypatch):
    runs = env[4]
    novel = np.arange(0, 100)
    df, member = run_novel(env, novel, monkeypatch)
    assert set(member[novel]) == {"test", "none"}
    for regime in ("full", "n128"):
        clean = {f"seed{sd}": clean_rows(runs[(regime, sd)], N) for sd in SEEDS}
        if regime == "full":                                   # clean = val_idx
            for sd in SEEDS:
                np.testing.assert_array_equal(clean[f"seed{sd}"], np.sort(runs[(regime, sd)]["val_idx"]))
        else:                                                  # clean = complement of train_idx
            for sd in SEEDS:
                assert set(clean[f"seed{sd}"]) == set(range(N)) - set(runs[(regime, sd)]["train_idx"])
        nov = set(novel)
        exp = {
            "task_all": {e: N for e in clean},
            "task_clean": {e: len(c) for e, c in clean.items()},
            "task_all_novel": {e: len(nov) for e in clean},
            "task_all_nonnovel": {e: N - len(nov) for e in clean},
            "task_clean_novel": {e: len(nov & set(c)) for e, c in clean.items()},
            "task_clean_nonnovel": {e: len(set(c) - nov) for e, c in clean.items()},
        }
        for view, want in exp.items():
            assert nrows(df, regime, view) == want, (regime, view)
    # full: seed117's clean rows (150..189) are all non-novel -> clean_novel has <30 rows for
    # one seed -> whole view (and its _pc twin) is NaN, no r2 rows; task_all_novel is real.
    for view in ("task_clean_novel", "task_clean_novel_pc"):
        x = df[(df.regime == "full") & (df.ckpt == "es") & (df.probe_set == view) & (df.measure != "n_rows")]
        assert list(x.measure) == ["delta_r2_mean"] and np.isnan(x.value.iloc[0]) and x.encoder.iloc[0] == "ft"
    assert nrows(df, "full", "task_clean_novel") == {"seed42": 40, "seed117": 0, "seed709": 40}
    for view in ("task_all_novel", "task_all_novel_pc", "task_all_nonnovel"):
        x = df[(df.regime == "full") & (df.ckpt == "es") & (df.probe_set == view) & (df.measure == "delta_r2_mean")]
        assert len(x) > 1 and x.value.notna().all()
    # n128: clean (98 rows each) is entirely novel -> real outputs
    x = df[(df.regime == "n128") & (df.ckpt == "es") & (df.probe_set == "task_clean_novel") & (df.measure == "r2")]
    assert x.value.notna().any()
    assert set(df[df.probe_set == "task_clean_nonnovel"].regime) == {"full", "n128"}


def test_all_novel_view_nan_when_too_few_novel_rows(env, monkeypatch):
    df, _ = run_novel(env, np.arange(0, 20), monkeypatch)      # only 20 novel rows
    for regime in ("full", "n128"):
        assert nrows(df, regime, "task_all_novel") == {f"seed{sd}": 20 for sd in SEEDS}
        for view in ("task_all_novel", "task_all_novel_pc", "task_clean_novel", "task_clean_novel_pc"):
            x = df[(df.regime == regime) & (df.ckpt == "es") & (df.probe_set == view) & (df.measure != "n_rows")]
            assert len(x) == 1 and np.isnan(x.value.iloc[0])
        x = df[(df.regime == regime) & (df.probe_set == "task_all_nonnovel") & (df.measure == "delta_r2_mean")]
        assert x.value.notna().all()


def test_measure_task_and_pc_task_row_selection(env, monkeypatch):
    S, pub, mods, pca, runs, tmp = env
    S.measure_task(NAME, pub, mods, {}, "cpu")
    d = S.OUT / S.slug(NAME)
    meta = json.loads((d / "done.json").read_text())
    assert meta["n_clean"] == {f"{r}/s{sd}": len(clean_rows(runs[(r, sd)], N)) for r, sd in runs}
    assert meta["n_clean"]["full/s42"] == 40 and meta["n_clean"]["n128/s42"] == 98
    long = pd.read_parquet(d / "drift_long.parquet")
    assert set(long.probe_set) >= {"pubchem", "task_all", "task_clean"}
    x = long[(long.regime == "full") & (long.ckpt == "es") & (long.probe_set == "task_clean")
             & (long.measure == "delta_r2")]
    assert set(x.encoder) == {"seed42", "seed117", "seed709"}
    S.pc_task(NAME, pub, pca)
    pc = pd.read_parquet(d / "drift_long_pc.parquet")
    assert set(pc.probe_set) == {"pubchem_pc", "task_all_pc", "task_clean_pc"}
    assert set(pc[pc.measure == "delta_r2_evrw"].descriptor) == {""}


def test_small_task_measured_with_full_regime_only(env, monkeypatch):
    S, pub, mods, pca, _, tmp = env
    n = 100                                                   # <= N_SMALL: no n128 runs exist
    task = fake_task(n, "regression")
    D = pd.DataFrame(np.random.default_rng(1).normal(size=(n, 6)), columns=DESC)
    D.assign(smiles=task.train_smiles).to_parquet(tmp / "descriptors" / f"{config.slug(NAME)}.parquet")
    runs_dir = tmp / "runs"
    for sd in SEEDS:
        d = runs_dir / config.slug(NAME) / "full" / "ft" / f"seed{sd}"
        d.mkdir(parents=True)
        (d / "done.json").write_text("{}")
        val = list(range(sd % 10, n, 5))
        (d / "run.json").write_text(json.dumps(dict(regime="full", seed=sd, val_idx=val,
                                                    train_idx=[i for i in range(n) if i not in val])))
    monkeypatch.setattr(S, "load_task", lambda name: task)
    monkeypatch.setattr(S, "RUNS", runs_dir)
    monkeypatch.setattr(S, "expected_runs", S.real_expected_runs)
    monkeypatch.setattr(S, "pretrain_membership", lambda smi: np.array(["none"] * 50 + ["train"] * 50, dtype=object))
    got = S.expected_runs(NAME)
    assert got is not None and len(got) == 3 and S.regimes_of(got) == ["full"]
    S.measure_task(NAME, pub, mods, {}, "cpu")
    d = S.OUT / S.slug(NAME)
    assert set(json.loads((d / "done.json").read_text())["n_clean"]) == {f"full/s{sd}" for sd in SEEDS}
    for fn, call in (("drift_long.parquet", None), ("drift_long_pc.parquet", lambda: S.pc_task(NAME, pub, pca)),
                     ("drift_long_novel.parquet", lambda: S.novel_task(NAME, pub, mods, pca)),
                     ("drift_long_ownalpha.parquet", lambda: S.ownalpha_task(NAME, pub))):
        if call:
            call()
        df = pd.read_parquet(d / fn)
        assert set(df[df.regime != ""].regime) == {"full"}, fn
    # a task with n128 runs missing but n_train > N_SMALL is still "not all done"
    monkeypatch.setattr(S, "load_task", lambda name: fake_task(N, "regression"))
    assert S.expected_runs(NAME) is None


def test_combine_requires_markers(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("measure_script_c", config.ROOT / "scripts/03_measure.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    out, res = tmp_path / "drift", tmp_path / "res"
    g = out / "_global"
    g.mkdir(parents=True); res.mkdir()
    monkeypatch.setattr(S, "OUT", out); monkeypatch.setattr(S, "GLOBAL", g); monkeypatch.setattr(S, "RESULTS", res)

    def pq(path, tag):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"tag": [tag]}).to_parquet(path)

    for f in ("drift_long", "families", "eta2", "posctrl", "drift_long_pc", "drift_long_ownalpha", "corr_clusters", "effdim"):
        pq(g / f"{f}.parquet", f"g_{f}")
    for m in ("done", "pca_done", "ownalpha_done"):
        (g / f"{m}.json").write_text("{}")
    # ok: everything; half: done.json but pc parquet written without pc_done; nodone: no done.json at all
    for t in ("ok", "half", "nodone"):
        for f in ("drift_long", "families", "eta2", "posctrl", "drift_long_pc", "drift_long_novel", "drift_long_ownalpha"):
            pq(out / t / f"{f}.parquet", f"{t}_{f}")
    for m in ("done", "pc_done", "novel_done", "ownalpha_done"):
        (out / "ok" / f"{m}.json").write_text("{}")
    for m in ("done", "novel_done"):
        (out / "half" / f"{m}.json").write_text("{}")
    S.combine()
    d = pd.read_parquet(res / "drift_long.parquet")["tag"].tolist()
    assert "ok_drift_long_pc" in d and "ok_drift_long_novel" in d and "ok_drift_long_ownalpha" in d
    assert "half_drift_long" in d and "half_drift_long_novel" in d
    assert "half_drift_long_pc" not in d and "half_drift_long_ownalpha" not in d
    assert not any(x.startswith("nodone") for x in d)
    assert "g_drift_long_pc" in d and "g_drift_long_ownalpha" in d
    assert not any(x.startswith("nodone") for x in pd.read_parquet(res / "families.parquet")["tag"])
    assert (res / "effdim.parquet").exists()


def test_combine_logs_summary_and_global_markers(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("measure_script_c2", config.ROOT / "scripts/03_measure.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    out, res = tmp_path / "drift", tmp_path / "res"
    g = out / "_global"
    g.mkdir(parents=True); res.mkdir()
    monkeypatch.setattr(S, "OUT", out); monkeypatch.setattr(S, "GLOBAL", g); monkeypatch.setattr(S, "RESULTS", res)
    for f in ("drift_long", "families", "eta2", "posctrl", "corr_clusters", "drift_long_pc"):
        pd.DataFrame({"tag": [f"g_{f}"]}).to_parquet(g / f"{f}.parquet")
    (g / "done.json").write_text("{}")   # no pca_done.json -> global pc parquet excluded
    (out / "t1").mkdir()
    for f in ("drift_long", "families", "eta2", "posctrl"):
        pd.DataFrame({"tag": [f"t_{f}"]}).to_parquet(out / "t1" / f"{f}.parquet")
    (out / "t1" / "done.json").write_text("{}")
    S.combine()
    printed = capsys.readouterr().out
    assert "combined 1 tasks; missing pc: ['t1'], novel: ['t1'], ownalpha: ['t1']" in printed
    assert "g_drift_long_pc" not in pd.read_parquet(res / "drift_long.parquet")["tag"].tolist()

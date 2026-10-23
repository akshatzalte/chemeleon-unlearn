import json
import logging
import shutil

import lightning
import numpy as np
import pytest
from conftest import fake_task_real_smiles

from drift import config
from drift import finetune as FT
from drift.phase2 import ladder as L

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning:lightning")


class StepRecorder(lightning.Callback):
    """Independent count of real optimizer steps (and the lr used at each) via on_before_optimizer_step."""

    def __init__(self):
        self.lrs, self.epochs = [], set()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.lrs.append(optimizer.param_groups[0]["lr"])
        self.epochs.add(trainer.current_epoch)


@pytest.fixture
def recorder(monkeypatch):
    rec = StepRecorder()
    real = FT.Trainer

    def trainer_with_recorder(*a, callbacks=(), **kw):
        return real(*a, callbacks=[*callbacks, rec], **kw)

    monkeypatch.setattr(FT, "Trainer", trainer_with_recorder)
    return rec


def _fixed(task, tmp_path, n, steps, **kw):
    idx = np.arange(n)
    return FT.run_one(task, "custom", "ft", 1, tmp_path, accelerator="cpu", train_idx=idx, fixed_steps=steps,
                      batch_size=min(n, 16), num_workers=0, **kw)


@pytest.mark.parametrize("n, epochs", [(16, 16), (64, 4)])
def test_fixed_steps_counts_optimizer_steps(tmp_path, recorder, n, epochs):
    t = fake_task_real_smiles(n=n, kind="regression")
    info = _fixed(t, tmp_path, n, 16)
    assert len(recorder.lrs) == 16 == info["n_steps"] == info["fixed_steps"]
    assert info["max_epochs"] == epochs and len(recorder.epochs) == epochs
    # warm-up is 2 steps: init_lr, halfway, then max_lr
    assert recorder.lrs[0] == pytest.approx(1e-4) and recorder.lrs[1] == pytest.approx(5.5e-4)
    assert recorder.lrs[2] == pytest.approx(1e-3)
    assert recorder.lrs[-1] < recorder.lrs[2]
    # no val split, es = final weights only
    assert info["n_val_rows"] == 0 and info["val_idx"] == [] and info["es_epoch"] is None
    assert (tmp_path / "pred_es.npy").exists() and (tmp_path / "es_mp.pt").exists()
    assert (tmp_path / "es_head.pt").exists() and (tmp_path / "done.json").exists()
    assert not list(tmp_path.glob("*last*")) and not (tmp_path / "metrics.csv").exists()


def test_fixed_steps_stops_mid_epoch_when_not_divisible(tmp_path, recorder):
    t = fake_task_real_smiles(n=32, kind="classification")
    info = _fixed(t, tmp_path, 32, 5)  # 2 batches/epoch -> 3 epochs allowed, stop after step 5
    assert len(recorder.lrs) == 5 == info["n_steps"] and info["max_epochs"] == 3


def test_fixed_steps_requires_custom_regime(tmp_path):
    t = fake_task_real_smiles(n=20, kind="regression")
    with pytest.raises(ValueError):
        FT.run_one(t, "full", "ft", 1, tmp_path, accelerator="cpu", fixed_steps=4)
    with pytest.raises(ValueError):
        FT.run_one(t, "custom", "ft", 1, tmp_path, accelerator="cpu")


def test_custom_without_fixed_steps_splits_inside_subsample(tmp_path):
    t = fake_task_real_smiles(n=40, kind="regression")
    sub = np.arange(5, 35)
    info = FT.run_one(t, "custom", "ft", 3, tmp_path, accelerator="cpu", max_epochs=1, train_idx=sub,
                      num_workers=0)
    assert sorted(info["train_idx"] + info["val_idx"]) == sub.tolist()
    assert len(info["val_idx"]) == 6 and (tmp_path / "pred_last.npy").exists()


def test_cache_graphs_flag(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = FT.SimpleMoleculeMolGraphFeaturizer.__call__

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(FT.SimpleMoleculeMolGraphFeaturizer, "__call__", counting)
    t = fake_task_real_smiles(n=32, kind="regression")
    n_test = len(t.test_smiles)

    def run(cache, sub):
        calls["n"] = 0
        FT.run_one(t, "custom", "ft", 1, tmp_path / sub, accelerator="cpu", train_idx=np.arange(32),
                   fixed_steps=8, batch_size=16, num_workers=0, cache_graphs=cache)  # 4 epochs
        return calls["n"]

    assert run(True, "c") == 32 + n_test  # each molecule featurized once, epochs 2..4 reuse the cache
    assert run(False, "u") == 4 * 32 + n_test  # featurized again every epoch
    assert json.loads((tmp_path / "c" / "run.json").read_text())["cache_graphs"] is True


def _ames_like(n=600, pos=0.2):
    t = fake_task_real_smiles(n=n, kind="classification")
    y = np.zeros(n)
    y[np.random.default_rng(0).choice(n, int(n * pos), replace=False)] = 1.0
    t.y_raw = y.copy()
    t.y_train = y.copy()
    return t


def test_subsample_paired_and_stratified(tmp_path):
    t = _ames_like()
    seeds_for = {16: 10, 32: 3, 128: 2, "full": 3}
    L.make_subsamples(t, sizes=(16, 32, 128, "full"), seeds_for=seeds_for, out_dir=tmp_path)
    d = tmp_path / "fake_classification_real"
    for s in range(10):
        rec = json.loads((d / f"N16_s{s}.json").read_text())
        idx = rec["idx"]
        assert len(idx) == len(set(idx)) == 16 and rec["N"] == 16 and rec["seed"] == s
        assert rec["sha1"] == L._sha1(idx)
        assert abs(t.y_raw[idx].sum() - 16 * 0.2) <= 1  # class ratio kept within 1
    assert json.loads((d / "N16_s0.json").read_text())["idx"] != json.loads((d / "N16_s1.json").read_text())["idx"]
    full = [json.loads((d / f"Nfull_s{s}.json").read_text())["idx"] for s in range(3)]
    assert full[0] == full[1] == full[2] == list(range(600))
    # idempotent: files untouched on a second call
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    L.make_subsamples(t, sizes=(16, 32, 128, "full"), seeds_for=seeds_for, out_dir=tmp_path)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_subsamples_exclude_nan_labels_and_regression(tmp_path):
    t = fake_task_real_smiles(n=200, kind="regression")
    t.y_raw[:50] = np.nan
    L.make_subsamples(t, sizes=(16, 128), seeds_for={16: 2, 128: 1}, out_dir=tmp_path)
    for f in (tmp_path / "fake_regression_real").glob("N*.json"):
        assert all(i >= 50 for i in json.loads(f.read_text())["idx"])


@pytest.fixture
def ladder_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    return tmp_path


def _make_arm(arm):
    d = L.pretrain_dir(arm)
    d.mkdir(parents=True)
    shutil.copy(config.CHEMELEON_MP, d / "mp.pt")
    (d / "done.json").write_text("{}")


def test_run_json_records_sha_and_arm(ladder_env):
    t = fake_task_real_smiles(n=40, kind="classification")
    _make_arm("T100")
    L.make_subsamples(t, sizes=(16,), seeds_for={16: 2})
    info = L.run_item(t.name, "T100", 16, 1, None, task_data=t)
    run = json.loads((L.run_dir(t.name, "T100", 16, 1) / "run.json").read_text())
    sub = L.load_subsample(t.name, 16, 1)
    assert run["arm"] == "T100" and run["subsample_sha1"] == sub["sha1"]
    assert run["mp_path"] == str(L.pretrain_dir("T100") / "mp.pt") and run["n_steps"] == 16 == info["n_steps"]
    assert run["train_idx"] == sub["idx"] and run["val_idx"] == [] and run["N"] == 16
    assert run["ft_seed"] == 1042 and run["batch_size"] == 16 and run["cache_graphs"] is True
    assert (L.run_dir(t.name, "T100", 16, 1) / "pred_es.npy").exists()
    # paired: another arm trains on exactly the same rows
    info2 = L.run_item(t.name, "chemeleon", 16, 1, None, task_data=t)
    assert info2["train_idx"] == info["train_idx"] and info2["subsample_sha1"] == run["subsample_sha1"]
    assert info2["mp_path"] == str(config.CHEMELEON_MP)


def test_run_item_protocol_routing(ladder_env, monkeypatch):
    t = fake_task_real_smiles(n=200, kind="regression")
    L.make_subsamples(t, sizes=(32, 128, "full"), seeds_for={32: 1, 128: 1, "full": 3})
    seen = []
    monkeypatch.setattr(L, "run_one", lambda td, regime, mode, seed, d, **kw: seen.append((regime, seed, kw)) or {})
    for n, s in ((32, 0), (128, 0), ("full", 2)):
        L.run_item(t.name, "chemeleon", n, s, None, task_data=t)
    (r32, s32, k32), (r128, s128, k128), (rf, sf, kf) = seen
    assert r32 == "custom" and k32["fixed_steps"] == 16 and k32["batch_size"] == 16 and len(k32["train_idx"]) == 32
    assert r128 == "custom" and "fixed_steps" not in k128 and "batch_size" not in k128
    assert len(k128["train_idx"]) == 128 and s128 == 42
    assert rf == "full" and "train_idx" not in kf and sf == 709  # phase-1 seeds for the three full runs
    assert all(k["cache_graphs"] for k in (k32, k128, kf))


def test_worklist_and_pending_skip_arms_without_encoder(ladder_env, caplog):
    wl = L.worklist()
    assert len(wl) == 2 * 8 * (3 * 10 + 2 * 5 + 3 * 3) and len({*wl}) == len(wl)
    assert {a for _, a, _, _ in wl} == set(L.ARMS) and wl[0][2] == "full"
    assert {n for _, _, n, s in wl if s >= 3} >= {16, 128}
    _make_arm("B")
    with caplog.at_level(logging.WARNING):
        todo = L.pending(wl)
    assert {a for _, a, _, _ in todo} == {"B", "chemeleon"}
    assert "arm T100: no finished encoder" in caplog.text
    done = todo[0]
    L.run_dir(*done).mkdir(parents=True)
    (L.run_dir(*done) / "done.json").write_text("{}")
    assert done not in L.pending(wl) and len(L.pending(wl)) == len(todo) - 1
    # a finished arm without mp.pt, or mp.pt without done.json, is not runnable
    (L.pretrain_dir("B") / "mp.pt").unlink()
    assert not L.arm_ready("B")


def test_ft_seed_mapping():
    assert [L.ft_seed("full", s) for s in range(3)] == [42, 117, 709]
    assert [L.ft_seed(16, s) for s in (0, 1, 9)] == [42, 1042, 9042]


def test_shards_partition_worklist_and_mix_arms():
    wl = L.worklist()
    for n in (2, 4, 8):
        parts = [L.shard_items(wl, i, n) for i in range(n)]
        assert set(sum(parts, [])) == set(wl) and sum(len(p) for p in parts) == len(wl)
        for p in parts:
            assert len({a for _, a, _, _ in p}) == len(L.ARMS)  # every shard sees every arm
    # sub-sharding (--jobs-per-gpu J): children of shard i/n partition it exactly
    n, J = 2, 3
    for i in range(n):
        kids = sum((L.shard_items(wl, i + n * j, n * J) for j in range(J)), [])
        assert set(kids) == set(L.shard_items(wl, i, n)) and len(kids) == len(L.shard_items(wl, i, n))

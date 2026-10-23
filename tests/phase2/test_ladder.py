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


POOL_BUILD = {"n_train": 180000, "n_val": 20000, "seed": 0, "wall_s": 26.2}


def _make_arm(arm, patience=15, n_train=180000, pool_build=None, config_json=True):
    d = L.pretrain_dir(arm)
    d.mkdir(parents=True)
    shutil.copy(config.CHEMELEON_MP, d / "mp.pt")
    (d / "done.json").write_text("{}")
    if config_json:
        (d / "config.json").write_text(json.dumps(dict(
            arm=arm, patience=patience, n_train=n_train, pool_build=POOL_BUILD if pool_build is None else pool_build)))


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


def test_run_dir_layout_by_mode_and_steps(ladder_env):
    base = config.RUNS / "phase2"
    t = "tdcommons/ames"
    assert L.run_dir(t, "B", 16, 1) == base / "ladder" / "tdcommons_ames" / "B" / "N16_s1"  # ft layout untouched
    assert L.run_dir(t, "B", 16, 1, "ft", 16) == L.run_dir(t, "B", 16, 1)
    assert L.run_dir(t, "B", 16, 1, mode="frozen") == base / "ladder_frozen" / "tdcommons_ames" / "B" / "N16_s1"
    assert L.run_dir(t, "B", 16, 1, steps=32) == base / "ladder_steps32" / "tdcommons_ames" / "B" / "N16_s1"
    assert L.run_dir(t, "B", 16, 1, "frozen", 32) == base / "ladder_frozen_steps32" / "tdcommons_ames" / "B" / "N16_s1"
    with pytest.raises(ValueError):
        L.run_dir(t, "B", 16, 1, mode="nope")


def test_pending_looks_in_the_run_dir_of_mode_and_steps(ladder_env):
    items = [("tdcommons/ames", "chemeleon", 16, 0), ("tdcommons/ames", "chemeleon", 16, 1)]
    for kw in ({}, dict(mode="frozen"), dict(steps=32)):
        assert L.pending(items, **kw) == items
    d = L.run_dir(*items[0], mode="frozen")
    d.mkdir(parents=True)
    (d / "done.json").write_text("{}")
    assert L.pending(items, mode="frozen") == items[1:] and L.pending(items) == items and L.pending(items, steps=32) == items


def test_run_item_frozen_and_steps_record_and_dirs(ladder_env):
    t = fake_task_real_smiles(n=40, kind="classification")
    L.make_subsamples(t, sizes=(16,), seeds_for={16: 1})
    ft = L.run_item(t.name, "chemeleon", 16, 0, None, task_data=t)
    assert ft["mode"] == "ft" and ft["fixed_steps"] == 16 and ft["n_steps"] == 16
    fr = L.run_item(t.name, "chemeleon", 16, 0, None, task_data=t, mode="frozen", steps=8)
    d = L.run_dir(t.name, "chemeleon", 16, 0, "frozen", 8)
    assert d == config.RUNS / "phase2" / "ladder_frozen_steps8" / "fake_classification_real" / "chemeleon" / "N16_s0"
    run = json.loads((d / "run.json").read_text())
    assert run["mode"] == "frozen" and run["fixed_steps"] == run["n_steps"] == fr["n_steps"] == 8
    assert run["batch_size"] == 16 and run["train_idx"] == ft["train_idx"] and run["subsample_sha1"] == ft["subsample_sha1"]
    assert (d / "pred_es.npy").exists() and not (d / "es_mp.pt").exists()  # frozen: head only, encoder not re-saved
    assert (L.run_dir(t.name, "chemeleon", 16, 0) / "es_mp.pt").exists()  # the ft run stays where it was
    assert json.loads((L.run_dir(t.name, "chemeleon", 16, 0) / "run.json").read_text())["mode"] == "ft"


def test_run_item_passes_mode_and_steps_to_run_one(ladder_env, monkeypatch):
    t = fake_task_real_smiles(n=200, kind="regression")
    L.make_subsamples(t, sizes=(32, 128), seeds_for={32: 1, 128: 1})
    seen = []
    monkeypatch.setattr(L, "run_one", lambda td, regime, mode, seed, d, **kw: seen.append((mode, d, kw)) or {})
    L.run_item(t.name, "chemeleon", 32, 0, None, task_data=t, mode="frozen", steps=40)
    L.run_item(t.name, "chemeleon", 128, 0, None, task_data=t, mode="frozen", steps=40)  # steps only affect N <= 64
    (m1, d1, k1), (m2, d2, k2) = seen
    assert m1 == m2 == "frozen" and k1["fixed_steps"] == 40 and k1["batch_size"] == 16
    assert "fixed_steps" not in k2 and d1.parts[-4] == d2.parts[-4] == "ladder_frozen_steps40"


def test_tailored_arms_follow_normal_readiness(ladder_env):
    assert not L.arm_ready("T100_ames") and not L.arm_ready("T100_ld50")
    _make_arm("T100_ames")
    assert L.arm_ready("T100_ames") and not L.arm_ready("T100_ld50")
    todo = L.pending(L.worklist(arms=("T100_ames", "T100_ld50")))
    assert {a for _, a, _, _ in todo} == {"T100_ames"}


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
    assert len(L.ARMS) == 15 and L.ARMS[:8] == ("B", "T25", "T100", "R25a", "R25b", "R100", "D535", "chemeleon")
    assert L.ARMS[8:] == ("rand", "B_s1", "T100_s1", "T100_ames", "T100_ld50", "B_full", "T100_full")
    assert len(wl) == 2 * 15 * (3 * 10 + 2 * 5 + 3 * 3) and len({*wl}) == len(wl)
    assert {a for _, a, _, _ in wl} == set(L.ARMS) and wl[0][2] == "full"
    assert {n for _, _, n, s in wl if s >= 3} >= {16, 128}
    _make_arm("B")
    with caplog.at_level(logging.WARNING):
        todo = L.pending(wl)
    assert {a for _, a, _, _ in todo} == {"B", "chemeleon"}  # rand has no mp.pt yet
    assert "arm T100: no finished encoder" in caplog.text
    done = todo[0]
    L.run_dir(*done).mkdir(parents=True)
    (L.run_dir(*done) / "done.json").write_text("{}")
    assert done not in L.pending(wl) and len(L.pending(wl)) == len(todo) - 1
    # a finished arm without mp.pt, or mp.pt without done.json, is not runnable
    (L.pretrain_dir("B") / "mp.pt").unlink()
    assert not L.arm_ready("B")


def test_arm_ready_requires_patience_15_full_pool_and_consistent_pool(ladder_env, caplog):
    assert L.arm_ready("chemeleon")  # always, no files needed
    _make_arm("B")
    _make_arm("T25")
    assert L.arm_ready("B") and L.arm_ready("T25")
    assert not L.arm_ready("T100")  # nothing on disk
    # stale patience-5 run (the first two arms were launched with patience 5)
    _make_arm("T100", patience=5)
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("T100")
    assert "T100" in caplog.text and "patience" in caplog.text and "5" in caplog.text
    # smoke-sized run
    caplog.clear()
    _make_arm("R25a", n_train=2000)
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("R25a")
    assert "R25a" in caplog.text and "n_train" in caplog.text and "2000" in caplog.text
    # no config.json at all
    _make_arm("R25b", config_json=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("R25b")
    assert "R25b" in caplog.text and "config.json" in caplog.text


def test_rand_arm_ready_bypasses_pretraining_checks(ladder_env, caplog):
    assert not L.arm_ready("rand")  # nothing on disk
    d = L.pretrain_dir("rand")
    d.mkdir(parents=True)
    shutil.copy(config.CHEMELEON_MP, d / "mp.pt")  # only the encoder: no done.json / config.json / n_train / pool_build
    with caplog.at_level(logging.WARNING):
        assert L.arm_ready("rand")
    assert "rand" not in caplog.text and L.arm_mp_path("rand") == d / "mp.pt"
    # a rand config with patience null / no pool_build must not poison the pool_build consistency check of real arms
    (d / "done.json").write_text("{}")
    (d / "config.json").write_text(json.dumps(dict(arm="rand", patience=None, note="random init, no pretraining")))
    _make_arm("B")
    _make_arm("B_s1")
    assert L.arm_ready("rand") and L.arm_ready("B") and L.arm_ready("B_s1")
    todo = L.pending(L.worklist())
    assert {a for _, a, _, _ in todo} == {"rand", "B", "B_s1", "chemeleon"}
    only = [it for it in L.shard_items(L.worklist(), 0, 3) if it[1] in {"rand", "B_s1", "T100_s1"}]
    assert {a for _, a, _, _ in L.pending(only)} == {"rand", "B_s1"}  # T100_s1 not finished yet
    done = next(it for it in only if it[1] == "rand")
    L.run_dir(*done).mkdir(parents=True)
    (L.run_dir(*done) / "done.json").write_text("{}")
    assert done not in L.pending(only)


def test_second_seed_arms_need_final_protocol(ladder_env):
    _make_arm("B_s1", patience=5)
    _make_arm("T100_s1")
    assert not L.arm_ready("B_s1") and L.arm_ready("T100_s1")


def test_arm_ready_rejects_pool_build_mismatch(ladder_env, caplog):
    _make_arm("B")
    _make_arm("T25", pool_build={**POOL_BUILD, "seed": 1})  # built from a different pool
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("T25") and not L.arm_ready("B")  # inconsistent pair: neither is trusted
    assert "pool_build" in caplog.text
    # re-pretraining the odd arm on the right pool fixes both
    shutil.rmtree(L.pretrain_dir("T25"))
    _make_arm("T25")
    assert L.arm_ready("B") and L.arm_ready("T25")


FULL_BUILD = {"n_train": 847000, "n_val": 94000, "seed": 0, "wall_s": 120.0}


def test_full_pool_arms_ready_within_their_own_group(ladder_env, caplog):
    _make_arm("B")  # 180k arm on POOL_BUILD
    _make_arm("B_full", n_train=847000, pool_build=FULL_BUILD)
    _make_arm("T100_full", n_train=847000, pool_build=FULL_BUILD)
    with caplog.at_level(logging.WARNING):
        # n_train 847000 / a different pool_build than the 180k arms is fine for the _full pair
        assert L.arm_ready("B_full") and L.arm_ready("T100_full") and L.arm_ready("B")
    assert not caplog.text
    # the _full arms must not poison (or be compared with) the 180k arms' consistency check
    _make_arm("T25")
    assert L.arm_ready("T25") and L.arm_ready("B")


def test_full_pool_arms_reject_mismatch_between_them(ladder_env, caplog):
    _make_arm("B_full", n_train=847000, pool_build=FULL_BUILD)
    _make_arm("T100_full", n_train=847000, pool_build={**FULL_BUILD, "seed": 1})
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("B_full") and not L.arm_ready("T100_full")
    assert "pool_build" in caplog.text and "T100_full" in caplog.text
    # differing n_train alone is also an inconsistency
    shutil.rmtree(L.pretrain_dir("T100_full"))
    _make_arm("T100_full", n_train=846000, pool_build=FULL_BUILD)
    assert not L.arm_ready("B_full") and not L.arm_ready("T100_full")
    shutil.rmtree(L.pretrain_dir("T100_full"))
    _make_arm("T100_full", n_train=847000, pool_build=FULL_BUILD)
    assert L.arm_ready("B_full") and L.arm_ready("T100_full")


def test_full_pool_arms_still_need_patience_15_and_pool_build(ladder_env, caplog):
    _make_arm("B_full", patience=5, n_train=847000, pool_build=FULL_BUILD)
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("B_full")
    assert "patience" in caplog.text
    shutil.rmtree(L.pretrain_dir("B_full"))
    _make_arm("B_full", n_train=847000)
    (L.pretrain_dir("B_full") / "config.json").write_text(json.dumps(dict(arm="B_full", patience=15, n_train=847000)))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert not L.arm_ready("B_full")
    assert "pool_build" in caplog.text
    # 180k arms keep the n_train rule even when a _full arm exists
    _make_arm("T100_full", n_train=847000, pool_build=FULL_BUILD)
    _make_arm("B", n_train=847000)
    assert not L.arm_ready("B")


def test_ft_seed_mapping():
    assert [L.ft_seed("full", s) for s in range(3)] == [42, 117, 709]
    assert [L.ft_seed(16, s) for s in (0, 1, 9)] == [42, 1042, 9042]


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
def test_shards_partition_worklist_and_mix_arms(n):
    wl = L.worklist()
    parts = [L.shard_items(wl, i, n) for i in range(n)]
    flat = sum(parts, [])
    assert len(flat) == len(wl) and set(flat) == set(wl)  # every item in exactly one shard
    for p in parts:
        assert len(p) >= len(L.ARMS) and len({a for _, a, _, _ in p}) == len(L.ARMS)  # every shard sees every arm
        # ... and every task and size too
        assert {t for t, _, _, _ in p} == set(L.TASKS) and {x for _, _, x, _ in p} == set(L.SIZES)
    assert max(map(len, parts)) - min(map(len, parts)) <= len(wl) // n * 0.05 + 1  # balanced


@pytest.mark.parametrize("n, J", [(1, 2), (1, 3), (2, 3), (3, 2), (2, 4), (4, 2), (1, 8), (3, 3), (1, 6)])
def test_subsharding_children_partition_parent(n, J):
    """--jobs-per-gpu J: shard i/n -> children (i + n*j)/(n*J); they partition the parent and each child mixes arms."""
    wl = L.worklist()
    for i in range(n):
        parent = L.shard_items(wl, i, n)
        kids = [L.shard_items(wl, i + n * j, n * J) for j in range(J)]
        flat = sum(kids, [])
        assert len(flat) == len(parent) and set(flat) == set(parent)
        for k in kids:
            assert {a for _, a, _, _ in k} == set(L.ARMS)


def test_first_eight_arms_keep_worklist_order():
    """New arms are appended: within each (task, N, seed) block the first 8 arms keep their relative order."""
    wl = L.worklist()
    assert [a for _, a, _, _ in wl[:len(L.ARMS)]] == list(L.ARMS)
    assert L.worklist(arms=L.ARMS[:8])[:8] == wl[:8]


def test_shard_key_does_not_pin_arms_for_6_shards():
    """The old key (k + k//8) % n pinned arms to children whenever n shared a factor with 9 (n = 3, 6)."""
    wl = L.worklist()
    for i in range(6):
        per_arm = {a: 0 for a in L.ARMS}
        for _, a, _, _ in L.shard_items(wl, i, 6):
            per_arm[a] += 1
        assert min(per_arm.values()) > 0.5 * len(wl) / 6 / len(L.ARMS)

import astartes
import numpy as np
from conftest import fake_task, fake_task_real_smiles, write_fake_cache

from drift import config
from drift.tasks import load_task, load_task_from, make_split


def test_full_split_matches_paper():
    t = fake_task(n=500, kind="regression")
    tr, va = make_split(t, "full", 42)
    tr2, va2 = astartes.train_test_split(np.arange(500), train_size=0.8, test_size=0.2, random_state=42)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)


def test_n128_sizes_and_determinism():
    t = fake_task(n=500, kind="regression")
    tr, va = make_split(t, "n128", 117)
    assert len(tr) + len(va) == 128 and len(set(tr) & set(va)) == 0
    assert np.array_equal(tr, make_split(t, "n128", 117)[0])


def test_n128_small_task_uses_all():
    t = fake_task(n=100, kind="regression")
    tr, va = make_split(t, "n128", 42)
    assert len(tr) + len(va) == 100


def test_n128_rare_class_has_both_classes():
    t = fake_task(n=600, kind="classification", pos_frac=0.08)
    for s in (42, 117, 709):
        tr, va = make_split(t, "n128", s)
        assert set(t.y_train[tr]) == {0, 1} and set(t.y_train[va]) == {0, 1}


def test_load_task_offline(tmp_path, monkeypatch):
    write_fake_cache(tmp_path / "polaris_x")
    monkeypatch.setattr("polaris.load_benchmark", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
    t = load_task_from(tmp_path / "polaris_x")
    assert t.task_type in ("regression", "classification")
    assert len(t.train_smiles) == 50 and len(t.test_smiles) > 0


def test_load_task_uses_config_data_and_fills_nan(tmp_path, monkeypatch):
    write_fake_cache(tmp_path / "tasks" / config.slug("polaris/foo-1"))
    import pandas as pd

    p = tmp_path / "tasks" / config.slug("polaris/foo-1") / "train.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "y"] = np.nan
    df.to_parquet(p)
    monkeypatch.setattr(config, "DATA", tmp_path)
    t = load_task("polaris/foo-1")
    assert np.isnan(t.y_raw[0]) and t.y_train[0] == np.nanmean(t.y_raw)


def test_real_smiles_helper_valid():
    from rdkit import Chem

    t = fake_task_real_smiles(200, "classification")
    assert len(t.train_smiles) == 200
    assert all(Chem.MolFromSmiles(s) is not None for s in t.train_smiles + t.test_smiles)

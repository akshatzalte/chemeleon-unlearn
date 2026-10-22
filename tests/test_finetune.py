import numpy as np
import torch
from conftest import fake_task_real_smiles

from drift import config
from drift.config import BENCHMARKS, CHEMELEON_MP
from drift.encoders import load_mp
from drift.finetune import PatienceBestCheckpoint, pending, run_dir, run_one, worklist


def test_patience_callback_logic():
    cb = PatienceBestCheckpoint(patience=5)
    vals = [1.0, 0.8, 0.9, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.5, 0.4]
    for e, v in enumerate(vals):
        cb._update(e, v, state={"w": torch.tensor(float(e))})
    assert cb.best_epoch == 3 and cb.stopped and cb.es_state["w"].item() == 3.0


def test_frozen_keeps_encoder(tmp_path):
    t = fake_task_real_smiles(n=60, kind="regression")
    run_one(t, "full", "frozen", 42, tmp_path, accelerator="cpu", max_epochs=2)
    torch.load(tmp_path / "es_head.pt")
    assert not (tmp_path / "es_mp.pt").exists() and not (tmp_path / "last_mp.pt").exists()
    assert (tmp_path / "done.json").exists()
    assert len(np.load(tmp_path / "pred_last.npy")) == len(t.test_smiles)


def test_ft_moves_encoder_and_writes_preds(tmp_path):
    t = fake_task_real_smiles(n=60, kind="classification")
    info = run_one(t, "full", "ft", 42, tmp_path, accelerator="cpu", max_epochs=2)
    assert np.load(tmp_path / "pred_es.npy").shape == (len(t.test_smiles),)
    p = np.load(tmp_path / "pred_last.npy")
    assert ((p >= 0) & (p <= 1)).all()
    mp0, mp1 = load_mp(CHEMELEON_MP), load_mp(tmp_path / "last_mp.pt")
    assert not torch.equal(mp0.state_dict()["W_h.weight"], mp1.state_dict()["W_h.weight"])
    load_mp(tmp_path / "es_mp.pt")
    assert sorted(info["train_idx"] + info["val_idx"]) == list(range(60))
    rows = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert rows[0] == "epoch,train_loss,val_loss" and len(rows) == 3
    assert 0 <= info["es_epoch"] < 2


def test_n128_subsample_and_nan_refill(tmp_path):
    t = fake_task_real_smiles(n=200, kind="regression")
    t.y_raw[:20] = np.nan
    info = run_one(t, "n128", "ft", 42, tmp_path, accelerator="cpu", max_epochs=1)
    assert info["n_train_rows"] + info["n_val_rows"] == 128


def test_worklist_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS", tmp_path)
    n = {b: 1000 + i for i, b in enumerate(BENCHMARKS)}
    n[BENCHMARKS[0]] = 100  # small: no n128
    wl = worklist(n)
    assert len(wl) == 28 * 12 - 6
    assert wl[0][0] == BENCHMARKS[-1]
    assert not any(t == BENCHMARKS[0] and r == "n128" for t, r, _, _ in wl)
    items = wl[:2]
    for it in items:
        run_dir(*it).mkdir(parents=True)
    (run_dir(*items[0]) / "done.json").write_text("{}")
    assert pending(items) == [items[1]]

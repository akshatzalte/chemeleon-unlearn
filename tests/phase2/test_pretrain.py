import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from chemprop.nn import BondMessagePassing

pytest.importorskip("cuik_molmaker")

from conftest import REAL_SMILES  # noqa: E402
from drift.phase2 import pretrain as PT  # noqa: E402
from drift.phase2 import pretrain_data as PD  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning:lightning"),
              pytest.mark.filterwarnings("ignore:.*treespec.*")]
PY1 = "/home/akshatz/anaconda3/envs/chemeleon_unlearn/bin/python"
CHEMELEON_MP = Path.home() / ".chemprop/chemeleon_mp.pt"
N_COLS = 1613
NAMES = ["c900", "c5", "c1200", "c100", "c7"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    pool, sets = root / "pool", root / "sets"
    pool.mkdir(), sets.mkdir()
    rng = np.random.default_rng(0)
    train, val = REAL_SMILES[:64], REAL_SMILES[64:83]
    (pool / "smiles_train.txt").write_text("\n".join(train) + "\n")
    (pool / "smiles_val.txt").write_text("\n".join(val) + "\n")
    dtr = rng.normal(size=(64, N_COLS)).astype(np.float32)
    dva = rng.normal(size=(19, N_COLS)).astype(np.float32)
    dtr[::7, 5] = np.nan
    dva[::5, 100] = np.nan
    np.save(pool / "desc_train.npy", dtr)
    np.save(pool / "desc_val.npy", dva)
    (pool / "columns.json").write_text(json.dumps([f"c{i}" for i in range(N_COLS)]))
    np.savez(pool / "stats.npz", mean=np.zeros(N_COLS), sd=np.ones(N_COLS))
    (pool / "build.json").write_text(json.dumps({"n_train": 64, "n_val": 19, "seed": 0}))
    (sets / "X.txt").write_text("\n".join(NAMES) + "\n")
    out = root / "out"
    info = PT.train("X", pool, sets, out, gpu=-1, max_epochs=1, num_workers=0)
    return dict(pool=pool, sets=sets, out=out, info=info, dva=dva, val=val)


def test_build_model_structure():
    m = PT.build_model(5)
    ffn = m.predictor.ffn
    assert [type(x).__name__ for x in ffn[0]] == ["Linear"] and ffn[0][0].weight.shape == (2048, 2048)
    assert ffn[1][-1].weight.shape == (5, 2048)  # Linear 2048 -> k after act (relu, dropout 0)
    mp = m.message_passing
    assert (mp.W_i.weight.shape, mp.W_h.weight.shape, mp.W_o.weight.shape) == (
        (2048, 86), (2048, 2048), (2048, 2120))
    assert isinstance(m.criterion, PD.MaskedDropoutMSE) and m.criterion.keep_prob == 0.7
    assert (m.init_lr, m.max_lr, m.final_lr, m.warmup_epochs) == (1e-4, 1e-3, 1e-4, 5)


@pytest.mark.skipif(not CHEMELEON_MP.exists(), reason="no chemeleon_mp.pt")
def test_hyperparameters_equal_public_checkpoint():
    pub = torch.load(CHEMELEON_MP, weights_only=True)["hyper_parameters"]
    assert PT.MP_HPARAMS == pub
    pub_sd = torch.load(CHEMELEON_MP, weights_only=True)["state_dict"]
    ours = BondMessagePassing(**PT.MP_HPARAMS).state_dict()
    assert {k: v.shape for k, v in ours.items()} == {k: v.shape for k, v in pub_sd.items()}


def test_train_outputs(tiny):
    out, info = tiny["out"], tiny["info"]
    for f in ["mp.pt", "head.pt", "metrics.csv", "config.json", "done.json"]:
        assert (out / f).exists(), f
    done_t = (out / "done.json").stat().st_mtime_ns
    assert all((out / f).stat().st_mtime_ns <= done_t for f in ["mp.pt", "head.pt", "metrics.csv", "config.json"])
    m = pd.read_csv(out / "metrics.csv")
    assert list(m.columns) == ["epoch", "train_loss", "val_loss", "seconds"] and len(m) == 1
    assert np.isfinite(m.val_loss[0]) and np.isfinite(m.train_loss[0]) and m.seconds[0] > 0
    assert info["epochs_run"] == 1 and info["s_per_epoch"] > 0
    cfg = json.loads((out / "config.json").read_text())
    for k in ["arm", "k", "batch_size", "keep_prob", "patience", "max_epochs", "init_lr", "max_lr", "final_lr",
              "warmup_epochs", "mp_hparams", "pool_build", "git_sha", "torch", "chemprop", "target_names"]:
        assert k in cfg, k
    assert cfg["k"] == 5 and cfg["target_names"] == NAMES and cfg["pool_build"]["n_train"] == 64
    head = torch.load(out / "head.pt", weights_only=True)
    assert head["ffn.0.0.weight"].shape == (2048, 2048) and head["ffn.1.2.weight"].shape == (5, 2048)


def test_val_loss_is_plain_masked_mse_of_saved_model(tiny):
    """val_loss logged by Lightning == plain masked MSE (no target dropout) of the saved best weights."""
    out = tiny["out"]
    ck = torch.load(out / "mp.pt", weights_only=True)
    model = PT.build_model(5)
    model.message_passing.load_state_dict(ck["state_dict"])
    model.predictor.load_state_dict(torch.load(out / "head.pt", weights_only=True))
    model.eval()
    dva = tiny["dva"]
    ds = PD.PoolDataset(tiny["val"], dva, PD.col_indices(NAMES, [f"c{i}" for i in range(N_COLS)]),
                        np.zeros(N_COLS), np.ones(N_COLS), batch_size=512)
    b = ds[0]
    with torch.no_grad():
        pred = model(b.bmg)
    mask = torch.isfinite(b.Y)
    expected = float(((pred - b.Y.nan_to_num(0)) ** 2)[mask].sum() / mask.sum())
    got = pd.read_csv(out / "metrics.csv").val_loss[0]
    assert got == pytest.approx(expected, rel=1e-4)


def test_checkpoint_loadable_by_phase1(tiny):
    out = tiny["out"]
    ck = torch.load(out / "mp.pt", weights_only=True)
    assert {"hyper_parameters", "state_dict", "target_names", "arm"} <= set(ck)
    assert ck["target_names"] == NAMES and ck["arm"] == "X"
    if CHEMELEON_MP.exists():
        assert ck["hyper_parameters"] == torch.load(CHEMELEON_MP, weights_only=True)["hyper_parameters"]
    code = (
        "import sys, torch, numpy as np\n"
        "from drift.encoders import load_mp, embed\n"
        "from chemprop.nn import BondMessagePassing\n"
        f"p = '{out / 'mp.pt'}'\n"
        "ck = torch.load(p, weights_only=True)\n"
        "BondMessagePassing(**ck['hyper_parameters']).load_state_dict(ck['state_dict'])\n"
        "mp = load_mp(p)\n"
        "e = embed(mp, ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN'], device='cpu')\n"
        "assert e.shape == (4, 2048) and np.isfinite(e).all()\n"
        "print('OK')\n"
    )
    r = subprocess.run([PY1, "-c", code], capture_output=True, text=True, env={"OMP_NUM_THREADS": "8", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]


def test_defaults_patience_15_seed_0_recorded_in_config(tiny):
    """The queue passes --patience 15; the function and script defaults must agree with it, and the seed is recorded."""
    import importlib.util
    import inspect

    sig = inspect.signature(PT.train).parameters
    assert sig["patience"].default == 15 and sig["seed"].default == 0
    spec = importlib.util.spec_from_file_location("p2_02", Path(__file__).parents[2] / "scripts" / "p2_02_pretrain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod.parse_args(["--arm", "B", "--gpu", "0"])
    assert a.patience == 15 and a.seed == 0
    assert mod.parse_args(["--arm", "B", "--gpu", "0", "--seed", "3"]).seed == 3
    cfg = json.loads((tiny["out"] / "config.json").read_text())
    assert cfg["patience"] == 15 and cfg["seed"] == 0


def test_seed_changes_permutation_and_is_recorded(tiny, tmp_path):
    out = tmp_path / "s7"
    PT.train("X", tiny["pool"], tiny["sets"], out, gpu=-1, max_epochs=1, num_workers=0, seed=7)
    assert json.loads((out / "config.json").read_text())["seed"] == 7
    cols = np.arange(5)
    mk = lambda seed: PD.PoolDataset(REAL_SMILES[:64], np.zeros((64, 5), np.float32), cols, np.zeros(5), np.ones(5),
                                     16, True, seed)
    assert not np.array_equal(mk(0).batch_rows(0), mk(7).batch_rows(0))

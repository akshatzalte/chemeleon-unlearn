import numpy as np
import pytest
import torch

pytest.importorskip("cuik_molmaker")

from conftest import REAL_SMILES  # noqa: E402
from drift.phase2 import pretrain_data as PD  # noqa: E402

N_COLS = 1613
SMILES = REAL_SMILES[:20]


@pytest.fixture
def pool(tmp_path):
    rng = np.random.default_rng(0)
    n = len(SMILES)
    desc = rng.normal(5.0, 3.0, size=(n, N_COLS)).astype(np.float32)
    mean = desc.mean(0).astype(np.float64)
    sd = desc.std(0).astype(np.float64)
    desc[2, 100] = np.nan
    desc[3, 5] = 1e4  # far outside +-6 sd -> clipped
    np.save(tmp_path / "desc.npy", desc)
    memmap = np.load(tmp_path / "desc.npy", mmap_mode="r")
    columns = [f"c{i}" for i in range(N_COLS)]
    return memmap, columns, mean, sd


def test_batch_shapes_and_name_selection(pool):
    memmap, columns, mean, sd = pool
    names = ["c900", "c5", "c1200", "c100", "c7"]  # non-contiguous, not sorted
    col_idx = PD.col_indices(names, columns)
    assert col_idx.tolist() == [900, 5, 1200, 100, 7]
    ds = PD.PoolDataset(SMILES, memmap, col_idx, mean, sd, batch_size=8, shuffle=False)
    assert len(ds) == 3  # 8 + 8 + 4
    b = ds[0]
    assert b.bmg.V.shape[1] == 72 and b.bmg.E.shape[1] == 14
    assert len(b.bmg) == 8
    assert b.Y.shape == (8, 5) and b.Y.dtype == torch.float32
    assert b.w.shape == (8, 1) and torch.all(b.w == 1)
    assert b.V_d is None and b.X_d is None and b.lt_mask is None and b.gt_mask is None
    expected = np.clip((np.asarray(memmap[:8])[:, col_idx] - mean[col_idx]) / sd[col_idx], -6, 6)
    np.testing.assert_allclose(b.Y.numpy(), expected, atol=1e-5)
    assert len(ds[2].bmg) == 4 and ds[2].Y.shape == (4, 5)


def test_nan_preserved_and_clipped(pool):
    memmap, columns, mean, sd = pool
    col_idx = PD.col_indices(["c100", "c5"], columns)
    ds = PD.PoolDataset(SMILES, memmap, col_idx, mean, sd, batch_size=20)
    Y = ds[0].Y
    assert torch.isnan(Y[2, 0]) and torch.isfinite(Y[2, 1])
    assert int(torch.isnan(Y).sum()) == 1
    assert Y[3, 1] == 6.0 and Y.abs().nan_to_num(0).max() <= 6.0


def test_unknown_name_raises(pool):
    _, columns, _, _ = pool
    with pytest.raises(AssertionError):
        PD.col_indices(["c1", "not_a_descriptor"], columns)


def test_shuffle_reshuffles_membership_each_epoch(pool):
    memmap, columns, mean, sd = pool
    col_idx = PD.col_indices(["c1"], columns)
    ds = PD.PoolDataset(SMILES, memmap, col_idx, mean, sd, batch_size=8, shuffle=True, seed=3)
    n_b = len(ds)

    def epoch_rows(e):
        return [ds.batch_rows(e * n_b + j).tolist() for j in range(n_b)]

    e0, e1 = epoch_rows(0), epoch_rows(1)
    assert sorted(sum(e0, [])) == list(range(20)) and sorted(sum(e1, [])) == list(range(20))
    assert e0 != e1 and epoch_rows(0) == e0  # deterministic in (seed, epoch)
    # SMILES and targets of an item belong to the same rows
    b = ds[1]
    rows = ds.batch_rows(1)
    expected = np.clip((np.asarray(memmap)[rows][:, col_idx] - mean[col_idx]) / sd[col_idx], -6, 6)
    np.testing.assert_allclose(b.Y.numpy(), expected, atol=1e-5)
    # no shuffle: fixed sequential batches
    ds2 = PD.PoolDataset(SMILES, memmap, col_idx, mean, sd, batch_size=8, shuffle=False)
    assert ds2.batch_rows(0).tolist() == list(range(8)) and ds2.batch_rows(n_b).tolist() == list(range(8))


def test_epoch_sampler_advances_epoch(pool):
    memmap, columns, mean, sd = pool
    col_idx = PD.col_indices(["c1"], columns)
    ds = PD.PoolDataset(SMILES, memmap, col_idx, mean, sd, batch_size=8, shuffle=True)
    s = PD.EpochSampler(ds)
    assert len(s) == 3
    assert list(s) == [0, 1, 2] and list(s) == [3, 4, 5]


def test_invalid_smiles_rejected(pool):
    memmap, columns, mean, sd = pool
    col_idx = PD.col_indices(["c1"], columns)
    bad = list(SMILES)
    bad[1] = "this is not smiles"
    ds = PD.PoolDataset(bad, memmap, col_idx, mean, sd, batch_size=8)
    with pytest.raises(Exception):
        ds[0]


def test_masked_loss_ignores_nan_and_drops():
    loss = PD.MaskedDropoutMSE(keep_prob=1.0)
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[0.0, float("nan")], [1.0, 1.0]])
    mask = targets.isfinite()
    val = loss(preds, targets.nan_to_num(0.0), mask, torch.ones(2, 1), None, None)
    plain = ((1 - 0) ** 2 + (3 - 1) ** 2 + (4 - 1) ** 2) / 3  # NaN cell contributes nothing, count is 3
    assert torch.isclose(val, torch.tensor(plain))
    # NaN targets handled even when the caller does not replace them
    loss2 = PD.MaskedDropoutMSE(keep_prob=1.0)
    val2 = loss2(preds, targets, None, torch.ones(2, 1), None, None)
    assert torch.isclose(val2, torch.tensor(plain))

    # dropout: kept count is the denominator, ~keep_prob of finite cells kept, loss gradient only through kept
    torch.manual_seed(0)
    big_p = torch.randn(4000, 10, requires_grad=True)
    big_t = torch.randn(4000, 10)
    drop = PD.MaskedDropoutMSE(keep_prob=0.7)
    drop.train()
    v = drop(big_p, big_t, None, torch.ones(4000, 1), None, None)
    v.backward()
    kept = (big_p.grad != 0).float().mean().item()
    assert 0.68 < kept < 0.72
    # unbiased: close to the plain MSE of the same cells
    assert abs(float(v.detach()) - float(((big_p - big_t) ** 2).mean().detach())) < 0.1


def test_dropout_only_when_training():
    loss = PD.MaskedDropoutMSE(keep_prob=0.7)
    loss.eval()
    p, t = torch.randn(500, 6), torch.randn(500, 6)
    v = loss(p, t, None, torch.ones(500, 1), None, None)
    assert torch.isclose(v, ((p - t) ** 2).mean(), rtol=1e-5)


def test_registered_under_new_name():
    from chemprop.nn.metrics import LossFunctionRegistry, MetricRegistry

    assert LossFunctionRegistry["masked_dropout_mse"] is PD.MaskedDropoutMSE
    assert MetricRegistry["masked_dropout_mse"] is PD.MaskedDropoutMSE

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("rdkit")

from drift.descriptors import descriptor_modules  # noqa: E402
from drift.phase2 import pool  # noqa: E402
from drift.pubchem import canonical  # noqa: E402

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O", "CN1CCC[C@H]1c1cccnc1", "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "c1ccccc1", "CCO", "CCCC", "CCN", "CCC(=O)O", "c1ccncc1", "CC(C)O", "NCCN", "OCCO", "CCCl", "CCBr",
    "C[C@H](N)C(=O)O", "CCOC(C)=O",
]
N_ROWS = 300


@pytest.fixture
def smiles_file(tmp_path):
    lines = [SMILES[i % len(SMILES)] for i in range(N_ROWS)]
    lines[7] = "this is not smiles"  # RDKit rejects
    p = tmp_path / "s.smiles"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_exclusion_by_canonical_smiles(smiles_file):
    # (S)-nicotine in the file; exclude its (R)/flat variant -> same non-isomeric canonical SMILES
    exclude = {canonical("CN1CCC[C@@H]1c1cccnc1")}
    assert canonical("CN1CCC[C@H]1c1cccnc1") in exclude
    rows, info = pool.sample_rows(smiles_file, n=10_000, seed=0, exclude=exclude)
    lines = smiles_file.read_text().splitlines()
    assert all(canonical(lines[i]) not in exclude for i in rows)
    assert info["n_excluded_test"] == sum(canonical(l) in exclude for l in lines)
    assert info["n_excluded_test"] > 0 and info["n_invalid"] == 1
    assert 7 not in rows
    assert len(rows) == N_ROWS - 1 - info["n_excluded_test"]
    assert set(pool.sample_pool(smiles_file, n=10_000, exclude=exclude)) == {l for i, l in enumerate(lines) if i in rows}


def test_sample_deterministic(smiles_file):
    a = pool.sample_pool(smiles_file, n=50, seed=3, exclude=set())
    b = pool.sample_pool(smiles_file, n=50, seed=3, exclude=set())
    c = pool.sample_pool(smiles_file, n=50, seed=4, exclude=set())
    assert a == b and a != c and len(a) == 50
    r1, _ = pool.sample_rows(smiles_file, n=50, seed=3)
    assert list(r1) == sorted(r1) and len(set(r1)) == 50
    tr, va = pool.split(list(range(100)), frac_val=0.1, seed=0)
    tr2, va2 = pool.split(list(range(100)), frac_val=0.1, seed=0)
    assert (tr, va) == (tr2, va2) and len(va) == 10 and len(tr) == 90 and not set(tr) & set(va)


def test_build_small(tmp_path, smiles_file):
    names = list(descriptor_modules().index)
    assert len(names) == 1613
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=(N_ROWS, 1613)).astype(np.float32)
    x[rng.random(x.shape) < 0.1] = np.nan
    x[:, 3] = 7.0  # constant column -> sd 1
    x[:, 4] = np.nan  # all-NaN column -> mean 0, sd 1
    allnan = [20, 33, 150]  # 33 in train; 20, 150 in val
    x[allnan] = np.nan
    val_rows = np.array([5, 20, 60, 100, 150, 200, 250])
    train_rows = np.setdiff1d(np.arange(N_ROWS), val_rows)
    z = zarr.open_array(str(tmp_path / "d.zarr"), mode="w", shape=x.shape, chunks=(16, 1613), dtype="float32")
    z[:] = x

    out = tmp_path / "pool"
    info = pool.build(out, tmp_path / "d.zarr", smiles_file, train_rows[::-1], val_rows)  # unsorted input ok
    rows = json.loads((out / "rows.json").read_text())
    keep_tr = [r for r in train_rows if r not in allnan]
    keep_va = [r for r in val_rows if r not in allnan]
    assert rows == {"train": keep_tr, "val": keep_va}
    assert info["n_train"] == len(keep_tr) and info["n_val"] == len(keep_va)
    assert info["n_dropped_allnan_train"] == 1 and info["n_dropped_allnan_val"] == 2
    assert info["zarr_shape"] == [N_ROWS, 1613]
    assert json.loads((out / "build.json").read_text())["n_train"] == len(keep_tr)

    dtr = np.load(out / "desc_train.npy", mmap_mode="r")
    dva = np.load(out / "desc_val.npy", mmap_mode="r")
    assert dtr.shape == (len(keep_tr), 1613) and dva.shape == (len(keep_va), 1613) and dtr.dtype == np.float32
    np.testing.assert_array_equal(dtr, x[keep_tr])  # NaN preserved (NaNs compare equal here)
    assert json.loads((out / "columns.json").read_text()) == names

    st = np.load(out / "stats.npz")
    np.testing.assert_allclose(st["mean"][10], np.nanmean(x[keep_tr, 10].astype(np.float64)), rtol=1e-6)
    np.testing.assert_allclose(st["sd"][10], np.nanstd(x[keep_tr, 10].astype(np.float64)), rtol=1e-6)
    # stats come from train rows only: differ from stats over train+val
    all_rows = keep_tr + keep_va
    assert abs(st["mean"][10] - np.nanmean(x[all_rows, 10].astype(np.float64))) > 1e-6
    assert st["sd"][3] == 1.0 and st["mean"][3] == 7.0
    assert st["mean"][4] == 0.0 and st["sd"][4] == 1.0
    assert (out / "smiles_train.txt").read_text().splitlines() == [smiles_file.read_text().splitlines()[i] for i in keep_tr]


def test_build_rejects_misaligned_zarr(tmp_path, smiles_file):
    z = zarr.open_array(str(tmp_path / "d.zarr"), mode="w", shape=(N_ROWS - 1, 1613), chunks=(16, 1613), dtype="float32")
    z[:] = 0
    with pytest.raises(AssertionError):
        pool.build(tmp_path / "p", tmp_path / "d.zarr", smiles_file, [0, 1], [2])


def test_check_alignment_detects_mismatch(tmp_path):
    import pandas as pd

    lines = [f"C{'C' * i}" for i in range(600)]
    sf = tmp_path / "all.smiles"
    sf.write_text("\n".join(lines) + "\n")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 1613)).astype(np.float32)
    x[:, 5] = np.nan
    zp = tmp_path / "d.zarr"
    z = zarr.open_array(str(zp), mode="w", shape=x.shape, chunks=(16, 1613), dtype="float32")
    z[:] = x
    pick = rng.permutation(600)[:520]
    rf = tmp_path / "ref.smiles"
    rf.write_text("\n".join(lines[i] for i in pick) + "\n")
    pq = tmp_path / "ref.parquet"
    pd.DataFrame(x[pick]).to_parquet(pq)
    res = pool.check_alignment(zp, sf, rf, pq)
    assert res["n_compared"] == 520 and res["row_match_rate"] == 1.0 and res["max_abs_diff"] == 0.0
    assert res["mismatch_cols"] == {} and res["nan_row_match_rate"] == 1.0
    assert pool.alignment_verdict(res) is None
    # a few columns with value mismatches (NaN pattern intact) are reported and tolerated
    y = x[pick].copy()
    y[:100, 7] += 1.0
    y[:, 9] += 1.0
    pd.DataFrame(y, columns=[f"c{j}" for j in range(1613)]).to_parquet(pq)
    res = pool.check_alignment(zp, sf, rf, pq)
    assert set(res["mismatch_cols"]) == {"c7", "c9"} and res["mismatch_cols"]["c9"] == 1.0
    assert res["nan_row_match_rate"] == 1.0
    assert "c9" in pool.alignment_verdict(res)  # c9 mismatches in 100% of rows > 50%
    assert pool.alignment_verdict({**res, "mismatch_cols": {"c7": 0.2}}) is None
    assert pool.alignment_verdict({**res, "mismatch_cols": {f"c{j}": 0.01 for j in range(101)}}) is not None
    pd.DataFrame(x[np.roll(pick, 1)]).to_parquet(pq)  # shifted rows -> not aligned
    res = pool.check_alignment(zp, sf, rf, pq)
    assert res["row_match_rate"] < 0.05 and pool.alignment_verdict(res) is not None

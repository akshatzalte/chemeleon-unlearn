import numpy as np
import pandas as pd

from drift.descriptors import compute_mordred, descriptor_modules, prepare_targets


def test_mordred_columns_and_bad_smiles():
    D = compute_mordred(["CCO", "c1ccccc1", "not_a_smiles", "[Na+].[Cl-]"], nproc=2)
    assert D.shape == (4, 1613) and D.iloc[2].isna().all()
    assert not D.iloc[0].isna().all()


def test_descriptor_modules_align_with_columns():
    D = compute_mordred(["CCO"], nproc=1)
    m = descriptor_modules()
    assert len(m) == 1613 and list(m.index) == list(D.columns)


def test_prepare_targets():
    rng = np.random.default_rng(0)
    D = pd.DataFrame({"a": rng.normal(5, 2, 100), "const": np.ones(100),
                      "nanny": np.where(rng.random(100) < 0.5, np.nan, 1.0), "out": np.r_[rng.normal(0, 1, 99), 1e6]})
    T = prepare_targets(D, np.arange(80))
    assert set(T.dropped) == {"const", "nanny"}
    a = T.Y[:80, T.names.index("a")]
    assert abs(a.mean()) < 1e-5 and abs(a.std() - 1) < 1e-2
    assert np.nanmax(T.Y[:, T.names.index("out")]) <= 6.0
    assert T.Y.dtype == np.float32


def test_prepare_targets_all_nan_column_dropped():
    D = pd.DataFrame({"a": np.arange(10.0), "z": np.full(10, np.nan)})
    T = prepare_targets(D, np.arange(10))
    assert T.names == ["a"] and T.dropped == ["z"]

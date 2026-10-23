import numpy as np
import pandas as pd
import pytest

from drift import config
from drift.descriptors import descriptor_modules
from drift.phase2 import sets as S

SUMMARY = config.RESULTS / "summary"
SCORES = SUMMARY / "phase2_relevance_scores.csv"
CLUSTERS = config.RESULTS / "corr_clusters.parquet"
SETS_JSON = SUMMARY / "phase2_descriptor_sets.json"


@pytest.fixture(scope="module")
def all_descriptors():
    return list(descriptor_modules().index)


@pytest.fixture(scope="module")
def built(all_descriptors):
    return S.build_sets(SCORES, CLUSTERS, SETS_JSON, all_descriptors)


def test_T_sets_nested_and_sized(built):
    t25, t100, d535 = built["T25"], built["T100"], set(built["D535"])
    assert len(t25) == 25 and len(t100) == 100
    assert len(set(t25)) == 25 and len(set(t100)) == 100
    assert set(t25) < set(t100)
    assert set(t100) <= d535


def test_T_follows_consensus_rank(built):
    rep = S.representatives(SETS_JSON, CLUSTERS)
    rk = pd.read_csv(SCORES, index_col=0).rank_consensus
    top = rk.nlargest(25).index
    assert built["T25"] == [rep[c] for c in top]


def test_R_sets_seeded(built, all_descriptors):
    assert len(built["R25a"]) == len(built["R25b"]) == 25 and len(built["R100"]) == 100
    assert built["R25a"] != built["R25b"]
    assert set(built["R25a"]) <= set(built["D535"]) and set(built["R100"]) <= set(built["D535"])
    again = S.build_sets(SCORES, CLUSTERS, SETS_JSON, all_descriptors)
    for k in ("R25a", "R25b", "R100"):
        assert again[k] == built[k]


def test_B_is_all_1613_in_module_order(built, all_descriptors):
    assert len(all_descriptors) == 1613
    assert built["B"] == all_descriptors


def test_representatives_one_per_cluster():
    rep = S.representatives(SETS_JSON, CLUSTERS)
    cl = pd.read_parquet(CLUSTERS).set_index("descriptor").cluster
    assert len(rep) == 535
    assert all(cl[d] == c for c, d in rep.items())


def _synthetic(n=60, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"d{i}" for i in range(p)])
    return X


def test_sanity_check_shape(monkeypatch):
    X = _synthetic()
    y_reg = X["d0"].to_numpy() + 0.1 * np.random.default_rng(1).normal(size=len(X))
    y_cls = (X["d1"].to_numpy() > 0).astype(float)
    data = {"t/cls": (X, y_cls, "classification"), "t/reg": (X, y_reg, "regression")}
    monkeypatch.setattr(S, "_load_xy", lambda task, cols: (data[task][0][cols], *data[task][1:]))
    sets = {"A": ["d0", "d1", "d2"], "B": ["d0", "d1", "d2", "d3", "d4"]}
    df = S.sanity_check(sets, tasks=("t/cls", "t/reg"), model="linear")
    assert list(df.columns) == ["task", "set", "n_descr", "score", "sd"]
    assert len(df) == 4
    assert set(df.n_descr) == {3, 5}
    assert df.score.notna().all() and (df.sd >= 0).all()
    # informative descriptors are included, so the scores are well above chance
    assert df[df.task == "cls"].score.min() > 0.8
    assert df[df.task == "reg"].score.min() > 0.8

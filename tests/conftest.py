"""Shared test helpers. Import in tests with `from conftest import fake_task, ...`
(pytest puts tests/ on sys.path)."""
import json

import numpy as np
import pandas as pd

from drift.tasks import TaskData

REAL_SMILES = [
    'CC(=O)Oc1ccccc1C(=O)O',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'CC(=O)Nc1ccc(O)cc1',
    'OC(=O)c1ccccc1O',
    'CN1CCC[C@H]1c1cccnc1',
    'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
    'CC(C)NCC(O)COc1cccc2ccccc12',
    'COc1ccc2[nH]cc(CCNC(C)=O)c2c1',
    'CCN(CC)CCNC(=O)c1ccc(N)cc1',
    'O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl',
    'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
    'Clc1ccc2c(c1)C(=NCC(=O)N2C)c1ccccc1',
    'CN(C)CCCN1c2ccccc2CCc2ccccc21',
    'CCOC(=O)c1ccc(N)cc1',
    'NC(=O)c1cnccn1',
    'OC(=O)CCCCC1CCSS1',
    'Cc1ccc(cc1)S(=O)(=O)N',
    'CC1=C(C(=O)Nc2ccccc2)SCCO1',
    'O=C(c1ccccc1)c1ccccc1',
    'c1ccc2c(c1)ccc1ccccc12',
    'Nc1ccc(cc1)S(=O)(=O)Nc1ncccn1',
    'COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2',
    'CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O',
    'CCCCC1=NC2(CCCC2)C(=O)N1Cc1ccc(cc1)-c1ccccc1-c1nn[nH]n1',
    'CC(=O)N[C@@H](CS)C(=O)O',
    'OCC1OC(O)C(O)C(O)C1O',
    'CN(C)C(=N)NC(=N)N',
    'Oc1ccc(cc1)C(=O)OCCCC',
    'CCCCCCCC(=O)O',
    'Fc1ccc(cc1)C(CCCN1CCC(CC1)n1c(=O)[nH]c2ccccc12)c1ccc(F)cc1',
    'O=C1CCCN1',
    'CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O',
    'CC(C)(C)c1ccc(O)cc1',
    'Cc1ccccc1N',
    'COC(=O)c1ccccc1',
    'CCOC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC',
    'Clc1ccc(cc1)C(c1ccccc1)N1CCN(CC1)C',
    'CN1CCN(CC1)c1nc2ccccc2[nH]1',
    'Nc1nc(N)c2nc(-c3ccccc3)c(N)nc2n1',
    'CC(C)n1cnc2c(NCc3ccccc3)nc(NCCO)nc21',
    'COc1ccc(cc1)C(=O)N',
    'O=C(Nc1ccccc1)Nc1ccccc1',
    'c1ccc(cc1)-c1ccccc1',
    'CC(=O)c1ccc(cc1)O',
    'O=Cc1ccc(O)c(OC)c1',
    'CC(C)(C)OC(=O)N1CCCCC1',
    'OC(=O)c1ccc(cc1)Cl',
    'CCc1ccc(cc1)C(=O)O',
    'NCCc1ccc(O)c(O)c1',
    'NCCc1c[nH]c2ccc(O)cc12',
    'Cn1c(=O)c2[nH]cnc2n(C)c1=O',
    'O=C(O)c1cccnc1',
    'CC(N)Cc1ccccc1',
    'COc1ccccc1OCC(O)CO',
    'OC(=O)C(O)c1ccccc1',
    'Nc1ccc(cc1)C(=O)O',
    'CC(=O)OCC[N+](C)(C)C',
    'ClC1=CC=C(C=C1)C(=O)O',
    'CCOc1ccc(NC(C)=O)cc1',
    'CC(C)(C)NC(=O)C1CCCCC1',
    'Cc1cc(C)nc(n1)Nc1ccccc1',
    'O=C(CCc1ccccc1)N1CCCC1',
    'CN1CCC(CC1)C(=O)c1ccccc1',
    'c1ccc2[nH]ccc2c1',
    'c1ccc2ncccc2c1',
    'Oc1ccc2ccccc2c1',
    'CC(=O)NC1=NN=C(S1)S(N)(=O)=O',
    'CC1=CC(=O)c2ccccc2C1=O',
    'O=C(O)C1CCCN1',
    'CC(C)(O)c1ccccc1',
    'NC(=O)OCC',
    'CCCN(CCC)CC',
    'COc1ccc(CCN)cc1OC',
    'O=C1NC(=O)c2ccccc12',
    'CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc21',
    'CC(C)Oc1ccc(cc1)C(=O)N',
    'CS(=O)(=O)c1ccc(cc1)C(O)=O',
    'Cc1ccc(cc1)C(=O)c1ccccc1',
    'FC(F)(F)c1ccc(cc1)N',
    'O=C(O)C=Cc1ccccc1',
    'CCCCN1CCCCC1C(=O)Nc1c(C)cccc1C',
    'CC(C)Cc1ccc(O)cc1',
]

TEST_SMILES = ["CCO", "c1ccccc1O", "CC(=O)N", "CCN(CC)CC", "O=C(O)c1ccccc1"]


def _labels(n, kind, pos_frac, rng):
    if kind == "classification":
        n_pos = max(2, int(round(n * pos_frac)))
        y = np.zeros(n)
        y[rng.choice(n, n_pos, replace=False)] = 1.0
        return y
    return rng.normal(size=n)


def _build(smiles, test_smiles, name, kind, pos_frac, seed=0):
    rng = np.random.default_rng(seed)
    y = _labels(len(smiles), kind, pos_frac, rng)
    return TaskData(
        name=name,
        task_type=kind,
        train_smiles=list(smiles),
        y_raw=y.copy(),
        y_train=y.copy(),
        test_smiles=list(test_smiles),
    )


def fake_task(n, kind, pos_frac=0.5) -> TaskData:
    """Synthetic task with dummy (unparseable-as-drug) SMILES; no NaNs."""
    smiles = [f"C{'C' * (i % 7)}O{i}" for i in range(n)]
    return _build(smiles, ["CCO"] * 10, f"fake/{kind}", kind, pos_frac)


def fake_task_real_smiles(n, kind) -> TaskData:
    """Task whose SMILES are valid RDKit molecules (REAL_SMILES cycled to length n)."""
    smiles = [REAL_SMILES[i % len(REAL_SMILES)] for i in range(n)]
    return _build(smiles, TEST_SMILES, f"fake/{kind}_real", kind, 0.5)


def write_fake_cache(path, n=50, kind="regression"):
    """Write train.parquet / test.parquet / meta.json in the on-disk cache layout."""
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    t = fake_task_real_smiles(n, kind)
    pd.DataFrame({"smiles": t.train_smiles, "y": t.y_raw}).to_parquet(path / "train.parquet")
    pd.DataFrame({"smiles": t.test_smiles}).to_parquet(path / "test.parquet")
    (path / "meta.json").write_text(
        json.dumps(dict(name="fake/x", task_type=kind, n_train=n, n_test=len(t.test_smiles)))
    )

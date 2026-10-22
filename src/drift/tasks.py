"""Polaris task cache and train/val splits."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import astartes
import numpy as np
import pandas as pd

from drift import config
from drift.config import N_SMALL, slug


@dataclass
class TaskData:
    name: str
    task_type: str  # "regression" | "classification"
    train_smiles: list[str]
    y_raw: np.ndarray  # (n,) float, NaNs kept
    y_train: np.ndarray  # (n,) NaN filled with full-train mean
    test_smiles: list[str]


def fetch_task(name: str, out_dir: Path) -> None:
    """Download a Polaris benchmark and cache train/test SMILES (+ train labels only)."""
    import polaris as po
    from polaris.utils.types import TargetType

    b = po.load_benchmark(name)
    smiles_col = list(b.input_cols)[0]
    target_cols = list(b.target_cols)
    assert len(target_cols) == 1, f"{name}: expected one target column, got {target_cols}"
    target_col = target_cols[0]
    train, test = b.get_train_test_split()
    train_df, test_df = train.as_dataframe(), test.as_dataframe()
    tt = b.target_types[target_col]
    task_type = "regression" if tt == TargetType.REGRESSION else "classification"
    if task_type == "classification":
        assert tt == TargetType.CLASSIFICATION, f"{name}: unexpected target type {tt}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "smiles": train_df[smiles_col].astype(str).to_numpy(),
            "y": train_df[target_col].astype(float).to_numpy(),
        }
    ).to_parquet(out_dir / "train.parquet")
    pd.DataFrame({"smiles": test_df[smiles_col].astype(str).to_numpy()}).to_parquet(
        out_dir / "test.parquet"
    )
    meta = dict(name=name, task_type=task_type, n_train=len(train_df), n_test=len(test_df))
    (out_dir / "meta.json").write_text(json.dumps(meta))


def load_task_from(path: Path) -> TaskData:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    train = pd.read_parquet(path / "train.parquet")
    test = pd.read_parquet(path / "test.parquet")
    y_raw = train["y"].to_numpy(dtype=float)
    y_train = np.where(np.isnan(y_raw), np.nanmean(y_raw), y_raw)
    return TaskData(
        name=meta["name"],
        task_type=meta["task_type"],
        train_smiles=train["smiles"].tolist(),
        y_raw=y_raw,
        y_train=y_train,
        test_smiles=test["smiles"].tolist(),
    )


def load_task(name: str) -> TaskData:
    """Read the cache only (no network). Reads config.DATA at call time."""
    return load_task_from(config.DATA / "tasks" / slug(name))


def _split(idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    tr, va = astartes.train_test_split(idx, train_size=0.8, test_size=0.2, random_state=seed)
    return np.asarray(tr), np.asarray(va)


def make_split(task: TaskData, regime: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices into task.train_smiles for (train, val)."""
    n = len(task.train_smiles)
    idx = np.arange(n)
    if regime == "n128" and n > N_SMALL:
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            sub = rng.choice(idx, N_SMALL, replace=False)
            tr, va = _split(sub, seed)
            if task.task_type != "classification":
                break
            ytr, yva = task.y_raw[tr], task.y_raw[va]
            ytr, yva = ytr[~np.isnan(ytr)], yva[~np.isnan(yva)]
            if len(set(ytr)) == 2 and len(set(yva)) == 2:
                break
        else:
            raise RuntimeError(f"{task.name}: no n128 split with both classes (seed {seed})")
        return tr, va
    if regime not in ("full", "n128"):
        raise ValueError(f"unknown regime {regime!r}")
    return _split(idx, seed)

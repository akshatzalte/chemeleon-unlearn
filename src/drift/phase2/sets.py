"""Descriptor sets for the tailored-pretraining arms, regenerated from saved selector outputs."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from drift import config
from drift.tasks import load_task

SUMMARY = config.RESULTS / "summary"
SCORES_CSV = SUMMARY / "phase2_relevance_scores.csv"
SETS_JSON = SUMMARY / "phase2_descriptor_sets.json"
CLUSTERS_PARQUET = config.RESULTS / "corr_clusters.parquet"

DEFAULT_TASKS = ("tdcommons/ames", "tdcommons/ld50-zhu")
SANITY_SEEDS = (0, 1, 2)


def representatives(sets_json: Path = SETS_JSON, clusters_parquet: Path = CLUSTERS_PARQUET) -> dict[int, str]:
    """cluster id -> its representative descriptor (the member of D535 in that cluster)."""
    d535 = json.loads(Path(sets_json).read_text())["D535"]
    cl = pd.read_parquet(clusters_parquet).set_index("descriptor").cluster
    rep: dict[int, str] = {}
    for name in d535:
        c = int(cl[name])
        assert c not in rep, f"cluster {c} has two D535 members: {rep[c]}, {name}"
        rep[c] = name
    assert set(rep) == set(cl.unique()), "D535 does not cover every cluster exactly once"
    return rep


def build_sets(
    scores_csv: Path,
    clusters_parquet: Path,
    sets_json: Path,
    all_descriptors: list[str],
) -> dict[str, list[str]]:
    """B (all descriptors), D535, T25/T100 (consensus top-k clusters), R25a/R25b/R100 (random clusters),
    T100_ames / T100_ld50 (per-task tailored: top-100 clusters by `auc_ames` / `r2_ld50` alone, same representatives)."""
    rep = representatives(sets_json, clusters_parquet)
    scores = pd.read_csv(scores_csv, index_col=0)
    rank = scores.rank_consensus
    out: dict[str, list[str]] = {"B": list(all_descriptors), "D535": [rep[c] for c in sorted(rep)]}
    for k in (25, 100):
        out[f"T{k}"] = [rep[int(c)] for c in rank.nlargest(k).index]
    cluster_ids = sorted(rep)
    for name, k, seed in (("R25a", 25, 0), ("R25b", 25, 1), ("R100", 100, 0)):
        picked = np.random.default_rng(seed).choice(cluster_ids, k, replace=False)
        out[name] = [rep[int(c)] for c in picked]
    for name, col in (("T100_ames", "auc_ames"), ("T100_ld50", "r2_ld50")):
        out[name] = [rep[int(c)] for c in scores[col].nlargest(100).index]
    return out


def _load_xy(task: str, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Descriptors (standardized, winsorized, NaN -> 0) and labels of the labelled Polaris train rows."""
    t = load_task(task)
    X = pd.read_parquet(config.DATA / "descriptors" / f"{config.slug(task)}.parquet", columns=cols)
    ok = ~np.isnan(t.y_raw)
    X = X[ok].replace([np.inf, -np.inf], np.nan)
    sd = X.std().replace(0, 1)
    X = ((X - X.mean()) / sd).clip(-config.WINSOR, config.WINSOR).fillna(0)
    return X, t.y_raw[ok], t.task_type


def _model(kind: str, model: str, seed: int):
    if kind == "classification":
        if model == "linear":
            return LogisticRegression(C=0.1, max_iter=1000)
        return MLPClassifier(hidden_layer_sizes=(config.FFN_HIDDEN,), alpha=1e-3, max_iter=300,
                             early_stopping=True, random_state=seed)
    if model == "linear":
        return Ridge(alpha=10.0)
    return MLPRegressor(hidden_layer_sizes=(config.FFN_HIDDEN,), alpha=1e-3, max_iter=300,
                        early_stopping=True, random_state=seed)


def sanity_check(
    sets: dict[str, list[str]],
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    model: str = "ffn",
) -> pd.DataFrame:
    """5-fold CV (3 seeds) of a small model on each set: ROC-AUC (classification) or R2 (regression)."""
    if model not in ("ffn", "linear"):
        raise ValueError(f"unknown model {model!r}")
    all_cols = sorted({c for cols in sets.values() for c in cols})
    rows = []
    for task in tasks:
        X, y, kind = _load_xy(task, all_cols)
        cls = kind == "classification"
        yy = (y > 0.5).astype(int) if cls else y
        for name, cols in sets.items():
            vals = []
            for seed in SANITY_SEEDS:
                cv = (StratifiedKFold if cls else KFold)(config.N_FOLDS, shuffle=True, random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = cross_val_score(_model(kind, model, seed), X[cols].to_numpy(), yy, cv=cv,
                                        scoring="roc_auc" if cls else "r2").mean()
                vals.append(s)
            rows.append(dict(task=task.split("/")[1], set=name, n_descr=len(cols),
                             score=float(np.mean(vals)), sd=float(np.std(vals))))
            print(f"  {rows[-1]['task']:9s} {name:5s} n={len(cols):4d}  score={rows[-1]['score']:.4f} +- {rows[-1]['sd']:.4f}",
                  flush=True)
    return pd.DataFrame(rows, columns=["task", "set", "n_descr", "score", "sd"])

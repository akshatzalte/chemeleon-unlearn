"""Linear and debiased linear CKA (ft / random encoders vs pre) on three row sets, from the embedding cache.

    06_cka.py [--only SLUG ...] [--out results/cka.parquet]

Row sets: pubchem (eval rows 8000..9999), task_all (all task train rows), task_clean (that seed's clean rows:
full -> val_idx, n128 -> rows outside train_idx). Random-encoder rows (encoder rand0..2) are computed on the SAME
row sets (task_clean: per regime x seed, ckpt ''; pubchem/task_all: regime '' and seed -1) so they form a
same-rows / same-size anchor. Only tasks with results/drift/<slug>/done.json and cached embeddings are used.
Output columns: task, regime, ckpt, seed, encoder, rows, n, cka_linear, cka_debiased.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift import config  # noqa: E402
from drift.config import BENCHMARKS, DATA, RESULTS, RUNS, SEEDS, slug  # noqa: E402
from drift.measure import clean_rows  # noqa: E402
from drift.probe import debiased_linear_cka, linear_cka  # noqa: E402

OUT = RESULTS / "drift"
EMB = DATA / "emb"
EVAL = np.arange(config.N_PUBCHEM_FIT, config.N_PUBCHEM)
COLS = ["task", "regime", "ckpt", "seed", "encoder", "rows", "n", "cka_linear", "cka_debiased"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(set_name: str, enc_id: str):
    p = EMB / set_name / f"{enc_id}.npy"
    return np.load(p).astype(np.float64) if p.exists() else None


def _row(base, rows, Zx, Zp, idx):
    return {**base, "rows": rows, "n": len(idx), "cka_linear": linear_cka(Zp[idx], Zx[idx]),
            "cka_debiased": debiased_linear_cka(Zp[idx], Zx[idx])}


def cka_task(name: str, regimes=config.REGIMES, ckpts=("es", "last")) -> pd.DataFrame:
    s = slug(name)
    Zt_pre, Zp_pre = load(s, "pre"), load("pubchem", "pre")
    if Zt_pre is None or Zp_pre is None:
        log(f"{name}: no cached pre embeddings, skipping")
        return pd.DataFrame(columns=COLS)
    n = len(Zt_pre)
    allrows = np.arange(n)
    out = []
    for r in range(3):
        Zt, Zp = load(s, f"rand{r}"), load("pubchem", f"rand{r}")
        if Zt is None or Zp is None:
            continue
        b = dict(task=name, regime="", ckpt="", seed=-1, encoder=f"rand{r}")
        out.append(_row(b, "pubchem", Zp, Zp_pre, EVAL))
        out.append(_row(b, "task_all", Zt, Zt_pre, allrows))
    for regime in regimes:
        for seed in SEEDS:
            rp = RUNS / s / regime / "ft" / f"seed{seed}" / "run.json"
            if not rp.exists():
                continue
            clean = clean_rows(json.loads(rp.read_text()), n)
            if len(clean) < 4:
                continue
            for r in range(3):
                Zt = load(s, f"rand{r}")
                if Zt is not None:
                    out.append(_row(dict(task=name, regime=regime, ckpt="", seed=seed, encoder=f"rand{r}"),
                                    "task_clean", Zt, Zt_pre, clean))
            for ckpt in ckpts:
                eid = f"{s}__{regime}__s{seed}__{ckpt}"
                Zt, Zp = load(s, eid), load("pubchem", eid)
                if Zt is None or Zp is None:
                    log(f"{name}: missing embeddings for {eid}, skipping")
                    continue
                b = dict(task=name, regime=regime, ckpt=ckpt, seed=seed, encoder=f"seed{seed}")
                out.append(_row(b, "pubchem", Zp, Zp_pre, EVAL))
                out.append(_row(b, "task_all", Zt, Zt_pre, allrows))
                out.append(_row(b, "task_clean", Zt, Zt_pre, clean))
    return pd.DataFrame(out, columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="task slugs")
    ap.add_argument("--out", default=str(RESULTS / "cka.parquet"))
    a = ap.parse_args()
    parts = []
    for name in BENCHMARKS:
        if a.only and slug(name) not in a.only:
            continue
        if not (OUT / slug(name) / "done.json").exists():
            log(f"{name}: no done.json, skipping")
            continue
        t0 = time.time()
        df = cka_task(name)
        log(f"{name}: {len(df)} rows ({time.time() - t0:.0f}s)")
        parts.append(df)
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out)
    log(f"wrote {a.out}: {len(res)} rows, {res['task'].nunique()} tasks")


if __name__ == "__main__":
    main()

"""Phase 2, step 1: build the pretraining pool (run under $PY2; needs zarr>=3, no polaris).

Samples 200k molecules (default_rng(0)) from data/pubchem/cleaned_pubchem_1MM.smiles, excluding the Ames and
LD50 Polaris TEST molecules (non-isomeric canonical SMILES match), drops rows that are all-NaN in CheMeleon's
published Zenodo descriptor zarr, splits 90/10 (seed 0), and writes data/phase2/pool/
(desc_{train,val}.npy, smiles_{train,val}.txt, rows.json, columns.json, stats.npz, build.json).
Descriptors are NOT recomputed. Aborts if the zarr does not reproduce the phase-1 Mordred descriptors of
pubchem10k (row-alignment / same-config check).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from drift import config
from drift.phase2 import pool
from drift.pubchem import canonical

TEST_TASKS = ("tdcommons/ames", "tdcommons/ld50-zhu")
SMILES_FILE = config.DATA / "pubchem" / "cleaned_pubchem_1MM.smiles"
ZARR = config.DATA / "phase2" / "zenodo" / "pubchem_1MM.zarr"
OUT = config.DATA / "phase2" / "pool"
MIN_MATCH = 0.99


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()

    assert (ZARR.parent / "extracted.ok").exists(), "Zenodo zarr not extracted yet"

    align = pool.check_alignment(
        ZARR, SMILES_FILE, config.DATA / "pubchem" / "pubchem10k.smiles",
        config.DATA / "descriptors" / "pubchem10k.parquet", min_rows=500)
    print("alignment vs phase-1 pubchem10k descriptors:", align, flush=True)
    if align["row_match_rate"] < MIN_MATCH:
        print(f"ABORT: zarr not aligned with the SMILES file / Mordred config (match < {MIN_MATCH})")
        return 2

    exclude = set()
    for t in TEST_TASKS:
        smi = pd.read_parquet(config.DATA / "tasks" / config.slug(t) / "test.parquet")["smiles"]
        exclude.update(c for c in map(canonical, smi) if c is not None)
    print(f"{len(exclude)} canonical test SMILES to exclude", flush=True)

    rows, info = pool.sample_rows(SMILES_FILE, args.n, args.seed, exclude, nproc=args.nproc)
    print("sampled", info, flush=True)
    import zarr

    z = zarr.open_array(str(ZARR), mode="r")
    kept = pool.drop_all_nan(z, rows)  # before splitting
    n_allnan = len(rows) - len(kept)
    train, val = pool.split(kept.tolist(), frac_val=0.1, seed=0)
    print(f"dropped {n_allnan} all-NaN rows; train {len(train)} val {len(val)}", flush=True)

    b = pool.build(args.out, ZARR, SMILES_FILE, train, val)
    meta = dict(b)
    meta.update(n_sampled=info["n_sampled"], n_excluded_test=info["n_excluded_test"], n_invalid=info["n_invalid"],
                n_lines=info["n_lines"], n_dropped_allnan_presplit=n_allnan, n_test_smiles_excluded_set=len(exclude),
                pool_n=args.n, seed=args.seed, alignment=align, wall_s=round(time.time() - t0, 1))
    (Path(args.out) / "build.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pretraining pool for the tailored-pretraining arms (phase 2).

Molecules are sampled (by line number) from CheMeleon's cleaned PubChem SMILES file; descriptors are NOT
recomputed but read from CheMeleon's published Zenodo zarr (record 15733575), whose row i belongs to line i
of that file. Standardization statistics come from the pool-train rows only.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

from drift.pubchem import canonical

N_DESCRIPTORS = 1613
READ_BLOCK = 16_200  # rows per zarr read; a multiple of the zarr chunk length (162)


def _read_lines(smiles_file) -> list[str]:
    return [l.strip() for l in open(smiles_file)]


def _canon_quiet(s):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    return canonical(s)


def canonical_lines(smiles: list[str], nproc: int = 1) -> list[str | None]:
    """Non-isomeric canonical SMILES per line (None where RDKit rejects it)."""
    if nproc > 1:
        from multiprocessing import Pool

        with Pool(nproc) as p:
            return p.map(_canon_quiet, smiles, chunksize=2000)
    return [_canon_quiet(s) for s in smiles]


def sample_rows(smiles_file, n: int = 200_000, seed: int = 0, exclude: set[str] | None = None,
                nproc: int = 1) -> tuple[np.ndarray, dict]:
    """Sorted line indices of n molecules sampled without replacement (default_rng(seed)) from the lines that
    RDKit accepts and whose canonical non-isomeric SMILES is not in `exclude`; plus the counts."""
    exclude = exclude or set()
    canon = canonical_lines(_read_lines(smiles_file), nproc)
    invalid = np.array([c is None for c in canon])
    excluded = np.array([c is not None and c in exclude for c in canon])
    candidates = np.flatnonzero(~invalid & ~excluded)
    n = min(n, len(candidates))
    rows = np.sort(np.random.default_rng(seed).choice(candidates, size=n, replace=False))
    return rows, dict(n_lines=len(canon), n_invalid=int(invalid.sum()), n_excluded_test=int(excluded.sum()),
                      n_sampled=int(n))


def sample_pool(smiles_file, n: int = 200_000, seed: int = 0, exclude: set[str] | None = None,
                nproc: int = 1) -> list[str]:
    """SMILES of the sampled pool (see `sample_rows`), in file order."""
    rows, _ = sample_rows(smiles_file, n, seed, exclude, nproc)
    lines = _read_lines(smiles_file)
    return [lines[i] for i in rows]


def split(items, frac_val: float = 0.1, seed: int = 0) -> tuple[list, list]:
    """Random (train, val) split with default_rng(seed); each part keeps the input order."""
    items = list(items)
    n_val = int(round(frac_val * len(items)))
    val_pos = set(np.random.default_rng(seed).permutation(len(items))[:n_val].tolist())
    train = [x for i, x in enumerate(items) if i not in val_pos]
    val = [x for i, x in enumerate(items) if i in val_pos]
    return train, val


def read_rows(z, rows, block: int = READ_BLOCK) -> np.ndarray:
    """Gather zarr rows (sorted, so reads stay chunk-aligned) in blocks -> float32 (len(rows), n_cols)."""
    rows = np.asarray(rows, dtype=np.int64)
    assert np.all(np.diff(rows) > 0), "rows must be strictly increasing"
    out = np.empty((len(rows), z.shape[1]), dtype=np.float32)
    for s in range(0, len(rows), block):
        sel = rows[s:s + block]
        # read the covering contiguous range (whole chunks) once, then pick the rows
        lo, hi = int(sel[0]), int(sel[-1]) + 1
        out[s:s + len(sel)] = z[lo:hi][sel - lo]
    return out


def _all_missing(x: np.ndarray) -> np.ndarray:
    return ~np.isfinite(x).any(axis=1)


def drop_all_nan(z, rows) -> np.ndarray:
    """Sorted `rows` minus those whose descriptor vector in the zarr has no finite value."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    keep = np.empty(len(rows), dtype=bool)
    for s in range(0, len(rows), READ_BLOCK):
        keep[s:s + READ_BLOCK] = ~_all_missing(read_rows(z, rows[s:s + READ_BLOCK]))
    return rows[keep]


def column_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """nanmean / nanstd per column over finite values (float64 accumulation); sd == 0 or non-finite -> 1,
    non-finite mean -> 0."""
    xf = np.where(np.isfinite(x), x, np.nan).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        mean, sd = np.nanmean(xf, axis=0), np.nanstd(xf, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return mean, sd


def _write_npy(path: Path, x: np.ndarray) -> None:
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=x.shape)
    mm[:] = x
    mm.flush()
    del mm


def build(out_dir, zarr_path, smiles_file, row_idx_train, row_idx_val) -> dict:
    """Gather the pool's descriptor rows from the zarr into out_dir:
    desc_train.npy / desc_val.npy (float32, NaN preserved), smiles_train.txt / smiles_val.txt, rows.json
    (zarr row indices in memmap order), columns.json, stats.npz (mean, sd from train rows only), build.json.
    All-NaN rows are dropped (and counted)."""
    import zarr

    from drift.descriptors import descriptor_modules

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = _read_lines(smiles_file)
    z = zarr.open_array(str(zarr_path), mode="r")
    assert z.shape[0] == len(lines), f"zarr rows {z.shape[0]} != SMILES lines {len(lines)}"
    assert z.shape[1] == N_DESCRIPTORS, f"zarr columns {z.shape[1]} != {N_DESCRIPTORS}"
    columns = list(descriptor_modules().index)
    assert len(columns) == N_DESCRIPTORS

    info: dict = {}
    data: dict[str, np.ndarray] = {}
    rows: dict[str, list[int]] = {}
    for part, idx in (("train", row_idx_train), ("val", row_idx_val)):
        idx = np.sort(np.asarray(idx, dtype=np.int64))
        assert len(np.unique(idx)) == len(idx), f"duplicate rows in {part}"
        x = read_rows(z, idx)
        keep = ~_all_missing(x)
        info[f"n_dropped_allnan_{part}"] = int((~keep).sum())
        data[part], rows[part] = x[keep], idx[keep].tolist()
    assert not set(rows["train"]) & set(rows["val"]), "train and val rows overlap"

    mean, sd = column_stats(data["train"])  # pool-train rows ONLY
    np.savez(out / "stats.npz", mean=mean, sd=sd)
    (out / "columns.json").write_text(json.dumps(columns))
    (out / "rows.json").write_text(json.dumps(rows))
    for part in ("train", "val"):
        _write_npy(out / f"desc_{part}.npy", data[part])
        (out / f"smiles_{part}.txt").write_text("".join(lines[i] + "\n" for i in rows[part]))
    info.update(n_train=len(rows["train"]), n_val=len(rows["val"]), zarr_shape=list(z.shape),
                wall_s=round(time.time() - t0, 1))
    (out / "build.json").write_text(json.dumps(info, indent=2))
    return info


def check_alignment(zarr_path, smiles_file, ref_smiles_file, ref_parquet, min_rows: int = 500,
                    rtol: float = 1e-4, atol: float = 1e-4) -> dict:
    """Compare zarr rows against independently computed descriptors (phase-1 pubchem10k): the reference
    molecules are located by line number in `smiles_file`, their zarr rows read, and compared
    (np.allclose, NaN == NaN). Returns row match rate and max abs diff over finite entries."""
    import pandas as pd
    import zarr

    z = zarr.open_array(str(zarr_path), mode="r")
    lines = _read_lines(smiles_file)
    assert z.shape[0] == len(lines)
    line_to_idx: dict[str, int] = {}
    for i, l in enumerate(lines):
        line_to_idx.setdefault(l, i)
    ref = _read_lines(ref_smiles_file)
    ref_df = pd.read_parquet(ref_parquet)
    assert len(ref) == len(ref_df)
    pos = [j for j, s in enumerate(ref) if s in line_to_idx]
    assert len(pos) >= min_rows, f"only {len(pos)} reference molecules found in {smiles_file}"
    rows = np.array([line_to_idx[ref[j]] for j in pos])
    order = np.argsort(rows)
    rows, pos = rows[order], np.array(pos)[order]
    zx = read_rows(z, rows)
    rx = ref_df.to_numpy(dtype=np.float32)[pos]
    ok = np.array([np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True) for a, b in zip(zx, rx)])
    both = np.isfinite(zx) & np.isfinite(rx)
    return dict(n_compared=len(rows), row_match_rate=float(ok.mean()),
                nan_pattern_equal=float(np.mean(np.isnan(zx) == np.isnan(rx))),
                max_abs_diff=float(np.abs(zx[both] - rx[both]).max()) if both.any() else 0.0)

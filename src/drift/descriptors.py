import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles

from drift import config

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

CHUNK = 1000


def _calculator() -> Calculator:
    calc = Calculator(descriptors, ignore_3D=True)
    calc.config(timeout=1)
    return calc


def descriptor_modules() -> pd.Series:
    calc = _calculator()
    return pd.Series({str(d): type(d).__module__.split(".")[-1] for d in calc.descriptors})


def compute_mordred(smiles: list[str], nproc: int = 64) -> pd.DataFrame:
    calc = _calculator()
    cols = [str(d) for d in calc.descriptors]
    out = np.full((len(smiles), len(cols)), np.nan, dtype=np.float32)
    block = rdBase.BlockLogs()
    valid, mols = [], []
    for i, s in enumerate(smiles):
        try:
            m = MolFromSmiles(s)
        except Exception:
            m = None
        if m is None:
            continue
        m.SetProp("_Name", "")  # avoid mordred's expensive default naming
        valid.append(i)
        mols.append(m)
    del block
    log.info("mordred: %d/%d SMILES unparseable (all-NaN rows)", len(smiles) - len(valid), len(smiles))
    t0 = time.time()
    for a in range(0, len(mols), CHUNK):
        chunk = mols[a:a + CHUNK]
        try:
            df = calc.pandas(chunk, nproc=nproc, quiet=True).fill_missing()
            arr = df.to_numpy(dtype=np.float64, na_value=np.nan)
            arr[~np.isfinite(arr)] = np.nan
            arr[np.abs(arr) > np.finfo(np.float32).max] = np.nan
            out[valid[a:a + CHUNK]] = arr.astype(np.float32)
        except Exception as e:  # never crash on an odd molecule: leave the chunk as NaN, retry per mol
            log.warning("mordred chunk %d failed (%r); retrying per molecule", a, e)
            for j, m in enumerate(chunk):
                try:
                    r = calc.pandas([m], nproc=1, quiet=True).fill_missing().to_numpy(dtype=np.float64, na_value=np.nan)[0]
                    r[~np.isfinite(r)] = np.nan
                    r[np.abs(r) > np.finfo(np.float32).max] = np.nan
                    out[valid[a + j]] = r.astype(np.float32)
                except Exception:
                    log.warning("mordred failed on molecule index %d", valid[a + j])
        log.info("mordred: %d/%d valid mols done (%.0fs)", min(a + CHUNK, len(mols)), len(mols), time.time() - t0)
    return pd.DataFrame(out, columns=cols)


@dataclass
class Targets:
    Y: np.ndarray
    names: list[str]
    dropped: list[str]


def prepare_targets(D: pd.DataFrame, fit_rows: np.ndarray) -> Targets:
    X = D.to_numpy(dtype=np.float64, na_value=np.nan)
    F = X[np.asarray(fit_rows)]
    nan_frac = np.isnan(F).mean(0)
    n_ok = (~np.isnan(F)).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(F, 0)
        sd = np.nanstd(F, 0)
    bad = (nan_frac > config.NAN_FRAC_MAX) | (n_ok == 0) | ~np.isfinite(sd) | (sd == 0)
    keep = ~bad
    Z = np.clip((X[:, keep] - mu[keep]) / sd[keep], -config.WINSOR, config.WINSOR)
    cols = list(D.columns)
    return Targets(Z.astype(np.float32), [c for c, k in zip(cols, keep) if k], [c for c, k in zip(cols, keep) if not k])

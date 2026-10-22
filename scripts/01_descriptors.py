"""Compute Mordred descriptors for the held-out PubChem sample and each task's Polaris TRAIN molecules."""
import logging
import time
import urllib.request

import pandas as pd

from drift import config
from drift.config import BENCHMARKS, slug
from drift.descriptors import compute_mordred, prepare_targets
from drift.pubchem import heldout_sample
from drift.tasks import load_task

URL = "https://zenodo.org/records/15733575/files/cleaned_pubchem_1MM.smiles?download=1"
NPROC = 64
log = logging.getLogger("01_descriptors")


def summarize(name: str, D: pd.DataFrame) -> None:
    import numpy as np
    T = prepare_targets(D, np.arange(len(D)))
    log.info("%s: rows=%d all-NaN rows=%d NaN frac=%.4f kept=%d dropped=%d", name, len(D),
             int(D.isna().all(axis=1).sum()), float(D.isna().to_numpy().mean()), len(T.names), len(T.dropped))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    pc = config.DATA / "pubchem"
    pc.mkdir(parents=True, exist_ok=True)
    full = pc / "cleaned_pubchem_1MM.smiles"
    if not full.exists():
        log.info("downloading %s", URL)
        urllib.request.urlretrieve(URL, full.with_suffix(".tmp"))
        full.with_suffix(".tmp").rename(full)
    log.info("pubchem file lines: %d", sum(1 for _ in open(full)))
    desc = config.DATA / "descriptors"
    desc.mkdir(parents=True, exist_ok=True)

    sample_file = pc / "pubchem10k.smiles"
    if not sample_file.exists():
        sample_file.write_text("\n".join(heldout_sample(full, config.N_PUBCHEM)) + "\n")
    out = desc / "pubchem10k.parquet"
    if not out.exists():
        smi = [l.strip() for l in open(sample_file)]
        D = compute_mordred(smi, NPROC)
        D.columns = D.columns.astype(str)
        D.to_parquet(out)
    summarize("pubchem10k", pd.read_parquet(out))

    for name in BENCHMARKS:
        out = desc / f"{slug(name)}.parquet"
        if not out.exists():
            smi = list(load_task(name).train_smiles)  # TRAIN rows only; test molecules never featurized
            D = compute_mordred(smi, NPROC)
            D["smiles"] = smi
            D.to_parquet(out)
        summarize(name, pd.read_parquet(out).drop(columns="smiles"))
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()

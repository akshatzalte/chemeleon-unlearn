from pathlib import Path

import numpy as np
import torch


def heldout_sample(smiles_file: Path, n: int = 10_000, seed: int = 0) -> list[str]:
    """Sample n SMILES from the test 10% of CheMeleon's pretraining random_split (seed 1701)."""
    lines = [l.strip() for l in open(smiles_file).readlines()]
    _, _, test = torch.utils.data.random_split(range(len(lines)), [0.7, 0.2, 0.1], torch.Generator().manual_seed(1701))
    idx = np.array(test.indices)
    pick = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
    return [lines[i] for i in pick]


_SPLIT_MAPS: dict = {}


def _canon(s):
    from rdkit import Chem

    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def _split_map(smiles_file: Path, cache: Path, nproc: int) -> dict:
    key = (str(smiles_file), str(cache))
    if key in _SPLIT_MAPS:
        return _SPLIT_MAPS[key]
    import pandas as pd

    if cache.exists():
        df = pd.read_parquet(cache)
        mp = dict(zip(df["smiles"], df["split"]))
    else:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        lines = [l.strip() for l in open(smiles_file).readlines()]
        tr, va, te = torch.utils.data.random_split(range(len(lines)), [0.7, 0.2, 0.1],
                                                   torch.Generator().manual_seed(1701))
        if nproc > 1:
            from multiprocessing import Pool

            with Pool(nproc) as p:
                cp = p.map(_canon, lines, chunksize=2000)
        else:
            cp = [_canon(l) for l in lines]
        mp = {}
        for name, idx in (("train", tr.indices), ("val", va.indices), ("test", te.indices)):
            for i in idx:
                if cp[i]:
                    mp.setdefault(cp[i], name)       # first split wins on duplicate canonical SMILES
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"smiles": list(mp), "split": list(mp.values())}).to_parquet(cache)
    _SPLIT_MAPS[key] = mp
    return mp


def pretrain_membership(smiles: list[str], smiles_file: Path | None = None, cache: Path | None = None,
                        nproc: int = 16) -> np.ndarray:
    """'train' | 'val' | 'test' | 'none' per molecule: which split of CheMeleon's pretraining
    random_split (seed 1701) the molecule's canonical (non-isomeric) SMILES falls in.
    The canonical->split map is cached in data/pubchem/pretrain_split_map.parquet."""
    from drift import config

    smiles_file = smiles_file or config.DATA / "pubchem" / "cleaned_pubchem_1MM.smiles"
    cache = cache or config.DATA / "pubchem" / "pretrain_split_map.parquet"
    mp = _split_map(Path(smiles_file), Path(cache), nproc)
    return np.array([mp.get(_canon(s), "none") for s in smiles], dtype=object)

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

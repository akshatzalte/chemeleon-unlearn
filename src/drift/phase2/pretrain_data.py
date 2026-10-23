"""Data pipeline and loss for the tailored-pretraining arms (phase 2).

`PoolDataset` yields whole batches (use with DataLoader(batch_size=None)); SMILES are featurized with
cuik-molmaker inside `__getitem__` (worker processes), nothing is cached. Targets are the pool's Mordred
descriptors of the arm's set, standardized with pool-train statistics and clipped to +-6, NaN preserved.
"""
from __future__ import annotations

import numpy as np
import torch
from chemprop.data.collate import TrainingBatch
from chemprop.featurizers.molgraph.molecule import CuikmolmakerMolGraphFeaturizer
from chemprop.nn.metrics import MSE, LossFunctionRegistry, MetricRegistry
from torch.utils.data import Dataset, Sampler

CLIP = 6.0


def read_set(path) -> list[str]:
    """Descriptor names of an arm's set file (one per line, order kept)."""
    names = [l.strip() for l in open(path) if l.strip()]
    assert len(names) == len(set(names)), f"duplicate descriptor names in {path}"
    return names


def col_indices(names: list[str], columns: list[str]) -> np.ndarray:
    """Positions of `names` in the pool's `columns`, in the order of `names`."""
    pos = {c: i for i, c in enumerate(columns)}
    missing = [n for n in names if n not in pos]
    assert not missing, f"descriptor names not in pool columns: {missing[:5]}"
    return np.array([pos[n] for n in names], dtype=np.int64)


class PoolDataset(Dataset):
    """One *batch* per item. With `shuffle=True`, item index i = epoch * len(self) + j is the j-th batch of
    a fresh permutation of the rows seeded by (seed, epoch); use `EpochSampler` to advance the epoch.
    Indices < len(self) are epoch 0, so plain `ds[j]` also works."""

    def __init__(self, smiles: list[str], desc_memmap, col_idx: np.ndarray, mean, sd,
                 batch_size: int = 512, shuffle: bool = False, seed: int = 0):
        assert len(smiles) == desc_memmap.shape[0], "SMILES / descriptor row mismatch"
        self.smiles = smiles
        self.desc = desc_memmap
        self.col_idx = np.asarray(col_idx, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float64)[self.col_idx]
        self.sd = np.asarray(sd, dtype=np.float64)[self.col_idx]
        assert np.all(self.sd > 0), "zero-variance target column"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n = len(smiles)
        self.featurizer = CuikmolmakerMolGraphFeaturizer("V2")
        self._perm_epoch = None
        self._perm = None

    def __len__(self) -> int:
        return -(-self.n // self.batch_size)

    def batch_rows(self, i: int) -> np.ndarray:
        epoch, j = divmod(i, len(self))
        if not self.shuffle:
            order = np.arange(self.n)
        else:
            if self._perm_epoch != epoch:  # per-process cache of the last epoch's permutation
                self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n)
                self._perm_epoch = epoch
            order = self._perm
        return np.sort(order[j * self.batch_size:(j + 1) * self.batch_size])  # sorted: sequential memmap reads

    def __getitem__(self, i: int) -> TrainingBatch:
        rows = self.batch_rows(i)
        Y = (np.asarray(self.desc[rows])[:, self.col_idx] - self.mean) / self.sd
        Y = np.clip(Y, -CLIP, CLIP).astype(np.float32)  # NaN preserved
        bmg = self.featurizer([self.smiles[r] for r in rows])
        # cuik-molmaker silently drops SMILES it cannot parse (empty graph, no error): every molecule must be present
        assert len(bmg) == len(rows) and bmg.batch.unique().numel() == len(rows), (
            f"featurizer dropped molecules from a batch (rows {rows[:3]}...)")
        return TrainingBatch(bmg, None, None, torch.from_numpy(Y), torch.ones(len(rows), 1), None, None)


class EpochSampler(Sampler):
    """Yields the item indices of the next epoch each time it is iterated (see `PoolDataset`)."""

    def __init__(self, dataset: PoolDataset):
        self.n_batches = len(dataset)
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        base = self.epoch * self.n_batches
        self.epoch += 1
        return iter(range(base, base + self.n_batches))


@LossFunctionRegistry.register("masked_dropout_mse")
@MetricRegistry.register("masked_dropout_mse")
class MaskedDropoutMSE(MSE):
    """Masked MSE over finite targets; in training mode each cell is additionally kept with prob `keep_prob`
    (random target dropout). The denominator is the number of kept cells. In eval mode (validation) every
    finite cell is kept, so the validation value is the plain masked MSE."""

    def __init__(self, keep_prob: float = 0.7, task_weights=1.0):
        super().__init__(task_weights)
        self.keep_prob = keep_prob

    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None) -> None:
        mask = torch.isfinite(targets) if mask is None else (mask & torch.isfinite(targets))
        if self.training and self.keep_prob < 1.0:
            mask = mask & (torch.rand_like(targets) < self.keep_prob)
        targets = torch.nan_to_num(targets, nan=0.0)  # after the mask, so no NaN reaches the arithmetic
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)

    def extra_repr(self) -> str:
        return f"keep_prob={self.keep_prob}, " + super().extra_repr()

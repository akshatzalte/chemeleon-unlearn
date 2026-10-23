"""Fine-tuning runner following the CheMeleon paper protocol (evaluate.py), plus drift bookkeeping."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import chemprop
import lightning
import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from lightning import Trainer

from drift import config
from drift.config import BENCHMARKS, CHEMELEON_MP, MAX_EPOCHS, MODES, PATIENCE, REGIMES, SEEDS, slug
from drift.encoders import build_mpnn, load_mp, save_mp
from drift.tasks import TaskData, _split, load_task, make_split


class PatienceBestCheckpoint(lightning.Callback):
    """Tracks the best-val epoch exactly as EarlyStopping(patience) + best checkpoint would,
    but never stops training. `es_state` is the (CPU, deep-copied) state at the best epoch
    reached before patience ran out. Also records per-epoch train/val loss in `history`."""

    def __init__(self, patience: int = PATIENCE):
        self.patience = patience
        self.best_epoch = -1
        self.best_val = float("inf")
        self.stopped = False
        self.es_state: dict | None = None
        self.history: list[dict] = []

    def _update(self, epoch: int, val: float, state: dict) -> None:
        if self.stopped:
            return
        if val < self.best_val:
            self.best_val, self.best_epoch = val, epoch
            self.es_state = {k: v.detach().cpu().clone() for k, v in copy.deepcopy(state).items()}
        elif epoch - self.best_epoch >= self.patience:
            self.stopped = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = float(trainer.callback_metrics["val_loss"])
        self._update(trainer.current_epoch, val, pl_module.state_dict())
        self._val = val

    def on_train_epoch_end(self, trainer, pl_module):
        # validation runs inside the training epoch, so _val belongs to this epoch
        m = trainer.callback_metrics
        train = m.get("train_loss_epoch", m.get("train_loss"))
        self.history.append(
            dict(epoch=trainer.current_epoch, train_loss=float(train), val_loss=getattr(self, "_val", float("nan")))
        )


class StopAtSteps(lightning.Callback):
    """Stops training once `trainer.global_step` (optimizer steps) reaches `n_steps`."""

    def __init__(self, n_steps: int):
        self.n_steps = n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.n_steps:
            trainer.should_stop = True


def run_dir(task: str, regime: str, mode: str, seed: int) -> Path:
    return config.RUNS / slug(task) / regime / mode / f"seed{seed}"


def pending(items: list) -> list:
    return [it for it in items if not (run_dir(*it) / "done.json").exists()]


def worklist(n_train: dict[str, int] | None = None) -> list[tuple[str, str, str, int]]:
    if n_train is None:
        n_train = {b: len(load_task(b).train_smiles) for b in BENCHMARKS}
    tasks = sorted(BENCHMARKS, key=lambda b: -n_train[b])  # stable: largest first
    return [
        (t, r, m, s)
        for t in tasks
        for r in REGIMES
        if not (r == "n128" and n_train[t] <= config.N_SMALL)
        for m in MODES
        for s in SEEDS
    ]


def _datapoints(smiles, y):
    return [MoleculeDatapoint.from_smi(s, np.array([v], dtype=float)) for s, v in zip(smiles, y)]


def _predict(model, loader, accelerator: str) -> np.ndarray:
    trainer = Trainer(
        accelerator=accelerator, devices=1, logger=False, enable_progress_bar=False,
        enable_checkpointing=False, enable_model_summary=False,
    )
    return torch.vstack(trainer.predict(model, loader)).numpy(force=True).flatten()


def run_one(
    task: TaskData, regime: str, mode: str, seed: int, out_dir: Path,
    accelerator: str = "gpu", max_epochs: int = MAX_EPOCHS,
    *, mp_path: Path | None = None, train_idx: np.ndarray | None = None, fixed_steps: int | None = None,
    batch_size: int | None = None, cache_graphs: bool = False, num_workers: int = 4,
    extra_info: dict | None = None,
) -> dict:
    """Fine-tune one (task, regime, mode, seed). Phase-1 behaviour unless the keyword-only extras are given:

    mp_path: encoder checkpoint (default CHEMELEON_MP).
    train_idx (needs regime="custom"): rows of task.train_smiles to train on, bypassing make_split. Without
        `fixed_steps` they get the paper protocol (80/20 train/val split INSIDE them, astartes, random_state=seed).
    fixed_steps (needs regime="custom"): no val split, no early stopping; train exactly this many optimizer steps
        (max_epochs = ceil(fixed_steps / n_batches); warm-up = 2 steps; `es` = final weights; only pred_es written).
    batch_size: training batch size (default 64, chemprop's). cache_graphs: pre-featurize each dataset once.
    extra_info: merged into run.json.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp_path = Path(mp_path) if mp_path is not None else CHEMELEON_MP
    if (regime == "custom") != (train_idx is not None):
        raise ValueError("regime='custom' and train_idx must be given together")
    if fixed_steps is not None and regime != "custom":
        raise ValueError("fixed_steps requires regime='custom' with train_idx")
    if regime == "custom":
        sub = np.asarray(train_idx, dtype=int)
        if fixed_steps is not None:
            tr_idx, va_idx = sub, np.array([], dtype=int)
        else:
            tr_idx, va_idx = _split(sub, seed)
        y = np.where(np.isnan(task.y_raw), np.nanmean(task.y_raw[sub]), task.y_raw)
    else:
        tr_idx, va_idx = make_split(task, regime, seed)
        if regime == "n128" and len(task.train_smiles) > config.N_SMALL:
            sub = np.concatenate([tr_idx, va_idx])
            y = np.where(np.isnan(task.y_raw), np.nanmean(task.y_raw[sub]), task.y_raw)
        else:
            y = task.y_train
    smiles = np.asarray(task.train_smiles, dtype=object)

    feat = SimpleMoleculeMolGraphFeaturizer()
    train_ds = MoleculeDataset(_datapoints(smiles[tr_idx], y[tr_idx]), feat)
    val_ds = MoleculeDataset(_datapoints(smiles[va_idx], y[va_idx]), feat) if fixed_steps is None else None
    test_ds = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in task.test_smiles], feat)
    if cache_graphs:
        for ds in (train_ds, val_ds, test_ds):
            if ds is not None:
                ds.cache = True
    scaler = None
    if task.task_type == "regression":
        scaler = train_ds.normalize_targets()
        if val_ds is not None:
            val_ds.normalize_targets(scaler)

    lightning.seed_everything(seed, workers=True)
    bs_kw = {} if batch_size is None else dict(batch_size=batch_size)
    train_dl = build_dataloader(train_ds, num_workers=num_workers, **bs_kw)
    val_dl = build_dataloader(val_ds, num_workers=num_workers, shuffle=False) if val_ds is not None else None
    test_dl = build_dataloader(test_ds, num_workers=num_workers, shuffle=False)

    mpnn_kwargs = None
    callbacks: list = []
    cb = None
    if fixed_steps is None:
        cb = PatienceBestCheckpoint(PATIENCE)
        callbacks.append(cb)
    else:
        n_batches = len(train_dl)
        max_epochs = math.ceil(fixed_steps / n_batches)
        mpnn_kwargs = dict(warmup_epochs=2 / n_batches)  # chemprop: warmup_steps = warmup_epochs * n_batches = 2
        callbacks.append(StopAtSteps(fixed_steps))

    mp = load_mp(mp_path)
    model = build_mpnn(mp, task.task_type, scaler, mpnn_kwargs)
    if mode == "frozen":
        for p in model.message_passing.parameters():
            p.requires_grad = False

    trainer = Trainer(
        max_epochs=max_epochs, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        accelerator=accelerator, devices=1, deterministic=False,
        check_val_every_n_epoch=1, callbacks=callbacks,
    )
    trainer.fit(model, train_dl, val_dl)
    n_steps = int(trainer.global_step)
    if fixed_steps is None:
        if cb.es_state is None:
            raise RuntimeError("validation never improved on inf (NaN val loss?)")
    elif n_steps != fixed_steps:
        raise RuntimeError(f"fixed_steps={fixed_steps} but trained {n_steps} optimizer steps")

    pretrained = load_mp(mp_path).state_dict()
    if mode == "frozen":
        for k, v in model.message_passing.state_dict().items():
            assert torch.equal(v.cpu(), pretrained[k]), f"frozen encoder changed: {k}"

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    states = [("es", final_state)] if fixed_steps is not None else [("es", cb.es_state), ("last", final_state)]
    for tag, state in states:
        model.load_state_dict(state)
        pred = _predict(model, test_dl, accelerator)
        np.save(out_dir / f"pred_{tag}.npy", pred)
        torch.save({k: v.cpu() for k, v in model.predictor.state_dict().items()}, out_dir / f"{tag}_head.pt")
        if mode == "ft":
            fresh = load_mp(mp_path)  # carries _drift_hparams for save_mp
            fresh.load_state_dict(model.message_passing.state_dict())
            save_mp(fresh, out_dir / f"{tag}_mp.pt")
    if cb is not None:
        pd.DataFrame(cb.history).to_csv(out_dir / "metrics.csv", index=False)

    info = dict(
        task=task.name, regime=regime, mode=mode, seed=seed, es_epoch=cb.best_epoch if cb else None,
        n_train_rows=len(tr_idx), n_val_rows=len(va_idx),
        train_idx=[int(i) for i in tr_idx], val_idx=[int(i) for i in va_idx],
        seconds=time.time() - t0, chemprop_version=chemprop.__version__, torch_version=torch.__version__,
        max_epochs=max_epochs, epochs_run=trainer.current_epoch,
        n_steps=n_steps, fixed_steps=fixed_steps, batch_size=batch_size or 64, cache_graphs=cache_graphs,
        mp_path=str(mp_path),
    )
    info.update(extra_info or {})
    (out_dir / "run.json").write_text(json.dumps(info))
    (out_dir / "done.json").write_text(json.dumps(dict(ok=True)))
    return info

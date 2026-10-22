"""Fine-tuning runner following the CheMeleon paper protocol (evaluate.py), plus drift bookkeeping."""
from __future__ import annotations

import copy
import json
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
from drift.tasks import TaskData, load_task, make_split


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
) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_idx, va_idx = make_split(task, regime, seed)
    if regime == "n128" and len(task.train_smiles) > config.N_SMALL:
        sub = np.concatenate([tr_idx, va_idx])
        y = np.where(np.isnan(task.y_raw), np.nanmean(task.y_raw[sub]), task.y_raw)
    else:
        y = task.y_train
    smiles = np.asarray(task.train_smiles, dtype=object)

    feat = SimpleMoleculeMolGraphFeaturizer()
    train_ds = MoleculeDataset(_datapoints(smiles[tr_idx], y[tr_idx]), feat)
    val_ds = MoleculeDataset(_datapoints(smiles[va_idx], y[va_idx]), feat)
    test_ds = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in task.test_smiles], feat)
    scaler = None
    if task.task_type == "regression":
        scaler = train_ds.normalize_targets()
        val_ds.normalize_targets(scaler)

    lightning.seed_everything(seed, workers=True)
    train_dl = build_dataloader(train_ds, num_workers=4)
    val_dl = build_dataloader(val_ds, num_workers=4, shuffle=False)
    test_dl = build_dataloader(test_ds, num_workers=4, shuffle=False)

    mp = load_mp(CHEMELEON_MP)
    model = build_mpnn(mp, task.task_type, scaler)
    if mode == "frozen":
        for p in model.message_passing.parameters():
            p.requires_grad = False

    cb = PatienceBestCheckpoint(PATIENCE)
    trainer = Trainer(
        max_epochs=max_epochs, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        accelerator=accelerator, devices=1, deterministic=False,
        check_val_every_n_epoch=1, callbacks=[cb],
    )
    trainer.fit(model, train_dl, val_dl)
    if cb.es_state is None:
        raise RuntimeError("validation never improved on inf (NaN val loss?)")

    pretrained = load_mp(CHEMELEON_MP).state_dict()
    if mode == "frozen":
        for k, v in model.message_passing.state_dict().items():
            assert torch.equal(v.cpu(), pretrained[k]), f"frozen encoder changed: {k}"

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    preds = {}
    for tag, state in (("es", cb.es_state), ("last", last_state)):
        model.load_state_dict(state)
        preds[tag] = _predict(model, test_dl, accelerator)
        np.save(out_dir / f"pred_{tag}.npy", preds[tag])
        torch.save({k: v.cpu() for k, v in model.predictor.state_dict().items()}, out_dir / f"{tag}_head.pt")
        if mode == "ft":
            fresh = load_mp(CHEMELEON_MP)  # carries _drift_hparams for save_mp
            fresh.load_state_dict(model.message_passing.state_dict())
            save_mp(fresh, out_dir / f"{tag}_mp.pt")
    pd.DataFrame(cb.history).to_csv(out_dir / "metrics.csv", index=False)

    info = dict(
        task=task.name, regime=regime, mode=mode, seed=seed, es_epoch=cb.best_epoch,
        n_train_rows=len(tr_idx), n_val_rows=len(va_idx),
        train_idx=[int(i) for i in tr_idx], val_idx=[int(i) for i in va_idx],
        seconds=time.time() - t0, chemprop_version=chemprop.__version__, torch_version=torch.__version__,
        max_epochs=max_epochs, epochs_run=trainer.current_epoch,
    )
    (out_dir / "run.json").write_text(json.dumps(info))
    (out_dir / "done.json").write_text(json.dumps(dict(ok=True)))
    return info

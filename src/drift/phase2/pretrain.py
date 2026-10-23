"""Pretraining of a CheMeleon-style D-MPNN encoder on one Mordred-descriptor target set (phase 2).

One identical pipeline for every arm; only the target columns (the arm's set file) differ. Pool, split,
featurization (cuik-molmaker "V2", no caching), loss (masked MSE with 30% random target dropout), schedule and
early stopping are fixed by the global constraints. Run through scripts/p2_02_pretrain.py.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import chemprop
import lightning
import numpy as np
import pandas as pd
import torch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from drift import config
from drift.phase2.pretrain_data import EpochSampler, MaskedDropoutMSE, PoolDataset, col_indices, read_set

# hyper_parameters of ~/.chemprop/chemeleon_mp.pt (asserted equal in the tests); mp.pt carries exactly these
MP_HPARAMS = dict(d_v=72, d_e=14, d_h=2048, bias=False, depth=6, dropout=0.0, activation="relu",
                  undirected=False, d_vd=None, V_d_transform=None, graph_transform=None)
HPARAMS = dict(batch_size=512, keep_prob=0.7, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4, warmup_epochs=5,
               ffn_hidden=2048, ffn_layers=1, clip=6.0, precision="32-true")


def build_model(k: int) -> MPNN:
    """CheMeleon-architecture D-MPNN + mean aggregation + FFN (Linear 2048->2048, ReLU, Linear 2048->k)."""
    return MPNN(
        BondMessagePassing(**MP_HPARAMS),
        MeanAggregation(),
        RegressionFFN(n_tasks=k, input_dim=MP_HPARAMS["d_h"], hidden_dim=HPARAMS["ffn_hidden"],
                      n_layers=HPARAMS["ffn_layers"], criterion=MaskedDropoutMSE(HPARAMS["keep_prob"])),
        batch_norm=False,
        metrics=[MaskedDropoutMSE(1.0)],  # plain masked MSE (extra val metric; val_loss is the same quantity)
        init_lr=HPARAMS["init_lr"], max_lr=HPARAMS["max_lr"], final_lr=HPARAMS["final_lr"],
        warmup_epochs=HPARAMS["warmup_epochs"],
    )


class EpochTimer(Callback):
    """Wall seconds per epoch (training + validation)."""

    def __init__(self):
        self.seconds: list[float] = []
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.seconds.append(time.time() - self._t0)
        print(f"[epoch {trainer.current_epoch}] {self.seconds[-1]:.1f}s "
              f"val_loss={float(trainer.callback_metrics.get('val_loss', float('nan'))):.5f}", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _loader(ds: PoolDataset, num_workers: int, shuffle: bool) -> DataLoader:
    kw = dict(batch_size=None, num_workers=num_workers, persistent_workers=num_workers > 0)
    if shuffle:
        return DataLoader(ds, sampler=EpochSampler(ds), **kw)
    return DataLoader(ds, shuffle=False, **kw)


def train(arm: str, pool_dir: Path, sets_dir: Path, out_dir: Path, gpu: int, max_epochs: int = 100,
          limit_rows: int | None = None, patience: int = 15, num_workers: int = 4, seed: int = 0) -> dict:
    """Pretrain on the pool for the arm's descriptor set; write mp.pt, head.pt, metrics.csv, config.json,
    done.json (last) to out_dir. gpu < 0 trains on CPU. Returns timing info (also written to done.json)."""
    pool_dir, sets_dir, out_dir = Path(pool_dir), Path(sets_dir), Path(out_dir)
    assert not (out_dir / "done.json").exists(), f"{out_dir} already holds a finished run; not overwriting"
    out_dir.mkdir(parents=True, exist_ok=True)
    lightning.seed_everything(seed, workers=True)

    columns = json.loads((pool_dir / "columns.json").read_text())
    stats = np.load(pool_dir / "stats.npz")
    names = read_set(sets_dir / f"{arm}.txt")
    col_idx = col_indices(names, columns)
    k = len(names)
    smiles = {s: [l.strip() for l in open(pool_dir / f"smiles_{s}.txt") if l.strip()] for s in ("train", "val")}
    desc = {s: np.load(pool_dir / f"desc_{s}.npy", mmap_mode="r") for s in ("train", "val")}
    n_tr = limit_rows if limit_rows else len(smiles["train"])
    n_va = max(limit_rows // 10, 512) if limit_rows else len(smiles["val"])
    n_tr, n_va = min(n_tr, len(smiles["train"])), min(n_va, len(smiles["val"]))
    smiles = {"train": smiles["train"][:n_tr], "val": smiles["val"][:n_va]}
    desc = {"train": desc["train"][:n_tr], "val": desc["val"][:n_va]}

    bs = HPARAMS["batch_size"]
    ds_tr = PoolDataset(smiles["train"], desc["train"], col_idx, stats["mean"], stats["sd"], bs, True, seed)
    ds_va = PoolDataset(smiles["val"], desc["val"], col_idx, stats["mean"], stats["sd"], bs, False)
    dl_tr, dl_va = _loader(ds_tr, num_workers, True), _loader(ds_va, num_workers, False)

    model = build_model(k)
    print(model, flush=True)
    cfg = dict(arm=arm, k=k, target_names=names, max_epochs=max_epochs, patience=patience, seed=seed,
               limit_rows=limit_rows, n_train=n_tr, n_val=n_va, num_workers=num_workers, gpu=gpu,
               mp_hparams=MP_HPARAMS, pool_build=json.loads((pool_dir / "build.json").read_text()),
               git_sha=_git_sha(), torch=torch.__version__, chemprop=chemprop.__version__,
               lightning=lightning.__version__, **HPARAMS)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    timer = EpochTimer()
    ckpt = ModelCheckpoint(dirpath=out_dir / "ckpt", filename="best", monitor="val_loss", mode="min",
                           save_top_k=1, save_weights_only=True)
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    logger = CSVLogger(out_dir, name="logs")
    trainer = Trainer(
        max_epochs=max_epochs, accelerator="cpu" if gpu < 0 else "gpu", devices=1 if gpu < 0 else [gpu],
        precision=HPARAMS["precision"], callbacks=[stopper, ckpt, timer], logger=logger,
        log_every_n_steps=50, enable_progress_bar=False, enable_model_summary=False,
    )
    t0 = time.time()
    trainer.fit(model, dl_tr, dl_va)
    wall = time.time() - t0

    # best epoch -> mp.pt / head.pt
    best = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(best)
    mp_sd = {k_: v.detach().cpu().contiguous() for k_, v in model.message_passing.state_dict().items()}
    torch.save({"hyper_parameters": dict(MP_HPARAMS), "state_dict": mp_sd, "target_names": names, "arm": arm},
               out_dir / "mp.pt")
    torch.save({k_: v.detach().cpu() for k_, v in model.predictor.state_dict().items()}, out_dir / "head.pt")

    log = pd.read_csv(Path(logger.log_dir) / "metrics.csv").groupby("epoch").first()
    m = pd.DataFrame({"epoch": log.index, "train_loss": log["train_loss_epoch"].values,
                      "val_loss": log["val_loss"].values})
    m["seconds"] = timer.seconds[:len(m)]
    m.to_csv(out_dir / "metrics.csv", index=False)

    steady = timer.seconds[1:] if len(timer.seconds) > 1 else timer.seconds  # drop first-epoch warm-up
    info = dict(arm=arm, k=k, epochs_run=len(timer.seconds), epoch_seconds=timer.seconds,
                s_per_epoch=float(np.mean(steady)), best_epoch=int(m.val_loss.idxmin()),
                best_val_loss=float(m.val_loss.min()), early_stopped=bool(stopper.stopped_epoch),
                wall_s=wall, n_train=n_tr, n_val=n_va)
    if gpu >= 0:
        info.update(gpu_max_allocated_mb=torch.cuda.max_memory_allocated(gpu) / 2**20,
                    gpu_max_reserved_mb=torch.cuda.max_memory_reserved(gpu) / 2**20)
    (out_dir / "done.json").write_text(json.dumps(info, indent=2))
    return info

"""Phase-2 fine-tuning ladder: every pretrained arm is fine-tuned on identical, paired subsamples of the
Ames / LD50 training rows (N in 16..1024 and full). Run through scripts/p2_03_ladder.py.

Protocols: N <= 64 -> 16 optimizer steps, batch min(N, 16), no val split, final weights = `es`;
N >= 128 -> CheMeleon paper protocol (80/20 split INSIDE the subsample, early stopping), N = 'full' -> all rows.

Items are (task, arm, N, seed). `seed` indexes the subsample (file N<N>_s<seed>.json, shared by all arms) and
determines the fine-tuning seed (`ft_seed`): 42 + 1000 * seed, except N == 'full' (one fixed row set), where the
three seeds 0/1/2 map to the phase-1 fine-tuning seeds 42/117/709 (drift.config.SEEDS).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from drift import config
from drift.config import slug
from drift.finetune import run_one
from drift.tasks import TaskData, load_task

log = logging.getLogger(__name__)

TASKS = ("tdcommons/ames", "tdcommons/ld50-zhu")
ARMS = ("B", "T25", "T100", "R25a", "R25b", "R100", "D535", "chemeleon")
SIZES = (16, 32, 64, 128, 256, 512, 1024, "full")
FIXED_STEPS = 16
MAX_FIXED_BATCH = 16
FULL = "full"


def n_seeds(n) -> int:
    n = n if n == FULL else int(n)
    return 10 if n != FULL and n <= 64 else 5 if n != FULL and n <= 256 else 3


SEEDS_FOR = {n: n_seeds(n) for n in SIZES}


def ft_seed(n, seed: int) -> int:
    return config.SEEDS[seed] if n == FULL else 42 + 1000 * seed


def subsample_dir(task: str, root: Path | None = None) -> Path:
    return (Path(root) if root is not None else config.DATA / "phase2" / "subsamples") / slug(task)


def subsample_path(task: str, n, seed: int, root: Path | None = None) -> Path:
    return subsample_dir(task, root) / f"N{n}_s{seed}.json"


def _sha1(idx: list[int]) -> str:
    return hashlib.sha1(json.dumps([int(i) for i in idx]).encode()).hexdigest()


def _eligible(task: TaskData) -> np.ndarray:
    """Rows with a label (NaN-labelled rows are never sampled)."""
    return np.flatnonzero(~np.isnan(task.y_raw))


def make_subsamples(task: TaskData, sizes=SIZES, seeds_for: dict | None = None, out_dir: Path | None = None) -> None:
    """Write data/phase2/subsamples/<slug>/N<N>_s<seed>.json = {idx, sha1, N, seed} (idx sorted row indices into
    task.train_smiles). Stratified by class for classification. 'full' = all rows, same for every seed.
    Idempotent: an existing file is kept (and its sha1 verified), never rewritten."""
    seeds_for = SEEDS_FOR if seeds_for is None else seeds_for
    elig = _eligible(task)
    strat = (task.y_raw[elig] > 0.5) if task.task_type == "classification" else None
    d = subsample_dir(task.name, out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        for seed in range(seeds_for[n]):
            p = d / f"N{n}_s{seed}.json"
            if p.exists():
                rec = json.loads(p.read_text())
                assert rec["sha1"] == _sha1(rec["idx"]), f"corrupt subsample {p}"
                continue
            if n == FULL:
                idx = np.arange(len(task.y_raw))
            elif n >= len(elig):
                log.warning("%s: N=%s >= %d labelled rows, skipped", task.name, n, len(elig))
                continue
            else:
                idx, _ = train_test_split(elig, train_size=n, stratify=strat, random_state=seed)
            idx = sorted(int(i) for i in idx)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(idx=idx, sha1=_sha1(idx), N=n, seed=seed)))
            tmp.replace(p)


def load_subsample(task: str, n, seed: int, root: Path | None = None) -> dict:
    return json.loads(subsample_path(task, n, seed, root).read_text())


def run_dir(task: str, arm: str, n, seed: int) -> Path:
    return config.RUNS / "phase2" / "ladder" / slug(task) / arm / f"N{n}_s{seed}"


def pretrain_dir(arm: str) -> Path:
    return config.RUNS / "phase2" / "pretrain" / arm


def arm_mp_path(arm: str) -> Path:
    return config.CHEMELEON_MP if arm == "chemeleon" else pretrain_dir(arm) / "mp.pt"


def arm_ready(arm: str) -> bool:
    """CheMeleon always; pretrained arms once their pretraining wrote done.json (and mp.pt)."""
    return arm == "chemeleon" or ((pretrain_dir(arm) / "done.json").exists() and arm_mp_path(arm).exists())


def worklist(tasks=TASKS, arms=ARMS, sizes=SIZES, seeds_for: dict | None = None) -> list[tuple[str, str, object, int]]:
    """All (task, arm, N, seed), largest N first (so shards balance); every arm is listed, ready or not."""
    seeds_for = SEEDS_FOR if seeds_for is None else seeds_for
    ordered = sorted(sizes, key=lambda n: -(10**9 if n == FULL else n))
    return [(t, a, n, s) for t in tasks for n in ordered for s in range(seeds_for[n]) for a in arms]


def shard_items(items: list, i: int, n: int) -> list:
    """Shard i of n of the FULL worklist (apply before any arm/task filter). Plain `items[i::n]` would give each
    shard a fixed subset of arms whenever n divides len(ARMS); the `k // len(ARMS)` skew rotates arms over shards.
    Sub-sharding is consistent: key % (n*J) == i + n*j implies key % n == i."""
    return [it for k, it in enumerate(items) if (k + k // len(ARMS)) % n == i]


def pending(items: list) -> list:
    """Items not done whose arm encoder exists; items of arms without a finished encoder are logged and skipped."""
    out, missing = [], {}
    for it in items:
        if (run_dir(*it) / "done.json").exists():
            continue
        if not arm_ready(it[1]):
            missing[it[1]] = missing.get(it[1], 0) + 1
            continue
        out.append(it)
    for arm, k in sorted(missing.items()):
        log.warning("arm %s: no finished encoder (%s), skipping %d items", arm, arm_mp_path(arm), k)
    return out


@lru_cache(maxsize=4)
def _task(name: str) -> TaskData:
    return load_task(name)


def run_item(task: str, arm: str, n, seed: int, gpu: int | None, task_data: TaskData | None = None,
             subsample_root: Path | None = None) -> dict:
    """Fine-tune `arm`'s encoder on subsample (task, n, seed). `gpu` is informational (the script restricts the
    visible devices via CUDA_VISIBLE_DEVICES); gpu=None runs on CPU. Removes a half-written run dir first."""
    td = task_data if task_data is not None else _task(task)
    sub = load_subsample(task, n, seed, subsample_root)
    assert sub["sha1"] == _sha1(sub["idx"]) and sub["N"] == n and sub["seed"] == seed
    d = run_dir(task, arm, n, seed)
    if d.exists():
        shutil.rmtree(d)
    kw = dict(accelerator="cpu" if gpu is None else "gpu", mp_path=arm_mp_path(arm), cache_graphs=True)
    if n == FULL:
        assert len(sub["idx"]) == len(td.train_smiles)
        regime = "full"
        kw.update(num_workers=2)
    else:
        regime = "custom"
        kw.update(train_idx=np.asarray(sub["idx"], dtype=int))
        if n <= 64:
            kw.update(fixed_steps=FIXED_STEPS, batch_size=min(n, MAX_FIXED_BATCH), num_workers=0)
        else:
            kw.update(num_workers=2)
    extra = dict(arm=arm, N=n, subsample_seed=seed, subsample_sha1=sub["sha1"], ft_seed=ft_seed(n, seed))
    return run_one(td, regime, "ft", ft_seed(n, seed), d, extra_info=extra, **kw)

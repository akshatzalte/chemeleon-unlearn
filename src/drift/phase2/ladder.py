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
# The first 8 arms keep their order (shard keys depend on it); later arms are appended: `rand` = never-pretrained
# encoder (scripts/p2_06_make_rand.py), `B_s1` / `T100_s1` = second pretraining seed of B / T100,
# `T100_ames` / `T100_ld50` = per-task tailored top-100 sets (drift.phase2.sets), `B_full` / `T100_full` = B / T100
# target sets pretrained on the FULL pool (data/phase2/pool_full, ~941k molecules; arm names ending `_full`).
ARMS = ("B", "T25", "T100", "R25a", "R25b", "R100", "D535", "chemeleon", "rand", "B_s1", "T100_s1",
        "T100_ames", "T100_ld50", "B_full", "T100_full")
NO_PRETRAIN_CHECKS = ("chemeleon", "rand")  # ready as soon as the encoder file exists (no patience/pool checks)
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


def ladder_root(mode: str = "ft", steps: int = FIXED_STEPS) -> Path:
    """runs/phase2/ladder (ft, 16 steps: the original layout); frozen -> ladder_frozen; steps != 16 -> ladder_steps<S>
    (both -> ladder_frozen_steps<S>)."""
    if mode not in ("ft", "frozen"):
        raise ValueError(f"unknown mode {mode!r}")
    name = "ladder" + ("_frozen" if mode == "frozen" else "") + ("" if steps == FIXED_STEPS else f"_steps{int(steps)}")
    return config.RUNS / "phase2" / name


def run_dir(task: str, arm: str, n, seed: int, mode: str = "ft", steps: int = FIXED_STEPS) -> Path:
    return ladder_root(mode, steps) / slug(task) / arm / f"N{n}_s{seed}"


def pretrain_dir(arm: str) -> Path:
    return config.RUNS / "phase2" / "pretrain" / arm


def arm_mp_path(arm: str) -> Path:
    return config.CHEMELEON_MP if arm == "chemeleon" else pretrain_dir(arm) / "mp.pt"


REQUIRED_PATIENCE = 15
REQUIRED_N_TRAIN = 180000


def _pretrain_config(arm: str) -> dict | None:
    p = pretrain_dir(arm) / "config.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pool_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg.get("pool_build"), sort_keys=True).encode()).hexdigest()[:12]


def _is_full(arm: str) -> bool:
    """Arms pretrained on the full pool: judged against each other, not against the 180k-row arms."""
    return arm.endswith("_full")


def _config_problem(arm: str) -> str | None:
    """Why this arm's pretraining config.json is not the final protocol (None = fine). `_full` arms are not held to
    the 180000-row n_train (their mutual consistency is checked in arm_ready)."""
    cfg = _pretrain_config(arm)
    if cfg is None:
        return f"no config.json in {pretrain_dir(arm)}"
    if cfg.get("patience") != REQUIRED_PATIENCE:
        return f"config.json patience={cfg.get('patience')} (need {REQUIRED_PATIENCE})"
    if not _is_full(arm) and cfg.get("n_train") != REQUIRED_N_TRAIN:
        return f"config.json n_train={cfg.get('n_train')} (need {REQUIRED_N_TRAIN})"
    if cfg.get("pool_build") is None:
        return "config.json has no pool_build"
    return None


def arm_ready(arm: str) -> bool:
    """CheMeleon always; `rand` as soon as its mp.pt exists. A pretrained arm is ready once it wrote done.json and mp.pt AND its config.json shows the
    final protocol (patience 15, the full 180000-row train pool) on the same pool_build as every other arm that
    passes those checks (a mismatch rejects all of the disagreeing arms). `_full` arms (full ~941k pool) need patience
    15 and a pool_build but not n_train == 180000; their n_train / pool_build must match each other only, and are never
    compared with the 180k arms. A rejected arm is logged with the reason."""
    if arm == "chemeleon":
        return True
    if arm == "rand":
        return arm_mp_path(arm).exists()
    if not ((pretrain_dir(arm) / "done.json").exists() and arm_mp_path(arm).exists()):
        return False
    problem = _config_problem(arm)
    if problem is None:
        cfg = _pretrain_config(arm)
        mine = (_pool_id(cfg), cfg.get("n_train"))
        # consistency only within the arm's own group: `_full` arms among themselves, all others among themselves
        others = {a: (_pool_id(c), c.get("n_train")) for a in ARMS
                  if a != arm and a not in NO_PRETRAIN_CHECKS and _is_full(a) == _is_full(arm)
                  if (pretrain_dir(a) / "done.json").exists() and _config_problem(a) is None
                  and (c := _pretrain_config(a)) is not None}
        differing = sorted(a for a, key in others.items() if key != mine)
        if differing:
            problem = (f"pool_build {mine[0]} / n_train {mine[1]} differs from that of arm(s) {', '.join(differing)}")
    if problem:
        log.warning("arm %s: finished but NOT ready: %s", arm, problem)
    return problem is None


def worklist(tasks=TASKS, arms=ARMS, sizes=SIZES, seeds_for: dict | None = None) -> list[tuple[str, str, object, int]]:
    """All (task, arm, N, seed), largest N first (so shards balance); every arm is listed, ready or not."""
    seeds_for = SEEDS_FOR if seeds_for is None else seeds_for
    ordered = sorted(sizes, key=lambda n: -(10**9 if n == FULL else n))
    return [(t, a, n, s) for t in tasks for n in ordered for s in range(seeds_for[n]) for a in arms]


def shard_items(items: list, i: int, n: int) -> list:
    """Shard i of n of the FULL worklist (apply before any arm/task filter). The worklist is a (block, arm) grid
    (arms vary fastest), so plain `items[i::n]` pins arms to shards whenever n shares a factor with len(ARMS).
    Instead item k = (block b, arm a) gets the key a * n_blocks + b (the transposed grid) and goes to shard
    `key % n`: along b the residues cycle through every value, so each shard sees every arm for every n.
    Sub-sharding is consistent because the key is independent of n: key % (n*J) == i + n*j implies key % n == i."""
    A = len(ARMS)
    n_blocks = -(-len(items) // A)
    return [it for k, it in enumerate(items) if ((k % A) * n_blocks + k // A) % n == i]


def pending(items: list, mode: str = "ft", steps: int = FIXED_STEPS) -> list:
    """Items not done (in the run dir of `mode` / `steps`) whose arm encoder exists; items of arms without a finished
    encoder are logged and skipped."""
    out, missing = [], {}
    for it in items:
        if (run_dir(*it, mode=mode, steps=steps) / "done.json").exists():
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
             subsample_root: Path | None = None, mode: str = "ft", steps: int = FIXED_STEPS) -> dict:
    """Fine-tune `arm`'s encoder on subsample (task, n, seed). `gpu` is informational (the script restricts the
    visible devices via CUDA_VISIBLE_DEVICES); gpu=None runs on CPU. Removes a half-written run dir first.
    mode "frozen" trains only the head (encoder weights asserted unchanged); `steps` is the fixed-step budget of the
    N <= 64 protocol. Both change the run dir (see ladder_root) and are recorded in run.json (mode, fixed_steps)."""
    td = task_data if task_data is not None else _task(task)
    sub = load_subsample(task, n, seed, subsample_root)
    assert sub["sha1"] == _sha1(sub["idx"]) and sub["N"] == n and sub["seed"] == seed
    d = run_dir(task, arm, n, seed, mode, steps)
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
            kw.update(fixed_steps=int(steps), batch_size=min(n, MAX_FIXED_BATCH), num_workers=0)
        else:
            kw.update(num_workers=2)
    extra = dict(arm=arm, N=n, subsample_seed=seed, subsample_sha1=sub["sha1"], ft_seed=ft_seed(n, seed))
    return run_one(td, regime, mode, ft_seed(n, seed), d, extra_info=extra, **kw)

"""Phase 2, step 3: fine-tuning ladder (run under $PY1).

    $PY1 scripts/p2_03_ladder.py --gpu G --shard i/n [--only-arm A ...] [--only-task T ...] [--jobs-per-gpu J]

Writes subsamples (idempotent) to data/phase2/subsamples, then fine-tunes every pending (task, arm, N, seed) of
this shard into runs/phase2/ladder/<slug>/<arm>/N<N>_s<seed>/. --mode frozen trains only the head (encoder frozen)
into runs/phase2/ladder_frozen/...; --steps S (default 16) sets the fixed-step budget of the N <= 64 protocol and, when
S != 16, restricts the worklist to N <= 64 (the only sizes it affects; --all-sizes keeps the rest) and writes to
runs/phase2/ladder_steps<S>/... (ladder_frozen_steps<S> with --mode frozen). Skips done runs, deletes half-written ones, and
skips arms whose pretraining has not finished (re-run later to pick them up). --jobs-per-gpu J splits the shard
into J subprocesses sharing GPU G (~3 GB each); shard i/n -> children (i + n*j)/(n*J), j = 0..J-1.
"""
import argparse
import logging
import os
import subprocess
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only-arm", nargs="*", default=None)
    ap.add_argument("--only-task", nargs="*", default=None, help="task names or slugs")
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    ap.add_argument("--mode", choices=["ft", "frozen"], default="ft", help="ft = full fine-tuning, frozen = head only")
    ap.add_argument("--steps", type=int, default=16, help="fixed-step budget for N <= 64 (default 16)")
    ap.add_argument("--all-sizes", action="store_true", help="with --steps != 16: also run N >= 128 (unaffected by --steps)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before torch/chemprop/drift import
    i, n = map(int, args.shard.split("/"))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from drift.config import slug
    from drift.phase2 import ladder

    for t in ladder.TASKS:
        if not args.only_task or t in args.only_task or slug(t) in args.only_task:
            ladder.make_subsamples(ladder._task(t))

    if args.jobs_per_gpu > 1:
        J = args.jobs_per_gpu
        procs = []
        for j in range(J):
            cmd = [sys.executable, __file__, "--gpu", str(args.gpu), "--shard", f"{i + n * j}/{n * J}"]
            if args.only_arm:
                cmd += ["--only-arm", *args.only_arm]
            if args.only_task:
                cmd += ["--only-task", *args.only_task]
            cmd += ["--mode", args.mode, "--steps", str(args.steps)]
            if args.all_sizes:
                cmd += ["--all-sizes"]
            procs.append(subprocess.Popen(cmd))
        return max(p.wait() for p in procs)

    items = ladder.shard_items(ladder.worklist(), i, n)
    if args.only_arm:
        items = [it for it in items if it[1] in set(args.only_arm)]
    if args.only_task:
        keep = set(args.only_task) | {slug(t) for t in args.only_task}
        items = [it for it in items if it[0] in keep or slug(it[0]) in keep]
    if args.steps != ladder.FIXED_STEPS and not args.all_sizes:
        items = [it for it in items if it[2] != ladder.FULL and int(it[2]) <= 64]
    todo = ladder.pending(items, mode=args.mode, steps=args.steps)
    print(f"shard {i}/{n}: mode={args.mode} steps={args.steps} -> {ladder.ladder_root(args.mode, args.steps)}: "
          f"{len(items)} items, {len(todo)} pending", flush=True)
    for k, it in enumerate(todo):
        t0 = time.time()
        info = ladder.run_item(*it, gpu=args.gpu, mode=args.mode, steps=args.steps)
        print(f"[{k + 1}/{len(todo)}] {it} steps={info['n_steps']} {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

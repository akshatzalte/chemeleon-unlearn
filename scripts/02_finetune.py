"""Run the fine-tuning worklist: --gpu G --shard I/N [--only SLUG ...] [--seeds ...]."""
import argparse
import os
import shutil
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", nargs="*", default=None, help="task slugs")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before torch/chemprop/drift import

    from drift.config import slug
    from drift.finetune import pending, run_dir, run_one, worklist
    from drift.tasks import load_task

    i, n = map(int, args.shard.split("/"))
    items = worklist()[i::n]
    if args.only:
        items = [it for it in items if slug(it[0]) in set(args.only)]
    if args.seeds:
        items = [it for it in items if it[3] in set(args.seeds)]
    todo = pending(items)
    print(f"shard {i}/{n}: {len(items)} items, {len(todo)} pending", flush=True)
    cache = {}
    for k, it in enumerate(todo):
        d = run_dir(*it)
        if d.exists():
            shutil.rmtree(d)  # half-written
        if it[0] not in cache:
            cache = {it[0]: load_task(it[0])}
        t0 = time.time()
        info = run_one(cache[it[0]], *it[1:], out_dir=d, accelerator="gpu")
        print(f"[{k + 1}/{len(todo)}] {it} es_epoch={info['es_epoch']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Phase 2, step 2: pretrain one arm (run under $PY2).

    $PY2 scripts/p2_02_pretrain.py --arm T100 --gpu 0 [--max-epochs 100] [--patience 15] [--seed 0] [--limit-rows R] [--out DIR]

Trains the CheMeleon-architecture D-MPNN on the pool (data/phase2/pool) to predict the descriptors listed in
data/phase2/sets/<arm>.txt. Output (default runs/phase2/pretrain/<arm>/): mp.pt (chemeleon_mp.pt format),
head.pt, metrics.csv, config.json, done.json. --limit-rows R uses the first R train rows and the first
max(R//10, 512) val rows (smoke tests / timing).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import sys
from pathlib import Path

from drift import config
from drift.phase2 import pretrain

POOL = config.DATA / "phase2" / "pool"
SETS = config.DATA / "phase2" / "sets"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0, help="torch/lightning seed and the training-row permutation seed")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool-dir", default=None, help="pool directory (default data/phase2/pool)")
    ap.add_argument("--set-name", default=None, help="set file name to use (default: the arm name)")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    out = Path(args.out) if args.out else config.RUNS / "phase2" / "pretrain" / args.arm
    pool = Path(args.pool_dir) if args.pool_dir else POOL
    info = pretrain.train(args.set_name or args.arm, pool, SETS, out, args.gpu, args.max_epochs, args.limit_rows,
                          args.patience, args.num_workers, args.seed)
    print(json.dumps({k: v for k, v in info.items() if k != "epoch_seconds"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

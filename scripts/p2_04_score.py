"""Phase 2, step 4: score finished ladder runs with Polaris (incremental). Run under $PY1; needs a Polaris login.

Same protocol as scripts/05_score.py (benchmark.evaluate with the benchmark's main metric; cached test-SMILES
order asserted against the Polaris test set), over runs/phase2/ladder/<slug>/<arm>/N<N>_s<seed>/.
Rows are keyed by (task, arm, N, seed, ckpt); rows already in results/phase2/scores.parquet are kept, runs without
done.json are skipped. `seed` is the run's fine-tuning seed (paired across arms by (task, N, seed)). `N` is stored
as a string ('16', ..., 'full') so the parquet column is homogeneous. `pred_last.npy` exists only for N >= 128.
"""
from __future__ import annotations

import json
import sys
import traceback

import numpy as np
import pandas as pd

from drift.config import RESULTS, RUNS, slug
from drift.phase2.ladder import TASKS

KEY = ["task", "arm", "N", "seed", "ckpt"]
COLS = KEY + ["metric", "direction", "score", "subsample_sha1"]
CKPTS = {"es": "pred_es.npy", "last": "pred_last.npy"}
OUT = RESULTS / "phase2" / "scores.parquet"


def finished_runs(task: str) -> list[dict]:
    out = []
    for done in sorted((RUNS / "phase2" / "ladder" / slug(task)).glob("*/N*_s*/done.json")):
        run = json.loads((done.parent / "run.json").read_text())
        assert run["task"] == task, f"{done.parent}: run.json task {run['task']!r} != {task!r}"
        run["dir"] = done.parent
        run["N"] = str(run["N"])
        out.append(run)
    return out


def todo_for(task: str, have: set) -> list[dict]:
    """(run, ckpt) pairs not yet scored whose prediction file exists."""
    return [
        dict(run=r, ckpt=c)
        for r in finished_runs(task)
        for c, f in CKPTS.items()
        if (task, r["arm"], r["N"], int(r["seed"]), c) not in have and (r["dir"] / f).exists()
    ]


def score_task(name: str, todo: list[dict]) -> list[dict]:
    """Loads the Polaris benchmark once and scores every (run, ckpt) in `todo`."""
    import polaris as po

    from drift.tasks import load_task

    b = po.load_benchmark(name)
    smiles_col = list(b.input_cols)[0]
    _, test = b.get_train_test_split()
    cached = load_task(name)
    assert list(test.as_dataframe()[smiles_col].astype(str)) == cached.test_smiles, (
        f"{name}: Polaris test SMILES order differs from cached test.parquet"
    )
    label = b.main_metric.label
    direction = b.main_metric.direction
    is_cls = cached.task_type == "classification"
    rows = []
    for t in todo:
        run, ckpt = t["run"], t["ckpt"]
        pred = np.load(run["dir"] / CKPTS[ckpt]).flatten()
        assert len(pred) == len(cached.test_smiles), f"{run['dir']}: wrong prediction length"
        res = b.evaluate(pred > 0.5, pred).results if is_cls else b.evaluate(pred).results
        score = float(res.query(f"Metric == '{label}'")["Score"].values[0])
        rows.append(dict(task=name, arm=run["arm"], N=run["N"], seed=int(run["seed"]), ckpt=ckpt, metric=label,
                         direction=direction, score=score, subsample_sha1=run.get("subsample_sha1")))
    return rows


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    have = set(map(tuple, old[KEY].astype({"seed": int}).itertuples(index=False, name=None)))
    new_rows, failed = [], []
    for name in TASKS:
        todo = todo_for(name, have)
        if not todo:
            continue
        try:
            new_rows += score_task(name, todo)
            print(f"scored {name}: {len(todo)} new", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAILED {name}", file=sys.stderr)
            traceback.print_exc()
    scores = pd.concat([old, pd.DataFrame(new_rows, columns=COLS)], ignore_index=True) if new_rows else old
    scores["N"] = scores["N"].astype(str)
    scores = scores.drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True)
    scores.to_parquet(OUT)
    print(f"{len(scores)} rows in {OUT}")
    print(scores.groupby(["task", "arm", "ckpt"]).size().unstack("ckpt").to_string())
    if failed:
        print("failed tasks:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

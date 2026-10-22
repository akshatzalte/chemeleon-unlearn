"""Score finished runs with Polaris (incremental) and write the paper sanity table.

Needs a valid Polaris login. Rows are keyed by (task, regime, mode, seed, ckpt);
rows already in results/scores.parquet are kept, runs without done.json are skipped.
"""
from __future__ import annotations

import json
import sys
import traceback

import numpy as np
import pandas as pd
import polaris as po

from drift import config
from drift.config import BENCHMARKS, RESULTS, RUNS, slug
from drift.score import parse_paper_results, range_check
from drift.tasks import load_task

PAPER_MD = config.ROOT / "external/CheMeleon/analysis/polaris_results/CheMeleon.md"
KEY = ["task", "regime", "mode", "seed", "ckpt"]
CKPTS = {"es": "pred_es.npy", "last": "pred_last.npy"}


def finished_runs(name: str) -> list[dict]:
    out = []
    for done in sorted((RUNS / slug(name)).glob("*/*/seed*/done.json")):
        run = json.loads((done.parent / "run.json").read_text())
        run["dir"] = done.parent
        out.append(run)
    return out


def score_task(name: str, todo: list[dict]) -> list[dict]:
    """todo: list of dict(run=..., ckpt=...). Loads the benchmark once."""
    from polaris.utils.types import TargetType

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
        rows.append(
            dict(task=name, regime=run["regime"], mode=run["mode"], seed=int(run["seed"]),
                 ckpt=ckpt, metric=label, score=score, direction=direction)
        )
    return rows


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "scores.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=KEY + ["metric", "score", "direction"])
    have = set(map(tuple, old[KEY].itertuples(index=False, name=None)))
    new_rows, failed = [], []
    for name in BENCHMARKS:
        todo = [
            dict(run=r, ckpt=c)
            for r in finished_runs(name)
            for c in CKPTS
            if (name, r["regime"], r["mode"], int(r["seed"]), c) not in have
        ]
        if not todo:
            continue
        try:
            new_rows += score_task(name, todo)
            print(f"scored {name}: {len(todo)} new", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAILED {name}", file=sys.stderr)
            traceback.print_exc()
    scores = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else old
    scores = scores.drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True)
    scores.to_parquet(path)

    paper = parse_paper_results(PAPER_MD)
    sel = scores[(scores.regime == "full") & (scores["mode"] == "ft") & (scores.ckpt == "es")]
    rows = []
    for r in sel.itertuples():
        rows.append(
            dict(task=r.task, seed=r.seed, ours=r.score, **range_check(r.score, paper[r.task]),
                 metric=r.metric, direction=r.direction,
                 higher_is_better=(r.direction == "max"))
        )
    sanity = pd.DataFrame(rows)
    sanity.to_csv(RESULTS / "sanity.csv", index=False)
    print(sanity.to_string(index=False))
    if len(sanity):
        print(f"\nfull/ft/es runs: {len(sanity)}; within_range {int(sanity.within_range.sum())}; "
              f"within_1sd_of_range {int(sanity.within_1sd_of_range.sum())}")
    if failed:
        print("failed tasks:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

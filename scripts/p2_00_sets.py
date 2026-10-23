"""Phase 2, step 0: write the descriptor sets and run the FFN sanity gate (spec 3.1).

Outputs: data/phase2/sets/<arm>.txt, data/phase2/sets/summary.json, results/phase2/sanity_ffn.csv
(summary.json and sanity_ffn.csv are also copied to docs/results/phase2/ for committing).
Run with OMP_NUM_THREADS=4 (default 64 BLAS threads thrash: >45 min vs ~6 min).
Gate: for every k, T_k scores >= R_k on at least one task (k=25: against both R25a and R25b).
"""
import argparse
import json
import shutil
import sys

import pandas as pd

from drift import config
from drift.descriptors import descriptor_modules
from drift.phase2 import sets as S

GATES = {"T25": ("R25a", "R25b"), "T100": ("R100",)}


def evaluate_gate(df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    score = df.set_index(["task", "set"]).score
    tasks = sorted(df.task.unique())
    out: dict[str, dict[str, dict]] = {}
    for t_arm, r_arms in GATES.items():
        out[t_arm] = {}
        for r_arm in r_arms:
            wins = {t: bool(score[t, t_arm] >= score[t, r_arm]) for t in tasks}
            out[t_arm][r_arm] = dict(per_task=wins, passed=any(wins.values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ffn", choices=["ffn", "linear"])
    args = ap.parse_args()

    all_desc = list(descriptor_modules().index)
    sets = S.build_sets(S.SCORES_CSV, S.CLUSTERS_PARQUET, S.SETS_JSON, all_desc)

    sets_dir = config.DATA / "phase2" / "sets"
    res_dir = config.RESULTS / "phase2"
    docs_dir = config.ROOT / "docs" / "results" / "phase2"
    for d in (sets_dir, res_dir, docs_dir):
        d.mkdir(parents=True, exist_ok=True)
    for arm, cols in sets.items():
        (sets_dir / f"{arm}.txt").write_text("\n".join(cols) + "\n")

    df = S.sanity_check(sets, model=args.model)
    df.to_csv(res_dir / "sanity_ffn.csv", index=False)
    print(df.pivot(index="set", columns="task", values="score").reindex(list(sets)).round(4).to_string())

    gate = evaluate_gate(df)
    rep = S.representatives()
    summary = dict(
        sizes={k: len(v) for k, v in sets.items()},
        n_clusters=len(rep),
        overlap={f"{a}&{b}": len(set(sets[a]) & set(sets[b]))
                 for a, b in [("T25", "R25a"), ("T25", "R25b"), ("T100", "R100"), ("R25a", "R25b"), ("T25", "T100")]},
        T25_subset_of_T100=set(sets["T25"]) < set(sets["T100"]),
        sanity_model=args.model,
        gate=gate,
        gate_passed=all(g["passed"] for per_r in gate.values() for g in per_r.values()),
    )
    (sets_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    shutil.copy(sets_dir / "summary.json", docs_dir / "sets_summary.json")
    shutil.copy(res_dir / "sanity_ffn.csv", docs_dir / "sanity_ffn.csv")
    print(json.dumps(summary, indent=1))
    print("GATE", "PASSED" if summary["gate_passed"] else "FAILED")
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

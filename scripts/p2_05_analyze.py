"""Phase 2, step 5: analysis of results/phase2/scores.parquet (run under $PY1; no Polaris needed).

Writes results/phase2/{gaps.csv, endpoint.json, curves.png, summary.md}. gaps.csv holds every contrast
(es and last checkpoints; gap > 0 = first arm better, sign-normalized). endpoint.json is the pre-registered
primary endpoint (spec §7: T100 - B, es checkpoint).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from drift.config import RESULTS, RUNS
from drift.phase2 import analysis as A

OUT = RESULTS / "phase2"
EXPECTED_DONE = 784  # len(ladder.worklist()): 2 tasks x 8 arms x 49 (N, seed) combinations
SECONDARY = (("T100", "T25"),)


def contrast_table(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = list(dict.fromkeys(list(A.CONTRASTS) + list(SECONDARY)))
    frames = []
    for ckpt in ("es", "last"):
        for a, b in pairs:
            g = A.gaps(scores, a, b, ckpt=ckpt)
            if len(g):
                frames.append(g.assign(contrast=f"{a}-{b}", ckpt=ckpt))
    cols = ["contrast", "ckpt", "task", "N", "gap", "sd", "n", "se"]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def _nan_to_none(o):
    """NaN is not valid JSON; write null instead."""
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return None if isinstance(o, float) and o != o else o


def _fmt(x, nd=4):
    return "nan" if x != x else f"{x:.{nd}f}"


def summary_md(ep: dict, table: pd.DataFrame, scores: pd.DataFrame) -> str:
    L = ["# Phase 2: tailored-descriptor pretraining, ladder summary", ""]
    L.append(f"Primary endpoint (T100 - B, es checkpoint, spec §7): **{'SUCCESS' if ep['success'] else 'NOT MET'}**")
    if not ep.get("complete_pairs", True):
        L.append("")
        L.append("**WARNING: paired scores incomplete or subsample hashes differ (--allow-incomplete); endpoint not final.**")
    L.append("")
    L.append("| task | points gap>SD (of N=16,32,64) | Spearman(gap, log N) | mean gap N<=64 | mean gap N>=512 | "
             "small>large | sizes | success |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in A.TASKS:
        e = ep[t]
        L.append(f"| {t} | {e['points_pass']} | {_fmt(e['spearman'], 3)} | {_fmt(e['gap_small'])} | {_fmt(e['gap_large'])} | "
                 f"{e['small_gt_large']} | {e['n_sizes']}/8 | {e['success']} |")
    L += ["", "Gaps are sign-normalized (higher = better; LD50 MAE negated); gap > 0 means the first arm is better.", ""]
    L.append("## Primary gap table (T100 - B, es)")
    L += ["", "| task | N | n pairs | gap | SD | SE | gap > SD |", "|---|---|---|---|---|---|---|"]
    for r in ep["gaps"]:
        L.append(f"| {r['task']} | {r['N']} | {r['n']} | {_fmt(r['gap'])} | {_fmt(r['sd'])} | {_fmt(r['se'])} | "
                 f"{bool(r['gap'] > r['sd'])} |")
    for ckpt in ("es", "last"):
        L += ["", f"## Secondary contrasts ({ckpt}): mean paired gap by N (mean +- SE)"]
        sub = table[(table.ckpt == ckpt) & (table.contrast != "T100-B")]
        if sub.empty:
            L += ["", "(none)"]
            continue
        for task in A.TASKS:
            st = sub[sub.task == task]
            if st.empty:
                continue
            ns = [n for n in A.N_ORDER if n in set(st.N)]
            L += ["", f"**{task}**", "", "| contrast | " + " | ".join(ns) + " |", "|---|" + "---|" * len(ns)]
            for c, g in st.groupby("contrast", sort=False):
                g = g.set_index("N")
                cells = [f"{g.gap[n]:+.4f} +- {_fmt(g.se[n], 4)}" if n in g.index else "-" for n in ns]
                L.append(f"| {c} | " + " | ".join(cells) + " |")
    n_pairs = scores.groupby(["task", "arm", "ckpt"]).size().unstack("ckpt").fillna(0).astype(int)
    L += ["", "## Score rows per (task, arm)", "", "```", n_pairs.to_string(), "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(OUT / "scores.parquet"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="compute the endpoint although paired scores are missing / subsamples differ "
                         "(stamps endpoint.json complete_pairs=false)")
    args = ap.parse_args()
    out = Path(args.out)
    scores = pd.read_parquet(args.scores)
    n_done = sum(1 for _ in (RUNS / "phase2" / "ladder").glob("*/*/*/done.json"))
    chk, complete = A.check_endpoint_pairs(scores)
    print(f"done.json under runs/phase2/ladder: {n_done} (expected {EXPECTED_DONE})")
    print(f"endpoint pair check ({' vs '.join(A.PRIMARY)}, es): paired scores per (task, N), identical subsample_sha1")
    print(chk.to_string(index=False))
    if not complete:
        print("INCOMPLETE: " + ("continuing (--allow-incomplete)" if args.allow_incomplete else "refusing to compute the endpoint"))
        if not args.allow_incomplete:
            return 2
    out.mkdir(parents=True, exist_ok=True)
    table = contrast_table(scores)
    table.to_csv(out / "gaps.csv", index=False)
    ep = A.endpoint(scores)
    ep["complete_pairs"] = complete
    (out / "endpoint.json").write_text(json.dumps(_nan_to_none(ep), indent=2))
    A.curves(scores, out / "curves.png")
    (out / "summary.md").write_text(summary_md(ep, table, scores))
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

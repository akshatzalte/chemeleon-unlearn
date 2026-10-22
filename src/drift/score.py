"""Pure helpers for scoring: paper-number parsing and within-range logic."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

PAPER_SEEDS = (42, 117, 709, 1701, 9001)
_BLOCK = re.compile(r"results_dict\s*=\s*(\{.*?\})", re.S)


def parse_paper_results(path: Path) -> dict[str, list[float]]:
    """task -> five scores, ordered by PAPER_SEEDS (order of the results_dict blocks)."""
    blocks = _BLOCK.findall(Path(path).read_text())
    assert len(blocks) == len(PAPER_SEEDS), f"expected 5 results_dict blocks, got {len(blocks)}"
    out: dict[str, list[float]] = {}
    for b in blocks:
        for task, v in json.loads(b).items():
            out.setdefault(task, []).append(float(v))
    return out


def range_check(ours: float, paper: list[float]) -> dict:
    """Paper min/max/mean/sd (ddof=1) and whether ours is in [min,max] / [min-sd,max+sd]."""
    p = np.asarray(paper, dtype=float)
    lo, hi, sd = float(p.min()), float(p.max()), float(p.std(ddof=1))
    return dict(
        paper_min=lo,
        paper_max=hi,
        paper_mean=float(p.mean()),
        paper_sd=sd,
        within_range=bool(lo <= ours <= hi),
        within_1sd_of_range=bool(lo - sd <= ours <= hi + sd),
    )

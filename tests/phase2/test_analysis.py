import numpy as np
import pandas as pd
import pytest

from drift.phase2 import analysis as A

AMES, LD50 = A.TASKS
SIZES = ["16", "32", "64", "128", "256", "512", "1024", "full"]


def _seeds(n):
    """Fine-tuning seeds as in ladder.ft_seed / SEEDS_FOR (n is a size string)."""
    if n == "full":
        return [42, 117, 709]
    k = 10 if n in ("16", "32", "64") else 5 if n in ("128", "256") else 3
    return [42 + 1000 * s for s in range(k)]


def _rows(task, arm, n, seed, score, ckpt="es"):
    metric, direction = ("roc_auc", "max") if task == AMES else ("mean_absolute_error", "min")
    return dict(task=task, arm=arm, N=n, seed=seed, ckpt=ckpt, metric=metric, direction=direction, score=score)


def _table(gap_by_n, noise=0.01, tasks=(AMES, LD50)):
    """B vs T100 over all sizes; T100 is better than B by gap_by_n[N] + alternating +-noise in the *better* direction
    (so the paired SD is ~noise). Better = higher AUC / lower MAE."""
    rows = []
    for task in tasks:
        sgn = 1 if task == AMES else -1
        for n in SIZES:
            for i, s in enumerate(_seeds(n)):
                base = 0.7 + 0.01 * i if task == AMES else 1.0 + 0.01 * i
                eps = noise * (1 if i % 2 == 0 else -1)
                rows.append(_rows(task, "B", n, s, base))
                rows.append(_rows(task, "T100", n, s, base + sgn * (gap_by_n[n] + eps)))
    return pd.DataFrame(rows)


DECREASING = dict(zip(SIZES, [0.10, 0.08, 0.06, 0.04, 0.03, 0.01, 0.005, 0.0]))


def test_gap_sign_for_min_metric():
    rows = []
    for s in (1, 2, 3):
        rows += [_rows(LD50, "B", "16", s, 1.00), _rows(LD50, "T100", "16", s, 0.90 - 0.01 * s)]  # lower MAE = better
        rows += [_rows(AMES, "B", "16", s, 0.70), _rows(AMES, "T100", "16", s, 0.80 + 0.01 * s)]
    g = A.gaps(pd.DataFrame(rows), "T100", "B").set_index("task")
    assert g.loc[LD50, "gap"] == pytest.approx(0.12)  # positive: ft score lower
    assert g.loc[AMES, "gap"] == pytest.approx(0.12)
    assert g.loc[LD50, "sd"] == pytest.approx(0.01) and g.loc[LD50, "n"] == 3
    assert g.loc[LD50, "se"] == pytest.approx(0.01 / np.sqrt(3))
    # reversing the arms flips the sign
    assert A.gaps(pd.DataFrame(rows), "B", "T100").set_index("task").loc[LD50, "gap"] == pytest.approx(-0.12)


def test_pairing_drops_unmatched_seeds():
    rows = []
    for s in (1, 2, 3, 4, 5):
        rows.append(_rows(AMES, "B", "16", s, 0.70))
    for s in (3, 4, 5, 6, 7):
        rows.append(_rows(AMES, "T100", "16", s, 0.75 if s != 7 else 0.0))  # seed 7 unmatched, must not count
    g = A.gaps(pd.DataFrame(rows), "T100", "B")
    assert len(g) == 1 and g.n.iloc[0] == 3
    assert g.gap.iloc[0] == pytest.approx(0.05)


def test_pairing_is_per_task_and_n():
    rows = [_rows(AMES, "B", "16", 1, 0.7), _rows(AMES, "T100", "32", 1, 0.9)]  # different N: nothing pairs
    assert A.gaps(pd.DataFrame(rows), "T100", "B").empty


def test_gaps_use_only_requested_ckpt_and_N_orders_with_full_last():
    t = _table(DECREASING)
    big = t[t.N.isin(["128", "256", "512", "1024", "full"])]
    last = big.assign(ckpt="last", score=0.0)  # B == T100 == 0 -> gap 0 if (wrongly) used
    both = pd.concat([t, last])
    g = A.gaps(both, "T100", "B")
    assert list(g[g.task == AMES].N) == SIZES
    assert g.gap.min() > 0  # the zeroed `last` rows were ignored
    gl = A.gaps(both, "T100", "B", ckpt="last")
    assert set(gl.N) == {"128", "256", "512", "1024", "full"} and (gl.gap == 0).all()


def test_integer_N_accepted():
    rows = [_rows(AMES, "B", 16, s, 0.7) for s in (1, 2)] + [_rows(AMES, "T100", 16, s, 0.8) for s in (1, 2)]
    g = A.gaps(pd.DataFrame(rows), "T100", "B")
    assert list(g.N) == ["16"]


def test_duplicate_rows_rejected():
    rows = [_rows(AMES, "B", "16", 1, 0.7)] * 2 + [_rows(AMES, "T100", "16", 1, 0.8)]
    with pytest.raises(ValueError, match="duplicate"):
        A.gaps(pd.DataFrame(rows), "T100", "B")


def test_endpoint_success_and_failure():
    ok = A.endpoint(_table(DECREASING))
    assert ok["success"] is True
    for t in (AMES, LD50):
        e = ok[t]
        assert e["points_pass"] == 3 and e["spearman"] == pytest.approx(-1.0)
        assert e["small_gt_large"] and e["success"]
    assert len(ok["gaps"]) == 16 and {"task", "N", "gap", "sd", "n", "se"} <= set(ok["gaps"][0])

    # gap grows with N: Spearman > 0, small < large -> fail on both tasks
    bad = A.endpoint(_table(dict(zip(SIZES, [0.0, 0.0, 0.0, 0.01, 0.02, 0.04, 0.06, 0.1]))))
    assert bad["success"] is False
    assert all(bad[t]["spearman"] > 0 and not bad[t]["small_gt_large"] and not bad[t]["success"] for t in (AMES, LD50))

    # gap below SD at small N (noise 0.05 > gap 0.01-0.03): points_pass 0
    noisy = A.endpoint(_table(dict(zip(SIZES, [0.03, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0])), noise=0.05))
    assert noisy[AMES]["points_pass"] == 0 and noisy["success"] is False


def test_endpoint_needs_both_tasks_and_two_of_three_points():
    # Ames passes, LD50 has gap > SD at only one of {16,32,64}
    g_ames = DECREASING
    full = _table(g_ames, tasks=(AMES,))
    g_ld = dict(zip(SIZES, [0.10, 0.005, 0.005, 0.004, 0.003, 0.001, 0.0, 0.0]))
    ld = _table(g_ld, tasks=(LD50,))
    e = A.endpoint(pd.concat([full, ld]))
    assert e[AMES]["success"] and e[LD50]["points_pass"] == 1 and not e[LD50]["success"]
    assert e["success"] is False
    # LD50 missing altogether
    e = A.endpoint(full)
    assert e[LD50]["success"] is False and e["success"] is False


def test_endpoint_incomplete_sizes_cannot_succeed():
    t = _table(DECREASING)
    e = A.endpoint(t[t.N != "1024"])
    assert not e[AMES]["complete"] and e[AMES]["success"] is False and e["success"] is False


def test_endpoint_uses_es_checkpoint_only():
    t = _table(DECREASING)
    wrong = t[t.N.isin(["128", "256", "512", "1024", "full"])].assign(ckpt="last")
    wrong.loc[wrong.arm == "T100", "score"] = 0.0
    assert A.endpoint(pd.concat([t, wrong]))["success"] is True


def test_tasks_match_ladder():
    from drift.phase2 import ladder

    assert A.TASKS == tuple(ladder.TASKS)


def test_curves_writes_png(tmp_path):
    t = _table(DECREASING)
    extra = t[t.arm == "B"].assign(arm="chemeleon", score=lambda d: d.score - 0.01)
    out = tmp_path / "curves.png"
    A.curves(pd.concat([t, extra]), out)
    assert out.stat().st_size > 5000


def _load_script(name):
    import importlib.util

    from drift import config

    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_script_selects_unscored_finished_runs(tmp_path, monkeypatch):
    """No Polaris: only the run discovery / incremental-keying logic of scripts/p2_04_score.py."""
    import json

    sc = _load_script("p2_04_score")
    monkeypatch.setattr(sc, "RUNS", tmp_path)
    base = tmp_path / "phase2" / "ladder" / "tdcommons_ames"

    def mk(arm, n, seed, done=True, last=False):
        d = base / arm / f"N{n}_s{seed // 1000}"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(dict(task="tdcommons/ames", arm=arm, N=n, seed=seed, subsample_sha1="h")))
        (d / "pred_es.npy").write_bytes(b"")
        if last:
            (d / "pred_last.npy").write_bytes(b"")
        if done:
            (d / "done.json").write_text("{}")

    mk("B", 16, 42)
    mk("B", "full", 42, last=True)
    mk("T100", 16, 42, done=False)  # unfinished: skipped
    todo = sc.todo_for("tdcommons/ames", have=set())
    assert sorted((t["run"]["arm"], t["run"]["N"], t["ckpt"]) for t in todo) == [
        ("B", "16", "es"), ("B", "full", "es"), ("B", "full", "last")]
    have = {("tdcommons/ames", "B", "16", 42, "es"), ("tdcommons/ames", "B", "full", 42, "es")}
    todo = sc.todo_for("tdcommons/ames", have)
    assert [(t["run"]["N"], t["ckpt"]) for t in todo] == [("full", "last")]

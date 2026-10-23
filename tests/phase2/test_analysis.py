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
    return dict(task=task, arm=arm, N=n, seed=seed, ckpt=ckpt, metric=metric, direction=direction, score=score,
                subsample_sha1=f"sha-{task}-{n}-{seed}")


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


def test_curves_with_rand_arm_and_pair_check_ignores_extra_arms(tmp_path):
    t = _table(DECREASING)
    rand = t[t.arm == "B"].assign(arm="rand", score=lambda d: d.score - 0.05 * np.where(d.task == AMES, 1, -1))
    s1 = t[t.arm == "B"].assign(arm="B_s1")  # second-seed arms share the subsample (same sha1)
    full = pd.concat([t, rand, s1])
    out = tmp_path / "curves_rand.png"
    A.curves(full, out)
    assert out.stat().st_size > 5000
    assert ("B", "rand") in A.CONTRASTS and A.ARM_COLOR["rand"] and A.CONTRAST_COLOR[("B", "rand")]
    g = A.gaps(full, "B", "rand")
    assert len(g) and (g.gap > 0).all()  # pretrained B beats the (worse) random arm in higher-is-better units
    chk, ok = A.check_endpoint_pairs(full)
    assert ok and list(chk.filter(like="n_").columns) == ["n_T100", "n_B"]
    # a rand row on a different subsample is caught by the sha1 consistency check
    bad = pd.concat([t, rand.assign(subsample_sha1=lambda d: np.where(d.index == d.index[0], "x", d.subsample_sha1))])
    assert not A.check_endpoint_pairs(bad)[1]
    assert A.endpoint(full)["success"] == A.endpoint(t)["success"]


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


# ------------------------------------------------------------------------------------- endpoint guards
def test_pair_check_passes_on_complete_table():
    chk, ok = A.check_endpoint_pairs(_table(DECREASING))
    assert ok and chk.ok.all() and len(chk) == 2 * len(SIZES)
    assert dict(zip(chk[chk.task == AMES].N, chk[chk.task == AMES].paired)) == A.EXPECTED_PAIRS
    assert A.EXPECTED_PAIRS == {"16": 10, "32": 10, "64": 10, "128": 5, "256": 5, "512": 3, "1024": 3, "full": 3}
    assert (chk.sha_bad == 0).all()


def test_pair_check_missing_pair_fails():
    t = _table(DECREASING)
    t = t.drop(t[(t.task == LD50) & (t.arm == "T100") & (t.N == "16") & (t.seed == 42)].index)  # T100 lacks a seed
    t = t.drop(t[(t.task == AMES) & (t.arm == "B") & (t.N == "512") & (t.seed == 2042)].index)  # B lacks a seed
    chk, ok = A.check_endpoint_pairs(t)
    assert not ok
    bad = chk[~chk.ok].set_index(["task", "N"])
    assert set(bad.index) == {(LD50, "16"), (AMES, "512")}
    assert bad.loc[(LD50, "16"), "paired"] == 9 and bad.loc[(AMES, "512"), "paired"] == 2
    assert bad.loc[(LD50, "16"), "n_T100"] == 9 and bad.loc[(LD50, "16"), "n_B"] == 10


def test_pair_check_missing_size_and_unpaired_extra_row():
    t = _table(DECREASING)
    chk, ok = A.check_endpoint_pairs(t[t.N != "full"])
    assert not ok and chk[~chk.ok].N.tolist() == ["full", "full"]
    extra = pd.DataFrame([_rows(AMES, "T100", "16", 99999, 0.5)])  # T100-only seed: counted unpaired, not paired
    chk, ok = A.check_endpoint_pairs(pd.concat([_table(DECREASING), extra], ignore_index=True))
    r = chk[(chk.task == AMES) & (chk.N == "16")].iloc[0]
    assert ok and r.paired == 10 and r.n_T100 == 11 and r.n_B == 10


def test_pair_check_sha_mismatch_fails():
    t = _table(DECREASING)
    i = t[(t.task == AMES) & (t.arm == "T100") & (t.N == "32") & (t.seed == 1042)].index
    t.loc[i, "subsample_sha1"] = "different"
    chk, ok = A.check_endpoint_pairs(t)
    r = chk[(chk.task == AMES) & (chk.N == "32")].iloc[0]
    assert not ok and r.sha_bad == 1 and not r.ok and r.paired == 10
    assert chk[~chk.ok].shape[0] == 1
    # a third arm with a different sha for the same subsample is also caught
    t3 = pd.concat([_table(DECREASING), pd.DataFrame([{**_rows(AMES, "D535", "16", 42, 0.6), "subsample_sha1": "x"}])])
    assert not A.check_endpoint_pairs(t3)[1]
    # missing sha (None) is a failure too; so is a table without the column
    t4 = _table(DECREASING)
    t4.loc[t4.index[0], "subsample_sha1"] = None
    assert not A.check_endpoint_pairs(t4)[1]
    assert not A.check_endpoint_pairs(_table(DECREASING).drop(columns="subsample_sha1"))[1]


def _scores_file(tmp_path, table):
    p = tmp_path / "scores.parquet"
    table.to_parquet(p)
    return p


def _analyze(tmp_path, monkeypatch, table, *flags, n_done=784):
    an = _load_script("p2_05_analyze")
    runs = tmp_path / "runs"
    for k in range(n_done):
        d = runs / "phase2" / "ladder" / "t" / "B" / f"N{k}_s0"
        d.mkdir(parents=True)
        (d / "done.json").write_text("{}")
    monkeypatch.setattr(an, "RUNS", runs)
    monkeypatch.setattr("sys.argv", ["p2_05", "--scores", str(_scores_file(tmp_path, table)), "--out", str(tmp_path / "out"),
                                     *flags])
    return an


def test_analyze_script_stamps_complete_pairs(tmp_path, monkeypatch, capsys):
    import json

    an = _analyze(tmp_path, monkeypatch, _table(DECREASING), n_done=5)
    assert an.main() == 0
    out = capsys.readouterr().out
    assert "done.json under runs/phase2/ladder: 5 (expected 784)" in out and "paired" in out
    assert json.loads((tmp_path / "out" / "endpoint.json").read_text())["complete_pairs"] is True


def test_analyze_script_refuses_incomplete_unless_overridden(tmp_path, monkeypatch, capsys):
    import json

    t = _table(DECREASING)
    t = t[~((t.task == AMES) & (t.arm == "T100") & (t.N == "16") & (t.seed == 42))]
    an = _analyze(tmp_path, monkeypatch, t)
    assert an.main() != 0
    assert "INCOMPLETE" in capsys.readouterr().out and not (tmp_path / "out" / "endpoint.json").exists()
    (tmp_path / "again").mkdir()
    an = _analyze(tmp_path / "again", monkeypatch, t, "--allow-incomplete", n_done=0)
    assert an.main() == 0
    assert json.loads((tmp_path / "again" / "out" / "endpoint.json").read_text())["complete_pairs"] is False


# ------------------------------------------------------------------------- tailored arms / absent arms / score --root
def test_tailored_contrasts_and_arm_colors():
    for c in (("T100_ames", "B"), ("T100_ld50", "B"), ("T100_ames", "T100"), ("T100_ld50", "T100"), ("B", "rand")):
        assert c in A.CONTRASTS and A.CONTRAST_COLOR[c]
    assert A.CONTRASTS.count(("B", "rand")) == 1
    for a in ("rand", "T100_ames", "T100_ld50"):
        assert a in A.ARM_ORDER and A.ARM_COLOR[a]
    assert len(set(A.ARM_COLOR.values())) == len(A.ARM_ORDER)
    assert sum(len(r) for r in A.CONTRAST_ROWS) == len(A.CONTRASTS)


def _with_tailored(t):
    ames = t[t.arm == "B"].assign(arm="T100_ames", score=lambda d: d.score + 0.01 * np.where(d.task == AMES, 1, -1))
    return pd.concat([t, ames])


def test_curves_adds_tailored_row_only_with_data(tmp_path):
    import matplotlib.image as mpimg

    t = _table(DECREASING)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    A.curves(t, a)
    A.curves(_with_tailored(t), b)
    assert mpimg.imread(b).shape[0] > mpimg.imread(a).shape[0]  # 5th row of gap panels
    g = A.gaps(_with_tailored(t), "T100_ames", "B")
    assert len(g) and (g.gap > 0).all() and A.gaps(_with_tailored(t), "T100_ld50", "B").empty


def test_contrast_table_skips_absent_arms_with_notes():
    an = _load_script("p2_05_analyze")
    t = _with_tailored(_table(DECREASING))  # arms: B, T100, T100_ames only
    table, notes = an.contrast_table_and_notes(t)
    assert set(table.contrast) == {"T100-B", "T100_ames-B", "T100_ames-T100"}
    skipped = {n.split(":")[0] for n in notes}
    assert {"T25-B", "B-rand", "B-chemeleon", "T100_ld50-B", "T100_ld50-T100", "T100-T25"} <= skipped
    assert all("not in the score table" in n for n in notes if n.startswith("T100_ld50-B"))
    assert an.contrast_table(t).equals(table)
    md = an.summary_md(A.endpoint(t), table, t, notes)
    assert "## Skipped contrasts" in md and "T100_ld50-B: skipped" in md


def test_analyze_script_runs_with_missing_arms(tmp_path, monkeypatch, capsys):
    import json

    t = _table(DECREASING)
    t = t[t.arm == "B"].assign(arm="T100_ames")  # no T100 / chemeleon / ... : only B and T100_ames
    t = pd.concat([_table(DECREASING)[lambda d: d.arm == "B"], t])
    an = _analyze(tmp_path, monkeypatch, t, "--allow-incomplete", n_done=0)
    assert an.main() == 0
    out = capsys.readouterr().out
    assert "NOTE T100-B: skipped" in out and "NOTE T100_ames-B" not in out
    assert (tmp_path / "out" / "curves.png").exists() and (tmp_path / "out" / "gaps.csv").exists()
    assert pd.read_csv(tmp_path / "out" / "gaps.csv").contrast.unique().tolist() == ["T100_ames-B"]
    ep = json.loads((tmp_path / "out" / "endpoint.json").read_text())
    assert ep["success"] is False and ep["complete_pairs"] is False
    assert "## Skipped contrasts" in (tmp_path / "out" / "summary.md").read_text()


def test_score_script_root_and_out_args(tmp_path, monkeypatch):
    import json

    sc = _load_script("p2_04_score")
    monkeypatch.setattr(sc, "RUNS", tmp_path / "default_runs")
    root = tmp_path / "ladder_frozen"

    def mk(base, arm):
        d = base / "tdcommons_ames" / arm / "N16_s0"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(dict(task="tdcommons/ames", arm=arm, N=16, seed=42, subsample_sha1="h")))
        (d / "pred_es.npy").write_bytes(b"")
        (d / "done.json").write_text("{}")

    mk(root, "B")
    mk(tmp_path / "default_runs" / "phase2" / "ladder", "T100")
    assert [r["arm"] for r in sc.finished_runs("tdcommons/ames", root)] == ["B"]
    assert [r["arm"] for r in sc.finished_runs("tdcommons/ames")] == ["T100"]  # default root unchanged
    assert [t["run"]["arm"] for t in sc.todo_for("tdcommons/ames", set(), root)] == ["B"]
    # main: --root/--out routing (scoring itself stubbed: no Polaris)
    seen = {}
    monkeypatch.setattr(sc, "score_task", lambda name, todo: seen.setdefault(name, todo) and [
        dict(task=name, arm=t["run"]["arm"], N=t["run"]["N"], seed=int(t["run"]["seed"]), ckpt=t["ckpt"], metric="roc_auc",
             direction="max", score=0.7, subsample_sha1="h") for t in todo])
    out = tmp_path / "res" / "scores_frozen.parquet"
    assert sc.main(["--root", str(root), "--out", str(out)]) == 0
    df = pd.read_parquet(out)
    assert df.arm.tolist() == ["B"] and df.N.tolist() == ["16"]
    assert not sc.OUT.exists() or sc.OUT != out  # the default parquet was not touched

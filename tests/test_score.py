import pytest

from drift import config
from drift.score import PAPER_SEEDS, parse_paper_results, range_check


def _md(tmp_path, n=5):
    txt = ""
    for i in range(n):
        txt += f"## Random Seed {i}\n\n```\nresults_dict = {{\n    \"a/x\": {i + 1}.0,\n    \"b/y\": {10 * i}.5\n}}\n```\n"
    p = tmp_path / "r.md"
    p.write_text(txt)
    return p


def test_parse_orders_by_block(tmp_path):
    out = parse_paper_results(_md(tmp_path))
    assert out["a/x"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(out["b/y"]) == len(PAPER_SEEDS)


def test_parse_requires_five_blocks(tmp_path):
    with pytest.raises(AssertionError):
        parse_paper_results(_md(tmp_path, n=4))


def test_parse_real_file_covers_all_benchmarks():
    out = parse_paper_results(config.ROOT / "external/CheMeleon/analysis/polaris_results/CheMeleon.md")
    assert set(config.BENCHMARKS) <= set(out)
    assert all(len(v) == 5 for v in out.values())


def test_range_check():
    paper = [1.0, 2.0, 3.0, 4.0, 5.0]  # sd = sqrt(2.5)
    r = range_check(3.0, paper)
    assert (r["paper_min"], r["paper_max"], r["paper_mean"]) == (1.0, 5.0, 3.0)
    assert r["within_range"] and r["within_1sd_of_range"]
    r = range_check(5.5, paper)
    assert not r["within_range"] and r["within_1sd_of_range"]
    r = range_check(7.0, paper)
    assert not r["within_range"] and not r["within_1sd_of_range"]

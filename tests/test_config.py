from drift import config as C
def test_benchmarks():
    assert len(C.BENCHMARKS) == 28 and len(set(C.BENCHMARKS)) == 28
    assert C.BENCHMARKS[0] == "polaris/pkis2-ret-wt-cls-v2" and C.BENCHMARKS[-1] == "tdcommons/ld50-zhu"
def test_slug():
    assert C.slug("polaris/adme-fang-solu-1") == "polaris_adme_fang_solu_1"
def test_chemeleon_present():
    assert C.CHEMELEON_MP.exists()

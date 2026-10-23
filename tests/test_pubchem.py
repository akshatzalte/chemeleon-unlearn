import torch

from drift.pubchem import heldout_sample


def test_heldout_disjoint_from_pretrain_train(tmp_path):
    lines = [f"C{'C'*(i%7)}O{i}" for i in range(1000)]
    f = tmp_path / "x.smiles"
    f.write_text("\n".join(lines) + "\n")
    s = heldout_sample(f, n=50)
    tr, va, te = torch.utils.data.random_split(range(1000), [0.7, 0.2, 0.1], torch.Generator().manual_seed(1701))
    assert len(s) == 50 and set(s) <= {lines[i] for i in te.indices}
    assert not set(s) & {lines[i] for i in tr.indices}


def test_pretrain_membership_synthetic(tmp_path):
    import numpy as np
    from drift.pubchem import pretrain_membership
    lines = [f"C{'C' * (i % 9)}O{'N' * (i % 5)}{'c1ccccc1' * (i % 3)}" for i in range(60)]
    lines = list(dict.fromkeys(lines))
    f = tmp_path / "x.smiles"
    f.write_text("\n".join(lines) + "\n")
    cache = tmp_path / "map.parquet"
    tr, va, te = torch.utils.data.random_split(range(len(lines)), [0.7, 0.2, 0.1], torch.Generator().manual_seed(1701))
    exp = {i: n for n, ix in (("train", tr.indices), ("val", va.indices), ("test", te.indices)) for i in ix}
    q = [lines[0], lines[len(lines) // 2], "CCCCCCCCCCCCCCCCCCCCCCCCC", "not_a_smiles"]
    out = pretrain_membership(q, f, cache, nproc=1)
    assert out[0] == exp[0] and out[1] == exp[len(lines) // 2] and out[2] == "none" and out[3] == "none"
    assert cache.exists()
    out2 = pretrain_membership(q, f, cache, nproc=1)           # served from cache
    assert list(out2) == list(out)
    assert set(pretrain_membership(lines, f, cache, nproc=1)) <= {"train", "val", "test"}

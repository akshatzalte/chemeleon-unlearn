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

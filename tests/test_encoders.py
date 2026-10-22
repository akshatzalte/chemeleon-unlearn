import numpy as np
import torch

from drift.config import CHEMELEON_MP
from drift.encoders import embed, load_mp, random_mp, save_mp

SMI = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "not_a_smiles"]


def test_roundtrip(tmp_path):
    mp = load_mp(CHEMELEON_MP)
    save_mp(mp, tmp_path / "m.pt")
    mp2 = load_mp(tmp_path / "m.pt")
    assert all(torch.equal(a, b) for a, b in zip(mp.state_dict().values(), mp2.state_dict().values()))
    saved = torch.load(tmp_path / "m.pt", weights_only=True)
    assert saved["hyper_parameters"] == torch.load(CHEMELEON_MP, weights_only=True)["hyper_parameters"]


def test_embed_shape_and_bad_smiles():
    Z = embed(load_mp(CHEMELEON_MP), SMI, device="cpu")
    assert Z.dtype == np.float32
    assert Z.shape == (4, 2048) and np.isfinite(Z[:3]).all() and np.isnan(Z[3]).all()


def test_embed_batching_consistent():
    mp = load_mp(CHEMELEON_MP)
    np.testing.assert_allclose(
        embed(mp, SMI, batch_size=2, device="cpu")[:3], embed(mp, SMI, device="cpu")[:3], atol=1e-5
    )


def test_random_differs_and_is_seeded():
    a, b, c = random_mp(0), random_mp(0), random_mp(1)
    sa = a.state_dict()["W_h.weight"]
    assert torch.equal(sa, b.state_dict()["W_h.weight"]) and not torch.equal(sa, c.state_dict()["W_h.weight"])


def test_random_mp_saveable(tmp_path):
    save_mp(random_mp(0), tmp_path / "r.pt")
    assert load_mp(tmp_path / "r.pt").output_dim == 2048


def test_build_mpnn():
    from sklearn.preprocessing import StandardScaler
    from drift.encoders import build_mpnn

    sc = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
    m = build_mpnn(load_mp(CHEMELEON_MP), "regression", sc)
    assert m.predictor.input_dim == 2048
    assert build_mpnn(random_mp(0), "classification").predictor.input_dim == 2048

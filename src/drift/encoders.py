"""CheMeleon message-passing encoder: load, save, random init, wrap, embed."""
from pathlib import Path

import numpy as np
import torch
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BinaryClassificationFFN, BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.nn.transforms import UnscaleTransform
from rdkit import Chem

from drift.config import CHEMELEON_MP, FFN_HIDDEN


def _make_mp(hparams: dict) -> BondMessagePassing:
    mp = BondMessagePassing(**hparams)
    mp._drift_hparams = dict(hparams)
    return mp


def load_mp(path: Path) -> BondMessagePassing:
    ckpt = torch.load(path, weights_only=True)
    mp = _make_mp(ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    return mp


def save_mp(mp: BondMessagePassing, path: Path) -> None:
    """Save in the same format as chemeleon_mp.pt (loadable with weights_only=True)."""
    state = {k: v.detach().cpu() for k, v in mp.state_dict().items()}
    torch.save({"hyper_parameters": dict(mp._drift_hparams), "state_dict": state}, path)


def random_mp(seed: int) -> BondMessagePassing:
    """Randomly initialised encoder with CheMeleon hyperparameters."""
    hparams = torch.load(CHEMELEON_MP, weights_only=True)["hyper_parameters"]
    torch.manual_seed(seed)
    return _make_mp(hparams)


def build_mpnn(mp: BondMessagePassing, task_type: str, scaler=None, mpnn_kwargs: dict | None = None) -> MPNN:
    """MPNN with MeanAggregation and a fresh FFN head (chemprop defaults otherwise).

    `mpnn_kwargs` is passed to the `MPNN` constructor (e.g. `warmup_epochs`, `max_lr`)."""
    kw = dict(input_dim=mp.output_dim, hidden_dim=FFN_HIDDEN)
    if task_type == "regression":
        transform = UnscaleTransform.from_standard_scaler(scaler) if scaler is not None else None
        ffn = RegressionFFN(output_transform=transform, **kw)
    else:
        ffn = BinaryClassificationFFN(**kw)
    return MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn, **(mpnn_kwargs or {}))


def embed(mp: BondMessagePassing, smiles: list[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray:
    """MeanAggregation(mp(mol graph)) per molecule, float32 (len(smiles), d_h).

    Rows for SMILES RDKit cannot parse are NaN. Runs under no_grad in eval mode;
    `mp` is moved to `device` and left there (train/eval mode is restored).
    """
    feat = SimpleMoleculeMolGraphFeaturizer()
    agg = MeanAggregation()
    was_training = mp.training
    mp.to(device).eval()
    out = np.full((len(smiles), mp.output_dim), np.nan, dtype=np.float32)
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
    valid = [i for i, m in enumerate(mols) if m is not None]
    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            idx = valid[start : start + batch_size]
            bmg = BatchMolGraph([feat(mols[i]) for i in idx])
            bmg.to(device)
            H = mp(bmg)
            Z = agg(H, bmg.batch)
            out[idx] = Z.float().cpu().numpy()
    mp.train(was_training)
    return out

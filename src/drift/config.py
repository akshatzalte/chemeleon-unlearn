from pathlib import Path

ROOT = Path("/data3/akshatz/chemeleon_unlearn")
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
CHEMELEON_MP = Path.home() / ".chemprop/chemeleon_mp.pt"

BENCHMARKS: tuple[str, ...] = (
    "polaris/pkis2-ret-wt-cls-v2",
    "polaris/pkis2-ret-wt-reg-v2",
    "polaris/pkis2-kit-wt-cls-v2",
    "polaris/pkis2-kit-wt-reg-v2",
    "polaris/pkis2-egfr-wt-reg-v2",
    "polaris/adme-fang-solu-1",
    "polaris/adme-fang-rppb-1",
    "polaris/adme-fang-hppb-1",
    "polaris/adme-fang-perm-1",
    "polaris/adme-fang-rclint-1",
    "polaris/adme-fang-hclint-1",
    "tdcommons/lipophilicity-astrazeneca",
    "tdcommons/ppbr-az",
    "tdcommons/clearance-hepatocyte-az",
    "tdcommons/cyp2d6-substrate-carbonmangels",
    "tdcommons/half-life-obach",
    "tdcommons/cyp2c9-substrate-carbonmangels",
    "tdcommons/clearance-microsome-az",
    "tdcommons/dili",
    "tdcommons/bioavailability-ma",
    "tdcommons/vdss-lombardo",
    "tdcommons/cyp3a4-substrate-carbonmangels",
    "tdcommons/pgp-broccatelli",
    "tdcommons/caco2-wang",
    "tdcommons/herg",
    "tdcommons/bbb-martins",
    "tdcommons/ames",
    "tdcommons/ld50-zhu",
)

SEEDS = (42, 117, 709)
REGIMES = ("full", "n128")
MODES = ("ft", "frozen")
N_SMALL = 128
MAX_EPOCHS = 50
PATIENCE = 5
FFN_HIDDEN = 256
WINSOR = 6.0
NAN_FRAC_MAX = 0.2
N_BOOT = 200
DROP_ABS = 0.05
DROP_SE_MULT = 3.0
CLUSTER_RHO = 0.9
N_PERM = 1000
TOP_K_POSCTRL = 20
N_PUBCHEM = 10_000
N_PUBCHEM_FIT = 8_000
N_FOLDS = 5


def slug(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")

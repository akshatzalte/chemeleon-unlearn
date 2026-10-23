"""Phase 2, step 6: the `rand` ladder arm = a never-pretrained encoder (run under $PY1; CPU, seconds).

    $PY1 scripts/p2_06_make_rand.py

Writes runs/phase2/pretrain/rand/{mp.pt, done.json, config.json}: drift.encoders.random_mp(0) (CheMeleon
hyper-parameters, random init) saved in chemeleon_mp.pt format. ladder.arm_ready treats `rand` like `chemeleon`
(ready once mp.pt exists, no patience / pool checks). Idempotent: overwrites the three files.
"""
import json
import sys

from drift import encoders
from drift.phase2 import ladder


def main() -> int:
    d = ladder.pretrain_dir("rand")
    d.mkdir(parents=True, exist_ok=True)
    encoders.save_mp(encoders.random_mp(0), d / "mp.pt")
    (d / "config.json").write_text(json.dumps(dict(arm="rand", patience=None, note="random init, no pretraining")))
    (d / "done.json").write_text(json.dumps(dict(arm="rand", seed=0, note="random init, no pretraining")))
    print(f"wrote {d}/{{mp.pt,config.json,done.json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

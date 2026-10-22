"""Fetch all 28 benchmarks from the Polaris hub into data/tasks/<slug>/ and print a summary."""
import json

from drift import config
from drift.config import BENCHMARKS, slug
from drift.tasks import fetch_task


def main() -> None:
    rows = []
    for name in BENCHMARKS:
        out = config.DATA / "tasks" / slug(name)
        if not (out / "meta.json").exists():
            fetch_task(name, out)
        m = json.loads((out / "meta.json").read_text())
        rows.append((name, m["task_type"], m["n_train"], m["n_test"]))
    print(f"{'name':45s} {'type':15s} {'n_train':>8s} {'n_test':>8s}")
    for r in rows:
        print(f"{r[0]:45s} {r[1]:15s} {r[2]:8d} {r[3]:8d}")
    print(f"{len(rows)} tasks; total train {sum(r[2] for r in rows)}, test {sum(r[3] for r in rows)}")


if __name__ == "__main__":
    main()

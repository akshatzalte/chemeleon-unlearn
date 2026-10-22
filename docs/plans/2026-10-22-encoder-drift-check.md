# CheMeleon Encoder-Drift Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune CheMeleon on the 28 Polaris/TDC benchmarks (full fine-tune and frozen-encoder, full-data and N=128, 3 seeds), then measure how far each fine-tuned encoder drifts from the pretrained one — CKA, weight change, and per-descriptor linear-probe R² on 1613 Mordred descriptors — with noise floors, a positive control and a pre-specified descriptor-specificity test, and produce the tables that decide the next phase.

**Architecture:** A small Python package `drift` (pure, unit-tested functions) plus numbered scripts that run the pipeline stages and write to `data/` and `results/`. Task data is fetched from Polaris once and cached, so fine-tuning and probing never touch the hub; only the final scoring step needs a Polaris login. Fine-tuning runs are independent jobs on a resumable worklist sharded across the two GPUs.

**Tech Stack:** Python 3.11, chemprop 2.2.4, lightning, torch (CUDA), polaris-lib 0.12.1, zarr<3, mordredcommunity, rdkit, astartes, scikit-learn, scipy, numpy, pandas, pyarrow, pytest.

**Spec:** `/data3/akshatz/chemeleon_unlearn/docs/specs/2026-10-22-encoder-drift-check-design.md`

## Global Constraints

- Project root: `/data3/akshatz/chemeleon_unlearn`. Conda env name: `chemeleon_unlearn`. All commands below run as `conda run -n chemeleon_unlearn ...` from the project root (abbreviated `$PY` = `conda run -n chemeleon_unlearn python`, `$PYTEST` = `conda run -n chemeleon_unlearn pytest`).
- `chemprop>=2.2,<2.3` (pin `2.2.4`); `polaris-lib==0.12.1` (the paper's version); `zarr<3`.
- Mordred: `Calculator(descriptors, ignore_3D=True)`, `calc.config(timeout=1)`, `.fill_missing()` → must yield exactly **1613** columns.
- CheMeleon encoder: `~/.chemprop/chemeleon_mp.pt`, loaded with `torch.load(..., weights_only=True)`, keys `hyper_parameters`, `state_dict`, into `chemprop.nn.BondMessagePassing`; aggregation `MeanAggregation`; featurizer `SimpleMoleculeMolGraphFeaturizer()` (defaults to V2 multi-hot atoms).
- Fine-tuning protocol (paper): Polaris train/test split; NaN targets → train mean; `astartes.train_test_split(train_size=0.8, test_size=0.2, random_state=seed)`; regression targets standardized + `UnscaleTransform`; `RegressionFFN`/`BinaryClassificationFFN` with `hidden_dim=256`; `MPNN(mp, agg, ffn)` chemprop defaults; `max_epochs=50`; early-stopping rule patience 5 on `val_loss`.
- Seeds: `(42, 117, 709)`. Regimes: `full`, `n128` (N=128). Modes: `ft`, `frozen`. Checkpoints: `es`, `last`.
- Probe targets: standardize with fit-split mean/SD, clip ±6 SD; drop descriptors constant or >20% NaN on the fit split; remaining NaNs masked, never imputed.
- PubChem probe set: 10,000 molecules from the **test** 10% of `random_split(range(N), [0.7, 0.2, 0.1], torch.Generator().manual_seed(1701))` over `cleaned_pubchem_1MM.smiles` (download from Zenodo 15733574; never copy from other users' directories). Split 8,000 fit / 2,000 eval.
- Task probe sets: Polaris **train** molecules only, 5-fold cross-fitting, folds fixed per task (seed 0). Per run, *clean rows* = train molecules that run took no gradient steps on (`full`: its val rows; `n128`: all rows outside its training rows). Seed ΔR² is computed on that seed's clean rows.
- Drop call: ΔR² < 0 in all 3 seeds AND seed-mean ΔR² < −3·SE AND seed-mean ΔR² < −0.05. Paired bootstrap, 200 resamples.
- Correlation families: |Spearman ρ| ≥ 0.9 on the PubChem fit set. η² permutation null: ≥1,000 permutations, family sizes preserved, 95th percentile.
- Positive control: top-20 descriptors by |Spearman ρ| with the label on Polaris train molecules only.
- Polaris test sets stay held out as-is: test molecules are used only to produce test predictions inside fine-tuning runs, and test labels only by `benchmark.evaluate` in `scripts/05_score.py`. No test molecule enters descriptor computation, probes, CKA, or positive controls.
- Large outputs live under `data/` and `results/` (git-ignored). `external/` is read-only reference.

## Review Focus

1. **Unparseable / odd molecules in task sets** (salts, mixtures, molecules RDKit or Mordred chokes on): the pipeline must not crash; such molecules get NaN descriptors (masked) and a count is logged per task. → test in Task 5.
2. **Classification subsample with a rare class at N=128** (e.g., a task with ~10% positives): resampling must terminate and both classes must be present in train and val. → test in Task 2.
3. **Interrupted or OOM fine-tuning job**: re-running the worklist must skip finished runs and redo half-written ones (a run is done only when `done.json` exists, written last). → test in Task 4.
4. **Polaris token expiring mid-pipeline**: nothing after Task 2's fetch may call the hub except `scripts/05_score.py`. → test in Task 2 (cached loader works with the hub unreachable).
5. **Duplicate SMILES within a task's train set**: keep every row; folds are assigned by row and clean rows are taken by row index from `run.json`, so a duplicate of a training molecule can appear among clean rows — log the count per run rather than silently dropping. → test in Task 6 (`make_folds` is row-based) and Task 9 (clean rows from `run.json`).

---

## File Structure

```
environment.yml                  # pinned env
pyproject.toml                   # package `drift` (src layout) + pytest config
.gitignore                       # data/, results/, external/, lightning_logs/
src/drift/
  config.py                      # constants & paths (benchmarks, seeds, thresholds)
  tasks.py                       # fetch/cache Polaris tasks; train/val splits per regime
  encoders.py                    # load/save/random BondMessagePassing; build MPNN; embed
  finetune.py                    # PatienceBestCheckpoint callback; run_one()
  descriptors.py                 # Mordred calc; probe-target preprocessing; families
  pubchem.py                     # held-out PubChem sample
  probe.py                       # linear CKA, masked ridge, cross-fitting, paired bootstrap
  stats.py                       # drop calls, η² + permutation null, correlation clusters, positive control
  measure.py                     # per-encoder measurement → long-format rows
scripts/
  00_fetch_tasks.py              # Polaris → data/tasks/<slug>/   (needs login)
  01_descriptors.py              # Mordred for task + PubChem molecules → data/descriptors/
  02_finetune.py                 # worklist runner (--gpu, --shard i/n, --only)
  03_measure.py                  # drift measures → results/drift_long.parquet
  04_summarize.py                # summary tables + figures → results/summary/
  05_score.py                    # benchmark.evaluate on saved predictions (needs login)
tests/                           # one test file per module
```

Shared on-disk layout (all tasks rely on it):
- `data/tasks/<slug>/train.parquet` (cols `smiles`, `y`), `test.parquet` (col `smiles`), `meta.json` (`name`, `task_type` ∈ {`regression`,`classification`}, `n_train`, `n_test`). `<slug>` = `"".join(c if c.isalnum() else "_" for c in name)`.
- `runs/<slug>/<regime>/<mode>/seed<seed>/`: `es_mp.pt`, `last_mp.pt` (ft only), `es_head.pt`, `last_head.pt`, `pred_es.npy`, `pred_last.npy` (test-set predictions; probabilities for classification), `metrics.csv` (per-epoch train/val loss), `run.json` (`es_epoch`, sizes, timings, `train_idx`, `val_idx` as lists of indices into the task's train rows), `done.json` (written last). `runs/` lives at `/data3/akshatz/chemeleon_unlearn/runs`.
- `data/descriptors/<slug>.parquet` and `data/descriptors/pubchem10k.parquet`: one row per input row (task: Polaris train rows only, same order as `train.parquet`; PubChem: sample order), 1613 descriptor columns in calculator order, plus `smiles`.

---

### Task 1: Environment, repo scaffold, config

**Files:**
- Create: `environment.yml`, `pyproject.toml`, `.gitignore`, `src/drift/__init__.py`, `src/drift/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces (`drift.config`): `ROOT: Path`, `DATA: Path`, `RUNS: Path`, `RESULTS: Path`, `CHEMELEON_MP: Path = Path.home()/".chemprop/chemeleon_mp.pt"`, `BENCHMARKS: tuple[str, ...]` (the 28, in the order of `external/CheMeleon/analysis/fine-tuning/scripts/finetune.py:POLARIS_BENCHMARKS`), `SEEDS = (42, 117, 709)`, `REGIMES = ("full", "n128")`, `MODES = ("ft", "frozen")`, `N_SMALL = 128`, `MAX_EPOCHS = 50`, `PATIENCE = 5`, `FFN_HIDDEN = 256`, `WINSOR = 6.0`, `NAN_FRAC_MAX = 0.2`, `N_BOOT = 200`, `DROP_ABS = 0.05`, `DROP_SE_MULT = 3.0`, `CLUSTER_RHO = 0.9`, `N_PERM = 1000`, `TOP_K_POSCTRL = 20`, `N_PUBCHEM = 10_000`, `N_PUBCHEM_FIT = 8_000`, `N_FOLDS = 5`, `def slug(name: str) -> str`.

- [ ] **Step 1:** `git init` in the project root; write `.gitignore` (`data/`, `runs/`, `results/`, `external/`, `lightning_logs/`, `__pycache__/`, `*.egg-info/`).
- [ ] **Step 2:** Write `environment.yml` (python=3.11, pip: `chemprop==2.2.4`, `polaris-lib==0.12.1`, `zarr<3`, `mordredcommunity`, `astartes`, `scikit-learn`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `pytest`; torch with CUDA 12.x wheels) and `pyproject.toml` (src layout, `[tool.pytest.ini_options] testpaths=["tests"]`). Create the env and `pip install -e .`.
- [ ] **Step 3:** Verify the stack: `$PY -c "import chemprop, polaris, torch; from mordred import Calculator, descriptors; print(chemprop.__version__, polaris.__version__, torch.cuda.is_available(), len(Calculator(descriptors, ignore_3D=True)))"` → `2.2.4 0.12.1 True 1613`. If the count is not 1613, pin `mordredcommunity==2.0.6` and re-check. Record `conda env export` to `environment.lock.yml`.
- [ ] **Step 4: Write the failing test** `tests/test_config.py`:
```python
from drift import config as C
def test_benchmarks():
    assert len(C.BENCHMARKS) == 28 and len(set(C.BENCHMARKS)) == 28
    assert C.BENCHMARKS[0] == "polaris/pkis2-ret-wt-cls-v2" and C.BENCHMARKS[-1] == "tdcommons/ld50-zhu"
def test_slug():
    assert C.slug("polaris/adme-fang-solu-1") == "polaris_adme_fang_solu_1"
def test_chemeleon_present():
    assert C.CHEMELEON_MP.exists()
```
- [ ] **Step 5:** Run `$PYTEST tests/test_config.py -v` → FAIL (module missing). Implement `config.py`. Re-run → PASS.
- [ ] **Step 6:** Ask the user to run `! conda run -n chemeleon_unlearn polaris login` and confirm.
- [ ] **Step 7: Commit** `git add -A && git commit -m "chore: env, scaffold, config"`.

### Task 2: Task data fetch, cache, and splits

**Files:**
- Create: `src/drift/tasks.py`, `scripts/00_fetch_tasks.py`
- Test: `tests/test_tasks.py`, `tests/conftest.py` (helpers `fake_task(n, kind, pos_frac=0.5) -> TaskData` with dummy SMILES, `write_fake_cache(path)`, and `fake_task_real_smiles(n, kind)` using ~80 hard-coded drug-like SMILES — reused by Tasks 4 and 9)

**Interfaces:**
- Consumes: `drift.config`.
- Produces (`drift.tasks`):
  - `@dataclass TaskData: name: str; task_type: str; train_smiles: list[str]; y_raw: np.ndarray  # (n,) float with NaNs; y_train: np.ndarray  # (n,) NaN filled with full-train mean; test_smiles: list[str]`
  - `fetch_task(name: str, out_dir: Path) -> None` — loads `po.load_benchmark(name)`, asserts exactly one target column, writes the cached files described in "Shared on-disk layout". Test labels are never written.
  - `load_task(name: str) -> TaskData` — reads only the cache; no network.
  - `make_split(task: TaskData, regime: str, seed: int) -> tuple[np.ndarray, np.ndarray]` — indices into `train_smiles` for (train, val).

`make_split` algorithm (pins the spec's choices):
```
idx = arange(n_train)
if regime == "n128" and n_train > 128:
    rng = np.random.default_rng(seed)
    for attempt in range(1000):
        sub = rng.choice(idx, 128, replace=False)
        tr, va = astartes.train_test_split(sub, train_size=0.8, test_size=0.2, random_state=seed)
        if task_type != "classification" or (both classes in y[tr] and in y[va]): break
    else: raise RuntimeError
else:
    tr, va = astartes.train_test_split(idx, train_size=0.8, test_size=0.2, random_state=seed)
```
For `n128`, the caller (Task 4) re-fills NaN targets from `y_raw` with the mean over `sub`.

- [ ] **Step 1: Write failing tests** (use a synthetic `TaskData`, no network):
```python
def test_full_split_matches_paper():   # same as astartes call in the paper
    t = fake_task(n=500, kind="regression")
    tr, va = make_split(t, "full", 42)
    tr2, va2 = astartes.train_test_split(np.arange(500), train_size=0.8, test_size=0.2, random_state=42)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
def test_n128_sizes_and_determinism():
    t = fake_task(n=500, kind="regression")
    tr, va = make_split(t, "n128", 117)
    assert len(tr) + len(va) == 128 and len(set(tr) & set(va)) == 0
    assert np.array_equal(tr, make_split(t, "n128", 117)[0])
def test_n128_small_task_uses_all():
    t = fake_task(n=100, kind="regression")
    tr, va = make_split(t, "n128", 42)
    assert len(tr) + len(va) == 100
def test_n128_rare_class_has_both_classes():          # Review Focus 2
    t = fake_task(n=600, kind="classification", pos_frac=0.08)
    for s in (42, 117, 709):
        tr, va = make_split(t, "n128", s)
        assert set(t.y_train[tr]) == {0, 1} and set(t.y_train[va]) == {0, 1}
def test_load_task_offline(tmp_path, monkeypatch):    # Review Focus 4
    write_fake_cache(tmp_path / "polaris_x")          # helper writes train/test/meta
    monkeypatch.setattr("polaris.load_benchmark", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
    t = load_task_from(tmp_path / "polaris_x")        # load_task delegates to this
    assert t.task_type in ("regression", "classification")
```
- [ ] **Step 2:** Run `$PYTEST tests/test_tasks.py -v` → FAIL.
- [ ] **Step 3:** Implement `tasks.py` (signatures above; add `load_task_from(path) -> TaskData`, used by `load_task`). `task_type` from `benchmark.target_types[target_col]` mapped to `"regression"`/`"classification"`.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Run `$PY scripts/00_fetch_tasks.py` (loops `BENCHMARKS`, calls `fetch_task`). Expected: 28 dirs under `data/tasks/`; print a table of name, type, n_train, n_test. Cross-check totals against `external/CheMeleon/analysis/polaris_results/PCA_MLP.md` ("Fitted on entire dataset" lines) — they must match.
- [ ] **Step 6: Commit** `feat: Polaris task cache and splits`.

### Task 3: Encoder utilities

**Files:**
- Create: `src/drift/encoders.py`
- Test: `tests/test_encoders.py`

**Interfaces:**
- Produces (`drift.encoders`):
  - `load_mp(path: Path) -> BondMessagePassing` (weights_only load of `{"hyper_parameters", "state_dict"}`).
  - `save_mp(mp: BondMessagePassing, path: Path) -> None` (same dict format as `chemeleon_mp.pt`; `hyper_parameters` copied from the loaded file, stored on the module as `mp._drift_hparams`).
  - `random_mp(seed: int) -> BondMessagePassing` (CheMeleon hyperparameters, `torch.manual_seed(seed)` before construction).
  - `build_mpnn(mp, task_type: str, scaler=None) -> MPNN` (MeanAggregation; FFN `hidden_dim=FFN_HIDDEN`, `input_dim=mp.output_dim`; `UnscaleTransform.from_standard_scaler(scaler)` for regression).
  - `embed(mp, smiles: list[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray` — float32 `(len(smiles), 2048)`, MeanAggregation output, `mp.eval()`, `torch.no_grad()`. Rows for molecules RDKit cannot parse are NaN.

- [ ] **Step 1: Write failing tests:**
```python
SMI = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "not_a_smiles"]
def test_roundtrip(tmp_path):
    mp = load_mp(CHEMELEON_MP); save_mp(mp, tmp_path/"m.pt"); mp2 = load_mp(tmp_path/"m.pt")
    assert all(torch.equal(a, b) for a, b in zip(mp.state_dict().values(), mp2.state_dict().values()))
def test_embed_shape_and_bad_smiles():
    Z = embed(load_mp(CHEMELEON_MP), SMI, device="cpu")
    assert Z.shape == (4, 2048) and np.isfinite(Z[:3]).all() and np.isnan(Z[3]).all()
def test_random_differs_and_is_seeded():
    a, b, c = random_mp(0), random_mp(0), random_mp(1)
    sa = a.state_dict()["W_h.weight"]
    assert torch.equal(sa, b.state_dict()["W_h.weight"]) and not torch.equal(sa, c.state_dict()["W_h.weight"])
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: encoder load/save/embed`.

### Task 4: Fine-tuning runner

**Files:**
- Create: `src/drift/finetune.py`, `scripts/02_finetune.py`
- Test: `tests/test_finetune.py`

**Interfaces:**
- Consumes: `TaskData`, `make_split`, `load_mp`, `save_mp`, `build_mpnn`, config constants.
- Produces (`drift.finetune`):
  - `class PatienceBestCheckpoint(lightning.Callback)`: `__init__(self, patience: int = 5)`; attributes `best_epoch: int`, `best_val: float`, `stopped: bool`, `es_state: dict | None` (deep copy of `pl_module.state_dict()` on CPU at the best epoch *before* patience ran out). Logic on each `on_validation_epoch_end`: if not stopped — if val < best_val: update best and snapshot; elif epoch − best_epoch ≥ patience: `stopped = True`. Never stops training. Skip the sanity-check validation pass (`trainer.sanity_checking`).
  - `run_one(task: TaskData, regime: str, mode: str, seed: int, out_dir: Path, accelerator: str = "gpu", max_epochs: int = MAX_EPOCHS) -> dict` — records `train_idx`/`val_idx` in `run.json`; builds data per the paper protocol (for `n128`, NaN targets refilled with the subsample mean using `y_raw`), `lightning.seed_everything(seed, workers=True)`, `deterministic=False` (chemprop scatter ops are not deterministic on CUDA), `build_dataloader(..., num_workers=4)` defaults otherwise. `mode == "frozen"`: set `requires_grad=False` on all `message_passing` params before fit. Trains `max_epochs`, then writes all files listed in "Shared on-disk layout" (`last_mp.pt`/`es_mp.pt` only for `ft`), predicting the test set with `es` and `last` weights, and writes `done.json` **last**. Returns the `run.json` dict.
  - `run_dir(task: str, regime: str, mode: str, seed: int) -> Path` (`RUNS/slug/regime/mode/seed<seed>`); `pending(items) -> list` (items without `done.json`; half-written dirs are deleted by the script before rerun).
  - `worklist() -> list[tuple[str, str, str, int]]` — `(task, regime, mode, seed)` over BENCHMARKS × REGIMES × MODES × SEEDS, skipping `n128` for tasks with `n_train ≤ 128`, ordered largest tasks first.
- `scripts/02_finetune.py --gpu G --shard I/N [--only SLUG ...] [--seeds ...]`: iterates `worklist()[I::N]`, skips items whose `done.json` exists, deletes and reruns dirs without it, sets `CUDA_VISIBLE_DEVICES=G`, logs timing per run.

- [ ] **Step 1: Write failing tests:**
```python
def test_patience_callback_logic():
    cb = PatienceBestCheckpoint(patience=5)
    vals = [1.0, 0.8, 0.9, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.5, 0.4]  # 0.5 at epoch 9 comes after patience ran out
    for e, v in enumerate(vals): cb._update(e, v, state={"w": torch.tensor(float(e))})
    assert cb.best_epoch == 3 and cb.stopped and cb.es_state["w"].item() == 3.0
def test_frozen_keeps_encoder(tmp_path):            # tiny CPU run on a synthetic task, 2 epochs
    t = fake_task_real_smiles(n=60, kind="regression")
    run_one(t, "full", "frozen", 42, tmp_path, accelerator="cpu", max_epochs=2)
    head = torch.load(tmp_path/"es_head.pt")        # exists
    assert not (tmp_path/"es_mp.pt").exists() and (tmp_path/"done.json").exists()
def test_ft_moves_encoder_and_writes_preds(tmp_path):
    t = fake_task_real_smiles(n=60, kind="classification")
    info = run_one(t, "full", "ft", 42, tmp_path, accelerator="cpu", max_epochs=2)
    assert np.load(tmp_path/"pred_es.npy").shape == (len(t.test_smiles),)
    mp0, mp1 = load_mp(CHEMELEON_MP), load_mp(tmp_path/"last_mp.pt")
    assert not torch.equal(mp0.state_dict()["W_h.weight"], mp1.state_dict()["W_h.weight"])
def test_worklist_resume(tmp_path, monkeypatch):    # Review Focus 3
    monkeypatch.setattr(config, "RUNS", tmp_path)
    items = worklist()[:2]
    for it in items: run_dir(*it).mkdir(parents=True)  # run_dir(task, regime, mode, seed) -> Path
    (run_dir(*items[0]) / "done.json").write_text("{}")
    assert pending(items) == [items[1]]                 # pending() skips done, keeps half-written
```
(`_update(epoch, val, state)` is the callback's internal method the Lightning hook calls; testing it directly avoids a trainer.) `fake_task_real_smiles` comes from `tests/conftest.py` (Task 2).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: fine-tuning runner with patience-best checkpoint`.

### Task 5: Mordred descriptors, PubChem sample, probe-target preprocessing

**Files:**
- Create: `src/drift/descriptors.py`, `src/drift/pubchem.py`, `scripts/01_descriptors.py`
- Test: `tests/test_descriptors.py`, `tests/test_pubchem.py`

**Interfaces:**
- Produces (`drift.descriptors`):
  - `compute_mordred(smiles: list[str], nproc: int = 64) -> pd.DataFrame` — 1613 columns in calculator order (`str(d)` names), one row per input; unparseable SMILES → all-NaN row; logs the count.
  - `descriptor_modules() -> pd.Series` — index = descriptor name, value = Mordred module short name (`type(d).__module__.split(".")[-1]`).
  - `@dataclass Targets: Y: np.ndarray (n, k) float32 with NaN; names: list[str]; dropped: list[str]`
  - `prepare_targets(D: pd.DataFrame, fit_rows: np.ndarray) -> Targets` — drop columns constant or >20% NaN on `fit_rows`; standardize with fit-row mean/SD (nan-aware); clip ±6.
- Produces (`drift.pubchem`): `heldout_sample(smiles_file: Path, n: int = 10_000, seed: int = 0) -> list[str]` — reconstructs the 0.7/0.2/0.1 `random_split` with `torch.Generator().manual_seed(1701)` over `range(len(lines))`, samples `n` from the test indices with `np.random.default_rng(seed)`.
- `scripts/01_descriptors.py`: downloads `cleaned_pubchem_1MM.smiles` from Zenodo 15733574 to `data/pubchem/` if absent (check line count 944,296 is not required — just record it); writes `data/pubchem/pubchem10k.smiles`, `data/descriptors/pubchem10k.parquet`, and `data/descriptors/<slug>.parquet` for each task (Polaris train rows only).

- [ ] **Step 1: Write failing tests:**
```python
def test_mordred_columns_and_bad_smiles():                  # Review Focus 1
    D = compute_mordred(["CCO", "c1ccccc1", "not_a_smiles", "[Na+].[Cl-]"], nproc=2)
    assert D.shape == (4, 1613) and D.iloc[2].isna().all()
def test_prepare_targets():
    rng = np.random.default_rng(0)
    D = pd.DataFrame({"a": rng.normal(5, 2, 100), "const": np.ones(100),
                      "nanny": np.where(rng.random(100) < 0.5, np.nan, 1.0), "out": np.r_[rng.normal(0,1,99), 1e6]})
    T = prepare_targets(D, np.arange(80))
    assert set(T.dropped) == {"const", "nanny"}
    a = T.Y[:80, T.names.index("a")]; assert abs(a.mean()) < 1e-5 and abs(a.std() - 1) < 1e-2
    assert np.nanmax(T.Y[:, T.names.index("out")]) <= 6.0
def test_heldout_disjoint_from_pretrain_train():
    lines = [f"C{'C'*(i%7)}O" for i in range(1000)]; f = write_tmp(lines)
    s = heldout_sample(f, n=50)
    tr, va, te = torch.utils.data.random_split(range(1000), [0.7, 0.2, 0.1], torch.Generator().manual_seed(1701))
    assert set(s) <= {lines[i] for i in te.indices}
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `$PY scripts/01_descriptors.py`. Expected: 29 parquet files; log of dropped/NaN counts per set.
- [ ] **Step 6: Commit** `feat: mordred descriptors, pubchem held-out sample`.

### Task 6: Probe and CKA math

**Files:**
- Create: `src/drift/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Produces (`drift.probe`):
  - `linear_cka(X: np.ndarray, Y: np.ndarray) -> float` — column-centered; `‖YᵀX‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F)`.
  - `choose_alpha(Z: np.ndarray, Y: np.ndarray, alphas=np.logspace(-2, 5, 15)) -> float` — 5-fold CV mean R² over descriptors (NaN-masked), on standardized Z.
  - `ridge_fit_predict(Z_fit, Y_fit, Z_eval, alpha) -> np.ndarray` — Z standardized with fit stats; exact NaN masking: group target columns by identical NaN pattern on fit rows and solve each group on its non-NaN rows (shared SVD per group).
  - `crossfit_predict(Z, Y, folds: np.ndarray, alpha) -> np.ndarray` — out-of-fold predictions; `folds` is an int array of fold ids per row.
  - `make_folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray` — row-based (Review Focus 5).
  - `r2_cols(Y, Yhat, rows=None) -> np.ndarray` — per-column R² over non-NaN rows (SST around the eval mean).
  - `paired_bootstrap_delta(Y, Yhat_pre, Yhat_list: list[np.ndarray], rows_list: list[np.ndarray], n_boot=200, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(delta_per_seed (S,k), delta_mean (k,), se_mean (k,))`. Seed s is evaluated on `rows_list[s]`; each resample is shared by `pre` and that seed (paired). `se_mean = sqrt(Σ_s se_s²) / S`. For PubChem pass the same eval rows for every seed.
  - α policy (spec 6.3): PubChem — `choose_alpha` on `pre`'s fit-split embeddings; task sets — `choose_alpha` on `pre`'s embeddings of the task's train rows. Then fixed for all encoders on that set.

- [ ] **Step 1: Write failing tests:**
```python
def test_cka_properties():
    X = rng.normal(size=(300, 50)); Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    assert abs(linear_cka(X, X) - 1) < 1e-9 and abs(linear_cka(X, 3 * X @ Q) - 1) < 1e-9
    assert linear_cka(X, rng.normal(size=(300, 50))) < 0.3
def test_ridge_recovers_linear_target_with_nans():
    Z = rng.normal(size=(500, 20)); W = rng.normal(size=(20, 3)); Y = Z @ W
    Y[rng.random(500) < 0.1, 1] = np.nan
    P = ridge_fit_predict(Z[:400], Y[:400], Z[400:], alpha=1e-3)
    assert (r2_cols(Y[400:], P) > 0.99).all()
def test_crossfit_is_out_of_fold():
    Z = rng.normal(size=(100, 5)); Y = rng.normal(size=(100, 2)); f = make_folds(100)
    P = crossfit_predict(Z, Y, f, alpha=1.0)
    assert r2_cols(Y, P).max() < 0.2        # pure noise → no in-sample leak
def test_paired_bootstrap_tighter_than_unpaired():
    Y = rng.normal(size=(400, 10)); eps = rng.normal(size=(400, 10))
    Ya, Yb = Y + eps, Y + eps + 0.01 * rng.normal(size=(400, 10))   # shared noise
    rows = np.arange(400)
    _, _, se_paired = paired_bootstrap_delta(Y, Ya, [Yb], [rows])
    # unpaired: bootstrap each R² independently, SE of difference = sqrt(se_a² + se_b²)
    se_unpaired = unpaired_se(Y, Ya, Yb, rows)               # test-local helper
    assert se_paired.mean() < 0.3 * se_unpaired.mean()
def test_duplicate_smiles_rows_kept():                 # Review Focus 5
    assert len(make_folds(10)) == 10
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: probe, CKA, paired bootstrap`.

### Task 7: Statistics — drop calls, families, η², positive control

**Files:**
- Create: `src/drift/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces (`drift.stats`):
  - `drop_calls(delta_per_seed (S,k), delta_mean (k,), se_mean (k,), abs_thr=0.05, se_mult=3.0) -> np.ndarray[bool]` — rule from Global Constraints.
  - `correlation_clusters(Y_fit: np.ndarray, names: list[str], rho=0.9) -> pd.Series` — Spearman (NaN-pairwise), distance `1 − |ρ|`, average linkage, `fcluster(t=1 − rho, criterion="distance")`; index = names.
  - `family_delta(delta_mean, drop, families: pd.Series) -> pd.DataFrame` — per family: mean ΔR², n members, fraction dropped, `dropped = frac > 0.5`.
  - `eta2(values: np.ndarray, labels: np.ndarray) -> float` — between-group SS / total SS.
  - `eta2_perm_test(values, labels, n_perm=1000, seed=0) -> tuple[float, float, float]` — `(eta2_obs, null_q95, p_value)`; permutes labels (sizes preserved).
  - `positive_control(D_train: pd.DataFrame, y_train: np.ndarray, names: list[str], k=20) -> list[str]` — top-k by |Spearman ρ| with label, NaN-aware, Polaris train molecules only.

- [ ] **Step 1: Write failing tests:**
```python
def test_drop_calls():
    dps = np.array([[-.10, -.10, .01, -.02], [-.12, -.09, -.20, -.03], [-.11, -.11, -.20, -.02]])
    dm = dps.mean(0); se = np.array([.01, .05, .01, .001])
    assert drop_calls(dps, dm, se).tolist() == [True, False, False, False]
def test_eta2_detects_structure():
    lab = np.repeat(np.arange(10), 30)
    v_struct = lab * 0.1 + rng.normal(0, .01, 300); v_flat = rng.normal(0, .1, 300)
    assert eta2_perm_test(v_struct, lab)[0] > eta2_perm_test(v_struct, lab)[1]
    assert eta2_perm_test(v_flat, lab)[2] > 0.01
def test_clusters_group_correlated():
    base = rng.normal(size=(500, 1)); Y = np.c_[base, 2*base + 1e-3*rng.normal(size=(500,1)), rng.normal(size=(500,1))]
    c = correlation_clusters(Y, ["a", "b", "c"]); assert c["a"] == c["b"] != c["c"]
def test_positive_control_picks_label_correlated():
    y = rng.normal(size=200); D = pd.DataFrame({"good": y + .1*rng.normal(size=200), "bad": rng.normal(size=200)})
    assert positive_control(D, y, ["good", "bad"], k=1) == ["good"]
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: drop calls, families, eta2 permutation test, positive control`.

### Task 8: Pilot fine-tuning, scoring, sanity check

**Files:**
- Create: `scripts/05_score.py`
- Test: none beyond the checks below (integration step).

**Interfaces:**
- Consumes: `scripts/02_finetune.py`, cached tasks, `runs/` layout.
- Produces: `results/scores.parquet` — columns `task, regime, mode, seed, ckpt, metric, score`; `results/sanity.csv` — per task: our `full/ft/es` score per seed, paper min/max/mean over 5 seeds (parsed from the five `results_dict = {...}` blocks in `external/CheMeleon/analysis/polaris_results/CheMeleon.md`), `within_range: bool`.
- `05_score.py`: for each task, `po.load_benchmark(name)` once, then for each finished run and ckpt: classification → `benchmark.evaluate(pred > 0.5, pred)`, regression → `benchmark.evaluate(pred)`; keep the row where `Metric == benchmark.main_metric.label` (same as paper `evaluate.py`). Before evaluating, assert the freshly loaded test SMILES list equals the cached `test.parquet` order — otherwise predictions are misaligned.

- [ ] **Step 1:** Pilot: `$PY scripts/02_finetune.py --gpu 0 --shard 0/1 --only tdcommons_lipophilicity_astrazeneca polaris_pkis2_kit_wt_cls_v2 --seeds 42`. Expected: 8 run dirs with `done.json` (2 tasks × {full,n128} × {ft,frozen}). Record wall-clock per run in `run.json`.
- [ ] **Step 2:** Run `$PY scripts/05_score.py` (needs a valid Polaris login; ask the user to re-login if expired). Expected: `results/sanity.csv` with `within_range == True` for both pilot tasks' seed-42 `full/ft/es` (or within ~1 paper-SD of the range). If not, stop and investigate the protocol before continuing — this is the spec 5.4 gate.
- [ ] **Step 3:** Extrapolate total GPU time from pilot timings; report it to the user before launching the full run.
- [ ] **Step 4:** Full run, in background, one shard per GPU: `$PY scripts/02_finetune.py --gpu 0 --shard 0/2` and `--gpu 1 --shard 1/2`. Check GPU 1's free memory first (it is shared). Expected: 336 minus skipped `n128` runs, all with `done.json`.
- [ ] **Step 5:** Re-run `05_score.py`; `results/sanity.csv` for all 28 tasks. Report any task outside the paper's range.
- [ ] **Step 6: Commit** `feat: scoring and sanity check` (code only; results are git-ignored, but commit `results/sanity.csv` copy into `docs/results/`).

### Task 9: Drift measurement

**Files:**
- Create: `src/drift/measure.py`, `scripts/03_measure.py`
- Test: `tests/test_measure.py`

**Interfaces:**
- Consumes: `embed`, `load_mp`, `random_mp`, `prepare_targets`, `descriptor_modules`, all of `probe` and `stats`.
- Produces (`drift.measure`):
  - `weight_change(mp, mp_pre) -> dict[str, float]` — per parameter tensor `‖W − W_pre‖_F / ‖W_pre‖_F`.
  - `measure_probe_set(Z_pre, Z_by_encoder: dict[str, np.ndarray], targets: Targets, eval_rows, mode: str, folds=None, alpha=None) -> dict` — `mode` ∈ {`"split"`, `"crossfit"`}; returns per-encoder predictions and R² per descriptor plus CKA vs `pre` on `eval_rows`.
  - Output `results/drift_long.parquet` columns: `task, regime, ckpt, probe_set ∈ {pubchem, task_all, task_clean}, encoder ∈ {pre, rand0..2, seed42, seed117, seed709}, measure ∈ {r2, delta_r2_mean, se, dropped, cka, cka_seedpair, wchange}, descriptor (or param name or ""), value`. `cka_seedpair` = mean CKA between the 3 fine-tuned seeds of the same task/regime/ckpt (the seed-noise anchor).
  - Also `results/families.parquet` (family-level, both definitions) and `results/eta2.parquet` (`task, regime, ckpt, probe_set, family_def, eta2, null_q95, p`), `results/posctrl.parquet` (`task, regime, ckpt, probe_set, descriptor, delta_r2_mean, se`).
- `scripts/03_measure.py`: for PubChem (embed once per encoder, cache embeddings in `data/emb/` as float16 if disk allows) and for each task (cross-fitted over train rows; `task_all` = all train rows, `task_clean` = each seed's clean rows from its `run.json`: `val_idx` for `full`, complement of `train_idx` for `n128`). `rand` and `pre` are evaluated on the same per-seed rows they are paired with. `ft` encoders only (`frozen` produces none); `rand` × 3 and `pre` on every probe set. Weight change for each `ft` encoder.

- [ ] **Step 1: Write failing test** — synthetic end-to-end with tiny fake embeddings (no GPU): encoder "same" = `Z_pre` → ΔR² ≈ 0, CKA = 1, nothing dropped; encoder "lost" = `Z_pre` with the 5 columns that carry descriptor 0 zeroed → descriptor 0 dropped in all seeds, others not.
```python
def test_measure_detects_planted_loss():
    out = measure_probe_set(Z_pre, {"seed42": Z_lost, "seed117": Z_lost, "seed709": Z_lost}, T, eval_rows, mode="crossfit", folds=make_folds(n))
    assert out["dropped"][0] and not out["dropped"][1:].any()
def test_measure_identity_no_change(): ...
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `$PY scripts/03_measure.py --only <pilot tasks>` first, inspect output sanity (pre R² distribution on PubChem: most descriptors should be well recovered, median R² > 0.5; `rand` lower), then the full run.
- [ ] **Step 6: Commit** `feat: drift measurement`.

### Task 10: Summaries and findings tables

**Files:**
- Create: `scripts/04_summarize.py`
- Test: `tests/test_summarize.py` (one test: on a tiny synthetic `drift_long.parquet`, the per-task summary has one row per task × regime × ckpt × probe_set and the cross-task count of drops equals the planted count).

**Interfaces:**
- Consumes: `results/*.parquet`, `results/scores.parquet`.
- Produces in `results/summary/`:
  - `per_task.csv` — task, regime, ckpt, probe_set, CKA (mean±SD over seeds), median weight change, median ΔR², n dropped descriptors, n dropped families (both defs), `es` epoch (mean), η² vs null (p), positive-control mean ΔR², `ft − frozen` test-score gap (mean±SD, sign-normalized so positive = fine-tuning better), `low_confidence` flag (task fit set < 1,000).
  - `cross_task.csv` — per descriptor and per family: number of the 28 tasks in which it is dropped, for each of the two co-primary views (`full`/`es`/`ft` × {`task_clean`, `pubchem`}) plus secondary views.
  - `heatmap_<view>.png` — task × Mordred-module ΔR² for each co-primary view.
  - `anchors.csv` — CKA and median ΔR² for `rand` and seed-vs-seed per task.

- [ ] **Step 1:** Write the failing test. **Step 2:** Run → FAIL. **Step 3:** Implement (use the `dataviz` skill before writing plotting code). **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `$PY scripts/04_summarize.py` on the full results.
- [ ] **Step 6:** Write `docs/results/2026-10-xx-drift-findings.md`: which spec section 9 branch the two co-primary views support (agree / split), with the key numbers (CKA, drops, η² tests, positive control, `ft − frozen` gaps at `full` and `n128`), and caveats (low-confidence tasks). Bring it to the user.
- [ ] **Step 7: Commit** `feat: summaries and findings`.

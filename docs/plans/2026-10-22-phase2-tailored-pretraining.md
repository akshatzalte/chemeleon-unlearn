# Phase 2: Tailored Descriptor Pretraining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pretrain CheMeleon-style encoders on five descriptor target sets (B=all 1613, T25, T100, R25, R100) with one identical pipeline, fine-tune each on Ames and LD50 across a paired downsampling ladder (16 → full), and evaluate the pre-registered endpoint (T100 − B at small N).

**Architecture:** New subpackage `drift.phase2` (sets, pool, pretrain data/model, ladder runner, analysis) plus scripts `p2_*`. Pretraining runs in a new conda env `chemeleon_pretrain` (chemprop 2.3.1 + cuik-molmaker 0.2); everything else (sets, pool descriptors, fine-tuning, scoring) stays in the phase-1 env `chemeleon_unlearn` (chemprop 2.2.4), which loads the new encoders through the existing `chemeleon_mp.pt` format. Fine-tuning reuses `drift.finetune` with a few new parameters.

**Tech Stack:** Python 3.11, chemprop 2.3.1 + cuik-molmaker 0.2 (pretraining only), chemprop 2.2.4 (fine-tuning), lightning, torch CUDA, mordredcommunity, rdkit, polaris-lib 0.12.1, numpy/pandas/scikit-learn, pytest.

**Spec:** `/data3/akshatz/chemeleon_unlearn/docs/specs/2026-10-22-phase2-tailored-pretraining-design.md` (decisions D1–D8 are binding).

## Global Constraints

- Project root `/data3/akshatz/chemeleon_unlearn`, branch `drift-check` (continue on it). Phase-1 code and results are read-only except the explicit `drift.finetune` extension in Task 4. Never modify `runs/<slug>/…` (phase-1 runs) or `results/drift*`.
- Envs: `$PY1 = /home/akshatz/anaconda3/envs/chemeleon_unlearn/bin/python` (phase 1; chemprop 2.2.4); `$PY2 = /home/akshatz/anaconda3/envs/chemeleon_pretrain/bin/python` (new; `chemprop==2.3.1`, `cuik-molmaker==0.2`, torch CUDA 12.x, lightning). Pretraining uses `$PY2`; all else `$PY1`.
- Architecture for every new arm (D5): `BondMessagePassing(d_v=72, d_e=14, d_h=2048, depth=6, bias=False, dropout=0.0, activation="relu")`, `MeanAggregation`, `RegressionFFN(n_tasks=k, input_dim=2048, hidden_dim=2048, n_layers=1)`; featurization `CuikmolmakerMolGraphFeaturizer(atom_featurizer_mode="V2")` on batches of SMILES, **no graph caching**.
- Pretraining hyperparameters, identical across arms: batch 512, `init_lr=1e-4, max_lr=1e-3, final_lr=1e-4, warmup_epochs=5`, max 100 epochs, EarlyStopping patience 5 on pool-val masked MSE, Adam (chemprop default). Loss = masked MSE over `finite(target) AND rand() < 0.7` (D7). Targets standardized per column on the pool train split and clipped to ±6.
- Arm B = all 1613 descriptors (D7); T/R from the 1367 that pass phase-1's filter (`results/corr_clusters.parquet` descriptor index). Sets: T25, T100 (consensus relevance rank), R25a (seed 0), R25b (seed 1), R100 (seed 0) — one representative per cluster from `results/summary/phase2_descriptor_sets.json["D535"]`. Optional D535.
- Pool: P=200k molecules sampled (row indices) from `data/pubchem/cleaned_pubchem_1MM.smiles` with `np.random.default_rng(0)`, Ames/LD50 Polaris **test** molecules removed by non-isomeric canonical SMILES match, 90/10 train/val split (seed 0); descriptors taken from CheMeleon's published Zenodo zarr (no recomputation); standardization stats from pool-train rows only. Same pool and split for all arms. P may be raised after the compute gate since descriptors are free.
- **Test everything before committing pretraining compute (user rule):** every pretraining component must pass its unit tests AND an end-to-end smoke run (`--limit-rows 2000 --max-epochs 2`, both GPUs, every arm's set file) that produces a checkpoint loadable by the phase-1 env and fine-tunable by `drift.finetune.run_one` on a tiny task, before the compute gate and before any full arm is launched.
- Ladder: N ∈ {16, 32, 64, 128, 256, 512, 1024, full}; seeds 10 (N ≤ 64), 5 (128, 256), 3 (≥ 512); one subsample per (task, N, seed), Ames stratified; all encoders (B, T25, T100, R25a, R25b, R100, CheMeleon) fine-tune on the identical subsample. N ≤ 64: batch min(N,16), 16 optimizer steps (2 warm-up), no val split, final weights = `es`. N ≥ 128: paper protocol (`drift.finetune.run_one`). Graph caching ON for fine-tuning. Head hidden 256.
- Scoring: `benchmark.evaluate` exactly as `scripts/05_score.py`; metrics `roc_auc` (Ames, max) and `mean_absolute_error` (LD50, min), sign-normalized in analysis. Polaris test labels/molecules are used only there.
- Primary endpoint (spec §7): per task and N, d_s = T100 − B paired per subsample (sign-normalized); gap(N)=mean, SD(N)=SD of d_s; success = gap > SD at ≥ 2 of N ∈ {16,32,64} in each task AND Spearman(gap, log N) < 0 AND mean gap(N≤64) > mean gap(N≥512); `es` checkpoint.
- Outputs under `data/phase2/`, `runs/phase2/`, `results/phase2/` (git-ignored); summaries copied to `docs/results/phase2/` and committed. Commits end with the attribution trailer from the system reminder.

## Review Focus

1. **A pool molecule that cuik-molmaker cannot featurize** (or RDKit rejects) must be dropped at pool-build time, not crash a batch mid-epoch. → test in Task 2 (pool builder filters with the same featurizer path) and Task 3 (dataset asserts no None).
2. **Column order mismatch between an arm's set file and the descriptor matrix** would silently train on the wrong targets. → Task 3 test: targets are selected by descriptor *name* against the saved column-name list, and the saved checkpoint records its target names.
3. **Paired subsampling drift**: an arm fine-tuned on a different subsample at the same (task, N, seed) invalidates the paired endpoint. → Task 4 test: subsample indices are generated once, saved, and every run records the hash of its indices.
4. **Fixed-step mode at N=16 with batch 16**: exactly 16 optimizer steps, LR warm-up over 2 steps, no validation loader. → Task 4 test counts optimizer steps.
5. **Sign normalization for LD50 (MAE, lower is better)** in gaps and in the endpoint. → Task 5 test with a synthetic scores table.

---

## File Structure

```
src/drift/phase2/
  __init__.py
  sets.py          # regenerate descriptor sets from saved scores; FFN/linear sanity check
  pool.py          # sample pool, exclude test molecules, split, run Mordred, save matrix + stats
  pretrain_data.py # (PY2) memmap-backed dataset → TrainingBatch with cuik featurizer; masked-MSE loss
  pretrain.py      # (PY2) build model, train, save chemeleon-format checkpoint
  ladder.py        # subsample generation; run_ladder_item() using drift.finetune; worklist/pending
  analysis.py      # gaps, endpoint, curves
scripts/
  p2_00_sets.py    # sets + sanity gate → data/phase2/sets/, results/phase2/sanity_ffn.csv
  p2_01_pool.py    # → data/phase2/pool/
  p2_02_pretrain.py  --arm ARM --gpu G [--max-epochs] [--limit-rows]   (PY2)
  p2_03_ladder.py    --gpu G --shard i/n [--only-arm ...]
  p2_04_score.py
  p2_05_analyze.py
tests/phase2/test_{sets,pool,pretrain_data,pretrain,ladder,analysis}.py
environment-pretrain.yml
```

On-disk layout:
- `data/phase2/sets/<arm>.txt` (one descriptor name per line; arms: B, T25, T100, R25a, R25b, R100, D535), `data/phase2/sets/summary.json`.
- `data/phase2/pool/{smiles_train.txt, smiles_val.txt, desc_train.npy, desc_val.npy (float32 memmap, rows × 1613), columns.json, stats.npz (mean, sd per column from train), build.json (counts, excluded)}`.
- `runs/phase2/pretrain/<arm>/{mp.pt (chemeleon format + "target_names"), head.pt, metrics.csv, config.json, done.json}`.
- `data/phase2/subsamples/<slug>/N<N>_s<seed>.json` (train row indices into the task's train set).
- `runs/phase2/ladder/<slug>/<arm>/N<N>_s<seed>/{pred_es.npy, run.json, done.json}` (+ `pred_last.npy` for N ≥ 128).
- `results/phase2/{sanity_ffn.csv, pretrain_timing.json, scores.parquet, gaps.csv, endpoint.json, curves.png}`.

---

### Task 1: Pretraining env, descriptor sets, sanity gate

**Files:** Create `environment-pretrain.yml`, `src/drift/phase2/__init__.py`, `src/drift/phase2/sets.py`, `scripts/p2_00_sets.py`, `tests/phase2/test_sets.py`.

**Interfaces (produces):**
- `sets.representatives() -> dict[int, str]` cluster id → representative descriptor (from `results/summary/phase2_descriptor_sets.json["D535"]` and `results/corr_clusters.parquet`).
- `sets.build_sets(scores_csv: Path, clusters_parquet: Path, sets_json: Path, all_descriptors: list[str]) -> dict[str, list[str]]` returning `{"B": [1613], "D535": [...], "T25", "T100", "R25a", "R25b", "R100"}`; T_k = top-k clusters by `rank_consensus` in `results/summary/phase2_relevance_scores.csv`; R_k = `np.random.default_rng(seed).choice(sorted(cluster_ids), k, replace=False)` with seed 0 (a) / 1 (b); B = `descriptor_modules().index` order.
- `sets.sanity_check(sets, tasks=("tdcommons/ames","tdcommons/ld50-zhu"), model="ffn"|"linear") -> pd.DataFrame` — 5-fold CV on Polaris train rows (`drift.tasks.load_task`, descriptors from `data/descriptors/<slug>.parquet`), 3 seeds, columns task,set,n_descr,score,sd; FFN = sklearn MLP (256,) early stopping, as in `docs/results/pick_and_ffn.py`.

- [ ] Step 1: Write failing tests: `test_T_sets_nested_and_sized` (T25 ⊂ T100, sizes 25/100, all in D535), `test_R_sets_seeded` (R25a ≠ R25b, deterministic), `test_B_is_all_1613_in_module_order`, `test_sanity_check_shape` (on a 60-row synthetic task via monkeypatch, linear model).
- [ ] Step 2: Run → FAIL. Step 3: Implement. Step 4: Run → PASS.
- [ ] Step 5: Create env `chemeleon_pretrain` from `environment-pretrain.yml` (python 3.11, `chemprop==2.3.1`, `cuik-molmaker==0.2`, torch cu126, lightning, numpy, pandas, pyarrow, rdkit); verify `$PY2 -c "from chemprop.featurizers.molgraph.molecule import CuikmolmakerMolGraphFeaturizer as F; print(F(atom_featurizer_mode='V2')(['CCO','c1ccccc1']).V.shape)"` → `(8, 72)` or similar (V rows = atoms, 72 features).
- [ ] Step 6: Run `$PY1 scripts/p2_00_sets.py` → `data/phase2/sets/*.txt`, `results/phase2/sanity_ffn.csv`. Gate (spec 3.1): T_k ≥ R_k on ≥ 1 task at each k (compare with both R25 draws). Print the table; commit sets' summary.json and sanity csv under `docs/results/phase2/`.
- [ ] Step 7: Commit.

### Task 2: Pretraining pool

**Files:** Create `src/drift/phase2/pool.py`, `scripts/p2_01_pool.py`, `tests/phase2/test_pool.py`.

**Interfaces (produces):**
- `pool.sample_pool(smiles_file, n=200_000, seed=0, exclude: set[str]) -> list[str]` — canonicalize with `drift.pubchem`'s non-isomeric canonical SMILES helper (reuse `pretrain_membership`'s canonicalization function; expose it as `drift.pubchem.canonical(smi) -> str|None` if not already public); exclude = canonical Ames + LD50 Polaris **test** SMILES (`load_task(name).test_smiles`); drop SMILES RDKit rejects; sample without replacement from the remaining lines in file order with `default_rng(seed)`.
- `pool.split(smiles, frac_val=0.1, seed=0) -> (train, val)`.
- **Descriptors are NOT recomputed** (user instruction): they come from CheMeleon's published pretraining descriptors, Zenodo record 15733575 `pubchem_1MM.zarr.tar.xz` (the controller is downloading/extracting it to `data/phase2/zenodo/`; a `pubchem_1MM.zarr` directory appears next to `extracted.ok`). It is a zarr **v3** array of shape (944296, 1613) float32, row-aligned with `data/pubchem/cleaned_pubchem_1MM.smiles` (assert `shape[0] == number of lines` and `shape[1] == 1613`); NaN = missing. Reading it needs `zarr>=3`, which conflicts with the phase-1 env (`zarr<3` for polaris) — so `scripts/p2_01_pool.py` runs under **`$PY2`** and must not import polaris; read the cached task test SMILES directly from `data/tasks/<slug>/test.parquet`.
- `pool.build(out_dir, zarr_path, smiles_file, row_idx_train, row_idx_val)` — gathers the sampled rows from the zarr (chunk-aligned reads) into memmaps `desc_train.npy`/`desc_val.npy` (float32, NaN preserved), writes `columns.json` (1613 names in calculator order from `drift.descriptors.descriptor_modules().index`, mordredcommunity is in the PY2 env), `stats.npz` (nanmean/nanstd per column over **pool train rows only**; sd==0 or non-finite → 1), `build.json` (n_sampled, n_excluded_test, n_invalid, n_train, n_val, zarr_shape, wall_s), and `rows.json` (the zarr row indices used).
- Scaling rule (user warning): standardization/winsorization stats are computed on the **pool train split**, never on the full 1M; every arm uses the same per-column stats, selecting its columns afterwards. Rows whose descriptor vector is all-NaN are dropped before splitting.
- Alignment sanity: match `data/pubchem/pubchem10k.smiles` molecules to their line numbers in the cleaned SMILES file, read those zarr rows, and compare with `data/descriptors/pubchem10k.parquet` (allclose, NaN-equal) on ≥ 500 rows — verifies the zarr is row-aligned with the SMILES file and uses the same Mordred config. Abort if it fails.

- [ ] Step 1: Failing tests (run under `$PY2`; `pytest.importorskip("zarr")`): `test_exclusion_by_canonical_smiles` (a stereo variant of an excluded molecule is excluded), `test_sample_deterministic`, `test_build_small` (a synthetic 300×1613 zarr v3 array + SMILES list → shapes, columns.json length 1613, stats from train rows only, all-NaN rows dropped, rows.json round-trips).
- [ ] Step 2–4: FAIL → implement → PASS.
- [ ] Step 5: Wait for `data/phase2/zenodo/extracted.ok`, then run `$PY2 scripts/p2_01_pool.py` (minutes, I/O bound; ~1.3 GB output). Record counts in `build.json`. Commit code (not data).

### Task 3: Pretraining script + compute gate

**Files:** Create `src/drift/phase2/pretrain_data.py`, `src/drift/phase2/pretrain.py`, `scripts/p2_02_pretrain.py`, `tests/phase2/test_pretrain_data.py`, `tests/phase2/test_pretrain.py` (tests run with `$PY2`; mark them `pytest.importorskip("cuik_molmaker")`).

**Interfaces (produces):**
- `pretrain_data.PoolDataset(smiles: list[str], desc_memmap: np.memmap, col_idx: np.ndarray, mean, sd, batch_size=512, shuffle=False, seed=0)` — `torch.utils.data.Dataset` returning one **batch** per item (`batch_size=None` in DataLoader): `TrainingBatch(bmg=CuikmolmakerMolGraphFeaturizer("V2")(smiles_batch), V_d=None, X_d=None, Y=standardized+clipped targets (NaN preserved), w=ones, lt_mask=None, gt_mask=None)`. Targets selected by `col_idx` derived from names. Shuffles batch membership each epoch when `shuffle=True` (reseed per epoch).
- `pretrain_data.MaskedDropoutMSE(keep_prob=0.7)` — chemprop `LossFunction` subclass: `mask = isfinite(Y) & (rand_like(Y) < keep_prob)`; loss = sum((pred−Y)²·mask)/mask.sum(); validation uses `keep_prob=1.0` (plain masked MSE), logged as `val_loss`.
- `pretrain.build_model(k: int) -> MPNN` per Global Constraints (criterion = MaskedDropoutMSE for the FFN).
- `pretrain.train(arm: str, pool_dir: Path, sets_dir: Path, out_dir: Path, gpu: int, max_epochs=100, limit_rows: int|None=None) -> dict` — trains, early-stops on `val_loss`, saves best `mp.pt` in `chemeleon_mp.pt` format (`{"hyper_parameters": {...same keys/values as chemeleon_mp.pt...}, "state_dict": mp.state_dict(), "target_names": [...], "arm": arm}`), `head.pt`, `metrics.csv` (epoch, train_loss, val_loss, seconds), `config.json` (all hyperparameters, pool build.json, git sha), `done.json` last. Returns timing (`s_per_epoch`, `epochs_run`).
- `scripts/p2_02_pretrain.py --arm ARM --gpu G [--max-epochs E] [--limit-rows R] [--out DIR]`.

- [ ] Step 1: Failing tests (`$PY2`): `test_batch_shapes_and_name_selection` (20 SMILES, 1613-col random memmap, set of 5 names in non-contiguous positions → Y has 5 columns in set order; bmg.V.shape[1]==72), `test_masked_loss_ignores_nan_and_drops` (NaN targets contribute 0; keep_prob=1 equals plain masked MSE), `test_checkpoint_loadable_by_phase1` (train 1 epoch on 64 rows CPU, then `$PY1`-compatible: `torch.load(weights_only=True)` has keys hyper_parameters/state_dict with the same hyper_parameters dict as `~/.chemprop/chemeleon_mp.pt` and `BondMessagePassing(**hp).load_state_dict(sd)` succeeds — run this part via subprocess with `$PY1`).
- [ ] Step 2–4: FAIL → implement → PASS.
- [ ] Step 5 (**compute gate**, needs Task 2's pool): `$PY2 scripts/p2_02_pretrain.py --arm B --gpu 0 --max-epochs 2 --out /tmp/…/gate` → `results/phase2/pretrain_timing.json` with s/epoch on 180k rows; extrapolate 100 epochs × arms. Report to the controller before launching anything else. Decision rule (spec §8): if B alone > 24 GPU-h, cut P to 100k (re-run Task 2 with n=100k) before proceeding.
- [ ] Step 6: Commit.

### Task 4: Fine-tuning ladder runner

**Files:** Modify `src/drift/finetune.py` (additive parameters only), create `src/drift/phase2/ladder.py`, `scripts/p2_03_ladder.py`, `tests/phase2/test_ladder.py`.

**Interfaces:**
- `drift.finetune.run_one(..., mp_path: Path | None = None, train_idx: np.ndarray | None = None, fixed_steps: int | None = None, batch_size: int | None = None, cache_graphs: bool = False)` — `mp_path` overrides `CHEMELEON_MP`; `train_idx` (with `regime="custom"`) bypasses `make_split` (train = given rows; for `fixed_steps` there is no val split and no early stopping: train for exactly `fixed_steps` optimizer steps — set `max_epochs = ceil(fixed_steps / n_batches)`, `warmup_epochs` so that warm-up ≈ 2 steps, stop via a `Callback` on `global_step == fixed_steps`; `es` = final weights; write `pred_es.npy` only). `cache_graphs=True` sets `dataset.cache = True` on train/val/test `MoleculeDataset`s (verify attribute exists in chemprop 2.2.4; otherwise pre-featurize datapoints once).
- `ladder.make_subsamples(task: TaskData, sizes, seeds_for, out_dir) -> None` — stratified for classification (`sklearn.model_selection.train_test_split(stratify=…)`), saved as JSON with `sha1` of the index list; idempotent.
- `ladder.worklist() -> list[(task, arm, N, seed)]`, `ladder.run_dir(...)`, `ladder.pending(items)`, `ladder.run_item(task, arm, N, seed, gpu)`: arm → `mp_path` (`runs/phase2/pretrain/<arm>/mp.pt`, or `CHEMELEON_MP` for arm `chemeleon`); N ≤ 64 → `fixed_steps=16, batch_size=min(N,16)`; N ≥ 128 → paper protocol with 80/20 split of the subsample; `N=full` → all rows. `run.json` records `subsample_sha1`, `arm`, `mp_path`, `n_steps`.
- `scripts/p2_03_ladder.py --gpu G --shard i/n [--only-arm A ...] [--only-task T ...] [--jobs-per-gpu J]`; skips done, deletes half-written.

- [ ] Step 1: Failing tests: `test_fixed_steps_counts_optimizer_steps` (N=16, batch 16 → exactly 16 steps; N=64 → 16 steps over 4 epochs; counted via a callback), `test_subsample_paired_and_stratified` (same indices across arms; Ames-like labels keep class ratio within 1 at N=16), `test_run_json_records_sha_and_arm`, `test_cache_graphs_flag` (second epoch does not re-featurize: monkeypatch featurizer call counter).
- [ ] Step 2–4: FAIL → implement → PASS (CPU, tiny, `fake_task_real_smiles`).
- [ ] Step 5: Commit.

### Task 5: Scoring, analysis, findings

**Files:** Create `src/drift/phase2/analysis.py`, `scripts/p2_04_score.py`, `scripts/p2_05_analyze.py`, `tests/phase2/test_analysis.py`.

**Interfaces:**
- `p2_04_score.py`: same as `scripts/05_score.py` over `runs/phase2/ladder/**` → `results/phase2/scores.parquet` (task, arm, N, seed, ckpt, metric, direction, score), incremental; asserts test-SMILES order.
- `analysis.gaps(scores, arm_a, arm_b) -> pd.DataFrame` (task, N, gap, sd, n, se) paired by (task, N, seed), sign-normalized by direction.
- `analysis.endpoint(scores) -> dict` implementing spec §7 exactly (keys: per task: `points_pass`, `spearman`, `small_gt_large`, `success`; overall `success` = both tasks).
- `analysis.curves(scores, out_png)` — score vs log N per arm (mean ± SE), one panel per task; and gap panels (T100−B, T25−B, T100−R100, T25−R25a/b, B−chemeleon). Load the `dataviz` skill before plotting.
- `scripts/p2_05_analyze.py` → `results/phase2/{gaps.csv, endpoint.json, curves.png, summary.md}`.

- [ ] Step 1: Failing tests: `test_gap_sign_for_min_metric` (MAE: lower ft score ⇒ positive gap), `test_endpoint_success_and_failure` (two synthetic score tables), `test_pairing_drops_unmatched_seeds`.
- [ ] Step 2–4: FAIL → implement → PASS.
- [ ] Step 5 (after the ladder finishes; Polaris login needed): score, analyze, write `docs/results/2026-10-xx-phase2-findings.md` (controller), commit summaries.

### Execution order (controller)
Task 1 → Task 2 (pool runs in background ~2 h) → Task 3 (tests on a small pool slice; gate on the real pool) → **compute gate decision** → launch pretraining arms two at a time (B and T100 first, then T25, R100, R25a, R25b; D535 only if cheap) → Task 4 (while pretraining runs) → ladder for `chemeleon` arm can start immediately; other arms as their checkpoints finish → Task 5 → score (Polaris login) → analyze → findings.

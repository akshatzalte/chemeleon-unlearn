# CheMeleon encoder-drift check — design

Date: 2026-10-22
Status: approved by user 2026-10-22

## 1. Why this exists

The larger research hypothesis (the user's): CheMeleon is pre-trained to predict all 1613
2D Mordred descriptors. Some of what that bakes into the encoder is irrelevant or harmful
for a given downstream task, and fine-tuning has to "unlearn" it, which costs data. If so,
re-pre-training on a task-tailored descriptor subset should help in low and ultra-low data
regimes.

That hypothesis rests on a premise we have not verified: **that fine-tuning actually
changes the CheMeleon encoder in a measurable, descriptor-specific way.** There is a
reason to doubt it. In the CheMeleon repo's own fine-tuning study
(`external/CheMeleon/analysis/fine-tuning/*.csv`, 28 tasks × 5 seeds), the best
validation loss with the message-passing encoder frozen is 0.455 vs. 0.438 with full
fine-tuning, and full fine-tuning overfits within a few epochs (mean val loss 0.89 at
epoch 30). With early stopping the encoder only trains for a handful of epochs, so it may
barely move.

This check answers, cheaply and before any re-pre-training is built:

1. How far does the CheMeleon encoder drift during fine-tuning, relative to meaningful
   reference points (seed-to-seed noise, a random encoder)?
2. Is any drop in descriptor recoverability descriptor-specific — concentrated in
   particular descriptors / descriptor families, consistently across seeds and tasks — or
   uniform/noise?

Its outcome decides the next phase (section 9).

## 2. Goals, non-goals, anti-goals

**Goals**
- Fine-tune CheMeleon on all 28 Polaris/TDC benchmarks used in the CheMeleon paper,
  under the paper's fine-tuning protocol, and keep the resulting encoders.
- Quantify encoder drift with three complementary measures (representation similarity,
  weight change, per-descriptor linear-probe recoverability).
- Produce per-task and cross-task summaries that identify descriptors / descriptor
  families whose recoverability drops beyond the noise floor.

**Non-goals (this phase)**
- Re-pre-training any CheMeleon variant.
- The full low-data ladder (multiple training sizes). This phase has exactly two regimes:
  full training set and N=128.
- Nonlinear (MLP) probes. Linear probes only; MLP probes are a possible follow-up.
- Improving CheMeleon's benchmark scores.

**Anti-goals (would make the result useless even if the code runs)**
- Drift measured differently for the pretrained vs. fine-tuned encoders (different probe
  data, standardization, regularization, or molecules). Every encoder — pretrained,
  fine-tuned, random — must go through the identical probe pipeline.
- A fine-tuning protocol that departs from the paper's, so the encoders are not the ones
  CheMeleon users actually get.
- Any use of the Polaris test sets — labels **or molecules** — anywhere except
  `benchmark.evaluate` test scoring (sections 5.4, 9). The test sets stay held out as-is
  (user's decision), so later phases (descriptor selection, re-pre-training) can still be
  evaluated on them without transductive leakage.
- Declaring a descriptor "forgotten" from a single seed or without comparison to the noise
  floor.

## 3. Context and hard constraints (verified)

- **Weights:** only the message-passing block is published (`chemeleon_mp.pt`, Zenodo
  15426600 / 15460715; cached at `~/.chemprop/chemeleon_mp.pt`). It is a
  `torch.load(..., weights_only=True)` dict with `hyper_parameters` and `state_dict` for
  `chemprop.nn.BondMessagePassing` (d_h=2048, depth=6). **No descriptor head and no
  feature means/variances are published** — hence the probe-on-both-sides design.
- **Architecture used downstream:** `BondMessagePassing` → `MeanAggregation` → FFN.
  Featurizer: chemprop `SimpleMoleculeMolGraphFeaturizer` with the V2 multi-hot atom
  featurizer (chemprop CLI enforces `--multi-hot-atom-featurizer-mode V2` for CheMeleon).
- **Paper fine-tuning protocol** (`external/CheMeleon/models/chemprop_mpnn/evaluate.py`):
  - Polaris `benchmark.get_train_test_split()`; NaN targets filled with train mean.
  - Train/val split of the Polaris train set: `astartes.train_test_split`,
    train_size=0.8, `random_state=seed`.
  - Regression targets standardized (`normalize_targets`) with `UnscaleTransform` on output.
  - Head: `RegressionFFN` or `BinaryClassificationFFN`, `hidden_dim=256`, other args default.
  - `MPNN(mp, agg, fnn)` with chemprop defaults (LR schedule, batch size, no batch norm).
  - Lightning `Trainer(max_epochs=50)`, `EarlyStopping(monitor="val_loss", patience=5)`,
    best checkpoint by `val_loss` reloaded for test prediction.
  - Seeds in the paper: 42, 117, 709, 1701, 9001. The paper never calls `seed_everything`;
    the seed only sets the astartes train/val split. We additionally seed everything
    (our call) for reproducibility, so per-seed scores will not match the paper exactly.
- **Benchmarks:** the 28 in `POLARIS_BENCHMARKS` in
  `external/CheMeleon/analysis/fine-tuning/scripts/finetune.py` (11 `polaris/*`,
  17 `tdcommons/*`; mix of regression and binary classification).
- **Reference scores** for sanity checking: `external/CheMeleon/analysis/polaris_results/CheMeleon.md`.
- **Pre-training molecules:** `cleaned_pubchem_1MM.smiles` (Zenodo 15733574, 44 MB;
  download it — do not use copies in other users' directories).
  The original pre-training split was `torch.utils.data.random_split(dataset,
  [0.7, 0.2, 0.1], torch.Generator().manual_seed(1701))`.
- **Descriptors:** `mordred.Calculator(descriptors, ignore_3D=True)` → 1613 descriptors,
  `calc.config(timeout=1)`, `.fill_missing()` (missing → NaN). Pre-training standardized
  each descriptor and clipped to ±6 SD (winsorization factor 6).
- **Machine:** 2× RTX 4090 (24 GB; one sometimes partly in use by others), 64 cores,
  503 GB RAM, ~291 GB free on `/data3`.
- **Environment:** the existing `chemprop` conda env is broken for this purpose (editable
  `~/chemprop` dev install needs `cuik_molmaker`). Create a new conda env
  `chemeleon_unlearn` with released `chemprop>=2.2,<2.3` (2.3 makes `cuik-molmaker` mandatory; 2.1.2→2.2.x has no changes to scheduler, batch size, FFN defaults or message passing), `polaris-lib`, `zarr<3`,
  `mordredcommunity`, `astartes`, `scikit-learn`, `pandas`, `pyarrow`. No zarr≥3 needed in
  this phase (no PubChem zarr store is used).
- **Polaris auth:** the cached hub token expired 2026-04-28. The user will run
  `! conda run -n chemeleon_unlearn polaris login` when asked, after the env exists.

## 4. Project layout

Project root: `/data3/akshatz/chemeleon_unlearn`.

- `external/` — cloned CheMeleon and how-to-train-your-chemeleon repos (read-only reference).
- `docs/specs/` — this document.
- Code, data, and results locations are the builder's call; keep code separate from
  generated data and keep large checkpoints under `/data3`.

## 5. Fine-tuning

### 5.1 Runs

28 tasks × 2 regimes × 3 seeds × 2 modes = **336 runs**.

- Modes: `ft` (full fine-tune, the paper protocol) and `frozen` (encoder `pre` frozen, only
  the FFN head trained, otherwise identical protocol). The `frozen` runs are what tell us
  whether drift *matters*: a probe-R² drop shows the encoder changed, not that the change
  was needed. The paper's `analysis/fine-tuning/frozen.csv` is not a substitute (30
  epochs, no early stopping, different protocol). `frozen` runs produce no new encoder,
  only test scores.

- Seeds: 42, 117, 709 (the first three of the paper's seeds).
- Regime `full`: the paper protocol exactly (section 3).
- Regime `n128`: before the 80/20 train/val split, subsample 128 molecules from the
  Polaris train set (seeded by the run seed); if a task's train set has ≤128 molecules,
  use all of it. Then the same 80/20 split and protocol. NaN targets are filled with the
  mean of the 128-molecule subsample (our call). For classification tasks, the
  subsample must contain both classes in both train and val (resample until it does —
  builder's call on the exact mechanism; record it).
- Tasks whose Polaris train set has ≤128 molecules (e.g. `adme-fang-rppb-1`, 135 total;
  `adme-fang-hppb-1`, 160 total — check their train sizes) make `n128` identical to
  `full`; run them once and do not count them twice in summaries.

### 5.2 What to keep per run

Training runs the full 50 epochs (no actual stopping) so that we can save both:

- **`es` encoder** — the checkpoint the paper's protocol would have used: best `val_loss`
  under the EarlyStopping(patience=5) rule, tracked online (i.e., the best checkpoint as of
  the epoch where patience would have run out). This needs a custom callback — a plain
  `ModelCheckpoint` over 50 epochs would return the global best instead. This is the
  primary object of study.
- **`last` encoder** — epoch 50. An upper bound on drift without early stopping.

Also record the epoch index of `es`, and the full per-epoch train/val loss curve.

Save only the message-passing `state_dict` (+ hyperparameters, same format as
`chemeleon_mp.pt`) for each, ~35 MB each → ~12 GB total. The FFN heads are only needed for
sanity-check scoring (5.4) and test scores — keep the `es` and `last` heads too, they are
small.

### 5.3 Reference encoders

- **`pre`** — the original CheMeleon encoder (`chemeleon_mp.pt`).
- **`rand`** — `BondMessagePassing` with the same hyperparameters, random init, 3 seeds.

### 5.4 Sanity check

Test scores: for every run (both modes, both regimes, `es` and `last`), compute the
benchmark test score via `benchmark.evaluate`. These feed the `ft` vs `frozen` comparison
(section 9).

For `full` regime `ft` `es` models, compute the benchmark test score via
`benchmark.evaluate` (same as the paper) and compare with
the spread across the 5 seeds in `analysis/polaris_results/CheMeleon.md` (exact per-seed
matches are impossible, see section 3). Expect our scores to fall within that spread; a systematic gap means the protocol was not reproduced and must be fixed
before any drift analysis is trusted.

## 6. Drift measurement

All embeddings are the 2048-d output of `MeanAggregation` over the encoder (what the FFN
sees), computed in eval mode.

### 6.1 Probe molecule sets

- **PubChem set:** 10,000 molecules sampled (fixed seed) from the **test** portion (10%)
  of the original pre-training split (reconstructed per section 3), so they were never
  trained on. Split 8,000 fit / 2,000 eval, fixed.
- **Task sets:** for each task, that task's Polaris **train** molecules only (labels
  unused; Polaris test molecules are never probed). **5-fold cross-fitting** (folds fixed
  per task, independent of fine-tuning seed): every molecule gets an out-of-fold
  prediction. A single 80/20 split would be too noisy: 19 of 28 tasks have under ~1,250
  molecules in total.
- **Clean stratum per run:** the fine-tuned encoder took gradient steps on its run's
  training rows and `pre` did not, so memorization alone can move R² on those rows. For
  each fine-tuning run, the *clean rows* are the Polaris train molecules that run never
  took gradient steps on: for `full`, its validation rows (~20%, used only for early
  stopping); for `n128`, everything outside its ~102 training rows. ΔR² for a seed is
  computed on that seed's clean rows (out-of-fold predictions; paired with `pre` on the
  same rows). Report ΔR² on all train rows too, as a secondary view. The clean stratum is
  what the decision uses (section 9). Note: for `full`, clean rows are ~20% of a small
  train set (often ~100 molecules), so the task view has limited power on small tasks —
  one reason PubChem is co-primary.

### 6.2 Descriptor targets

- Compute 1613 Mordred descriptors (exact config in section 3) for the PubChem set and the
  Polaris train molecules of every task. Verify the calculator yields 1613 columns.
- Per probe set: standardize each descriptor with mean/SD from that set's **fit** split,
  then clip to ±6 SD.
- Drop descriptors that are constant or have >20% NaN on the fit split (builder may pick a
  different threshold; record which and how many were dropped per set). Remaining NaNs are
  masked out of the loss / metric, not imputed.

### 6.3 Measures (computed for every encoder: `pre`, `rand`×3, each `es`, each `last`)

1. **Linear CKA** between the encoder's embeddings and `pre`'s embeddings, on the PubChem
   eval set and on the task set (all train rows and the run's clean rows).
2. **Relative weight change** per parameter tensor: ‖W − W_pre‖_F / ‖W_pre‖_F
   (fine-tuned encoders only).
3. **Linear-probe recoverability:** ridge regression from embedding (standardized using
   the fit split) to all retained descriptors, R² per descriptor on held-out molecules
   (PubChem: the fixed eval split; task sets: cross-fitted out-of-fold, section 6.1). The ridge penalty α is chosen once by cross-validation on `pre`'s
   fit embeddings **separately for each probe set** (PubChem, and each task set), then held
   fixed for every encoder on that probe set — so differences reflect the representation,
   not regularization (our call). Task probe sets whose fit split has fewer than ~1,000
   molecules (fewer than or near the 2,048 embedding dims) are flagged as lower-confidence
   in all summaries. (Builder may additionally report per-encoder CV'd α as a robustness
   check.)
   ΔR²_d = R²_d(encoder) − R²_d(pre).

### 6.4 Noise floor

- **Probe noise:** bootstrap the eval molecules (e.g., 200 resamples) to get a standard
  error for each R²_d and each ΔR²_d. The bootstrap is **paired**: ΔR²_d is computed on the
  same resample for both encoders (unpaired SEs are inflated and would hide drops).
  On task sets each seed has its own clean rows, so each seed is bootstrapped (paired with
  `pre`) on its own rows and the SE of the seed-mean is sqrt(Σ SE_s²)/S.
- **Fine-tuning seed noise:** spread of ΔR²_d across the 3 seeds; CKA between the encoders
  from different seeds of the same task/regime.
- **Scale anchors:** `pre` (CKA=1, ΔR²=0) and `rand` (maximum drift). `rand` is a scale
  anchor, not a per-descriptor floor: random D-MPNN features still decode size-like
  descriptors fairly well.
- **Positive control:** for each task, the descriptors most correlated with the task label
  (|Spearman ρ| on the Polaris train set only; e.g. top 20) should *gain* R² after
  fine-tuning. If none do anywhere, the probe cannot detect representational change and a
  "little drift" result is uninformative. This also gives a per-task descriptor-relevance
  ranking for the re-pre-training phase.

### 6.5 Calling a descriptor "dropped" for a task/regime/checkpoint

A descriptor counts as dropped if ΔR²_d < 0 for all 3 seeds **and** the seed-mean ΔR²_d is
more negative than both 3× its bootstrap SE and an absolute threshold of −0.05. These
thresholds are an initial setting; report sensitivity to them.

### 6.6 Descriptor families

Report results both per descriptor and per family, with two family definitions:
- **Mordred module** (e.g., `Autocorrelation`, `ABCIndex`, …, from the descriptor
  instance's module).
- **Correlation clusters:** hierarchical clustering on |Spearman ρ| over the PubChem fit
  set, cut at |ρ| ≥ 0.9.

Family-level ΔR² = mean of member ΔR²; family counted as dropped if a majority of members
are dropped.

**Descriptor-specificity statistic (pre-specified):** per task/regime/checkpoint, η² = the
fraction of variance in ΔR²_d (seed-mean) explained by family label, compared against a
permutation null (shuffle family labels, ≥1,000 permutations; family sizes preserved).
Computed for both family definitions. "Descriptor-specific" means η² exceeds the 95th
percentile of the null; "uniform" means it does not.

## 7. Outputs

- A tidy long-format results table (parquet/CSV): task, regime, seed, checkpoint, probe
  set, measure, descriptor (or family), value, SE.
- Per-task summary: CKA, weight change, median ΔR², number/list of dropped descriptors and
  families, `es` epoch.
- Cross-task summary: for each descriptor/family, in how many of the 28 tasks it is
  dropped (per regime/checkpoint); a task × family heatmap of ΔR².
- Sanity-check table (5.4).
- Per task × regime: test score `ft` vs `frozen` (mean ± SD over seeds), and the gap.
- Positive-control table (6.4) and η² vs permutation null (6.6).
- A short written findings note (markdown) stating which decision branch (section 9) the
  data supports, with the key numbers.

## 8. Compute estimate (rough, to be confirmed after the first few runs)

- 168 `ft` fine-tunes of a 2048-d, depth-6 D-MPNN for 50 epochs on datasets of ~100–7,000
  molecules: roughly 10–15 GPU-hours total; run across both GPUs with one run per GPU.
  The 168 `frozen` runs are much cheaper (no encoder backward pass; embeddings can be
  precomputed once per task).
- Mordred for ~10k PubChem + all task molecules: minutes to tens of minutes on 64 cores.
- Embeddings + ridge probes for ~340 encoders × 2 probe sets: small; GPU embedding
  inference, CPU/GPU ridge.
- Disk: ~12 GB checkpoints plus embeddings (embeddings can be recomputed rather than stored
  if space is tight).

Run order: the full pipeline on 2 tasks × 1 seed first, end-to-end through the probe, and
check timings and the sanity score before launching everything.

## 9. Decision rule (what the result means for the next phase)

**Primary views (user's decision, co-primary):** the branch is decided by the `full`
regime, `es` checkpoint, `ft` mode, on **two co-primary probe sets**:
(a) each task's own probe set (Polaris train molecules), clean stratum, cross-fitted
(section 6.1); and
(b) the PubChem probe set (8k fit / 2k eval). A branch is called only when (a) and (b)
agree; if they disagree, the findings note reports a split result and says which view
supports which branch (e.g., drift on task chemistry but not on general chemistry).
Rationale: unlearning needs enough training to happen at all; (a) is closest to what the
task needs, (b) is larger and cleaner. The other views (`n128` regime, `last` checkpoint,
all task train rows including gradient-trained ones) are reported alongside as supporting
evidence.
The cross-task "dropped in k of 28 tasks" table and the heatmap are built from both
primary views side by side, with the others as secondary panels. Tasks flagged as
lower-confidence (section 6.3) are shown but marked.


Each branch is read together with the `ft` vs `frozen` test-score gap per task. Drift
only counts as evidence for the hypothesis in tasks where full fine-tuning beats the frozen
encoder by more than seed noise — otherwise the encoder moved at no benefit (free drift),
and descriptor drops there say nothing about costly unlearning. The `n128` gap is reported
alongside: the hypothesis predicts encoder updates help less (or hurt) at low data.

- **Little drift:** `es` encoders have CKA ≳ 0.95 vs `pre` on PubChem and few or no dropped
  descriptors beyond the noise floor in most tasks → early-stopped fine-tuning does not
  meaningfully unlearn. Next phase: drop the forgetting analysis and test task-tailored
  pre-training directly (descriptor subsets chosen by relevance, with full-set and
  random-subset re-pre-training controls).
- **Real, descriptor-specific drift:** η² beats the permutation null (6.6), dropped
  descriptors/families are consistent across seeds, appear in tasks where `ft` beats
  `frozen`, and show structure across tasks (e.g., shared within task groups like ADME vs
  kinase) → proceed to the full probe study (MLP probes, low-data ladder) and choose 2–3
  tasks / task groups for the re-pre-training experiment.
- **Uniform drift:** encoders move but η² is within the permutation null (6.6) → drift is
  real but not descriptor-specific; revisit the hypothesis before re-pre-training.

The 0.95 and other thresholds are initial and will be read alongside the `rand` and
seed-to-seed anchors, not applied mechanically.

## 10. Left to the builder

- Code structure, file formats, results directory layout, job scheduling across GPUs.
- Exact conda env versions (record them; pin in an environment file).
- How classification-subsample resampling is implemented (5.1).
- Mordred package version, as long as it yields exactly 1613 descriptors with the section 3
  config (pin `mordredcommunity==2.0.6` if the latest does not).
- Clustering linkage for correlation families (6.6).
- NaN-threshold for dropping descriptors (6.2), bootstrap count (6.4).
- Whether to cache embeddings on disk or recompute.

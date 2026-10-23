# Phase 2: task-tailored descriptor pretraining — design

Date: 2026-10-22. Status: builder-checked; awaiting user approval.
Builds on phase 1: `docs/results/2026-10-22-drift-findings.md`, selector analyses in
`results/summary/phase2_*.csv|json`.

## 1. Hypothesis and what this phase tests

User's hypothesis: CheMeleon pretrains on 1613 Mordred descriptors; much of that is
unimportant or harmful for a given task, and fine-tuning must unlearn it. With little data
there is not enough signal to unlearn, so a model **pretrained on a task-tailored
descriptor set** should fine-tune better, with the advantage largest in the low / ultra-low
data regime.

Phase 1 established the premise and the selector: full-data fine-tuning erodes ~16% of
pretraining-added descriptor information (barely any at 128 molecules); what erodes is
mostly generic, and the generically *preserved* descriptors are the least task-relevant
(shallow-FFN check: G100 is the worst set on both demo tasks). A label-relevance selector at
cluster level is stable (Jaccard ≥ 0.96 across CV seeds) and keeps the task's information
in far fewer targets (LD50: 100 tailored descriptors beat all 1613 in a shallow FFN).

**Phase 2 tests, on a tox demo pair (Ames mutagenicity, LD50 acute toxicity):** does
re-pretraining CheMeleon on a small, tailored descriptor set give better fine-tuned models
than (a) the full 1613 and (b) a size-matched random set, and does the advantage grow as
training data shrinks?

**Decisions I made — change any of these (section 10 lists the rest):**
- D1. Tailored set = label relevance, *consensus* of Ames and LD50, one representative per
  correlation cluster ("tox-tailored" pretraining). Not phase-1 drift-derived (unstable:
  Jaccard 0.15–0.6 across checkpoints/views).
- D2. Set size is a two-level factor, k ∈ {25, 100} (user decision), each with its own
  size-matched random control.
- D3. Ultra-low-N fine-tuning uses a fixed epoch budget instead of early stopping on a
  3-molecule validation set.
- D4. Ames/LD50 Polaris **test** molecules are excluded from the pretraining pool; train
  molecules may remain (consistent with the phase-1 fairness ruling).
- D5. All new arms use the **DEFAULT (V2 multi-hot) featurizer and MeanAggregation**, i.e.
  public CheMeleon's architecture, not the how-to-train repo's defaults (RIGR +
  NormAggregation). Reason: the phase-1 fine-tuning runner and the public-CheMeleon
  reference line both assume V2 + mean pooling; a different featurizer/pooling would make
  the reference incomparable and require a second fine-tuning path.
- D6. Ultra-low-N budget is in **optimizer steps, not epochs**: for N ≤ 64, batch size
  min(N, 16) and **S = 16 steps** (2 warm-up + one-cycle decay over 14), identical for every
  arm. Rationale: phase-1's best n128 checkpoints sat at ~12 steps (median es_epoch 6 ×
  2 batches); with the default batch size of 64, "6 epochs" at N=16 would be 6 steps.
  N ≥ 128 keeps the paper protocol (80/20 + patience-5), accepting that its validation set
  is small at 128/256 — that is what the paper and phase 1 did, and it keeps comparability.
- D7. Arm B = **all 1613** descriptors (CheMeleon's actual target set, the baseline the
  hypothesis is stated against); T/R/D draw from the 1367 that pass the NaN/constant filter.
  NaN targets are masked out of the loss in every arm. The how-to-train
  `RandomDropoutMSE` combines its random mask with the finite-mask by `logical_or`, which
  (a) keeps every finite target (dropout inert) and (b) can include NaN targets; the
  builder replaces it with `logical_and` (keep-probability 0.7 AND finite) for all arms.
- D8. Random controls: **two independent draws of R25** (seeds 0, 1) and one R100, if the
  compute gate allows (R25 models are cheap); otherwise one draw each and the single-draw
  caveat is stated. T25/T100/R25/R100 are regenerated from the saved scores (the saved
  `phase2_descriptor_sets.json` holds the exploratory k=50/100 sets, not the final ones).

## 2. Goals, non-goals, anti-goals

**Goals**
- Pretrain five CheMeleon-style models with one identical pipeline, differing only in the
  descriptor target set: B (all 1613), T25, R25, T100, R100.
- Fine-tune each on Ames and LD50 across a downsampling ladder with paired subsamples; score
  on the untouched Polaris test sets.
- Decide the pre-registered primary endpoint (section 7).

**Non-goals**
- Reproducing public CheMeleon's absolute performance (a 100–200k-molecule budget with the
  how-to-train pipeline will underperform it; all arms share that handicap). Public
  CheMeleon rides along as a reference line only.
- Per-task tailored sets, task-family generalization, other task pairs (optional follow-ups).
- Any change to phase-1 code or results.

**Anti-goals (would invalidate the result)**
- Any difference between arms other than the target set (molecules, budget, masking, LR,
  batch size, architecture, early stopping, featurizer).
- Selecting descriptors with any use of Polaris test labels or test molecules.
- Arms fine-tuned on different subsamples at the same (N, seed).
- Early stopping on validation sets too small to be meaningful.

## 3. Descriptor sets (frozen before any pretraining)

All sets are defined over the 1613 Mordred descriptors; cluster map =
`results/corr_clusters.parquet` (535 clusters at |Spearman| ≥ 0.9 on PubChem).

- **Representative per cluster:** the member with the highest mean |Spearman| to the other
  members on the PubChem fit split (medoid); singletons are their own representative.
  `results/summary/phase2_descriptor_sets.json["D535"]` lists them.
- **Relevance score per cluster:** mean of (a) 5-fold CV AUC of a logistic model on the
  cluster's members → Ames label and (b) 5-fold CV R² of ridge → LD50 label, on Polaris
  **train** rows only; consensus = mean percentile rank of (a) and (b)
  (`results/summary/phase2_relevance_scores.csv`).
- **T_k:** representatives of the top-k clusters by consensus rank. **R_k:** representatives
  of k clusters drawn uniformly at random (seed 0), excluding none. k ∈ {25, 100}.
- **B:** all 1613 (CheMeleon's set). Optional **D535**: one representative per cluster
  (dedup only), if compute allows (section 8).
- The builder regenerates T25/T100/R25/R100 from the saved scores and writes them to
  `data/phase2/sets/<arm>.txt`; the sanity check in 3.1 must pass before pretraining.

### 3.1 Pre-pretraining sanity check (user requirement)
Shallow FFN (1×256 hidden, fastprop-style; a linear probe is acceptable) on each set,
5-fold CV on the Polaris train split, 3 seeds, for both tasks. Expected from the exploratory
run: T100 ≥ B on LD50 (0.50 vs 0.45 R²), T100 ≈ B on Ames (0.87 vs 0.88 AUC); T_k > R_k at
both k on LD50, at k=25 on Ames. Record the table; proceed only if T_k ≥ R_k on at least one
task at each k (expected to hold via LD50; Ames at k=25 is untested and may not). If a k
fails on both tasks, keep the arm but flag it: the FFN check is a feature-level proxy, the
pretraining experiment is the test.

## 4. Pretraining (one pipeline, five arms)

- **Code:** `how-to-train-your-chemeleon` (`external/how-to-train-your-chemeleon`) as the
  base, chemprop ≥ 2.3 in a separate conda env from phase 1. Featurization in the pretraining
  dataset must use chemprop 2.3's `CuikmolmakerMolGraphFeaturizer` (batch of SMILES →
  `BatchCuikMolGraph`) instead of the repo's per-molecule `SimpleMoleculeMolGraphFeaturizer`
  loop, with **no graph caching** (user rule). Featurizer mode V2 / mean aggregation per D5
  (`d_v=72, d_e=14`, same as `chemeleon_mp.pt`).
- **Molecule pool:** a fixed random subset of `cleaned_pubchem_1MM.smiles` of size
  **P = 200k** (fallback 100k if the compute gate fails), seed fixed, with Ames and LD50
  Polaris test molecules removed by non-isomeric canonical SMILES match. Same pool, same
  train/val split for every arm. Descriptors are **not recomputed**: they are read from CheMeleon's
  published pretraining descriptors (Zenodo 15733575, `pubchem_1MM.zarr`, row-aligned with
  the cleaned SMILES file), standardized per descriptor on the **pool's train split only**
  (never the full 1M), winsorized at ±6 SD; each arm then takes its column subset. Because
  descriptors are free, P can be raised after the compute gate if GPU time allows.
- **Hyperparameters:** identical across arms, taken from the how-to-train defaults
  (MP hidden 2048, depth 6, FFN 2048×1, random-dropout MSE 0.30, warmup/one-cycle LR,
  patience 5, max epochs 100); the FFN output dimension follows the set size. Record the
  exact config; it applies to all arms.
- **Budget:** same max epochs and patience for all arms (early stopping on pool validation
  loss is fine here — it is not the task's data).
- **Seeds:** one pretraining seed per arm in the first pass; a second seed for B and T25 if
  compute allows (section 8).
- **Output:** per arm, a chemprop-loadable message-passing checkpoint in the
  `chemeleon_mp.pt` format (usable by `drift.encoders.load_mp` and phase-1 fine-tuning).

## 5. Fine-tuning ladder

- **Tasks:** `tdcommons/ames` (classification, 5821 train) and `tdcommons/ld50-zhu`
  (regression, 5907 train); Polaris test sets untouched except for scoring.
- **Sizes:** N ∈ {16, 32, 64, 128, 256, 512, 1024, full}.
- **Paired subsampling:** for each (N, seed) draw one subsample (Ames stratified by class)
  from the Polaris train set; **every arm** — B, T25, R25, T100, R100, and public CheMeleon
  as reference — fine-tunes on that identical subsample. Seeds: 10 at N ≤ 64, 5 at
  128–256, 3 at ≥ 512.
- **Protocol (phase-1 runner, `drift.finetune.run_one`, with graph caching ON):**
  - N ≥ 128: paper protocol (80/20 train/val, max 50 epochs, patience-5 early stopping,
    es checkpoint).
  - N ≤ 64 (D3): no validation split; train on all N for a fixed number of epochs
    E = 6 (D6; chemprop's default 2 warm-up epochs then one-cycle decay over the remaining
    4). Record it. The "es" checkpoint is the final weights.
  - Head: paper head (hidden 256); regression targets standardized on the subsample.
- **Scoring:** `scripts/05_score.py` protocol; benchmark main metrics are `roc_auc`
  (Ames, higher better) and `mean_absolute_error` (LD50, lower better) — sign-normalize.

## 6. Analysis

- Per task, per N: mean ± SD over seeds of each arm's test score; paired differences
  T_k − B, T_k − R_k, T100 − T25, B − CheMeleon (same subsample), with seed-SD.
- Curves: score vs log N per arm; gap (T_k − B) vs log N.
- Secondary: freeze-encoder variant at N ≤ 64 for B and T100 (does tailoring reduce the
  need to move the encoder?); probe drift of each arm after fine-tuning is NOT required.

## 7. Pre-registered primary endpoint

**Primary:** for each task and N, d_s = score_T100(s) − score_B(s) on paired subsample s,
sign-normalized so higher = better (LD50 MAE negated); gap(N) = mean_s d_s; SD(N) = SD_s d_s
(the paired-difference SD, in the metric's units, per N). **Success** = gap(N) > SD(N) at
≥ 2 of the 3 points N ∈ {16, 32, 64} **in each task**, and gap decreases with N: Spearman
ρ(gap, log N) < 0 over the 8 sizes and mean gap at N ≤ 64 > mean gap at N ≥ 512. Evaluated
on the `es` checkpoint (fixed-budget final weights at N ≤ 64); `last` reported as
secondary.
**Secondary contrasts:** T_k − R_k at both k (is it the *right* targets or just fewer?);
T25 − B (does the lean set hold up?); T100 − T25.
**Interpretation grid:** T > R and T > B → tailoring works; T ≈ R > B → fewer targets
help (rebalancing, not tailoring); T ≈ R ≈ B → no effect at this budget; T < B → tailoring
discards needed information.

## 8. Compute (gate first)

- **Step 1 of execution:** time one B-arm pretraining run on the P=200k pool (one GPU,
  cuik-molmaker, no cache). Decide P and the number of seeds from that number. Cut order if
  it is bad: second pretraining seeds → D535 → P=100k. Never cut B or T100.
- Pretraining: 5 arms (+D535, +2 extra seeds optional) on two RTX 4090s, two runs at a
  time (one per GPU).
- Fine-tuning: 8 sizes × (10/5/3 seeds) × 6 models × 2 tasks ≈ 500 runs; with graph caching
  and several jobs per GPU, well under the phase-1 sweep's wall time.
- Disk: pool descriptors ~1.3 GB (200k × 1613 float32); checkpoints 35 MB each.

## 9. Caveats to carry into the write-up

- The selector uses Ames/LD50 **training** labels (full set) — legitimate for a
  "tox-tailored pretraining" story, but a real ultra-low-data user would select from a
  related task, not the target. Report the Jaccard between consensus T and each single-task
  T (0.47 / 0.47 at k=100) and note it.
- 63% of Ames/LD50 molecules are in CheMeleon's pretraining set; our P-subset keeps the
  same fairness-over-novelty rule for train molecules (test molecules removed).
- Ames information is diffuse: at k=100 random matches tailored in the FFN check. The
  tailoring hypothesis is expected to show mainly on LD50 and on Ames only at k=25.
- Shallow-FFN scores are CV on the train split, not Polaris test scores.

## 10. Left to the builder

- Env for pretraining (chemprop ≥ 2.3, cuik-molmaker, how-to-train deps), pool subset
  seed, descriptor computation parallelism, file layout under `data/phase2/` and
  `runs/phase2/`.
- Graph-caching mechanism for fine-tuning and job packing per GPU.
- Subsample seeding scheme (record all indices).
- Plot styling (use the `dataviz` skill).

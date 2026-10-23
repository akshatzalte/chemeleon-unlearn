# Phase 1 findings: does fine-tuning CheMeleon "unlearn" Mordred descriptors?

Date: 2026-10-22. Spec: `docs/specs/2026-10-22-encoder-drift-check-design.md`.
Inputs: 28 Polaris/TDC tasks × {full, n128} × {ft, frozen} × 3 seeds = 324 fine-tunes;
linear probes on 1613 Mordred descriptors; full tables in `results/summary/`.

## Headline

1. **Full-data fine-tuning erodes pretraining-specific descriptor information, moderately.**
   Median CKA vs the pretrained encoder 0.85 (PubChem) / 0.74 (task molecules); median
   ΔR² −0.04; 25–32% of eligible descriptors "dropped" (ΔR² < −0.05, all seeds, >3 SE).
   Normalized by what pretraining added over a random encoder, a median **16%** of
   pretraining-specific information becomes linearly unrecoverable (drift fraction 0.17–0.26).
2. **At 128 molecules the encoder barely moves.** Median ΔR² −0.02, 6–12% dropped, drift
   fraction 0.08–0.11. Consistent with the hypothesis' premise that low-data fine-tuning
   cannot reshape the encoder.
3. **Most of *what* is lost is the same for every task.** Across the 28 tasks the
   per-descriptor normalized-loss profiles correlate at Spearman **0.81** (q10 0.66), vs
   0.95 between seeds of the same task. A reproducible task-specific component exists but
   explains only **18%** of the variance (seed-to-seed reproducibility of the residual:
   median 0.51 at full data, 0.22 at n128).
4. **Almost nothing improves.** Descriptors whose R² rises after fine-tuning (all seeds,
   >2 SE): median 0 per task, max 2 on PubChem. The positive control (top-20
   label-correlated descriptors should gain) fails in every task.
5. **Loss is concentrated in fine detail.** PCA of the descriptor targets: mean ΔR² −0.03
   for PCs 1–10, −0.06 for 11–50, −0.12 for 51–150, −0.13 for the tail. By family, the
   most eroded are the ones pretraining taught most (RingCount, TopologicalCharge,
   Autocorrelation); simple counts and path/walk counts are barely touched.

## Pre-registered branch rule (spec §9)

Agreed across both co-primary views (full/es, ft): descriptor_specific 14, uniform 12,
little_drift 0, split 2. **This rule is not informative as written**: the η² permutation
gate (family structure) is anti-conservative with correlated descriptors and passed in
every task (p = 0.001 floor), so the specific/uniform split was decided only by the
ft−frozen score gap. All four sensitivity variants (debiased CKA, strict η² ≥ 1.5× null,
no η²) give identical counts. The useful descriptor-level outputs are items 3–5 above and
the cluster tables (`results/summary/cross_task*.csv`, `families.parquet`).

## Robustness

- **Probe regularization:** per-encoder α gives the same median ΔR² (−0.039) and drop
  fraction (32%) as the fixed α → not an artifact of tuning on `pre`.
- **Pretraining overlap:** 35% of task molecules were in CheMeleon's pretraining train
  split (60–80% for several TDC sets). On molecules CheMeleon never saw, drift is the same
  or slightly larger (median ΔR² −0.043 novel vs −0.037 non-novel) and `pre` has no R²
  advantage (0.950 vs 0.939) → memorization does not inflate the drops.
- **Small-n CKA bias:** debiased CKA differs from linear CKA by ≤ 0.008 on real embeddings.
- **Sanity vs the paper:** 77/84 full/ft/es test scores within 1 SD of the paper's 5-seed
  range (53 inside the range); 23/28 tasks clean on all seeds. Outliers are single seeds
  (notably pkis2-ret-cls seed 117: 0.42 vs 0.53–0.64).
- **ft vs frozen (side diagnostic):** full data — ft better by >1 SD in 15/28, frozen
  better in 2/28; n128 — ft better in 4/26, frozen never. Matches the paper.

## Caveats

- Linear probe: "lost" = no longer *linearly* readable, not necessarily gone.
- n128 clean rows include the 26 early-stopping validation rows.
- Mordred is heavily redundant: 1,367 usable descriptors have 283 PCs at 99% variance and
  a participation ratio of 9.8; Autocorrelation alone is 606 targets (37% of the
  pretraining loss). Two correlation clusters (169 and 83 members) are essentially
  "molecular size". Descriptor counts above therefore over-count; cluster-level tables are
  the honest unit.
- Task pairing and the descriptor-selection analyses below are post-hoc.

## Implications for phase 2

- Selecting a tailored set by "what this task forgets" mostly recovers the generic
  pattern. Use the **task-relative** component (what this task spares vs others) at
  **cluster** level, plus label relevance; compare against a generic-pruned set and a
  size-matched random set, all re-pretrained with one pipeline (public CheMeleon is not a
  valid control).
- Task pairs with the most similar task-specific profiles (mean Spearman over 4 views):
  hPPB~rPPB 0.71, hCLint~rCLint 0.69 (same compound library), **Ames~LD50 0.64 with the
  highest floor (0.58) and only 22% shared molecules**. Decision (user): phase-2 demo on
  **Ames + LD50**; Fang species pairs optional second demo.
- Low-data runs barely move the encoder, so the ladder (16 → full) is where a tailored
  pretraining should show up, if it does.

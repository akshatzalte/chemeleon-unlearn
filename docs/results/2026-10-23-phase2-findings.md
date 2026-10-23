# Phase 2 findings: does pretraining on task-tailored descriptor targets help low-data fine-tuning?

Date: 2026-10-23. Spec: `docs/specs/2026-10-22-phase2-tailored-pretraining-design.md`.
Tables/figures: `docs/results/phase2/` (`summary.md`, `gaps.csv`, `endpoint.json`, `curves.png`,
`scores.csv`, `pretrain_metrics_*.csv`). Build ledger: `docs/results/2026-10-23-phase2-build-ledger.md`.

Terminology: *targets* = the descriptor columns an encoder is pretrained to predict; the
*featurizer* (atom/bond input encoding, Chemprop V2 multi-hot) was identical in every arm.

## Verdict

**The pre-registered endpoint is not met, and the idea is closed.** Pretraining CheMeleon-style
encoders on 100 task-tailored Mordred targets gives no detectable advantage over all 1613 at
any fine-tuning size on Ames or LD50; 25 tailored targets are no better — on Ames slightly
worse — than 25 real descriptors chosen at random. The pretraining target set is not a lever
for low-data performance in this setting.

## What was run

- Pool: 200k PubChem molecules (CheMeleon's published Zenodo descriptors; Ames/LD50 Polaris
  test molecules removed), 180k/20k split, per-column stats from the train split, ±6 SD clip.
- Arms (one pipeline; only the target set differs): **B** all 1613; **T100**, **T25** top-k
  correlation clusters by consensus label relevance (Ames AUC + LD50 R², train labels only),
  one representative per cluster; **R100**, **R25a**, **R25b** size-matched random clusters;
  **D535** one representative per cluster (dedup only); **chemeleon** = public weights.
  D-MPNN 2048×6, V2 featurizer, mean aggregation, masked-dropout MSE (keep 0.7), batch 512,
  one-cycle LR 1e-4→1e-3→1e-4, patience 15, max 100 epochs, 1 RTX 4090 per arm.
- Ladder: N ∈ {16, 32, 64, 128, 256, 512, 1024, full}; 10/10/10/5/5/3/3/3 seeds; one
  subsample per (task, N, seed) shared by every arm (Ames stratified). N ≤ 64: batch
  min(N,16), 16 optimizer steps, final weights. N ≥ 128: the paper protocol (80/20, patience
  5). Scoring: Polaris `benchmark.evaluate` (Ames ROC-AUC ↑, LD50 MAE ↓). 784 runs.

## Primary endpoint (T100 − B, paired per subsample, `es` checkpoint)

| task | gap > SD at N∈{16,32,64} | Spearman(gap, N) | mean gap N≤64 | mean gap N≥512 | success |
|---|---|---|---|---|---|
| Ames | 0 / 3 | +0.12 | −0.002 AUC | +0.007 | no |
| LD50 | 0 / 3 | +0.14 | −0.011 (−MAE) | −0.007 | no |

Paired SE at N ≤ 64 is ≈ 0.008 (10 seeds), so gaps below ~0.02 AUC / 0.02 MAE are undetectable:
this is "no detectable advantage," not "no effect." The gap does not shrink with N — the
sign of the Spearman is opposite to the hypothesis' prediction.

## Secondary contrasts (mean paired gap, higher = better)

| contrast | Ames N≤64 | Ames N≥512 | LD50 N≤64 | LD50 N≥512 | reading |
|---|---|---|---|---|---|
| T100 − R100 | −0.010 | +0.010 | −0.008 | −0.004 | tailored ≈ random |
| T25 − R25a / R25b | −0.029 / −0.022 | −0.023 / −0.012 | +0.004 / −0.000 | −0.013 / −0.010 | tailored ≤ random |
| T25 − B | −0.011 | −0.016 | −0.008 | −0.016 | 25 targets slightly worse than 1613 |
| B − chemeleon | −0.022 | +0.004 | +0.010 | +0.001 | 200k re-pretraining ≈ public 1M model |
| T100 − T25 | +0.009 | +0.023 | −0.004 | +0.009 | 100 ≥ 25 |

Per-arm mean scores (`summary.md`): all eight arms lie within ~0.03 AUC / ~0.02 MAE of each
other at every N; the only cell outside that band is D535 on LD50 at N=full
(MAE 0.517 ± 0.007 vs B 0.546 ± 0.026) — a lead for *deduplicated* targets at full data,
untested further.

Pipeline validation: B ≈ public CheMeleon across the ladder (|gap| ≤ 0.03; the N=full
`chemeleon` arm reproduces phase-1 full-data scores within 0.01 on 5 of 6 seeds, the sixth
differing by 0.05 MAE on one seed with early stopping at epoch 4 vs later). A 200k-molecule,
single-GPU, ~1-hour pretraining matches the 1M-molecule public model downstream.

## Why the feature-level selector did not transfer

The shallow-FFN gate showed the tailored sets *carry* the task information (T100 beats all
1613 on LD50 as input features). But a pretraining target does something else: it forces
the encoder to learn whatever representation computes the target from the graph. Mordred's
1613 descriptors are so redundant (535 clusters at |ρ| ≥ 0.9; 283 PCs for 99% variance;
two clusters are "molecular size") that any 25 real descriptors already demand the same
general chemistry. Relevance *as a feature* and value *as a supervision signal* are
different properties; phase 2 tested the first as a proxy for the second.

Read together with phase 1 (fine-tuning erodes ~16% of pretraining-specific detail,
generically, and no descriptor becomes more recoverable; the encoder barely moves at
N ≤ 128): what pretraining puts in the encoder — within the descriptor-target family — does
not change the downstream outcome. The low-data regime here is limited by the fine-tuning
data, not by anything that has to be unlearned.

## Independent corroboration

OpenADMET's "Throwing Everything AND the Kitchen Sink at CheMeleon" (West, Burns et al.,
Aug 2026) swept 15 target sets (osmordred, PCA-reduced osmordred, RDKit 2D, fingerprints,
3D descriptors, MiniMol embeddings, surrogate-ADMET pseudo-descriptors) at full data on
ADME/tox endpoints with 5–20 seeds: no target set improved every endpoint; CheMeleon is
"robust to pre-training methodologies"; even fingerprint targets "performed reasonably."
Their positive signals were elsewhere: a ~40k generated analog set around the target
chemistry beat a 10M max-diversity pool, and property-like targets (25 surrogate-ADMET
predictions; MiniMol embeddings) gave ~6% MAE reduction. Our contribution relative to
theirs: size-matched random-target controls, the 25-target floor, and the low-N ladder
with paired subsamples.

## Critique experiments (mechanism-level checks)

**Random-initialized encoder (`rand`, no pretraining, same ladder).** Pretraining is clearly
detectable from N = 32–64 upward: B − rand = +0.07 AUC at N = 64–128 and +0.03 at full on
Ames; +0.04–0.06 MAE at N = 16–128 and +0.01 at full on LD50 (paired SE 0.01–0.02). The one
exception is Ames at N = 16, where `rand` ≈ B (−0.02 ± 0.01): that cell cannot distinguish
any encoder and is uninformative. So "all pretrained arms perform the same" is a statement
about encoders that *are* distinguishable from no pretraining — the target set is irrelevant
even where pretraining itself matters.

**Second pretraining seeds (B_s1, T100_s1).** Pretraining-seed differences are small:
B_s1 − B and T100_s1 − T100 lie within ±0.015 at every N (mostly < 0.01), the same scale as
the T100 − B gap itself. The replicate contrast T100_s1 − B_s1 is ≈ 0 on Ames and slightly
*negative* on LD50 at N = 16–32 (−0.017, −0.013; SE 0.006, 0.002). The primary result is not
hiding under pretraining-seed noise.

Public CheMeleon on Ames at N = 16–32 is the one cell where a pretraining-side effect shows
(+0.03 over our B, which is indistinguishable from `rand` there): something about the
original pretraining (molecules, length) helps at ultra-low N that our 200k recipe does not
reproduce. Worth carrying into the lite project; it is not a descriptor-targets effect.

*Pending — appended when scored:* frozen-encoder ladder for B/T100/R100/chemeleon (with the
encoder frozen nothing can be "unlearned", so a tailored encoder should matter most — the most
direct test of the mechanism); 64-step budget at N ≤ 64 (does a longer small-N budget separate
arms?). Per-task tailored sets (Ames-only / LD50-only selection) were cancelled: OpenADMET's
sweep already establishes robustness to target choice.

## Caveats

- One task pair (Ames, LD50); Ames information is diffuse (random 100 ≈ tailored 100 even as
  features). The tailored set was selected on the target tasks' own training labels — the
  most favourable version of the hypothesis — and still did not help.
- Two random draws at k=25, one at k=100; one pretraining seed per primary arm (second seeds
  pending above).
- Arms stop at different epochs under per-arm early stopping with a fixed 100-epoch one-cycle
  LR (epochs run / best epoch: B 80/64, T100 86/70, T25 41/25, R100 84/68, R25a 100/90,
  R25b 100/98, D535 66/50; R25a/b hit the cap but were flat over their last 10–20 epochs;
  T25 is spiky — small target sets train less stably). Patience was raised from 5 to 15
  after the first T100 run stopped at epoch 19 on loss spikes while B kept improving; the
  patience-5 runs are kept as `*_p5`.
- Subsamples are nested across N at a given seed, so gap(N) points are positively correlated.
- 16-step protocol at N ≤ 64 (chemprop's default batch of 64 would have meant 6 steps);
  N ≥ 128 early-stops on a small validation set, as in the paper.
- Public CheMeleon was pretrained on a pool containing the Ames/LD50 test molecules and on the
  original Zenodo targets (58 columns differ slightly from our recomputed values); reference
  only.

## What the project keeps

- A reproducible CheMeleon-style pretraining recipe: 200k molecules, 25–1613 targets,
  ~35–90 min on one RTX 4090, matching the public model downstream.
- A paired downsampling-ladder harness (subsamples shared across arms, fixed-step small-N
  protocol, Polaris scoring, pre-registered endpoint).
- Phase-1 probing tools (exact ridge probe, paired bootstrap, drift measures).

Next direction (user decision, 2026-10-23): a **CheMeleon-lite** — fast to pretrain, small,
easy to deploy, minimal fine-tuning — as a separate project. Leads carried forward: keep and
use the descriptor head at fine-tuning (auxiliary loss / stacked predicted descriptors);
better estimators on frozen embeddings (GP, kernel ridge, k-NN, ensembles, tabular
foundation models on the CheMeleon fingerprint); tailor the pretraining *molecules*
(coverage of the task's chemical space) and *property-like targets* (surrogate predictors,
alert-based SMARTS), possibly agent-designed and judged on a validation ladder only.

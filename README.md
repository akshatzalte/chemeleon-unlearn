# chemeleon-unlearn

Does fine-tuning a descriptor-pretrained molecular foundation model "unlearn" what
pretraining taught — and would pretraining on a **task-tailored descriptor set** give better
models in the low-data regime?

This repository tests that hypothesis on [CheMeleon](https://github.com/JacksonBurns/CheMeleon),
a Chemprop D-MPNN pretrained to predict 1613 Mordred descriptors, using the 28 Polaris/TDC
benchmarks from the CheMeleon paper.

## Phase 1 — encoder-drift check (complete)

Fine-tune CheMeleon on all 28 tasks (full data and 128 molecules, fine-tuned and
frozen-encoder, 3 seeds), then probe every encoder for the 1613 descriptors with an exact
ridge probe on held-out PubChem molecules and on each task's own molecules.

Headline (`docs/results/2026-10-22-drift-findings.md`):
- Full-data fine-tuning erodes ~16% of the descriptor information pretraining added over a
  random encoder; at 128 molecules the encoder barely moves.
- ~80% of *what* is lost is the same for every task (cross-task Spearman 0.81); a
  reproducible task-specific component explains ~18% of the variance.
- Almost no descriptor becomes *more* recoverable after fine-tuning; loss concentrates in
  low-variance detail (ring counts, topological charge, autocorrelations).
- The descriptors fine-tuning generically preserves are the least task-relevant ones.

Design: `docs/specs/2026-10-22-encoder-drift-check-design.md`; plan:
`docs/plans/2026-10-22-encoder-drift-check.md`; build ledger with every ruling:
`docs/results/2026-10-22-phase1-build-ledger.md`.

## Phase 2 — tailored-descriptor pretraining (in progress)

Re-pretrain CheMeleon-style encoders with one pipeline on five target sets — all 1613,
25 and 100 tailored (label-relevance, one representative per correlation cluster), and
size-matched random sets — then fine-tune on Ames and LD50 across a paired downsampling
ladder (16 → full). Pre-registered endpoint: tailored − all at N ≤ 64.

Design: `docs/specs/2026-10-22-phase2-tailored-pretraining-design.md`; plan:
`docs/plans/2026-10-22-phase2-tailored-pretraining.md`.

## Layout

```
src/drift/        package: tasks, encoders, finetune, descriptors, probe, stats, measure, score, summarize, phase2/
scripts/          numbered pipeline stages (00–06 phase 1, p2_* phase 2)
tests/            pytest (phase-1 suite runs in the chemprop 2.2.4 env)
docs/specs|plans  design documents and implementation plans
docs/results      findings notes, summary tables, figures
```

Environments: `environment.yml` (phase 1 / fine-tuning: chemprop 2.2.4, polaris-lib 0.12.1)
and `environment-pretrain.yml` (phase 2 pretraining: chemprop 2.3.1 + cuik-molmaker).
Large data, run outputs and raw results are not tracked.

## Reproducing

```
conda env create -f environment.yml && pip install -e .
polaris login
python scripts/00_fetch_tasks.py && python scripts/01_descriptors.py
python scripts/02_finetune.py --gpu 0 --shard 0/2   # (and 1/2 on a second GPU)
python scripts/05_score.py
python scripts/03_measure.py && python scripts/03_measure.py --combine
python scripts/06_cka.py && python scripts/04_summarize.py
```

Cite CheMeleon (DOI 10.1021/acs.jcim.6c01546) when using the pretrained weights.

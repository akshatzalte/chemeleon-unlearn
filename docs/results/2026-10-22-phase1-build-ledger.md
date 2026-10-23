# SDD ledger — plan: /data3/akshatz/chemeleon_unlearn/docs/plans/2026-10-22-encoder-drift-check.md
Spec: /data3/akshatz/chemeleon_unlearn/docs/specs/2026-10-22-encoder-drift-check-design.md
Branch: drift-check (base ffc899c on main). Models: all subagents sonnet (user instruction).

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 self | git init + .gitignore | Already done by controller at setup (ffc899c). |
| T1→all | config constants, slug() | consistent |
| T2 self | tests use conftest helpers it creates | ok |
| T2→T4 | TaskData(y_raw,y_train), make_split | ok after y_raw fix |
| T2→T5 | train.parquet order = descriptor row order | ok |
| T3→T4,T9 | load_mp/save_mp/random_mp/build_mpnn/embed | ok |
| T4 self | test_worklist_resume monkeypatches config.RUNS; run_dir must read config.RUNS at call time | note to implementer |
| T4→T8,T9 | runs/ layout, run.json train_idx/val_idx | ok |
| T5→T9 | Targets, descriptor_modules, parquet layout | ok |
| T6 self | paired_bootstrap_delta takes rows_list (per seed) | ok |
| T6→T9 | measure_probe_set signature says single `eval_rows` but clean rows are per-seed | CONFLICT |
| T7→T9,T10 | drop_calls, eta2_perm_test, family_delta, positive_control, correlation_clusters | ok |
| T8 self | no unit tests; launches long GPU jobs | controller launches full run in background |
| T9→T10 | parquet schemas | ok |
| Login | T2 Step5 + T8 need Polaris login | ask user after T1 env exists |

Ruling: T1 Step 1 (git init/.gitignore) skipped — done at setup — costs nothing if wrong.
Ruling: T9 measure_probe_set takes `rows_by_encoder: dict[str, np.ndarray]` (eval rows per encoder; pre/rand paired per seed) instead of a single `eval_rows` — spec 6.1 clean rows are per run — if wrong, small signature change in T9 only.
Ruling: fix-loop rounds 4-5 and final review also use sonnet (user: subagents sonnet, not haiku) rather than escalating tiers — costs possible extra fix rounds.
Ruling: Task 8's full 336-run fine-tune is launched by the controller in background after the pilot passes the sanity gate; Task 9 implementation proceeds in parallel on synthetic data — if wrong, only ordering changes.
Task 1: dispatched (base ffc899c, implementer ad773b06ca3413bca, sonnet)
Task 1: implemented 900f9ce (DONE; chemprop 2.2.4, polaris 0.12.1, torch 2.14.1+cu126, mordred 1613). Review dispatched.
Task 1: verified ⚠️ items myself (cuda True, mordred 1613).
Task 1: minor (deferred): environment.lock.yml pins local `drift==0.1.0` and channel `defaults` — recreating env from lock would fail on that line.
Task 1: minor (deferred): ROOT hard-coded absolute path in config.py.
Task 1: complete (commits ffc899c..900f9ce, review clean)
Polaris login: user attempted via `!` (non-interactive, aborted); asked user to run `polaris login --overwrite` in a real terminal. Task 2 Step 5 (fetch) waits on it.
Task 2: dispatched (base 900f9ce, implementer ab28feaeab7ab1833, sonnet; Step 5 fetch deferred to controller after login)
Polaris login: verified working 2026-10-22 16:51 (ames 5821/1457). Token lifetime ~4h — fetch soon; scoring (Task 8) may need re-login.
Task 2: review approved (94f5823). minor (deferred): test_load_task_offline monkeypatch near-vacuous + causes 3 polaris DeprecationWarnings; no tests for RuntimeError/unknown-regime/NaN-label n128 branch; no RED evidence captured.
Task 2: ⚠️ live fetch running (controller); completion pending 28 dirs + PCA_MLP totals check.
Task 3: dispatched (base 94f5823, implementer a964dab228343101e, sonnet)
Task 3: implemented d985057 (DONE). Review dispatched.
Task 3: ⚠️ FFN_HIDDEN=256 confirmed in config.py by controller.
Task 3: minor (deferred): save_mp AttributeError if mp lacks _drift_hparams (carry to Task 4: save from the loaded mp object); build_mpnn accepts any non-"regression" as classification; test_build_mpnn only asserts input_dim; no RED evidence.
Task 3: complete (commits 94f5823..d985057, review clean)
Ruling: Task 4 worklist(n_train: dict|None=None) reads meta.json when None so tests don't depend on cached data — costs a signature tweak if wrong.
Task 4: dispatched (base d985057, implementer a5b60f71f192cec9a, sonnet)
Task 4: implemented cc39f27 (DONE). Review dispatched.
Task 2: ⚠️ resolved — live fetch done: 28 tasks, train 37739 / test 9416, totals match PCA_MLP.md for all 28. n_train<=128: rppb, hppb (n128 skipped for those).
Task 2: complete (commits 900f9ce..94f5823, review clean)
Task 4: minor (deferred): metrics.csv test doesn't assert val_loss non-NaN / es_epoch==argmin; double copy in _update (deepcopy+clone, extra GPU mem); frozen check uses bare assert; worklist() loads full parquets for n_train; no per-item try/except in 02_finetune (one crash aborts the shard).
Task 4: complete (commits d985057..cc39f27, review clean)
Ruling: controller starts the Task 8 pilot (2 tasks x seed 42) now on GPU while Tasks 5-7 are built — fine-tuning code is reviewed; scoring script comes later — costs only a rerun if Task 4 code changes.
Task 5: dispatched (base cc39f27, sonnet). Pilot (Task 8 step 1) running on GPU 0 in background.
Task 8 pilot: 8/8 runs done on GPU0; 22-74 s/run (lipo full ft 74s, es_epoch 10; n128 ft es_epoch 2-5); 299 MB for 8 runs.
Ruling: full fine-tune sweep (2 shards, GPU0+GPU1) launched before the sanity-score gate — pilot timing implies ~5-6 GPU-h total, cheap to rerun if the gate fails; scoring script (Task 8) gets written after Task 5 to avoid parallel implementers — costs ~6 GPU-h if the protocol turns out wrong.
Task 5: implemented ad8bb5c (DONE; 1918s mordred run; pubchem 944,296 lines; Zenodo file served from record 15733575, size 43,931,153 = listing under 15733574). Review dispatched.
Task 5: ⚠️ verified by controller: 29 parquets present.
Task 5: minor (deferred): PubChem parquet lacks `smiles` column while task parquets have it — Task 9 must select descriptor columns by descriptor_modules() index (carry to Task 9 brief); local numpy import in summarize; except-retry path untested.
Task 5: complete (commits cc39f27..ad8bb5c, review clean)
Ruling: Task 6 ridge = exact per-column ridge w/ unpenalized intercept on non-NaN fit rows (sklearn-equivalent), Z standardized on all fit rows; choose_alpha may use NaN-free columns only — alpha is fixed across encoders so selection detail can't bias deltas.
Ruling: Task 6 bootstrap SE — identical rows across seeds (PubChem) → joint resample, SD of seed-mean delta; different rows (task clean rows) → independent per-seed paired resamples, sqrt(Σse²)/S — plan's single formula would mis-state PubChem SE (seeds correlated) — if wrong, SEs off by ~sqrt(3).
Task 6: dispatched (base ad8bb5c, sonnet)
Task 6: implemented 91cafb4 (DONE; ridge 9-40s at 8000x2048->2000x1400; choose_alpha 63s). Review dispatched.
Task 6: review Needs fixes — Important: r2_cols constant-column SST rounding → −1e15 instead of NaN; crossfit leakage test vacuous (plan-mandated weakness). ⚠️ choose_alpha raises if no NaN-free column.
Ruling: choose_alpha falls back to fewest-NaN columns with fit-fold mean imputation (alpha selection only) instead of raising — alpha is shared across encoders, so this can't bias deltas.
Task 6: minor (deferred): test_duplicate_smiles_rows_kept near-vacuous (plan-mandated); no RED evidence.
Task 6: fix round 1 dispatched to implementer a2ea910f77fd6bdcd (FIX_BASE 91cafb4)
Task 6: fix round 1/5 (3 addressed, 0 open; commits 91cafb4..434ae7f)
Task 6: complete (commits ad8bb5c..434ae7f, review clean)
Task 7: dispatched (base 434ae7f, sonnet). Rulings in dispatch: NaN→not dropped; eta2 drops NaN; p=(1+#≥)/(1+n); clusters pairwise Spearman, NaN ρ→0, average linkage.
Ruling: correlation_clusters ranks each column once then pairwise-complete Pearson (≤0.03 diff vs pandas per-pair re-ranking with NaNs; 14s vs >12min) — clusters cut at |rho|>=0.9 barely affected — costs slightly different family borders if wrong.
Task 7: minor (deferred): eta2_perm_test raises on all-NaN input (carry to Task 9: guard / return NaNs); exact float compare null>=obs; module-level rng in tests; no RED evidence.
Task 7: complete (commits 434ae7f..515faa3, review clean)
Task 8: dispatched (base 515faa3, sonnet) — 05_score.py + sanity on finished runs; pilot+sweep already run by controller.
Task 8: implemented 2259ce9 (DONE). Partial sanity on 79 runs: 21 full/ft/es rows, 17/21 within paper range, 20/21 within 1 SD (miss: adme-fang-hclint-1 s709 0.7053 vs min 0.7132, sd 0.0059). Gate: PASS (partial). Review dispatched.
Task 8: minor (deferred): no run["task"]==name guard; rescoring skips re-run dirs keyed already; score_task untested (needs Polaris); docs/results matched by .gitignore `results/` rule (force-added).
Task 8: code complete (commits 515faa3..2259ce9, review clean). Pending: full 28-task sanity after sweep (needs Polaris re-login after ~20:45).
Task 9: dispatched (base 2259ce9, sonnet) with task-9-controller-decisions.md (rulings 1-13: targets fit rows, alpha per probe set, views pubchem/task_all/task_clean, rand anchor in pubchem+task_all only, clusters once from PubChem fit, emb cache, per-task resumable outputs, pilot on lipo+ames).
Task 9: implemented 1cf4324 (DONE_WITH_CONCERNS). Pilot (lipo, ames): pre R² median PubChem 0.937, rand 0.587; ames full/es pubchem median ΔR² −0.080, 931 dropped, CKA 0.72; poscontrol 0–1/20 gain; ~25 min/task.
Ruling: probe Z standardization floors column SD at 1% of median column SD and clips standardized Z to ±6 (all encoders, fit+eval) — near-constant ReLU dims gave |z|~7.5e4 and exploding preds (lipo seed117 ΔR² −2.15) — costs a slightly non-sklearn ridge if wrong.
Ruling: drop calls only for descriptors with pre R²>0 on that view's rows (all seeds' clean rows for task_clean); n_eligible recorded; η²/families on eligible only — can't "forget" what pre never encoded; removes VR1_A-type −1e6 R² — costs excluding weakly-encoded descriptors.
Task 9: concerns fix dispatched to implementer adab6df3a14a2f3a2 (FIX_BASE 1cf4324)
Task 9: fix commit 2e92856 (R1-R3). Pilot after fix: lipo full/es median ΔR² ≈ −0.05 (pubchem/all/clean), CKA .84, ~55% eligible dropped; ames full/es −0.073 pubchem, 900/1328 dropped, CKA .72; n128/es ≈ −0.01, few drops; rand anchor −0.11 to −0.22. Seed117 outlier resolved. Positive control: 0/20 gain (label-correlated descriptors also lose R²).
Ruling: add per-encoder-α PubChem view (`pubchem_ownalpha`) for pre, es ft encoders, rand — broad uniform drops could be a fixed-α artifact if fine-tuning changes the spectrum; spec 6.3 lists this as optional robustness, now necessary — costs ~1.5h compute.
Task 9: R4 sent to implementer adab6df3a14a2f3a2
User decision (2026-10-22 ~20:00): add PCA (decorrelated) probe view + effective dimensionality to phase 1. Sent as R5 to Task 9 implementer: PCA on PubChem fit-split standardized targets to 99% var; PC targets for all probe sets via PubChem stats; probe_set suffix _pc; evr-weighted ΔR²; effdim per Mordred module (n90/n95/n99, participation ratio).
Controller: started overlap check (task molecules vs pretraining train/val/test splits) in scratchpad/overlap.py.
Finding: task molecules in CheMeleon pretraining TRAIN split (non-isomeric canonical match): 35.4% overall (train+test rows); TDC cyp/dili/half-life/bioavail ~71-79%; ames/ld50/bbb ~63%; Polaris pkis2 1.9%, adme-fang 3-14%.
Ruling: add `_novel` task views (rows not in pretraining train/val) as extra views; primary view definition change left to user — pre may be advantaged via memorized descriptors on overlapping molecules.
Task 9: R6 sent (pretrain_membership + task_*_novel views).
User decision: keep primary task view as-is (not novel-only); _novel views secondary. Spec §9 updated.
Ruling: full measurement (26 remaining tasks, 4 workers, 2 per GPU) launched before Task 9 review — embeddings cached, so a review-driven fix only reruns probe stages — costs ~2-3h compute if a bug forces full rerun. Sweep complete: 324/324.
Task 9: R4-R6 committed (85c1eb8, 0657251, d63cdb5). Full-task review dispatched over 2259ce9..d63cdb5.
Task 9: review Approved (pairing, clean rows, novel, eligibility, PCA numerically verified, own-alpha, no test leakage, resumability). Important: I1 evrw includes ineligible PCs; I2 script-level view logic untested.
Ruling: I1 handled in Task 10 summary (evrw over eligible PCs = PCs whose `dropped` is not NaN; stored all-PC value kept and labeled) — avoids touching running measurement — costs nothing if wrong (both reported).
Ruling: I2 tests-only fix while measurement runs (no behavior change).
Task 9: minor (deferred): M1 combine doesn't check *_done markers for pc/novel/ownalpha/_global; M2 n_rows only via novel_task; M3 duplicated loops in 03_measure.py (528 lines); M4 loose spike-test bound; M5 family_delta on empty eligible set unguarded.
Caveats for write-up: n128 clean rows include 26 early-stopping val rows (es selected on them); 1,597/10,000 PubChem probe molecules match pretraining-train molecules under non-isomeric SMILES (stereoisomers; identical 2D descriptors) — user's fairness decision covers it; report.
Task 9: fix round 1 dispatched (I2 tests only; FIX_BASE d63cdb5)
Measurement launch fix: first two launches exited silently — zsh does not word-split unquoted $grp, so --only got one bogus slug. Relaunched 20:59 with ${=grp}; 4 workers alive (2/GPU).
Task 9: fix round 1 (I2 tests c6a69dc) — re-review dispatched.
Ruling: Task 10 pre-registered branch rule — little_drift: CKA≥0.95 & dropped/eligible≤5%; descriptor_specific: η² p<0.05 for both family defs AND ft beats frozen by >1 seed-SD; uniform otherwise; co-primary views must agree — spec gave qualitative branches only; thresholds fixed before seeing full results — costs a re-labeling if user prefers other thresholds.
Ruling: findings note written by controller, not the Task 10 implementer.
Task 10: dispatched (base c6a69dc, sonnet) with task-10-controller-decisions.md
Task 9: fix round 1/5 (1 addressed, 0 open; commits 9de1509..c6a69dc)
Task 9: complete (commits 2259ce9..c6a69dc, review clean)
Task 10: implemented e66b421 (DONE). Pilot branches: ames uniform (gap not >1 SD), lipo descriptor_specific; η² p=0.001 everywhere (note: test likely over-powered — report η² effect size); PC spectrum: tail PCs lose most. Review dispatched.
Measurement relaunch #3 (21:10) with OMP/MKL/OPENBLAS=12 threads per worker — previous workers oversubscribed (≈15 cores each, load 150). Embedding cache kept.
Task 9: bug — expected_runs requires n128 for rppb/hppb (n_train≤128) → skipped. Fix round 2 dispatched to implementer; it will measure only those 2 tasks after fixing.
Task 10: review Approved. minor: ft/frozen gap not seed-matched; drift_fraction medians over different eligible sets (rand task_all vs view); robustness tables untested/inline; per_task gap/branch columns not asserted end-to-end; load_meta silently drops tasks.
Ruling: fix the two numeric minors (seed matching, common eligible set for drift_fraction) after Task 9 round 2 finishes — they change reported numbers slightly — costs one small fix round.
Task 9: fix round 2 commit e447082 (rppb/hppb measured, done.json present); re-review dispatched. Measurement progress 21:26: 14 done.json.
Task 10: fix round 1 dispatched (seed-matched gap; common-eligible drift_fraction; FIX_BASE e447082).
Task 9: fix round 2/5 (1 addressed, 0 open; commits e66b421..e447082). Note for write-up: rppb/hppb task_clean_novel NaN (<30 rows).
Task 9: complete (commits 2259ce9..e447082, review clean)
Task 10: fix round 1/5 (2 addressed, 0 open; commits e447082..72f7380)
Task 10: complete (commits c6a69dc..72f7380, review clean). Findings note (step 6) pending full measurement + rescoring.
Final whole-branch review dispatched (ffc899c..72f7380, sonnet) while measurement runs.
Final review: "With fixes". Important: (1) linear CKA biased upward at small n (clean rows ~100) → little_drift gate; rand CKA anchor on different rows; (2) η² permutation gate anti-conservative, ≈always passes → branch effectively decided by gap; (3) pc_spectrum averages over varying task sets for task views; (4) combine lacks *_done marker checks (M1). Minors: missing branch→split; load_meta NaN→low_conf False; unbound ckpt lambda; README anchor note.
Ruling: primary pre-registered branch rule kept; F1 debiased CKA (same-rows rand anchor) + F2 η²/null_q95 ratio and strict/no-η² variants added as SENSITIVITY columns — changing the rule post hoc would be HARKing — costs extra columns to read.
Final fix wave dispatched (one fresh sonnet implementer, final-fix-brief.md, FIX_BASE 72f7380).
Final fix wave committed (7f456e5, 061b5d3, d2407ef). Debiased CKA ≈ linear on real data (≤0.008). Finding: ames ft-vs-pre CKA on task rows (0.53) below rand-vs-pre (0.67). Scoped re-review dispatched. Measurement 18 done.json.
Final fix wave: re-review — all findings addressed; primary branch rule unchanged (verified). Residual minors (parked): stale comment on VARIANTS/CORR_DEFS; no_eta2 variant None when η² p missing — Ruling: park, cosmetic/edge-only — costs nothing on real data.
Code complete. Remaining: measurement finish → 06_cka.py → 03 --combine → Polaris re-login + 05_score.py → 04_summarize.py → findings note.
Full sanity (648 score rows): 84 full/ft/es runs — 53/84 within paper range, 77/84 within 1 SD; 23/28 tasks all seeds within 1 SD. Outliers: hclint s709, ret-cls s117 (0.418 vs 0.53–0.64), cyp2c9 s709 (better), herg s117, vdss (2 seeds better, 1 worse). Gate PASS.
Findings note written: docs/results/2026-10-22-drift-findings.md (904e67f). Phase 1 complete.

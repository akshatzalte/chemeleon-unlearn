# SDD ledger — plan: /data3/akshatz/chemeleon_unlearn/docs/plans/2026-10-22-phase2-tailored-pretraining.md
Spec: docs/specs/2026-10-22-phase2-tailored-pretraining-design.md. Branch drift-check. Subagents: sonnet (user rule).

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1→T3 | sets/<arm>.txt names; T3 selects columns by name vs pool columns.json | ok (Review Focus 2) |
| T2→T3 | pool memmaps + columns.json + stats.npz | ok |
| T3→T4 | mp.pt chemeleon format loadable by drift.encoders.load_mp (PY1) | ok; T3 test does PY1 subprocess load |
| T4 self | run_one additive params; regime="custom" new value — make_split must not be called | note to implementer |
| T4→T5 | runs/phase2/ladder layout, run.json arm/N/seed | ok |
| T1 env | PY2 env creation is in T1 but used only by T3 | ok |
| Two envs | tests/phase2/test_pretrain* need PY2; pytest default config runs with PY1 → importorskip | ok |
Ruling: pretraining tests live under tests/phase2 and skip when cuik_molmaker is absent, so the phase-1 suite stays green under PY1 — costs nothing if wrong.
Task 1: dispatched (base 6ec247b, sonnet)
Pushed to https://github.com/akshatzalte/chemeleon-unlearn (public; default branch drift-check; 32 commits). Policy: push after each completed phase-2 task.
Ruling (user): no Mordred recomputation — pool descriptors from Zenodo 15733575 zarr (v3; PY2); stats from pool-train only. User rule: test everything (unit + e2e smoke on every arm) before committing pretraining compute. Download started 23:1x to data/phase2/zenodo/.
Task 1: implemented 404a482 (DONE_WITH_CONCERNS). Gate passed via LD50 (T25>R25a,b; T100>R100); Ames T≈R as expected. cuik_molmaker from conda-forge (0.2.1); libstdc++ .pth fixup script. Review dispatched.
Zenodo file is zstd (not xz); extracting with zstd -dc | tar at 00:01.
Zenodo zarr extracted: (944296, 1613) float32, chunks (162,1613), rows == SMILES lines. Ready for Task 2.
Task 1: review Approved. minor (deferred): gate margin thin (T25 vs R25a +0.010 R² LD50; Ames inverted at k=25) — interpretation caveat; OMP_NUM_THREADS only documented; .pth unguarded CDLL; tests read untracked results/; evaluate_gate untested.
Task 1: complete (commits e32a20c..404a482, review clean)
Task 2: dispatched (base 404a482, sonnet; PY2, Zenodo zarr)
Task 2: BLOCKED on alignment gate — Zenodo zarr vs phase-1 recompute: NaN pattern 100% identical, 99.95% cells match, 58 columns differ (RDKit/mordred version drift in authors' run; nHBAcc +1..+5 on 15% rows; IC family 0.2–3.6%).
Ruling (advisor-confirmed): option (a) use zarr as-is (it IS CheMeleon's targets; B stays faithful); gate narrowed: strict NaN-pattern alignment + recorded mismatch columns (abort if >100 cols or any >50%). Caveat for spec §9: selector used recomputed values, pretraining uses zarr values (affected: T25 MIC3; T100 6 cols incl nHBAcc).
Task 2: DONE a6d7036 — pool 180k/20k, 2,829 test lines excluded, build 29 s; gate narrowed per ruling. Push fixed via gh auth setup-git (credential helper was missing). Review dispatched.
Task 2: review Approved (stats train-only, row order, exclusion, phase-1 suite all verified on real build). minor (deferred): read_rows read amplification; drop_all_nan/script untested; check_alignment assumes column order.
Ruling: heavy-tailed columns (VR1_A etc., sd~1e18) are left under mean/sd + ±6 clip exactly as CheMeleon's pretraining did (faithful B; identical across arms) — they become near-constant targets, a known property of the original pipeline — costs nothing for T/R if those columns are absent from the sets.
Task 2: complete (commits 404a482..a6d7036, review clean)
Task 3: dispatched (base a6d7036, sonnet) with task-3-controller-decisions.md (smoke every arm + gate B/T25 2 epochs full pool; no full launches).
Task 3: DONE fc73fbc. Gate: B 54.5 s/epoch (GPU0), T25 63.4 s/epoch (GPU1, shared) on 180k; 6.7 GB alloc / 15.7 GB reserved → one arm per GPU. 100 ep × 7 arms ≈ 10.6–12.3 GPU-h; 30 ep ≈ 3.2–3.7 GPU-h. Smoke: all 7 arms OK, mp.pt loads in PY1.
Ruling (advisor-confirmed): keep P=200k for the primary arms (finishes tonight even at 100 epochs); launch order B+T100 → T25+R100 → R25a+R25b → D535, one per GPU, GPU index recorded; headroom after: second seeds for B/T100/T25 (swap GPUs), then optional full-pool B/T100 replication (user may veto). Launch only after Task 3 review approves (user: test before compute).
Note: smoke val_loss table (2k rows, 2 epochs) is noise — do not interpret.
Task 4: dispatched (base fc73fbc, sonnet) in parallel with Task 3 review (disjoint files). Ruling: Task 4 may start before Task 3 review closes — file sets disjoint (finetune/ladder vs pretrain*).
Task 3: review Approved (loss divisor=kept count, NaN masking, val_loss plain masked MSE, name-based selection, reshuffle, no cache, bad-SMILES guard, checkpoint format, hparam parity all verified). minor (deferred): assert smiles/desc length before slicing; weak invalid-smiles test; sampler epoch off-by-one (cosmetic); git_sha ignores dirty tree.
Task 3: complete (commits a6d7036..fc73fbc, review clean)
PRETRAINING LAUNCHED 00:36 (machine clock): GPU0 queue B→T25→R25a→D535; GPU1 queue T100→R100→R25b (results/phase2/pretrain_queue.sh; logs results/phase2/logs/). Watch: first finished arm's metrics.csv — if patience-5 stops before ~epoch 15 with a noisy descending curve, consider patience 10 rerun.
Task 4: DONE_WITH_CONCERNS a9638a1. Ruling: accept warmup_epochs=2/n_batches (float) over the brief's integer rule — gives exactly 2 warm-up steps per Global Constraints; brief's rule was wrong at N=64. Implementer accidentally started+killed 2 ladder jobs (no outputs kept; pretraining untouched). Review dispatched.
Task 4: review Approved. minor (deferred): shard_items pins arms when n divides 9 (fix key to (k%A + k//A)%n, add n=3 test); `if cb` truthiness; private _split import; no both-class retry at N≥128.
Task 4: complete (commits fc73fbc..a9638a1, review clean)
LADDER LAUNCHED: results/phase2/ladder_loop.sh per GPU (shard i/2, 2 jobs/GPU), re-invoked every 15 min until all 7 arms done and 0 pending. Avoid n=3 sharding (minor bug).
INCIDENT 00:56: ladder jobs OOM'd on both GPUs (pretraining allocator reserves ~22 GB); ladder killed within ~2 min, pretraining unaffected (no errors, epochs continuing), one completed chemeleon ladder run kept. Ruling: ladder loops now wait for pretraining on that GPU to finish before starting (3 jobs/GPU). Costs ladder wall time (starts after queue), fits before morning.
Task 5: DONE 4b99201 (implementer ran p2_04_score once against Polaris despite instruction — harmless, output deleted; noted). Review dispatched.
Task 5: review Approved (endpoint verified numerically by reviewer). minor (deferred): incremental scorer keeps stale rows if a run dir is rewritten; NaN on one side of a pair counted in n; no boundary tests.
Task 5: complete (commits a9638a1..4b99201, review clean). Step 5 (real scoring/analysis/findings) pending ladder + Polaris login.
Finding 01:10: T100 (patience 5) stopped at epoch 19 after train-loss spikes (0.127, 0.289) and noisy val; best epoch 14 (0.0276). B smooth, still descending at epoch 29. Patience-5 confound: B would train 2–3× longer than small-k arms.
Ruling (advisor-confirmed): patience 15 for every arm, max_epochs 100, LR untouched. Non-destructive path (auto-mode classifier denied kill+rm): queue/ladder shell loops stopped; running B and R100 (patience 5) left to finish and kept as B_p5/R100_p5 with T100_p5 (patience comparison); then all 7 arms rerun with --patience 15; ladder relaunched after. Cost ≈ +1 GPU-h waiting + longer arms.
Final whole-branch review (phase 2 code, e32a20c..4b99201) dispatched while compute runs.
Final review (phase 2): "With fixes". Important: shard key pins arms at n·J=6 (loops use 2×3); patience default still 5 + no --seed + no readiness guard on config; endpoint computed on whatever pairs exist (need pair-count + sha1 asserts); residual training-length confound (report epochs per arm). Minors: ladder_loop termination heuristic, relaunch waits for both p5 arms, freeze-encoder secondary unimplemented, nested subsamples, reference arm not apples-to-apples, spec §5 E=6 stale.
Ruling: fix wave F1–F4 dispatched now (code only; takes effect for queued jobs since each process re-imports). Freeze-encoder secondary deferred (optional, after primary results). Verify 784 done.json before scoring; cross-check chemeleon N=full vs phase-1 full-ft.
Final fix wave: 4c18f19 515c1c5 6e73d95 fa1abd6. Scoped re-review dispatched.
Final fix wave: re-review — all addressed, training path unchanged. Phase-2 CODE COMPLETE (ffc899c..fa1abd6). Remaining: compute (p15 arms → ladder) → scoring (Polaris login) → analysis → findings note; cross-check chemeleon N=full vs phase-1 full-ft.

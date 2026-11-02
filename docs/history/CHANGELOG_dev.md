# Changelog

## 2026-11-02 (v3.7): code v9, lean HF-README benchmark prepared (nothing run on Kaggle, uploaded or published)

- Srimon (t48u): proper but lean, publishable benchmark for the HF README. Scope locked: 14 GLiClass-v3 card sets on FULL test splits (avg14 + 12-set zero-shot avg), Banking77/CLINC-150 accuracy, long-Jev@8192; models v0.1, gliclass-modern-base-v3.0, gliclass-base-v3.0.
- `scripts/hf_readme_eval.py` (prep / worker / aggregate): thin wrapper on eval.py + eval_long_jev.py; per-(model, set, label order) resumable npz logits, 2-GPU work-stealing queue, length-sorted batches, OOM halving, hard stop; aggregate = macro-F1/acc + bootstrap CIs, paired bootstrap on avg gaps, McNemar, long-Jev paired compare, card reproduction check, TABLES.md.
- `kaggle_eval_readme.ipynb` (`scripts/build_hf_readme_eval_notebook.py`); `scripts/eval_long_jev.py --add_prefix_space`.
- Job spec: `notes/HF_README_EVAL_JOB.md`. CPU plumbing smoke + notebook glue test passed.

## 2026-10-25 (v3.6): stage 3 (long context 8,192 + up to 256 options) prepared; smoke on the current account, full run deferred to account #2 (nothing run on Kaggle, uploaded or published)

- Srimon: "we still need stage 3 to prove anything"; then (25 Oct) smoke + real data prep on the current Kaggle account only, full train later on a 2nd account.
- `configs/data_stage3.yaml` (v2): long Jev (`full` train, state >= 6,000 chars, license_use commercial, no -sa/by-sa/gpl/odbl rows; 300 val carved) + 40,000-row seeded replay of S2 `data/stage2_full/train.jsonl` (S2 label pools copied). Box dry run: **61,184 train rows (21,184 long Jev + 40,000 replay), 300 val**, md5 d35eb55c...; 774 train rows sharing a val text dropped; 0 train texts in the long-Jev eval slice / val / slice 3k-prefixes. Procedural-style share 13.9 %.
- `prepare_data.py`: `min_state_chars` vectorised pre-filter, `exclude_license_contains`, `format: jsonl` replay (`take_jsonl`, prefixes source `replay:`), `pools_file_from`.
- `configs/stage3_longctx.yaml`: init runs/stage2_full_C/final_model, max_num_classes 256, max_length 8,192, many-label k_max 250, consistency 0.25/0.1 skl, batch 1 x accum 32 x 2, ckpt on, LR 5e-6/1.5e-5, **956 steps**, output runs/stage3_full.
- Eval: `calib/jev_long_test.jsonl` (Jev full test+validation, states >= 6k chars, 431 rows / 344 groups; `scripts/build_long_jev_slice.py`), `scripts/eval_long_jev.py` (accuracy, cluster-bootstrap CI, paired `--compare` with McNemar; bar +0.03).
- `kaggle_stage3.ipynb` (`scripts/build_stage3_notebook.py`, from the S2 full notebook): MODE smoke (real data, S2 baseline, 1.0-h train cap -> checkpoint + needs_resume, checkpoint long-Jev eval; session cap 1.8 h) / full (resume the smoke output, final evals + compare). OOM -> one resume at max_length 4096. S2 data symlinked read-only and unlinked before save.
- CPU verification (`scripts/run_stage3_notebook_cpu.py`): smoke -> needs_resume at step 5 with real data (md5 identical to the box build); full with the smoke output attached -> restore, md5 match, resume from checkpoint-5, complete, evals + compare. Plain `train.py` 3-step CPU run OK.
- Docs: `STAGE3_PLAN.md`, `STAGE3_JOB_SPEC.md`. Code bundle v8 NOT built yet.

## 2026-10-24 (v3.5): code v7, train/val text dedupe for the full stage-2 data (prepared only; nothing run on Kaggle or published)

- Kaggle full run v2 (code v6) stopped at the data cell: 835 train rows had an exact-text duplicate in `calib/stage2_pilot_val.jsonl` (limit 1 % of 1,640 distinct val texts = 16). Other checks passed (1,732,370 rows, procedural 5.25 %, 3,000 typed rows).
- Cause: `take()` dedupes on (text + prompt, sorted labels), so the same TEXT under another label set still reaches train. gliclass-v2.0 holds many texts several times with different label sets (and includes texts that are also in gliclass-v3-logic); logic rows are repeated x3. The val rows are carved first, the other label-set copies land in train. At pilot size (gliclass-v2.0 ~59k rows) this was ~30 rows; at 1.09 M it is ~18x more.
- `prepare_data.py`: new `dedupe_train_vs_val(train, val, recipe, per_source)`, opt-in by recipe key `dedupe_train_vs_val: true` (+ `dedupe_val_files`). Runs after sampling/taxonomy finalisation and before the shuffle/write; drops every train row whose `text` equals (exactly, the notebook's rule) a text of this run's val or of the listed val files; logs and writes `manifest.train_val_dedupe` (dropped by train source and by val source); updates per-source `train_rows` (+ `train_rows_before_dedupe`, `dropped_val_text`). Rows carry an internal `_skey` during the build, stripped before writing. Val files are never changed. Older recipes (no key) build byte-identical data.
- `scripts/make_stage2_full_recipe.py` / `make_full_smoke_recipe.py`: set `dedupe_train_vs_val: true`, `dedupe_val_files: [calib/stage2_pilot_val.jsonl]`; regenerated `configs/data_stage2_full.yaml` and `_smoke.yaml` (only these keys + a header line changed). Notebook unchanged; its leak assertion is kept as is.
- Full-size CPU build of `configs/data_stage2_full.yaml` on the box (non-streaming, cached sources, ~9 min, ~4.7 GB RSS): before dedupe 1,732,370 rows (= Kaggle v2 exactly); dropped 865 (211 distinct val texts): gliclass-v2.0 758, gliclass-v3-logic 90 (30 texts x repeat 3), Jev 15, go_emotions 2. After: 1,731,505 rows (need 1,728,000, margin 3,505), procedural-style rows 90,923 (5.25 %, unchanged), typed TRAIN rows 1,500 + 1,500, dedicated procedural 0; the notebook's leak count reads **0**. 865 > 835 because it also removes ~30 rows matching the recipe's own carved val, which differs from the pilot val in 39 texts (34 logic, 5 procedural: the larger gliclass sample shifts the logic carve via the shared `seen` set); training uses the pilot val file. train.jsonl md5 on the box d82991a305052e585050954f1ac6778f (Kaggle may differ if a Hub dataset revision differs; the notebook records its own md5 in session 1).
- Pilot/ablation overlap (box rebuilds of their train sets vs `calib/stage2_pilot_val.jsonl`): pilot 30 rows (26 distinct val texts), E1 33 (29), E1z 33 (29), E2 32 (28), all gliclass-v2.0 / logic. These notebooks had no leak assertion. ~1.6-1.8 % of val texts were seen in training under another label set: a small optimistic bias on their eval_loss, the same for every arm, so arm comparisons hold.
- Tests: new `scripts/test_train_val_dedupe.py` ALL PASS; `scripts/test_stage2.py` ALL PASS; notebook CPU smoke (fresh session, mock mounts with code v7 + S1 output, smoke recipe, 12 steps) ran end to end: data asserts passed (leak 0), `complete`, final quick eval rc 0. Logs: `smoke_logs/v7/`.
- `scripts/make_code_bundle.py` -> code dataset **v7** `/workspace/ultrabert/ultragliclass_code_v7/` (+ `.zip`).

## 2026-10-24 (v3.4): full stage-2 run package, arm C, fast settings (prepared only; nothing run on Kaggle or pushed)

- Srimon: skip the speed test and run the full stage 2 with the measured fast settings. New `configs/stage2_full_C.yaml` (pilot C recipe, S1 init, seed 42; per-device 4 x accum 8 = 64, gradient checkpointing off, fp16, length_group off, DataLoader workers 0, 27,000 steps, save/eval every 500, keep 2 checkpoints, time budget 10.4 h).
- New `configs/data_stage2_full.yaml` (`scripts/make_stage2_full_recipe.py`): E2 mix at data_stage2 sizes, ~1.74 M rows. DBpedia/Bitext off, dedicated procedural source dropped (Jev kept, ~4.9 % procedural-style), LocalLLaMA/typed-decisions TRAIN (customer_service + agent_trace_observability, 3,000 rows), HWU64 kept, forbid_ids kept, licences unchanged. Pilot val (n_val, seed 43) kept, so eval_loss stays comparable.
- New `kaggle_stage2_full.ipynb` (`scripts/build_stage2_full_notebook.py`), built from the ablation notebook: session-aware resume (`PREV_OUTPUT="auto"` restores runs/data/results and checks the data md5), session caps ARM 10.6 / TOTAL 10.8 h, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, one OOM retry at b2 x accum 16, NaN/stall/1-GPU fallback kept, mid eval per session end, final quick eval (500/set, 3 orders, max_length 1024, NOT_zero_shot notes), and data-mix asserts.
- `scripts/estimate_stage2_cost.py --plan_full`: session plan from the measured 2.4 s/step x 1.05 -> `results/stage2_full_cost_plan.json`: S1 ~10.8 h, S2 ~10.0 h, total ~20.8 GPU-h.
- `scripts/make_code_bundle.py` -> code dataset v6. Smoke tools: `scripts/make_full_smoke_recipe.py`, `scripts/run_full_notebook_cpu.py`. Spec: `STAGE2_FULL_JOB_SPEC.md`.
- Notebook CPU smoke, two sessions through the real notebook with read-only mock /kaggle/input (code v6 + S1 output), smoke recipe (3,846 rows), fp32, 1 process, shrunk eval: **S1** built the data (2.9 min, streaming), all data asserts passed (dedicated procedural 0, semantic 3,000 in the 2 workflows, 0 train/val text overlap), hit the forced time cap at step 11 -> `needs_resume`, mid eval on checkpoint-11 rc 0 -> `stage2_full_C_mid_step11.json`, summary_s1. **S2** (S1 output mounted as previous output, fresh working dir): PREV_OUTPUT auto-found, restored runs/data/results (1.72 GB in 2 s), train.jsonl md5 matched, SESSION 2 detected, resumed from checkpoint-11 (optimizer/scheduler/RNG restored) to step 30 -> `complete`, final eval rc 0 with NOT_zero_shot notes, checkpoints dropped only after that. The first attempt caught a smoke-recipe bug (procedural n_train 0 -> 8) through the dedicated-procedural assert; fixed. Logs: `smoke_logs/full/`. NOT verified: GPU, fp16, DDP/NCCL, speed, host RAM and prep time at 1.74 M rows.

## 2026-10-24 (v3.3): length-grouped batching + fast stage-2 configs (prepared only; nothing run on Kaggle or on a GPU)

- New `ugc/length_group.py`: seeded, resume-deterministic, DDP-safe mega-batch length-grouped sampler (`data.length_group: {enabled, mega_batch_mult (50), loss_norm (true)}`, **default off**), cached token lengths, `LengthModel` (adds the planned many-label expansion to the base length), `calibrate_label_norm`.
- `ugc/many_label.py`: optional *planning mode* (expansion RNG seeded from (seed, epoch, row); `planned_expansion` mirrors `expand_row`). Unchanged when planning is off.
- `ugc/trainer.py`: `StrictTrainer(train_sampler=, label_norm=)`; `set_epoch` per epoch; fixed-denominator loss when `label_norm` is set (training only).
- `train.py`: length_group wiring; v4 patch (peak_reserved_gb + per-rank "mem" log line).
- New: `scripts/test_length_group.py` (44 checks), `scripts/test_length_group_e2e.sh`, `scripts/estimate_padding.py`, `scripts/build_speedtest_fast_notebook.py`, `kaggle_speedtest_fast.ipynb`, `configs/stage2_fast_{A,B,C,C_nogroup}.yaml`, `configs/speedtest_fast_C_{nogroup,grouped}.yaml`, `configs/stage2_smoke_lg.yaml`, `notes/LENGTH_GROUP.md`, `STAGE2_FAST_SPEEDTEST_SPEC.md`.
- Estimate (6,000 pilot rows, real tokenizer, micro-batch 4): padding share 50.8 % -> 12.9 %, 1.77x fewer computed tokens (1.94x as DDP cost). Implied about 2.2-2.9x vs the pilot setting together with ckpt off + b4. ESTIMATE; the Kaggle speed test in the spec measures it.
- Tests (CPU): new suite ALL PASS, existing `test_stage2.py` ALL PASS, e2e resume + 2-proc gloo DDP OK, 6-step smoke train with the flag on completes. NOT verified: GPU, fp16, NCCL, memory, real speed.
- Fast configs keep data, seed 42, init, LR/schedule, max_steps 1,500 and effective batch 64 of the pilots; they are not bit-identical (batch composition). No existing config enables the flag.

## 2026-10-24 (v3.2): stage 2 ablation package (prepared only; nothing run on Kaggle)

- `prepare_data.py`: `answers_col` per source, `exclude_source_contains` (Jev), and a guard that aborts if the LocalLLaMA typed-decisions source is anything but its train split.
- `eval.py`: typed eval also writes `accuracy_by_workflow`.
- New: `scripts/make_ablation_recipes.py`, `configs/data_stage2_abl_*.yaml`, `configs/stage2_abl_*.yaml` (E1, E1z, E2, E3, C0; override only data/val/output of pilot C), `calib/stage2_pilot_val.jsonl`, `scripts/check_ablation_data.py` (ALL OK), `scripts/estimate_ablation_cost.py`, `scripts/build_ablation_notebook.py`, `kaggle_stage2_ablation.ipynb`, `STAGE2_ABLATION_JOB_SPEC.md`.
- Finding: Jev already carries procedural-style rows; S1 had ~4.9 %, pilot 17.2 %. Notes corrected.
- Notebook CPU smoke (mock Kaggle mounts, arms E1 and E2, 4 steps, fp32, shrunk eval): ran end to end, both arms complete, eval rc 0, results and summary JSON written. NOT verified: GPU, fp16, DDP, real Kaggle prep time. E2/E3 gated by `SEMANTIC_DATA_APPROVED=False` (owner decision D10).

## 2026-10-22 (v2): resume-hang hardening after Kaggle S0b

**What happened.** S0a (T4x2, fp16) trained to step 75 and stopped on its time budget. S0b, resumed from S0a's
checkpoint-75, ran for ~50 min with no output and an empty log (`[]`). It was cancelled by hand.

**Root cause: a hypothesis, not confirmed.** S0b left no log, so nothing below is proven.
- Most likely: a distributed hang (NCCL init or the first collective), with no timeout that actually fired. We
  found that the `ddp_timeout` handed to HF was not enforced in our `train.py` path. On the CPU repro, a rank
  stuck at a barrier with `ddp_timeout=60` was still waiting after 3 min. So a stuck collective could block for
  good. 2 of 4 T4x2 launches so far stalled silently at the 2-process launch (S0a attempt 1 and S0b).
- Ruled out as the cause: a copy race. The old notebook already copied in one process, before torchrun.
- Ruled out as the cause of a *hang*: the read-only input. The repro shows it crashes loudly (PermissionError on
  STATUS.json, then gloo "connection closed").
- Latent bugs that the repro found and that are now fixed:
  1. Resuming an fp16 checkpoint without a GradScaler (CPU/bf16/fp32) crashed in `_load_scaler`.
  2. `copytree` kept read-only dir modes from `/kaggle/input`.
  3. `destroy_process_group()` was never called.
  4. Nothing bounded a hang in the notebook.

**Changes**
- `train.py`
  - Sets `PYTHONUNBUFFERED=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1`.
  - **Creates the process group itself, early, with an explicit timeout** (`train.ddp_timeout`, default
    600 s). NCCL gets an eager `device_id`, so the communicator is built, or fails, right there. Verified: a
    stuck rank now raises after the timeout.
  - Rank-tagged, flushed trace lines (`[ugc HH:MM:SS rankR/W] ...`) for: process up, PG init, trainer built,
    the checkpoint summary, each barrier (waiting/passed), model/optimizer/RNG load on every rank,
    train_begin, and the first step after resume (begins/done).
  - Only rank 0 prints everything else: replica stdout is muted during model and dataset build, and
    `log_level_replica=error`.
  - Write probe on `output_dir` (fails fast with a chmod hint), barriers before train and after STATUS, and
    `destroy_process_group()`.
  - `HEARTBEAT.json` is written on every log step.
  - Test hook `UGC_DEBUG_HANG_RANK`.
- `ugc/trainer.py`
  - Skips `scaler.pt` when the run has no GradScaler (precision switch on resume).
  - Times every resume load step, per rank.
- `scripts/dist_check.py` (new): a 20-s torchrun pre-flight (init + all_reduce + barrier, 90 s timeout).
- `kaggle_stage1.ipynb` (built by `scripts/build_notebook.py`)
  - Auto-finds the code dataset and the previous output. This folds in the Handler's
    `/kaggle/input/datasets/<user>/ultragliclass-code` fix, generalized.
  - Restores the previous output **in the notebook process, before torchrun**. It uses plain copies, checks
    sizes, logs per file and then runs `chmod -R u+w`.
  - Every subprocess runs through `run_streamed`: live output in the notebook plus a log file in
    `/kaggle/working/logs/`.
  - A **stall watchdog** kills the process group if nothing is printed for `STALL_MINUTES`.
  - The NCCL pre-flight is retried with `NCCL_P2P_DISABLE=1`, then falls back to 1 GPU.
  - `USE_SINGLE_GPU` knob. Automatic 1-GPU resume (grad-accum doubled, same effective batch) if the 2-GPU run
    crashes or stalls.

**Tests (CPU, 2-process gloo DDP, previous output in a read-only /kaggle/input mimic):** see `SMOKE_TEST.md` § v2.

## v3: stage-2 code (23 Oct 2026) — see STAGE2_PLAN.md
- `ugc/many_label.py` (new/extended): `LabelPools` (flat or grouped pools, sibling lookup, TF-IDF char n-gram neighbours),
  `expand_row` (hard-negative label expansion, log-uniform K in [k_min, k_max], gold + near-duplicate banning, token-budgeted so the
  text always keeps >= `text_keep_tokens`), `ManyLabelDataset` (re-expands on every fetch, never mutates stored rows, optional second
  label order as a `pair`), `PairedCollator`, `consistency_loss` (symmetric KL aligned by label identity).
- `ugc/trainer.py`: `consistency_weight` / `consistency_kind`, logs `cons_loss` and pairs per log window.
- `train.py`: reads `data.many_label` and `data.consistency` (not compatible with `static_shapes`); precomputes neighbour tables
  before DataLoader workers fork.
  **Bug fix:** the `UGC_DEBUG_HANG_RANK` hook compared `None == None` when the variable was unset, so every single-process run
  (not torchrun, which sets RANK) slept forever. Stage-1 Kaggle runs used torchrun and were unaffected.
- `ugc/convert.py`, `prepare_data.py`: taxonomy sources (`from_taxonomy`, `resolve_names`, label-phrasing `schemes`), label pools
  (`label_pools.json`), `enabled: false` switch per source, extra manifest keys (pool sizes, eval_overlap, license_note).
- `configs/data_stage2.yaml`, `configs/stage2_scale.yaml` rewritten; `configs/data_stage2_smoke.yaml`, `configs/stage2_smoke.yaml` added.
- scripts: `make_calib_val.py`, `analyze_perm.py`, `compare_results.py`, `estimate_stage2_cost.py`, `test_stage2.py`.

## v3.1: stage-2 PILOTS A/B/C (23 Oct 2026) — see STAGE2_PILOT_JOB_SPEC.md
Decision (Srimon): run the 3 stage-2 pilots; DBpedia and Bitext are excluded (licence decision D2).
- **Bug fix (`ugc/many_label.py`)**: `PairedCollator` sized the pair permutation from the primary view only; when label
  augmentation added a label to the second view it raised `IndexError` (would have crashed pilot C). Now sized from both views.
  Regression test added to `scripts/test_stage2.py` (section 4); verified to fail on the old code.
- **Data**: `configs/data_stage2_pilot.yaml` (96,035 train rows, 1,850 val; natural 86.4% / taxonomy 13.6%; HWU64, LEDGAR, fin-topics,
  GoEmotions, fin-sentiment; no DBpedia/Bitext; same val for all arms). `prepare_data.py`: `forbid_ids` guard (refuses banking77/clinc/massive/
  snips/financial_phrasebank/sst5 etc.), `disabled_sources` and `license_note` recorded in the manifest.
- **Arms**: `configs/stage2_pilot_{A,B,C}.yaml` (B inherits A, C inherits B; the only differences are `many_label.enabled`,
  `consistency.enabled`, `output_dir`). 1,500 steps x 64 rows, LR 1e-5/3e-5, linear, warmup_ratio 0.02, save/eval every 500, seed 42.
- **Notebook**: new `kaggle_stage2_pilot.ipynb` (built by `scripts/build_notebook.py` from `scripts/pilot_cells.py`; `kaggle_stage1.ipynb` regenerated,
  unchanged apart from cell ids). Knobs: `ARMS`, `INIT_MODEL` ("auto"/path), `PREV_OUTPUT`, `ARM_MAX_HOURS`, `TOTAL_MAX_HOURS`, quick-eval knobs.
  Keeps all v2 safety features; auto-runs `eval.py` after each arm -> `results/stage2_pilot_{A,B,C}_quick.json`.
- `train.py`: `UGC_DUMP_ORDER=<prefix>` logs the fetched row indices (debug hook for the same-order check; off by default).
- `calib/stage1_val.jsonl` (the S1 val set, md5 cacbb981...) for the quick-eval temperature fit.
- New scripts: `make_pilot_smoke_recipe.py`, `check_pilot_arms.py`, `estimate_pilot_cost.py`, `run_notebook_cpu.py`, `make_code_bundle.py`.
- Smoke: see SMOKE_TEST.md v3.1. Bundle: `/workspace/ultrabert/ultragliclass_code_v3.zip` (not uploaded).

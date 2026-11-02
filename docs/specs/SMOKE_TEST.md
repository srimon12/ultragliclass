# CPU smoke test: 22 Oct 2026 (box: 8 CPU cores, 15 GB RAM, no GPU)

Environment: Python 3.13, torch 2.14.1+cpu, transformers 5.20.2, gliclass 0.1.21, datasets 5.1.0, accelerate 1.15.0.
Backbone: the real `dmis-lab/UltraBERT-150M` (not a tiny stand-in). Reproduce with `bash scripts/smoke_cpu.sh`.
Logs are in `smoke_logs/`, and the final STATUS.json of every run is in `smoke_logs/status/`.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | `prepare_data.py --streaming` on the stage-0 recipe (gliclass-v2.0 + logic + jev, commercial filter) | PASS: 240 train / 32 val rows in 35 s | 01_prepare_stage0.log, data/stage0/manifest.json |
| 2 | GLiClass init leaves the 218 UltraBERT tensors untouched; the 3 new special tokens are seeded from "label"/"text"/"example" | PASS: 0 of 218 changed | 00_check_init.log |
| 3 | Training, 6 steps (fp32 on CPU), eval and checkpoint every 3 steps, final_model + STATUS.json | PASS: loss 0.23 → 0.28 (6 steps, noise), eval_loss 0.260 → 0.256 | 02_train_uninterrupted.log |
| 4 | Resume from checkpoint-3 (optimizer, scheduler, RNG, data position) | PASS: exact. With dropout 0 and no label shuffling the resumed steps 4-6 match the uninterrupted run digit for digit (0.2059 / 0.2653 / 0.2821). With dropout on they differ in the 3rd decimal (stochastic) | 03_*, 04_det_* |
| 5 | Time budget: stop, save and exit cleanly, then resume to the end in a new process | PASS: stopped at step 11/20 with `needs_resume`, checkpoint-11 saved; rerun resumed from 11 and finished `complete` | 05_budget_session{1,2}.log |
| 6 | DDP, 2 processes (torchrun, gloo backend) | **FAIL, then fixed.** `model.logit_scale` gets no gradient (it is only used when normalize_features=True), so DDP raised. The official Trainer *swallows* the exception ("Skipping iteration due to error") and kept training on nothing: loss 0, grad_norm 0. Fixes: freeze the unused `logit_scale` (ugc/model_init.py) and use `StrictTrainer`, which raises instead of skipping (ugc/trainer.py). After the fix: PASS (4 steps, world=2) | 06_ddp_gloo_2proc.log, 06b_unused_params.log |
| 7 | DDP resume from checkpoint-2 | PASS | 07_ddp_resume_from2.log |
| 8 | Stage chaining: `model.init_from` = previous stage's final_model | PASS | 10_init_from_chain.log |
| 9 | bf16 switch (CPU autocast) | PASS (2 steps). fp16 needs CUDA, so it was not testable here; it is the first thing stage 0 on Kaggle checks | 11_bf16_cpu.log |
| 10 | Static-shape collator (TPU path): pad to max_length and a fixed 50 label slots | PASS: eval-mode loss equal to dynamic padding (diff 1.5e-8, max logit diff 2.5e-7) | 12_static_shapes.log |
| 11 | Gradient checkpointing on UltraBERT | PASS (all 25 modules flagged; identical loss) | 13_gradient_checkpointing.log |
| 12 | Control config (ModernBERT-base backbone) trains through the same code | PASS | 14_control_modernbert.log |
| 13 | eval.py end to end: 14 GLiClass sets + Banking77 (77 labels) + CLINC-150 (150 labels) + typed-decisions, 3 label permutations, ECE before/after T | PASS on tiny subsets (scores are meaningless after 6 steps) | 08_eval_smoke.log, results/smoke_stage0.json |
| 14 | Eval harness validated on a published model (`knowledgator/gliclass-modern-base-v3.0`, 200-example subsets) | PASS: CR 0.887 / sst2 0.910 / ag_news 0.685 / rotten_tomatoes 0.685 macro-F1 vs card 0.890 / 0.896 / 0.666 / 0.707 | 09_eval_control_modernbase_v3.log |
| 15 | Data recipes for stages 1-3 parse and stream (20 rows per source) | PASS. Found: only ~1 in 215 jev `full` rows has a state of 6,000+ chars, so stage 3 takes that source whole (recipe updated) | 15_recipe_dryruns.log |
| 16 | `kaggle_stage1.ipynb` run end to end under nbclient with Kaggle paths remapped: session 1 (install → streamed data → train → status → cleanup), then session 2 with the previous output attached (restores runs/ and data/, skips data prep, resumes from checkpoint-2 → `complete`) | PASS (CPU, 2 + 2 steps) | run interactively, not logged |

Extra findings
- **Truncation trap.** At max_length 512 the Banking77 label prompt alone (563 tokens on ModernBERT, 560 on UltraBERT) does not fit, and the published model scored 0.0005 macro-F1. At 1024 it scored 0.26 (n=100; card 0.356 on the full set). eval.py now warns when the label prompt exceeds max_length. Token counts of the label prompt on the UltraBERT tokenizer: Banking77 560, CLINC-150 667, MASSIVE 311, so stage 1's 1024 fits all of them.
- **Baseline we measured** (full typed-decisions test, 2,000 decisions): gliclass-modern-base-v3.0 accuracy 0.391, KL 1.907, Brier 0.544, ECE 0.413. That is below the dataset's input-blind Prior (0.470). Order sensitivity on Banking77: 34% of predictions flip under label shuffling.
- **Gradient scale.** At step 1 the UltraBERT grad-norm is about 6 vs 0.56 for ModernBERT on the same batch. With max_grad_norm 1.0 most early UltraBERT steps are clipped. This fits the card's "use up to 10x lower LR" note; watch grad_norm in stage 0/1.
- Mean tokenized length of the stage-0 mix: 160 tokens (p95 about 570), so most batches pad far below 1024.

Not tested here (needs the real hardware): fp16 numerics on T4, 2-GPU NCCL, real throughput and memory, TPU/XLA, flash-attention-2.

## v2: resume-hang tests (22 Oct 2026, 20:00–20:35 IST)
Setup:
- 2-process **gloo** DDP on CPU, driven by the real `kaggle_stage1.ipynb`, with paths remapped:
  `/kaggle/input` → `/tmp/kin` (made read-only with `chmod -R a-w`) and `/kaggle/working` → `/tmp/kw`.
- `/tmp/kin` holds the code dataset at `datasets/srimondanguria/ultragliclass-code` and the real S0a output
  (checkpoint-75: fp16 and world size 2).
- Overrides to fit the box's RAM: batch 2, accumulation 4, max_length 256, budget 0.05 h.

Logs are in `smoke_logs/v2_resume/`.

| test | result |
|---|---|
| old code, read-only restore | **crash, not a hang**: PermissionError on STATUS.json (rank 0), gloo "connection closed" (rank 1), rc 1 |
| old code, writable | **crash**: `_load_scaler` AttributeError (fp16 scaler.pt, no scaler on CPU) |
| simA / simA2: v2 normal resume | auto-found code and prev output; restored 1.72 GB in 3 s; DIST_CHECK_OK; per-rank model/optimizer/RNG load traced; scaler skipped; first step 75→76 in 7 s; budget stop at 101, `needs_resume`, rc 0. Same train_loss (0.20207) with and without the explicit PG init |
| simB: v1-style PG init, rank 1 stuck, `ddp_timeout=60` | **timeout did not fire**: rank 0 waited >3 min at the barrier. The watchdog killed it at 3 min, then the 1-GPU fallback resumed 75→96, rc 0 |
| direct: v2 explicit PG init, rank 1 stuck, timeout 30 s | rank 0 raised `Timed out waiting 30000ms` after 30 s, torchrun rc 1. This also resumed a world-size-1 checkpoint on 2 ranks |
| simC: v2 notebook, rank 1 stuck, `ddp_timeout=60` | 2-GPU run failed after 70 s (EXIT 1, timeout). The notebook resumed on 1 process with accumulation 8, 75→94, rc 0 |

## v3: stage-2 code paths (23 Oct 2026, CPU)
- `python scripts/test_stage2.py`: ALL PASS. Covers: consistency loss is 0 under a pure reorder (skl, logit_mse) and > 0 when scores differ;
  gradients flow and are zero on padding; fp16 ok; stored rows never mutated; 92% of taxonomy rows reach >= 20 labels (mean 36 at
  max_length 512); the label prompt never crowds out the text (>= 100 tokens after SEP on long texts); distinct draws per fetch;
  no near-duplicate-of-gold labels; 2 DataLoader workers yield pairs and draw different expansions; `pair_perm` alignment.
- `python prepare_data.py --recipe configs/data_stage2_smoke.yaml` -> `data/stage2_smoke/` (about 12.3k rows, label pools: intents 68,
  legal 100, dbpedia 206, fin-topics 20, emotions 28, glic-v2 40).
- `python train.py --config configs/stage2_smoke.yaml` (max_length 512, 6 steps, pair_prob 0.5): completed. `cons_loss` about 0.007-0.011,
  eval_loss 0.34, checkpoint-6 and final_model written.
- The first attempt hung because of the `UGC_DEBUG_HANG_RANK` `None == None` bug (fixed, see CHANGELOG v3).
Not tested: GPU/fp16 numerics of the new loss on T4, 2-GPU DDP with pairs (the pair is stacked into one forward so it is DDP-safe by construction).

## v3.1: stage-2 PILOT path (23 Oct 2026, CPU, no Kaggle)
Logs are in `smoke_logs/pilot/`. Tools: `scripts/make_pilot_smoke_recipe.py` (tiny recipe), `scripts/run_notebook_cpu.py` (runs the real
`kaggle_stage2_pilot.ipynb` under nbclient with `/kaggle/input` -> `/tmp/kin`, `/kaggle/working` -> `/tmp/kw`), `scripts/check_pilot_arms.py`.

| test | result |
|---|---|
| `python scripts/test_stage2.py` | ALL PASS, incl. the new section 4 (second view with MORE labels than the primary). The new test **fails on the old code**: `PairedCollator` raised `IndexError` when label augmentation added a label to the second view. That would have crashed pilot C on Kaggle. Fixed (`c` = max over both views). |
| `prepare_data.py --streaming` on the tiny pilot recipe (1/60 of the pilot) | 1,611 train / 185 val rows, labels/row mean 6.2 max 30, pools: HWU64 64, LEDGAR 100, fin-topics 20, GoEmotions 28, glic-v2 33. DBpedia/Bitext disabled and listed in the manifest; the `forbid_ids` guard is active. 157 s. |
| notebook, mock mounts: code dataset at `/tmp/kin/datasets/srimondanguria/ultragliclass-code` (new bundle), S1 kernel output at `/tmp/kin/notebooks/srimondanguria/ultragliclass-s1/ultragliclass/runs/stage1_ultrabert150m/final_model` (the S1 output also holds an OLD code copy, to test the code finder) | `CODE_INPUT` resolved to the dataset (not the S1 old code copy); `INIT_MODEL="auto"` found the S1 final_model, validated it (590 MB, GLiClassModel), copied to `/tmp/ugc_init_model`; read-only mounts OK. |
| notebook, arms A, B, C, 4 steps each (batch 2 x accum 2, max_length 512, 1 process, fp32), then quick eval per arm (`--max_per_dataset 6 --permutations 2 --calib_n 30 --typed_max_cases 4`) | all 3 arms `complete` (step 4/4, eval_loss 0.0216 / 0.0211 / 0.0216), `results/stage2_pilot_{A,B,C}_quick.json`, `stage2_pilot_summary.json`, `stage2_pilot_data_manifest.json` written; the C log shows `cons_loss` and `pairs_since_last_log`; every arm logged `Initialized from GLiClass checkpoint /tmp/ugc_init_model`. Whole notebook 10.6 min CPU. The eval numbers are noise (4 steps, 6 rows/set): avg macro-F1 0.589 for all arms. |
| notebook resume: `PREV_OUTPUT="auto"` pointing at a mock earlier output (arm A interrupted at checkpoint-2, arm B complete) | restored 2.3 GB; data not rebuilt; A resumed from `checkpoint-2` (weights, optimizer, RNG) and finished 4/4; B reported `already complete`, skipped. |
| row order: `UGC_DUMP_ORDER` (new hook in `train.py`), 6 steps, 0 workers, arms A/B/C | the fetched dataset-index sequence is **byte-identical** across the three arms (md5 258362270f99...): same base data order; only expansion/pairing differ. |
| `scripts/check_pilot_arms.py` (100 steps of the smoke data, per-step stats) | step counts equal; B and C shift loss weight to many-label rows (loss-weight share of rows with >= 20 labels: A 44%, B 66%, C 69%); C has +8% forward rows (second views). See `STAGE2_PILOT_JOB_SPEC.md` "Comparability". |
| `scripts/estimate_pilot_cost.py` | A 3.50 s/step (1.46 h), B 3.59 s/step (1.50 h), C 3.67 s/step (1.53 h) for 1,500 steps on T4x2. |
Not tested (cannot be on CPU): fp16/DDP on the T4s with pilot rows (B/C peak memory with 150-label rows at max_length 1024 is unmeasured; S1 peaked at 3.95 GB, so there is headroom),
real-size data prep time for the 96k-row pilot recipe (streaming; the full-recipe time is unknown, spec budgets 0.5 h).
Box note: during the notebook smoke the box venv's `pip` lost its `pip/_internal/operations/build` files (cause unknown, not touched by our code); restored from the pip 25.1.1 wheel. Unrelated to Kaggle.

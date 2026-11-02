# Stage-3 job spec: SMOKE on the current Kaggle account; full run deferred to account #2 (for the Kaggle Handler)

Status, 25 Oct 2026: **prepared on the box, nothing uploaded, no Kaggle GPU used, nothing published.**
Srimon decided (25 Oct): **smoke and proper data prep on the CURRENT account only.** The full stage-3 train **waits for a second Kaggle account** with fresh quota, which Srimon will share later.
**Do not plan or start a full train on this account.** UNVERIFIED marks estimates; everything else was measured on the box.

## 1. What runs
`kaggle_stage3.ipynb` (built by `scripts/build_stage3_notebook.py` from the Kaggle-proven `kaggle_stage2_full.ipynb` cells: watchdog, NCCL pre-flight, restore/resume, data md5). One knob: **`MODE`**.

| MODE | account | what | GPU-h |
|---|---|---|---|
| **`"smoke"`** (the job now) | current, 11.66 h left until Sat 31 Oct 05:30 IST | 1. **REAL data** build (`configs/data_stage3.yaml`, the same recipe the full run uses), with leak asserts. 2. **S2 baseline** on the long-Jev slice at 8,192 → `results/stage3_baseline_s2_longjev_8192.json`. 3. `train.py` on the **REAL config and schedule** (`configs/stage3_longctx.yaml`, 956 steps, `runs/stage3_full`, init = S2 final_model), **capped at 1.0 h of training**. That is a few hundred steps at max_length 8,192; it ends with a checkpoint and `needs_resume`. 4. A long-Jev eval of that checkpoint plus a paired comparison with the baseline. This is an early signal, NOT the go/no-go. | **~1.5 (1.3–1.8) UNVERIFIED**; hard caps: train 1.0 h, session 1.8 h, plus ~0.1 h upload, so ≤ 2 h |
| `"full"` (DEFERRED) | #2 | The same notebook. `PREV_OUTPUT="auto"` restores the smoke output (data md5-checked, checkpoint and optimizer state) and resumes to step 956. Then the final quick eval, the long-Jev eval and the comparison (§6). | ~3.3 h resumed (2.3–5.5) UNVERIFIED; ~4.5 h from scratch |

**Resume-safe:**
* The smoke is the first chunk of the real run: same data md5, same LR schedule, same output dir. The full run continues it.
* This was tested on CPU: smoke → `needs_resume` at step 5 → full mode restored that output, matched the data md5, resumed from `checkpoint-5` and reached `complete` (§9).
* If account #2 cannot see the smoke output, the full run starts fresh with identical data. That costs ~1 h more, and the smoke-only number is lost.

**Init:** `runs/stage2_full_C/final_model` (615 MB) from kernel output **`srimondanguria/ultragliclass-s2full` version 4**.
* The notebook auto-globs `/kaggle/input/**/runs/stage2_full_C/final_model` and copies it to `/tmp/ugc_init_model`.
* Box copy: `/workspace/kaggle/ultragliclass/S2full/session2/output/ultragliclass/runs/stage2_full_C/final_model`.

## 2. Data mix (`configs/data_stage3.yaml`, seed 44; box dry run 25 Oct, 171 s)

| part | rows | notes |
|---|---|---|
| Long Jev | **21,184** train / 300 val | Jev `full` train with states ≥ 6,000 chars, `license_use == commercial`, and no rows whose license contains `-sa`, `by-sa`, `gpl` or `odbl`. Kinds: choice 18,310, noul 2,185, score 1,463 (before dedupe). **774** train rows that share a text with val were dropped. |
| Replay | **40,000** | A seeded sample of S2 `data/stage2_full/train.jsonl`: gliclass-v2.0 25,039, Jev ≤ 4K 10,205, lex_glue 1,344, go_emotions 943, fin-tweets 1,499, HWU64 637, logic 270, LocalLLaMA typed **TRAIN** 63. S2's `label_pools.json` is copied for many-label expansion. |
| **Total** | **61,184** train, 300 val | `data/stage3/train.jsonl` md5 `d35eb55c3fc957d1c951aa6ab57c1028`. The Kaggle rebuild should match unless the HF dataset changes (not asserted). |

* The EST before the dry run was about 63k rows and about 985 steps. The real figure is lower because of text-hash dedupe and the val-state drop.
* **Eval guards:**
  * No eval-set source is used (`forbid_ids`, as in S2), and there are no dedicated procedural sources.
  * Leak checks gave **0** train rows equal to a long-Jev slice text, **0** equal to a val text, and **0** sharing a 3,000-char prefix with the slice. The Jev test/validation long states share 0 states and 0 groups with Jev train (full scan).
  * The typed LocalLLaMA TEST split is never read for training (prepare_data guard). Replay carries 63 of S2's approved TRAIN rows, so typed results are still **partly in-domain**.
* Procedural-style rows are **13.9 %** (8,480: 6,279 from long Jev such as record_eligibility and needle_retrieval, the rest from replay). S2 full was about 4.9 %, and E1 suggested that a high share hurts typed "choice". This is kept on purpose, because those rows are the long-context signal. If typed choice drops, this is the first suspect.
* Long-Jev tokens (labels + prompt + text, UltraBERT tokenizer): mean 3,938, p50 3,137, p90 7,184. **1.5 % > 8,192** (tail truncated) and 37 % > 4,096. Replay rows have a mean of about 200 tokens.

## 3. Train settings (`configs/stage3_longctx.yaml`, base `stage2_pilot_C.yaml`)
* **Lengths and labels:** max_length 8,192, `max_num_classes` 256 (data cap). The saved model `config.json` will still say 50, which is cosmetic: S2 trained with up to 150 labels under the same value.
* **Many-label:** many-label on (`data/stage3/label_pools.json`, k_max 250, glic-v2 pool override as in S2).
* **Consistency:** on (pair_prob 0.25, weight 0.1, symmetric KL). Pairs need ≥ 8 labels, so they land mostly on replay rows.
* **Other data settings:** length_group off. Augmentation: label removal 0.05, label addition 0.05, text addition 0.15, others 0.
* **Batching:** batch 1 × accum 32 × 2 GPUs = **64 rows per step, 956 steps (1 epoch)**. Gradient checkpointing **on**, fp16.
* **Optimizer:** LR encoder 5e-6, others 1.5e-5, warmup 3 %, linear schedule. Seed 42.
* **Saving, eval and logging:** save every 100 steps (keep 2), eval every 200 (val = 300 long Jev + `calib/stage2_pilot_val.jsonl`, max 2,150 rows), log every 10.
* **Runtime guards:** workers 0, `max_oom_skips 0`, `time_budget_hours` set by the notebook.

## 4. Smoke cost and checks (Kaggle GPU-h = session wall hours)

| item | EST | basis |
|---|---|---|
| setup + pip + init copy | 0.05 h | S2 |
| REAL data build (HF download of Jev `full` ~1.95 GB + filter + replay) | 0.1–0.2 h | box: 171 s end to end; the notebook rebuild on the box gave the identical md5 |
| S2 baseline long-Jev eval (431 rows, 8,192, fp16, batch 1, 1 GPU) | 0.05–0.1 h UNVERIFIED | |
| train (cap) | **1.0 h → ~250 steps (150–400)** | ~14 s/step EST (9–24) UNVERIFIED; one in-train eval at step 200 (~2 min, 2,150 rows) |
| checkpoint long-Jev eval + compare | 0.05–0.1 h UNVERIFIED | |
| optional suite quick eval of the checkpoint (`SMOKE_QUICK_EVAL=True`) | +0.12 h | off by default |
| output upload (~1.7 GB checkpoint + 0.4 GB data) | ~0.1 h | |
| **total** | **~1.5 h (1.3–1.8), ≤ 2 h by the caps** | quota after the smoke: ~10 h left on this account, which stays unused |

s/step basis (UNVERIFIED): per GPU per step about 11.1 long rows × 3,904 tokens × 1.45 full-attention factor, plus 21 replay rows ≈ 72k token-equivalents.
* The S2-calibrated rate is about 7.2k token-equivalents/s/GPU, ×0.75 for gradient checkpointing ≈ 5.4k/s, plus batch-1 overhead.
* Memory EST: 5–8 GB of 15 (UNVERIFIED). S2 T0 with batch 8 × 1,024 and checkpointing was 4.8 GB.

**What the smoke must show (Handler reads `results/stage3_smoke_summary_s1.json` → `speed_mem.metric_lines`, `logs/train_s3_*.log`):**
1. **Speed:** median `sec_per_step` after step 10.
   * ≤ 25 → full at 8,192 fits a ~3.5–7 h session on account #2.
   * 25–40 → full at 8,192 is still possible (≤ 10.6 h) but tight; report.
   * \> 40 → the full run should use the 4,096 fallback (about 0.66× cost); report.
2. **Memory:** `peak_reserved_gb` per rank ≤ 14.5 GB with no OOM. If an OOM happens, the notebook resumes **once** with `data.max_length=4096`. Report that as a deviation: the remainder of the smoke ran at 4,096, and the full run would need the same override. A second OOM stops the run.
3. **No NaN:** `STATUS.state` is `needs_resume` (expected) and not `nan_stop`. Train loss is finite, and the eval_loss at step 200 is not above the S2-final-like start by more than 0.05.
4. **Resume-ready:** `runs/stage3_full/checkpoint-<N>` exists and STATUS says `needs_resume`.
5. **Early signal:** `results/stage3_mid_step<N>_longjev_compare.json`. This is informational; a partial run proves nothing yet.

## 5. Stop rules
**In the notebook:**
* **NaN** → `nan_stop`; stop and report.
* **OOM** → one resume at `data.max_length=4096`; a second OOM → stop.
* **Stall:** no output for 15 minutes → kill. The 1-GPU fallback is full mode only.
* **Caps:** the smoke stops training at 1.0 h (checkpoint saved) and the whole session at 1.8 h. Note the in-train save-every-100 too.
* **Data asserts** (stop before training):
  * no eval or procedural-dedicated source;
  * **0** slice texts and **0** val texts in train;
  * long Jev between 20,000 and 26,000 rows, and replay between 39,000 and 40,000;
  * the slice md5 equals `cc664b35683a6703d4a8be15983fd5cb`;
  * in full mode, the restored `data/stage3/train.jsonl` md5 equals the smoke's (`results/stage3_data_md5.json`).

**Handler:** stop and report if the session clock passes 2 h for any reason. Do not start `MODE="full"` on this account.

## 6. Go / no-go eval protocol for the FULL run (fixed now; it runs on account #2)
**A. Long-Jev slice** (`calib/jev_long_test.jsonl`, built by `scripts/build_long_jev_slice.py`, manifest `calib/jev_long_test_manifest.json`):
* Contents: Jev `full` **test (177) + validation (254)**, states ≥ 6,000 chars, all `license_use` values (eval only), with 1 tied-gold row dropped. That gives **431 rows / 347 states / 344 groups**: choice 390, noul 28, score 13; commercial 324, unspecified 83, non-commercial 24.
* Test alone (177) is too small for a 3-pt bar. Validation is used for evaluation only, because stage 3 carves its own val from train.
* **Baseline:** S2 final_model, `python scripts/eval_long_jev.py --model <S2 final_model> --slice calib/jev_long_test.jsonl --max_length 8192 --batch_size 1 --dtype fp16 --out results/stage3_baseline_s2_longjev_8192.json`. The notebook runs this before training.
  * S2 was trained at 1,024 tokens, but the baseline is evaluated at 8,192, the same as the candidate. The 32K RoPE makes this valid input.
  * Also report S2 at 1,024 as a reference if there is time. It is optional and not part of the bar.
* **Candidate:** the same command on `runs/stage3_full/final_model` → `results/stage3_final_longjev_8192.json`.
* **Compare:** `python scripts/eval_long_jev.py --compare BASE CAND`. This gives the paired delta, a cluster bootstrap 95 % CI by group, exact McNemar and wins/losses. **Bar 1 met = point delta ≥ +0.030.** Also report whether the CI excludes 0 (with n = 431 the CI half-width is about ±3–4 pts, UNVERIFIED), plus breakdowns by kind, by length bucket and by truncation.
* Metric: argmax accuracy against the gold argmax, with 1 label order by default. `--permutations 3` is optional for a flip rate.
* **CPU check on the box (25 Oct):** S2 final, 4 longest rows at max_length 2,048, ran end to end (7 s). This verifies the code path only; the number is meaningless.

**B. Suite + CLINC:**
* Run the same quick protocol as S2: `eval.py --suites gliclass intent typed --max_per_dataset 500 --permutations 3 --calib_file calib/stage1_val.jsonl --calib_n 2000 --max_length 1024` → `results/stage3_final_quick.json`.
* Compare with S2's `results/stage2_full_C_final_quick.json` (avg F1 0.6829, CLINC-150 acc 0.650 with macro-F1 0.604, Banking77 0.554, typed T1 0.6155).
* **Bar 2:** CLINC-150 accuracy ≥ 0.650. Note that 500 rows give an SE of about ±2.1 pts, so a tie within noise should be reported as such.
* **Bar 3:** avg macro-F1 ≥ 0.673.
* Typed results are reported but are not a bar (partly in-domain).

**GO = bars 1–3 all met.** Single run, single seed; seed noise is unmeasured.

## 7. What to send back after the smoke
* `results/stage3_smoke_summary_s1.json`, which includes `speed_mem` (sec_per_step and peak_reserved_gb lines), STATUS, the OOM and 4096 flags, the baseline, and `longjev_mid` with its compare
* `results/stage3_baseline_s2_longjev_8192.json`
* `results/stage3_mid_step*_longjev_8192.json` and `_compare.json`
* `results/stage3_data_manifest.json`, `_data_check.json` and `_data_md5.json`
* `runs/stage3_full/STATUS.json` and `HEARTBEAT.json`
* the last 40 lines of `logs/train_s3_*.log`
* the full `logs/notebook_*.log`
* data-prep minutes, session hours and the kernel version slug

**Keep this smoke output version.** The full run on account #2 resumes from it. No publishing and no sharing without Srimon.

## 8. Files the Handler needs
**Code dataset `srimondanguria/ultragliclass-code`, NEW VERSION 8.**
* Build it with `python scripts/make_code_bundle.py /workspace/ultrabert/ultragliclass_code_v8`, run from the project root. That gives the directory plus a `.zip`. It contains everything except runs/, data/, results/, smoke_logs/. It is not built yet; the Handler or parent builds it when the upload is approved.
* New or changed files in v8 compared with v7:
  * `kaggle_stage3.ipynb`
  * `configs/stage3_longctx.yaml`, `configs/data_stage3.yaml`
  * `prepare_data.py` (min_state_chars pre-filter, `exclude_license_contains`, `format: jsonl` replay via `take_jsonl`, `pools_file_from`)
  * `calib/jev_long_test.jsonl` and `calib/jev_long_test_manifest.json`
  * `scripts/eval_long_jev.py`, `scripts/build_long_jev_slice.py`, `scripts/build_stage3_notebook.py`, `scripts/run_stage3_notebook_cpu.py`
  * `STAGE3_PLAN.md`, this spec, CHANGELOG v3.6

**Attach (smoke):**
1. code v8;
2. kernel output **`srimondanguria/ultragliclass-s2full` v4**. It provides the init model and `data/stage2_full/{train.jsonl,label_pools.json}` for replay; the notebook symlinks it read-only and removes the link before saving.

**Kernel settings:** GPU **T4 x2**, Internet **on**, persistence off. Knobs to edit: none (`MODE="smoke"` is the default).

**Attach (full, account #2, later):**
* code v8, s2full v4 and the smoke output, all visible to account #2. Set `MODE="full"`.
* **Blocker:** the account #2 handover. These are private Kaggle artifacts of the current account, so account #2 needs access to them. That means sharing or re-uploading them, which is a sharing action and needs Srimon's explicit OK.

## 9. Verification on the box (CPU, 25 Oct)
* `prepare_data.py --recipe configs/data_stage3.yaml` ran end to end (row counts in §2; leak checks 0).
* A `train.py` 3-step CPU run on a 160-row stage-3 subset (max_length 1,024, batch 1 × accum 2) worked: S2 init loaded, the 5 S2 label pools loaded, consistency on, eval and save ran, and the state was `complete`.
* `eval_long_jev.py` on S2 final (4 rows, 2,048) ran.
* **Notebook CPU runs** (`scripts/run_stage3_notebook_cpu.py`, mock `/kaggle/input` with the bundled code and the S2 v4 output; max_length 512, accum 2):
  * **(a) MODE smoke:**
    * The REAL data build in the notebook gave 61,184 rows, md5 `d35eb55c…`, identical to the box dry run.
    * Leak asserts were 0.
    * The baseline was written.
    * train.py hit the tiny time cap at step 5 (`needs_resume`, `checkpoint-5` saved).
    * The checkpoint long-Jev eval and compare ran.
    * The S2 data symlink was removed before saving.
  * **(b) MODE full**, with (a)'s output attached:
    * The restore ran (1.72 GB checkpoint plus data) and the data md5 matched.
    * The baseline was reused.
    * Training resumed from `checkpoint-5` (optimizer and scheduler state loaded) and reached `complete` at step 7.
    * The final quick eval, the long-Jev eval and the paired compare ran.
  * The CPU accuracies at 512 tokens are meaningless (they only exercise the code path).
* **Not verifiable on CPU:** s/step, memory at 8,192, SDPA kernel choice on T4, and 2-GPU DDP at batch 1 with checkpointing on. The Kaggle smoke covers these.

## 10. Risks
* **License:**
  * New long-Jev rows are commercial only, with no share-alike or copyleft rows (about 1.8k removed). Their licenses are mainly apache, cc-by-4.0 and mit (see the manifest).
  * **Replay inherits the S2 caveats:** S2's mix contains Jev CC-BY-SA rows and the approved typed TRAIN rows. So the stage-3 model is not free of the CC-BY-SA caveat. The Jev commercial filter covers the new rows only.
  * The eval slice includes non-commercial and unspecified rows. They are used for evaluation only, never for training.
* **Data counts:**
  * The rows are 61,184 compared with the 63k EST (−3 %). If the HF dataset changes, the Kaggle rebuild can differ. The asserts allow 20–26k long rows, and the md5 is recorded in session 1.
  * Procedural share is 13.9 % (§2).
  * The long-Jev source mix is dominated by procedural (6.3k), HelpSteer3 (2.8k), tab (2.8k), quality (2.7k) and cuad (2.4k). The slice mix is similar.
* **Statistical:** 431 rows and 344 groups. A +3-pt bar is near the noise floor of a single run. Report the CI and McNemar, not just the point delta.
* **Cost:** s/step and memory are UNVERIFIED until the smoke. The 4,096 fallback changes the experiment.
* **Account #2 handover:** the full run needs code v8, the s2full v4 output and the smoke output visible to account #2. That is a sharing or re-upload step and needs Srimon's OK. Without the smoke output, the full run starts fresh (about 1 h more).
* **Smoke truncation:** a 1-h cap means about 250 of 956 steps (UNVERIFIED). The checkpoint long-Jev number is mid-schedule, at a high LR, so do not read it as the result.

# Stage-2 typed-decisions ABLATION job spec (for the Kaggle Handler)

**Status: PREPARED ONLY. Nothing has been uploaded or run on Kaggle, no GPU spent, nothing published or pushed.** Built and smoke-tested on CPU (see section 9).
Uploading the new code-dataset version and starting the kernel are Kaggle actions: they need Srimon's go-ahead (and the Handler's usual approval step).
**E2 and E3 additionally need Srimon's separate decision D10 (section 3)** and stay blocked in the notebook until `SEMANTIC_DATA_APPROVED = True`.

## 1. Question and design
Stage-2 pilot arms lost typed-decisions accuracy vs S1 (0.503 -> A 0.456 / B 0.440 / C 0.472): `choice` -6 to -7 pts in all arms, and the
security_incidents / invoice_processing workflows (notes/S2_PILOT_INVESTIGATION.md, Steps 4 and 6). Hypothesis (**unproven**): the procedural typed rows
(non-semantic option strings, long key:value states) hurt semantic `choice`. Every arm below = **pilot arm C** (many-label + consistency, seed 42, 1,500 x 64 rows,
init = S1 `stage1_ultrabert150m/final_model`, same LR schedule); only the training data mix differs. All arms: 96,035 train rows, evaluated on the **same val file**
`calib/stage2_pilot_val.jsonl` (the pilot's actual val set), so `eval_loss` is comparable with the pilot arms.

| arm | data change vs pilot C | procedural-style share* | config / recipe | needs approval |
|---|---|---|---|---|
| **E1** | dedicated `procedural-typed-decisions` rows (11,750) removed, replaced by gliclass-v2.0 rows (47,100 -> 58,850) | 4.9 % | `configs/stage2_abl_E1.yaml` / `data_stage2_abl_E1.yaml` | no |
| **E1z** | E1, and the procedural-style rows inside the Jev source also excluded (Jev still 23,500 rows, drawn from the non-procedural rest) | 0 % | `..._E1z` | no |
| **E2** | E1 + 3,000 semantic prose-option rows (LocalLLaMA/typed-decisions **train**, customer_service + agent_trace_observability); gliclass 47,100 -> 55,850 | 4.9 % | `..._E2` | **D10** |
| **E3** | pilot C + the same 3,000 semantic rows (gliclass 47,100 -> 44,100; procedural kept) | 17.2 % | `..._E3` | **D10** |
| C0 | pilot C unchanged (reproducibility / noise rerun, **optional**) | 17.2 % | `configs/stage2_abl_C0.yaml` (pilot recipe) | no |

\*Share of train rows whose source tag contains `procedural-typed-decisions`. **Finding while building this:** the Jev source itself carries procedural-style rows
(4,728 of its 23,500 pilot rows, 20.1 %), so S1 already saw about 20k of them (~4.9 % of its 410k rows). The pilot added the dedicated dataset on top (11,750), taking the share to
**17.2 %** (16,478 rows), about 3.5x S1's. So E1 = "back to ~S1's procedural share", E1z = "none at all" (dose-response). The reference "with procedural" is the **existing pilot arm C**
(same code, config, seed and init; its Kaggle result is already on the box). C0 only measures run-to-run noise: skip it unless E1/E1z land in the grey zone of section 6.

## 2. Priority and sessions (GPU-hours: see section 4)
1. **E1** (answers the hypothesis; ~1.8 h).  2. **E2** (only if D10 approved) or **E1z** (if not).  3. the other of the two.  4. E3, C0 (optional).
Never more than 6.0 GPU-h per session (`TOTAL_MAX_HOURS`, `ARM_MAX_HOURS = 2.0` per arm; train.py saves and exits `needs_resume` at the arm cap).

| plan | `ARMS` | est. session hours (range) |
|---|---|---|
| **P1** | `["E1"]` | **~1.8** (1.5-2.1) |
| **P2 (recommended after D10)** | `["E1", "E2"]` | **~3.3** (2.9-3.9) |
| P2' (D10 declined) | `["E1", "E1z"]` | ~3.2 (2.8-3.7) |
| P3 | `["E1", "E1z", "E2"]` | ~4.7 (4.1-5.4) |
| P4 (everything) | two sessions: `["E1","E1z","E2"]` then `["E3","C0"]` | ~8.5 total (4.7 + 3.8) |

## 3. Attach list and decisions
1. **Dataset `srimondanguria/ultragliclass-code`, NEW VERSION (v4)**: build `python scripts/make_code_bundle.py <build-box>/ultragliclass_code_v4` (done: `<build-box>/ultragliclass_code_v4/` + `.zip`, ~2.3 MB; upload the directory as the dataset root).
   It contains the new configs, `prepare_data.py` (new keys `answers_col`, `exclude_source_contains`, and a **guard that aborts if a LocalLLaMA/typed-decisions source uses the test split or a val_split**),
   `eval.py` (typed eval now also reports `accuracy_by_workflow`), `kaggle_stage2_ablation.ipynb`, `calib/stage2_pilot_val.jsonl`. The v3 code dataset does NOT have these.
2. **Kernel output `srimondanguria/ultragliclass-s1`** (S1 final_model), exactly as for the pilot. Do not attach the pilot output (not needed; `INIT_MODEL="auto"` finds S1).
3. Resume only: the previous version of THIS kernel's output with `PREV_OUTPUT = "auto"`.
Notebook to push: **`kaggle_stage2_ablation.ipynb`**, GPU **T4 x2**, Internet **on**, persistence off. Cell 1 defaults: `ARMS = ["E1"]`, `SEMANTIC_DATA_APPROVED = False`
(E2/E3 assert-fail while it is False; set it to True ONLY with Srimon's D10 approval). Other knobs as in the pilot spec (watchdog 15 min, NCCL pre-flight, 1-GPU fallback, NaN/OOM stops).

**D10 (Srimon to decide) - training on the `LocalLLaMA/typed-decisions` train split.** Facts, verified 24 Oct 2026:
- **Licence: apache-2.0** (dataset card, synthetic data). Usable for training, including commercially. **It is not off limits.**
- **Train split is separate from test**: 1,200 cases (300 per workflow, `all` config = 6,000 decisions: choice 1,800 / noul 1,800 / score 2,400), "from a separate run at a different seed; packaging refuses to build if any case id or state appears in both splits" (card). I checked: **0 exact state matches** between train and our 400 eval states. The recipe reads **only `split: train`** and the code aborts otherwise; the test split is never used for training.
- **But it is in-schema for our eval**: the train split has the same 4 workflows, the same question schemas and the **same 57 option strings** as the test split we evaluate on. Training on it turns our typed-decisions number into a *fitted-on-train specialist* score. The benchmark itself keeps such models in a separate table
  ("Fitted or fine-tuned on `train`"), not comparable with zero-shot rows. Our earlier project rule was "we do not train on it". Hence the ablation uses only 2 of 4 workflows (customer_service, agent_trace_observability; 600 cases -> 3,000 rows = 3.1 %), and reports invoice_processing + security_incidents (**unseen workflows**, where the drop was largest) separately. Gains on the unseen workflows are the evidence of generalisation; gains on the two seen workflows are in-schema fitting and must NOT be quoted as zero-shot.
- No other licensed semantic-prose-option typed source was found; Jev/procedural (options like entailment / S1..S5) are the opposite of prose. (Not searched exhaustively: UNVERIFIED that none exists.)
- Any E2/E3-trained model must be labelled "partly fitted on LocalLLaMA/typed-decisions train" in every report; E2/E3 are diagnostics and not release candidates unless Srimon decides otherwise.

## 4. GPU-hour estimate (EST; Kaggle GPU-hours = session wall hours)
Base: **measured pilot arm C = 4.008 s/step** (STATUS.json; A 3.669, B 3.722, same session; peak 4.27 GB; train 1.67 h, wall incl. quick eval and save 1.79 h). Per-arm step cost is scaled from C by a
token-cost simulation on each arm's real built `train.jsonl` (`scripts/estimate_ablation_cost.py`, many-label + pair_prob 0.25): procedural rows are long (mean ~1,500-1,700 chars), so removing them makes steps cheaper.
Uncertainty about +/-15 %; the same method estimated pilot C at 3.68 s/step vs 4.008 measured (+9 %), so I scaled to the measurement.

| arm | rel. step cost vs C | s/step | train h (1,500 steps) | arm wall h (train + ~0.12 quick eval/save) |
|---|---|---|---|---|
| E1 | 0.857 | 3.44 | 1.43 | ~1.55 |
| E1z | 0.794 | 3.18 | 1.33 | ~1.45 |
| E2 | 0.840 | 3.37 | 1.40 | ~1.52 |
| E3 | 0.977 | 3.92 | 1.63 | ~1.75 |
| C0 | 1.000 | 4.01 | 1.67 | ~1.79 |
Session setup (pip, S1 copy, data prep ~30 s per arm on CPU here; Kaggle streaming slower; pilot overhead was small) budgeted at 0.2 h. Totals in section 2. Quota: 11.5/30 h used before the weekly reset (5:30 AM IST Oct 24); P2 uses about 3.3 h of the new week.
Output per arm: final_model 0.6 GB + manifest/quick JSON; checkpoints of completed arms are deleted (`DROP_CKPT_AFTER_ARM`).

## 5. Stop conditions (as the pilot, all implemented in the notebook)
NaN -> `nan_stop`, stop and send logs. OOM (peak ~4.3 GB so unlikely; semantic rows are short) -> do not retry blindly; report. Arm > 2 h -> `needs_resume`, report, Srimon decides on resuming.
Total session clock > 6 h -> no new arm. Data cell **asserts** per arm: 96,035 rows, no DBpedia/Bitext/eval-set source, procedural share as designed (E1 ~4.9 %, E1z 0, E2 ~4.9 %, E3/C0 ~17.2 %), 3,000 semantic rows exactly in E2/E3 and none elsewhere.

## 6. How to read the result (decision rules, set before running)
Reference (same quick eval, 500/set, 3 orders): **S1** typed 0.503 / choice 0.485 / noul 0.645 / score 0.410, avg F1 0.586. **Pilot C** typed 0.472 / choice 0.412 / noul 0.650 / score 0.384, avg F1 0.660, many-label flips 0.258/0.248/0.250/0.258.
Noise: single runs; case-clustered SE of a paired accuracy difference is ~2.3 pts for `choice` (n=600) and ~1.7 for overall; GPU/seed run-to-run noise is UNMEASURED (that is what C0 is for).
- **Hypothesis supported** if E1 *or* E1z reaches `choice` >= 0.455 (+4.3 pts over C, about 2 SE) **and** typed overall >= 0.490, **without regression**: avg F1 >= 0.650 (C minus 0.01) and all four many-label flip rates <= 0.30.
  Then the full stage 2 mix should cut the dedicated procedural share (E1-style); E1z vs E1 says whether to go to zero.
- **Hypothesis not supported** if both E1 and E1z have `choice` <= 0.435 (< +2.3 pts over C): the cause is elsewhere (candidates: the 13.6 % taxonomy rows, the LR re-warm / 1,500-step drift from S1, Jev sampling): plan a different ablation before any full run.
- **Grey zone** (0.435-0.455): run C0 for the noise level and/or E1z before concluding.
- **E2 vs E1** (D10 only): semantic rows "help" only if the **unseen workflows** (invoice_processing, security_incidents) improve by >= 3 pts over E1; seen-workflow gains are in-schema and prove nothing about generalisation. Also check noul/score did not fall.
- Always report the per-workflow table (`typed_by_workflow` in each `stage2_abl_<arm>_quick.json`), not only the overall accuracy. A and S1 per-workflow reference: security_incidents S1 0.638 -> A 0.494; invoice S1 0.474 -> A 0.406; customer_service 0.490 -> 0.530 (notes Step 6; pilot C/B per-workflow was not finished).

## 7. What to send back (after every session, even a failed one)
- `results/stage2_abl_summary.json` and `results/stage2_abl_<arm>_quick.json` (the deliverable; contains typed accuracy by type AND by workflow, KL, Brier, ECE)
- `results/stage2_abl_<arm>_data_manifest.json`, `runs/stage2_abl_<arm>/STATUS.json` + `HEARTBEAT.json`
- last 20 lines of each `logs/train_*.log` (`sec_per_step`, `peak_mem_gb`, `eval_loss`), the full `logs/notebook_*.log`, measured s/step per arm (to correct section 4), session hours, any NaN/OOM/stall
- kernel version slug (models stay in the kernel output; the final_models are then pulled to the shared box for CPU per-item analysis as in the pilot, `scripts/peritem_cpu.py`)
No other Kaggle action, no publishing, no sharing.

## 8. Files added/changed (all local)
`configs/stage2_abl_{E1,E1z,E2,E3,C0}.yaml`, `configs/data_stage2_abl_{E1,E1z,E2,E3}.yaml` (generated by `scripts/make_ablation_recipes.py`), `calib/stage2_pilot_val.jsonl`,
`scripts/{make_ablation_recipes,check_ablation_data,estimate_ablation_cost,build_ablation_notebook}.py`, `kaggle_stage2_ablation.ipynb`, `results/stage2_ablation_cost_estimate.json`,
`prepare_data.py` (answers_col, exclude_source_contains, typed-decisions test-split guard), `eval.py` (accuracy_by_workflow). Pilot configs/notebook untouched.

## 9. Verification done on CPU (no GPU)
- `scripts/check_ablation_data.py` (output in `notes/ablation_data_check.txt`): all four recipes built from the real sources on CPU: 96,035 train rows each; val rows equal the pilot's val; procedural share E1/E2 4.9 %, E1z 0 %, E3 17.2 %; semantic rows 3,000 in E2/E3 only; 0 exact matches between any training state and the 400 typed-decisions eval states.
- Rebuild fidelity: my CPU rebuild of the pilot mix is row-identical (same rows, same order) to the actual Kaggle pilot data; only ~1.3 % of rows differ in label-list order/expansion.
- Config diff: each E arm differs from pilot C only in train_files / val_files / pools_file / output_dir (checked programmatically).
- Notebook CPU smoke with mock Kaggle mounts: see CHANGELOG v3.2 (result recorded there).
Not testable on CPU: GPU/fp16 numerics, real Kaggle data-prep time, DDP.

# Stage-2 PILOT job spec (for the Kaggle Handler)

**Status: nothing has been uploaded or run on Kaggle.** Built and smoke-tested on CPU only (SMOKE_TEST.md v3.1).
Approved by Srimon: run the three stage-2 pilots; **DBpedia and Bitext are excluded** (licence decision D2).
Uploading the new code-dataset version and starting the kernel are still Kaggle actions: they need the Handler to do them under the usual approval.

## 1. What the pilots are
Same 1,500 steps x 64 rows, same data order, same seed and LR schedule, starting from the **stage-1 model** (`stage1_ultrabert150m/final_model`, 0.586 avg macro-F1 on the quick eval).

| arm | adds to "S1 continued" | config | est. s/step | est. train time |
|---|---|---|---|---|
| A | nothing (control) | `configs/stage2_pilot_A.yaml` | 3.50 | 1.46 h |
| B | many-label expansion (K 20-150, hard negatives) | `configs/stage2_pilot_B.yaml` | 3.59 | 1.50 h |
| C | B + permutation-consistency term (paired label orders, symmetric KL, pair_prob 0.25, weight 0.1) | `configs/stage2_pilot_C.yaml` | 3.68 | 1.53 h |

B inherits A and C inherits B; the only config differences are `many_label.enabled`, `consistency.enabled`, `output_dir`.
Estimates: `scripts/estimate_pilot_cost.py`, calibrated on the **measured S1 3.0 s/step** (T4 x2, fp16, grad-checkpointing). For comparison, the same simulation gives
3.67 s/step for the natural rows alone (more jev/procedural than S1, longer texts) and **3.50 s/step with the taxonomy rows (13.6%)** added: taxonomy rows are mostly short
(HWU64, tweets, GoEmotions) so they do not raise the cost; LEDGAR is the long one. Expansion adds about 0.09 s/step, pairing about 0.09 s/step more. Uncertainty about +/-15%.

## 2. Attach list (Kaggle notebook inputs)
1. **Dataset `srimondanguria/ultragliclass-code`, NEW VERSION (v3).** The existing version is stage-1 code and does NOT contain the pilot configs (the notebook refuses it).
   Build: `python scripts/make_code_bundle.py` -> `<build-box>/ultragliclass_code_v3/` (upload this directory as the dataset root; it already has `dataset-metadata.json`)
   and the same files as `<build-box>/ultragliclass_code_v3.zip` (about 1.5 MB, 71 files).
   File list = everything in `ultragliclass/` EXCEPT `runs/ data/ results/ smoke_logs/ __pycache__` (and `*.pyc`, `*.zip`): `train.py eval.py prepare_data.py requirements.txt`, `ugc/*.py`,
   `configs/*.yaml` (incl. `data_stage2_pilot.yaml`, `stage2_pilot_{A,B,C}.yaml`), `scripts/*`, `calib/stage1_val.jsonl` (4,300 rows, the S1 val set), `kaggle_stage1.ipynb`, `kaggle_stage2_pilot.ipynb`, docs and `notes/`.
2. **Kernel output `srimondanguria/ultragliclass-s1`** (the finished S1 run; it contains `ultragliclass/runs/stage1_ultrabert150m/final_model`, 590 MB). It also holds an old copy of the stage-1 code; the notebook ignores it.
3. Only for a RESUME session: the previous version of THIS pilot kernel's output (in addition to 1 and 2).
Notebook to push: **`kaggle_stage2_pilot.ipynb`**, accelerator **GPU T4 x2**, Internet **on**, persistence off, "Save & Run All".

## 3. Cell-1 settings
Default (one session, all three arms; recommended):
```python
ARMS = ["A", "B", "C"]; INIT_MODEL = "auto"; PREV_OUTPUT = None
DATA_RECIPE = "configs/data_stage2_pilot.yaml"; DATA_DIR = "data/stage2_pilot"; PREP_ARGS = ""
ARM_MAX_HOURS = 2.0; TOTAL_MAX_HOURS = 6.0; QUICK_EVAL = True    # eval: 500/set, 3 permutations, calib calib/stage1_val.jsonl
```
`INIT_MODEL="auto"` globs `/kaggle/input/**/runs/stage1_ultrabert150m/final_model/config.json`; it also accepts the exact `final_model` directory or the kernel-output root
(the path of the S1 kernel output mount, whatever Kaggle names it). It validates the model, copies it to `/tmp/ugc_init_model` and passes `model.init_from=...` to every arm.

| session | ARMS | PREV_OUTPUT | notes |
|---|---|---|---|
| P1 (default) | `["A","B","C"]` | `None` | data prep, then A, B, C in order; quick eval after each |
| P1 split (if one session is judged too long) | `["A","B"]` then `["C"]` | `None`, then `"auto"` | the second session restores data + earlier arms' final models from the attached P1 output |
| resume after a stop | the unfinished arm and the rest | `"auto"` | resumes from the newest checkpoint; completed arms are skipped (not re-run, not re-evaluated) |
| single arm | e.g. `["B"]` | `None` | for a re-run; data is rebuilt (about the prep time below) |

The arms run strictly in sequence; the notebook stops at the first arm that does not finish `complete`.

## 4. Expected GPU-hours (EST; Kaggle GPU-hours = session wall hours)
| part | hours |
|---|---|
| setup, pip, data prep (streaming ~96k rows; prep time of the full recipe is unmeasured; budget) | 0.3-0.6 |
| arm A: train 1.46 + quick eval 0.11 (S1 took 6.5 min) + load/save 0.03 | about 1.6 |
| arm B | about 1.65 |
| arm C | about 1.7 |
| **total** | **about 5.3 h (range 4.6-6.2)**. Weekly quota use is about 5-6 of 30 h |
Output size: 3 final models (0.6 GB each) + data (about 0.2 GB) + at most 1.8 GB of checkpoints of an unfinished arm; under 6 GB.

## 5. Stop conditions (all implemented in the notebook; the Handler also enforces them)
- **NaN**: STATUS `nan_stop` -> the notebook stops. Send logs; do not retry.
- **OOM**: B and C use up to 150 labels at max_length 1024 (peak memory unmeasured; S1 peaked at 3.95 GB, T4 has 15 GB). The notebook does NOT fall back to 1 GPU on OOM. Rerun that arm with
  `ARM_SET = {"B": ["train.per_device_train_batch_size=4", "train.gradient_accumulation_steps=8"]}` (same effective batch 64) and say so in the report.
- **> 2 h in one arm** (`ARM_MAX_HOURS`): train.py saves and exits `needs_resume`; the notebook stops. Report; the decision to resume is Srimon's.
- **Total session clock > 6 GPU-h** (`TOTAL_MAX_HOURS`): no new arm starts after 5.6 h. Never let a session run past 6 h without asking.
- No output for 15 min (watchdog), NCCL pre-flight failure -> 1-GPU fallback (v2 behaviour; slower, about 2x per step: tell us and check the 2 h cap).
- Data prep failing or producing fewer than max_steps x 64 rows, or any dbpedia/bitext/eval source contributing rows: the DATA cell asserts and stops.

## 6. What to send back (after every session, even a failed one)
- `results/stage2_pilot_summary.json` and `results/stage2_pilot_{A,B,C}_quick.json` (the eval output; this is the deliverable)
- `results/stage2_pilot_data_manifest.json` (row counts, per-source licences, label pools)
- `runs/stage2_pilot_{A,B,C}/STATUS.json` and `HEARTBEAT.json`
- the last 20 lines of each `logs/train_*.log` (`sec_per_step`, `peak_mem_gb`, `loss`, `cons_loss`, `eval_loss`) and the full `logs/notebook_*.log`
- measured s/step and peak memory per arm (so the full-run estimate in STAGE2_PLAN.md can be corrected), total session hours, any NaN/OOM/stall
- the kernel version slug of each run (so the models can be attached later). Do not copy the models anywhere else.
No other Kaggle action, no publishing, no sharing.

## 7. Pilot data (`configs/data_stage2_pilot.yaml`; 96,035 train rows, 1,850 val; same for all arms; seed 43)
Natural : taxonomy = 86.4 : 13.6, the same proportions as `data_stage2.yaml` with DBpedia and Bitext removed (their weight went to LEDGAR, GoEmotions, fin-topics, fin-sentiment).
| source | train rows | licence recorded | note |
|---|---|---|---|
| knowledgator/gliclass-v2.0 | 47,100 | apache-2.0 (hub tag) | upstream sources undocumented |
| knowledgator/gliclass-v3-logic-dataset | 205 x3 | apache-2.0 | |
| tasksource-jev-typed-decisions | 23,500 | `license_use: commercial` bucket | the bucket includes CC-BY-SA text: flag |
| procedural-typed-decisions | 11,750 | apache-2.0 | |
| DeepPavlov/hwu64 (intent pool, 64 labels) | 600 x3 | CC-BY-4.0 | |
| lex_glue/ledgar (100 labels) | 4,160 | CC-BY-4.0 | |
| zeroshot/twitter-financial-news-topic (20 labels) | 1,180 x2 | MIT | tweet platform terms: open decision D3 |
| go_emotions (28 labels) | 2,770 | Apache-2.0 | Reddit comments: D3 |
| zeroshot/twitter-financial-news-sentiment | 660 x3 (3 label schemes) | MIT | D3 |
Disabled in the recipe with reasons: DBpedia and Bitext (D2), banking77 and CLINC (D1: eval sets). `forbid_ids` makes `prepare_data.py` abort if banking77/clinc/massive/snips/financial_phrasebank/sst5/etc. is ever enabled.
**K range possible:** HWU64 has only 64 labels, so intent-type rows reach K=20-64 (banking77's 77 and CLINC's 150 cannot be matched). LEDGAR gives K up to 100 (the maximum for taxonomy rows); GoEmotions 28; fin-topics 20 (fixed).
gliclass-v2.0 rows get a mild K=12-60. Only about 20% of forward rows have >= 20 labels in B/C, so the pilot is low-powered for the many-label sets (see section 8).

## 8. Reading the result: comparability and the gate
- The three arms see the **same row order** (verified: identical fetched-index sequence). They differ in loss weighting: the GLiClass loss is a per-micro-batch average over label slots, so B and C move loss weight to many-label rows
  (share of loss weight on rows with >= 20 labels: A 44%, B 66%, C 69%, smoke data), and C trains on about 8% more forward rows (second views). **Logged train loss is therefore NOT comparable across arms; use `eval_loss` on the same val set and the quick eval.**
- Gate to go to the full stage 2 (suggested): best arm >= 0.606 avg macro-F1 (stage-2 bar), many-label flip rate <= 0.40 (S1 and ModernBERT-v3 numbers are in the plan page), and no arm more than 0.01 avg below A on the 14 sets.
- Fair baselines on the same eval: ModernBERT-v3 0.556, S1 0.586.
- Pilot gains smaller than the eval noise (500 rows/set) must not be over-read; A is the control for "just 1,500 more steps".

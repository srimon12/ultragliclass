# How UltraGliClass v0.1 was made (end-to-end recipe)

One readable answer to "how did we create 0.1?". Every number here is read from a file in this repo
(`proofs/…`, `configs/…`) or from the Kaggle artifacts on the build box (paths in [`PROOFS.md`](PROOFS.md)).
Where a number is an estimate or unverified it says so.

## 0. The model in one paragraph

GLiClass **uni-encoder** (labels and text in one sequence, `<<LABEL>>l1<<LABEL>>l2…<<SEP>>[prompt]text`, prompt first,
first-token pooling, MLP scorer, dropout 0.3) on **`dmis-lab/UltraBERT-150M`** (24 layers, hidden 768, sliding-window
attention 128 with every third layer global, RoPE, 32,768 positions, 49,152-token BPE + 3 GLiClass specials).
Loss: GLiClass focal BCE over label slots (α 0.75, γ −1 = α-weighting only), soft `{label: prob}` targets for Jev
score/probability rows. From Stage 2 on: a **permutation-consistency** term (same row, second label order, symmetric KL,
weight 0.1, pair probability 0.25). Model/loss settings: [`configs/base.yaml`](../configs/base.yaml) (sources R1/R3/R6 in `REFERENCES.md`).

## 1. Shared training settings (all stages)

| Setting | Value | Where |
|---|---|---|
| Hardware | Kaggle **2× T4**, DDP (torchrun), SDPA attention | notebooks/, `proofs/stage_summaries/*/STATUS.json` (`world_size: 2`) |
| Precision | **fp16** (T4 has no bf16; UltraBERT was pre-trained in bf16 — bf16 for this checkpoint is untested) | `configs/base.yaml` |
| Effective batch | **64 rows / optimizer step** (per-device × accumulation × 2 GPUs) | stage configs |
| Optimizer | AdamW β (0.9, 0.98), ε 1e-6, max grad norm 1.0 | `configs/base.yaml` |
| Weight decay | encoder 0.1, heads 0.01, **none on RMSNorm gains** (our fix) | `configs/base.yaml`, `src/ugc/model_init.py` |
| Schedule | linear decay; warmup per stage below; fixed `max_steps` so the schedule spans Kaggle sessions | stage configs |
| Seed | training seed 42 in every stage; data seeds 42 / 43 / 44 for S1 / S2 / S3 | configs |
| Software | `torch 2.11.0+cu128` (Kaggle image; seen in the S1, S2 and S3 notebook logs), `transformers==5.20.2`, `gliclass==0.1.21` | logs, `pyproject.toml`, `proofs/hf_readme_eval/env.json` |
| Trainer | `scripts/train.py` + `src/ugc/trainer.py` (`StrictTrainer`) | |

Changes vs the official GLiClass `train.py` (all in `docs/PLAN.md` §2): seeded deterministic data; Trainer-managed
devices for DDP; resume + time budget + `max_steps`; `prediction_loss_only` (the official `compute_metrics` breaks on
ragged label counts); RMSNorm gains excluded from weight decay; `logit_scale` frozen (unused, broke DDP); a **strict
training step** (the official step swallows exceptions and silently trains on zeros — found in our CPU smoke test).

## 2. Stages

### Stage 0 — smoke (not part of the weights)
`configs/stage0_kaggle.yaml` + `configs/data_stage0_kaggle.yaml`: 10k streamed rows, 150 steps, resume tested across two
Kaggle sessions (S0a → S0b). ~0.44 GPU-h over several attempts (`proofs/stage_summaries/gpu_hours_log.tsv`).
Led to the v2 resume-hang hardening (`docs/history/CHANGELOG_dev.md`, 22 Oct).

### Stage 1 — pilot (1,024 tokens, ≤ 50 labels)

| | |
|---|---|
| Data recipe | [`configs/data_stage1.yaml`](../configs/data_stage1.yaml) (seed 42, `max_options` 50, text ≤ 6,000 chars) |
| Train rows | **410,422** = gliclass-v2.0 300,000 + gliclass-v3-logic 10,422 (×2 repeat) + Jev typed decisions 100,000 (`license_use == commercial`, state ≤ 4,000 chars) — `proofs/stage_summaries/s1/stage1_data_manifest.json` |
| Val rows | 4,300 (2,000 + 300 + 2,000); same file as `calib/stage1_val.jsonl` (md5 `cacbb981…`) |
| train.jsonl md5 | `a9de950386ab7c99bdd6f1646254f30b` (Kaggle S1 output) |
| Train config | [`configs/stage1_pilot.yaml`](../configs/stage1_pilot.yaml): 8 × accum 4 × 2 GPUs, grad checkpointing on, no augmentation |
| Steps / LR | **6,413** steps (1 epoch, ceil(410,422 / 64)); encoder **1e-5**, heads **3e-5**, warmup 5 % |
| Result | `STATUS.json`: complete, final val loss **0.0305**, peak 3.97 GB/GPU. ≤500-row quick eval: suite avg macro-F1 **0.586**, Banking77 0.454, CLINC-150 0.398, typed 0.503 (`stage1_ultrabert150m_quick.json`) |
| Compute | **5.71** GPU-h, one session |
| Code | an earlier code version; the files that differ from this repo are in `proofs/code_snapshots/s1/` (see `PROVENANCE.md`) |

### Stage 2 — scale (1,024 tokens, 20–150 labels, arm C)

How arm C was chosen: three 1,500-step pilots (A = S1 continued, B = + many-label expansion, C = B + consistency;
`configs/stage2_pilot_{A,B,C}.yaml`, 5.14 GPU-h), then data ablations E1/E1z/E2 (3.35 GPU-h) and a speed test (0.22 GPU-h).
Outcome: arm C recipe with the E2 data mix (no dedicated procedural source — it hurt typed "choice"; + 3,000 typed-decision
TRAIN rows), batch 4 × accum 8 with checkpointing off. Write-ups: `docs/specs/STAGE2_PLAN.md`, `STAGE2_PILOT_JOB_SPEC.md`,
`STAGE2_ABLATION_JOB_SPEC.md`, `docs/history/S2_PILOT_INVESTIGATION.md`.

| | |
|---|---|
| Data recipe | [`configs/data_stage2_full.yaml`](../configs/data_stage2_full.yaml), generated by `scripts/make_stage2_full_recipe.py` from `data_stage2_abl_E2.yaml` + sizes of `data_stage2.yaml`; seed 43; `forbid_ids` eval guard; `dedupe_train_vs_val` |
| Train rows | **1,731,505** (after dropping 865 rows whose text equals a val text) — `proofs/stage_summaries/s2/stage2_full_data_manifest.json`: |
| | gliclass-v2.0 1,089,242 · gliclass-v3-logic 12,891 (×3) · Jev (commercial, state ≤ 4k) 436,985 · LEDGAR 59,930 · go_emotions 39,998 · fin-news topic 33,980 · fin-news sentiment 28,626 · HWU64 26,853 · LocalLLaMA/typed-decisions **train** 3,000 (customer_service 1,500 + agent_trace_observability 1,500) · dedicated procedural 0 |
| Val | `calib/stage2_pilot_val.jsonl`, 1,850 rows (md5 `8e5e644e…`) — same as the pilots so `eval_loss` is comparable |
| train.jsonl md5 | `d82991a305052e585050954f1ac6778f` — the Kaggle build and an independent box rebuild agree (`stage2_full_data_md5.json`) |
| Train config | [`configs/stage2_full_C.yaml`](../configs/stage2_full_C.yaml) ← `stage2_pilot_C` ← `B` ← `A` ← `base` |
| Many-label expansion | expand_prob 0.7, K log-uniform in [20, 150], hard-negative fraction 0.5, near-dup ban sim 0.80, keep ≥ 256 text tokens; gliclass-v2.0 rows milder (p 0.10, K 12–60) |
| Consistency | pair_prob 0.25, symmetric KL, weight 0.1 (pairs need ≥ 8 labels) |
| Augmentation | official GLiClass: label removal 0.05, label addition 0.05, text addition 0.05 |
| Steps / LR | **27,000** steps (≈1 epoch: 1,728,000 rows); encoder **1e-5**, heads **3e-5**, warmup 2 %; init = S1 final model |
| Batch | 4 × accum 8 × 2 GPUs, grad checkpointing **off**, DataLoader workers 0 |
| Sessions | session 1 → `needs_resume` at step 15,910 (10.72 h); session 2 resumed from checkpoint-15910 → 27,000 (7.48 h) |
| Result | final val loss **0.0775**, peak reserved 9.61 GB. ≤500-row quick eval: suite **0.683**, Banking77 0.554, CLINC-150 **0.650**, typed 0.616 (`stage2_full_C_final_quick.json`) |
| Compute | **18.20** GPU-h (+0.14 for a run stopped by the train/val leak assert before training) |

### Stage 3 — long context (8,192 tokens, up to 256 labels) → **v0.1**

| | |
|---|---|
| Data recipe | [`configs/data_stage3.yaml`](../configs/data_stage3.yaml), seed 44, `max_options` 256, `max_chars` 30,000 |
| Train rows | **61,184** = 21,184 long Jev rows (config `full`, state ≥ 6,000 chars, `license_use == commercial`, **rows whose licence contains `-sa`, `by-sa`, `gpl` or `odbl` excluded**, ~1.8k removed) + **40,000 replay** rows sampled from the S2 train file (S2 label pools copied) — `proofs/stage_summaries/s3/stage3_data_manifest.json` |
| Hygiene | 774 train rows sharing a val text dropped; **0** train texts in the long-Jev eval slice or val (`stage3_data_check.json`) |
| Val | 300 long Jev rows carved from train + the S2 val (1,850) |
| train.jsonl md5 | `d35eb55c3fc957d1c951aa6ab57c1028` — Kaggle and box builds agree |
| Train config | [`configs/stage3_longctx.yaml`](../configs/stage3_longctx.yaml) (base `stage2_pilot_C`) |
| Context / labels | `max_length` **8,192**, `max_num_classes` 256, many-label K up to 250 |
| Augmentation | label removal 0.05, label addition 0.05, **text addition 0.15** (S2: 0.05) |
| Steps / LR | **956** steps (floor(61,184 / 64)); encoder **5e-6**, heads **1.5e-5** (half of S2, forgetting guard), warmup 3 %; init = S2 final model |
| Batch | 1 × accum 32 × 2 GPUs, grad checkpointing on |
| Sessions | smoke session → `needs_resume` at step 150 (1.16 h); full session resumed from checkpoint-150 → 956 (6.11 h) |
| Result | final val loss 0.1070 (different val set from S2), peak reserved 5.47 GB, median ~24 s/step. Long-Jev@8192: S2 **0.552 → 0.664** (+0.111, cluster CI [+0.068, +0.157], McNemar p = 1.9e-7, 67 wins / 19 losses; `stage3_longjev_compare.json`). ≤500-row quick eval: suite **0.689**, Banking77 0.548, CLINC-150 **0.640**, typed 0.574 (`stage3_final_quick.json`) |
| Compute | **7.27** GPU-h |

**Go/no-go and what was accepted.** The pre-registered Stage 3 bars (`docs/specs/STAGE3_PLAN.md`) were: long-Jev
+0.03 over S2 (**met**, +0.111) and CLINC-150 ≥ the S2 value 0.650 on the quick protocol (**missed: 0.640, −0.010**).
The miss was **accepted for v0.1** as a small short-input loss traded for the long-context gain. On the later full
4,500-row test split v0.1 scores 0.636 (95% CI 0.622–0.650). Typed decisions also dropped vs S2 (0.616 → 0.574,
mostly invoice_processing); a suspected (unconfirmed) cause is the higher procedural-style share in S3 (~13.9 % vs ~4.9–5.25 % in S2).

## 3. Final weights

`runs/stage3_full/final_model/` from Kaggle kernel `srimondanguria/ultragliclass-s3full` (box copy:
`/workspace/kaggle/ultragliclass/S3full/final/ultragliclass/runs/stage3_full/final_model/`):

| file | bytes | sha256 |
|---|---|---|
| `model.safetensors` (fp32) | 615,323,864 | `8964cf620013b4b1307baf7f9b132ca13051b314f7cb34b0bcae557e803751fa` |
| `config.json` | 4,588 | `748a02d905618b1dec94511bdcfc0a743e0f4748aedc16ca3f5af608a684785e` |
| `tokenizer.json` | 3,524,287 | `b22a0247b08dc8420d1fd950f51bf903b4d579a851f081e3962e8cb1bc53d225` |
| `tokenizer_config.json` | 1,006 | `949f9464da03d1e6a54deb9d50ba9016304ee5c3d8fb0d5b89f52e2a82951387` |

Intermediate weights (box): S2 `…/S2full/session2/output/ultragliclass/runs/stage2_full_C/final_model/model.safetensors`
sha256 `19ae4b90…4c8c`; S1 `…/S1/ultragliclass/runs/stage1_ultrabert150m/final_model/model.safetensors` sha256 `b76e9e1e…40b8`.
Weights are **not** in git; they go to a GitHub Release or the Hugging Face Hub at launch.

## 4. Compute totals (Kaggle GPU-h = session wall hours on a 2× T4 machine)

From `proofs/stage_summaries/gpu_hours_log.tsv`:

| Item | GPU-h |
|---|---|
| Stage 0 smoke attempts | 0.44 |
| Stage 1 | 5.71 |
| Stage 2 pilots / speed test / ablation | 5.14 + 0.22 + 3.35 = 8.71 |
| Stage 2 full (2 sessions + 1 failed start) | 10.72 + 7.48 + 0.14 = 18.34 |
| Stage 3 (smoke + full) | 1.16 + 6.11 = 7.27 |
| README benchmark | 1.22 |
| **Total logged** | **41.69** |
| **Direct lineage of the v0.1 weights (S1 + S2 full + S3)** | **5.71 + 18.20 + 7.27 = 31.18** |

Re-running only the final lineage costs ~31 Kaggle GPU-h (≈ one weekly 30 h quota and a bit). Physical GPU-hours are
about twice these numbers (two T4s per session).

## 5. Exact data transforms (from `docs/DATASETS.md`, `src/ugc/convert.py`, the configs)

Every row becomes GLiClass JSON: `{"text", "all_labels", "true_labels" (list or {label: prob}), "prompt"?, "source", "kind", "license_use"?}`.

- **gliclass-v2.0 / gliclass-v3-logic** (`format: gliclass`): native format; drop empty or duplicate labels; text capped
  at `max_chars` (6,000 in S1/S2); rows with > `max_options` labels dropped (50 in S1/S2; S3 allows 256).
- **Jev typed decisions** (`format: jev`): filter `license_use == "commercial"` (S3 also drops `-sa/by-sa/gpl/odbl`
  licences); **choice/score** → options as labels, soft target = option probabilities, question as prompt;
  **noul** → labels `["yes","no"]`, target `{yes: p, no: 1−p}`; JSON states rendered as `key: value` lines;
  state ≤ 4,000 chars (S1/S2) or ≥ 6,000 chars (S3, config `full`).
- **LocalLLaMA/typed-decisions** (`format: system_one`, S2 only): **train split only** (a guard in `prepare_data.py`
  aborts if the test split is read); one row per question; choice = option descriptions, noul = true/false
  descriptions, score = level descriptions in order; instructions as prompt; target = answer probabilities.
- **Taxonomy sources** (HWU64, LEDGAR, fin-news topic/sentiment, go_emotions; `format: taxonomy`, S2): class ids →
  names (`humanize` where set), label-phrasing schemes, label pools for hard-negative expansion (`label_pools.json`).
- **Hygiene:** `forbid_ids` refuses every eval-set id; train rows whose exact text equals a val/eval-slice text are dropped
  (`dedupe_train_vs_val`, counts in the manifests); data md5s are recorded per stage.

**Doc drift to know about:** the "Mix ratios" table in `DATASETS.md` is the *pre-run plan* (it lists a dedicated
procedural source in S2/S3 and S3 label-addition p = 0.5). The runs dropped the dedicated procedural source and used
label addition 0.05 in S3. The manifests and configs above are what actually ran.

## 6. Reproduce

See the README "Reproduce the training" block. Order matters: S2 needs `calib/stage2_pilot_val.jsonl`; S3 needs
`data/stage2_full/train.jsonl` (replay) and `calib/jev_long_test.jsonl` (dedupe guard). How to rebuild those:
[`data/README.md`](../data/README.md). Expected md5s are listed above; a mismatch usually means a Hub dataset revision changed.
On Kaggle, use the notebooks in `notebooks/` (they expect the flat code-bundle layout — see `notebooks/README.md`).

# UltraGliClass: training plan (v1, 22 Oct 2026)

Goal: a GLiClass zero-shot / decision classifier on `dmis-lab/UltraBERT-150M`. It is trained the official GLiClass
way (Knowledgator `train.py` recipe, data format, focal loss, pipeline) and uses the dmis-lab fine-tuning settings
for UltraBERT. Training happens in small staged, resumable checkpoints on Kaggle, never all in one go.
The references [R#] are in REFERENCES.md, the datasets in DATASETS.md, and the CPU smoke results in SMOKE_TEST.md.

## 1. Architecture choice

- **Uni-encoder first** (`architecture_type: uni-encoder`): labels and text in one sequence,
  `<<LABEL>>l1<<LABEL>>l2…<<SEP>>[prompt]text`, scored by an MLP scorer on first-token pooling. This is the
  architecture of every current GLiClass release, including gliclass-v4-edge [R6], and the default of `train.py` [R1].
  Labels attend to each other and to the text, which matters for many-option and instruction-style decisions.
- Head settings follow the v4-edge config [R6]: pooler `first`, class_token_pooling `first`, scorer `mlp`,
  dropout 0.3, focal loss α 0.75 / γ −1, prompt_first, and label shuffling on every example.
- **Why UltraBERT fits:** native in transformers 5.20 (no remote code) [R3, R10]. Local ±128 attention in 2 of every
  3 layers gives cheap long inputs, with a 32K context. The vocabulary is 49K BPE. It has 151M parameters, the same
  size as ModernBERT-base, which makes for a clean A/B.
- **Bi-encoder later** (not in v1): `bi-encoder` / `bi-encoder-fused` in GLiClass give cacheable label embeddings for
  1,000+ label routing. It is worth trying after stage 2 if the many-option latency in stage 3 is too high.
- Out of scope for v1: the v4 RL stage (`train_rl.py`, PPO-style) and the learned calibrator (`calibrate.py`) [R6].
  In v1, calibration is plain temperature scaling, and RL plus the calibrator form a possible stage 4.

## 2. Recipe (all stages)

| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW β (0.9, 0.98), ε 1e-6 | UltraBERT card [R3] |
| LR | encoder 1e-5, heads 3e-5 (GLiClass keeps heads at about 2.5× the encoder) | [R3] MNLI best 1e-5; [R1] 2e-5/5e-5 ratio |
| Weight decay | encoder 0.1, heads 0.01, none on RMSNorm gains | [R3]; [R1]; our fix (GLiClass default misses RMSNorm) |
| Schedule | linear, 5% warmup (2-3% for later stages), max_grad_norm 1.0 | [R3]; [R1]; [R4] |
| Batch | 8 per GPU × accumulation 4 × 2 GPUs = 64 | sized for T4 16 GB at 1024 tokens (verify in stage 0) |
| Precision | fp16 on T4 (no bf16 there); bf16 switch for A100/TPU; fp32 master weights | UltraBERT was trained in bf16; **fp16 is untested** [R3] |
| Attention | SDPA (FlashAttention-2 needs Ampere or newer; T4 is Turing) | [R3] |
| Memory | gradient checkpointing on | |
| Loss | GLiClass focal BCE over label slots, soft targets for jev/score rows | [R1, R2, R6] |
| Checkpoints | every 500 steps + at the time budget + final; keep 2; full optimizer/RNG state | ours |
| Sessions | fixed `max_steps` per stage, so the LR schedule spans every session; time budget 11 h of 12 h | Kaggle limits [R30] |

Faithfulness notes, i.e. where we changed the official train.py:
- Seeded, deterministic data (the official script calls `random.shuffle` unseeded).
- Trainer-managed devices instead of a hard-coded `cuda:0`, for DDP.
- Resume, time-budget and max_steps support.
- `prediction_loss_only`, because the official compute_metrics breaks on ragged label counts.
- RMSNorm gains excluded from weight decay.
- `logit_scale` frozen: it is unused with normalize_features=False and broke DDP.
- A strict training step: the official one swallows exceptions and silently trains on zeros (found in the smoke test).

## 3. Stages

| Stage | Data (rows, EST) | Length / options | Steps at batch 64 | T4×2 time (EST) | Starts from |
|---|---|---|---|---|---|
| 0 smoke | 10k streamed (stage-1 mix) | 1024 / ≤50 | 150 | ≤1 h incl. setup | UltraBERT-150M |
| 1 pilot | ~415k: gliclass-v2.0 300k + logic 7.5k ×2 + jev commercial 100k | 1024 / ≤50 | ~6.5k | 3.5-7 h, 1 session | UltraBERT-150M |
| 1-control | the same data and code, ModernBERT-base | 1024 / ≤50 | ~6.5k | 3-6 h, 1 session | ModernBERT-base |
| 2 scale | ~2.1M: all gliclass-v2.0 + logic ×3 + all commercial jev + 300k procedural + GLiClass augmentation | 1024 / ≤50 | ~33k | 18-36 h, 2-4 sessions | stage 1 final |
| 3 long + many options | ~60-90k: long jev states ≥6k chars, needle retrieval, procedural, 30k replay; label/text-addition augmentation | 8192 / ≤256 | ~1.4k | 8-16 h, 1-2 sessions | stage 2 final |

The time estimates are FLOP-based, assuming about 8 effective TFLOPS per T4 at about 400 padded tokens per row. They are not measured;
stage 0 measures `sec_per_step` and `peak_mem_gb`, and the notebook prints the ETA. GPU budget: about 30 h/week on Kaggle [R30],
so week 1 = stage 0 + stage 1 + control + eval (about 15 h).

### Go / no-go bars

| After | Go if all hold | If not |
|---|---|---|
| Stage 0 | (1) no NaN/inf in 150 fp16 steps; (2) both GPUs used (world=2); (3) the time-budget stop and the resume reach `complete`; (4) peak memory < 14.5 GB per T4 at 8×1024; (5) projected stage-1 time ≤ 10 h | NaN: lower the LR to 5e-6 or add fp32 warmup steps, or move to TPU/bf16. OOM: batch 4 × accum 8. Too slow: max_length 512 for stage 1 (all eval label sets except CLINC still fit) |
| Stage 1 | (1) val loss falls monotonically per eval; (2) GLiClass-suite average macro-F1 ≥ the 1-control − 0.01; (3) typed-decisions accuracy ≥ 0.470 (input-blind Prior) and above the 1-control; (4) label-shuffle flip rate ≤ the 1-control | UltraBERT trails the control by >0.02: rerun quarter-size pilots at LR 5e-6 and 2e-5 (about 1.5 GPU-h each) before dropping the backbone |
| Stage 2 (revised 23 Oct, see STAGE2_PLAN.md §5) | (1) 14-set suite avg macro-F1 >= 0.606 (fair ModernBERT 0.556 + 0.05); (2) no set > 0.03 below ModernBERT; (3) mean flip on banking77/clinc150/massive/20ng <= 0.40; (4) Banking77/CLINC accuracy >= stage 1; (5) typed acc >= max(0.503, ModernBERT), ECE after T <= 0.10; (6) mean ECE after T <= ModernBERT's | Drop DBPedia/LEDGAR first; ship with 3-order logit averaging if flips stay high |
| Stage 3 | (1) +3 pts accuracy on the long-state (≥6k chars) jev test slice vs stage 2; (2) CLINC-150 ≥ stage 2; (3) GLiClass-suite regression ≤ 0.01 | Lower the stage-3 LR or add more replay |

## 4. Evaluation suite (`eval.py`)

- **GLiClass zero-shot suite** [R9]: the 13 sets of the official `test_gliclass.py` plus snips (from the cards).
  Single-label, softmax, macro-F1, same label strings.
  Parquet mirrors replace the script-based PolyAI/banking77, AmazonScience/massive and takala/financial_phrasebank (see DATASETS.md).
- **All-label intent**: Banking77 with all 77 labels, CLINC-150 with all 150 in-scope labels (`--clinc_oos` adds `oos`).
- **Option-order robustness**: K=3 runs (canonical order + 2 seeded shuffles): mean/std accuracy and flip rate.
- **Calibration**: ECE (15 bins) of the raw softmax; ECE after one global temperature fitted on the stage's
  in-distribution val split; and a 2-fold per-dataset temperature (optimistic bound).
- **Typed decisions** [R20]: LocalLLaMA/typed-decisions test, 400 cases / 2,000 decisions, using the official scorer
  definitions (accuracy = argmax vs gold argmax, KL from gold, Brier, ECE with 10 bins) and the GLiClass-v4 System One mapping
  (choice/noul/score descriptions as labels, instructions as prompt) [R8].
  Reference points: Prior 0.470, teacher self-agreement 0.735, gliclass-v4-edge 0.493, gliclass-modern-base-v3.0 0.391 (we measured it).
- **Controls**:
  - a ModernBERT-base GLiClass trained with the identical data and code (`stage1_control_modernbert.yaml`);
  - the published gliclass-modern-base-v3.0 and gliclass-v4-edge;
  - Laya via `--backend laya` (adapter written from the README, **unverified**; Laya's card reports Banking77 0.425, ECE 0.466, and typed-decisions 0.766 *after fitting on train*, which is not comparable to zero-shot) [R25, R26].
- Never trained on: any eval set. Typed-decisions `train` is not used; a fine-tuned variant would be reported separately
  in the "fitted on train" table.
- Latency: p50 ms per example (batched) is logged; batch-1 GPU latency gets measured on Kaggle.

## 5. Risks

1. **fp16 on T4.** UltraBERT was only trained and tested in bf16, and grad-norms are about 10× ModernBERT's at init. Mitigation: NaN guard
   (stops after 3 bad logs), stage 0 gate, fallbacks (lower LR, TPU/bf16).
2. **Hardware TBD** (2×T4 vs TPU v3-8; the Kaggle Handler is checking). The code is GPU-first. TPU path: `data.static_shapes: true`
   (smoke-tested, numerically identical), `precision: bf16`, torch_xla launch; **untested on XLA**. TPU sessions are 9 h and 20 h/week [R30].
3. **Truncation.** Labels come first and text is truncated at max_length, so long label sets eat the text budget
   (Banking77 needs about 560 tokens of labels). Kept at 1024 for stage 1; stage 3 goes to 8192.
4. **Data licenses.** gliclass-v2.0 is tagged apache-2.0, but its upstream sources are undocumented. Jev rows are mixed and filtered
   to `license_use == commercial`, which still includes share-alike sources (cc-by-sa). See DATASETS.md.
5. **Benchmark contamination.** The GLiClass training sets may contain eval-domain data (e.g. sentiment, intents). We dedupe our own
   splits but cannot audit upstream overlap; the zero-shot claims carry that caveat, as the published cards do.
6. **Weekly quota.** About 30 GPU-h/week. Stage 2 spans 1-2 weeks; checkpoints carry over between sessions (20 GB output limit:
   a 1.8 GB checkpoint × 2 + final 0.6 GB + data is fine).
7. **Upstream drift.** Versions are pinned (transformers 5.20.2, gliclass 0.1.21), and pip is constrained to keep Kaggle's CUDA torch.
8. **Laya adapter unverified**, and the controls' numbers come from their cards unless we re-measure them (we re-measure the GLiClass ones).

## 6. Kaggle hardware note

The default target is **GPU T4 ×2**: `torchrun --nproc_per_node 2`, NCCL, fp16, SDPA. On 1 GPU (P100) the notebook
falls back to `python train.py`, and the effective batch then halves unless `gradient_accumulation_steps` is doubled.
On TPU v3-8: set `precision: bf16` and `data.static_shapes: true`, and launch with torch_xla. That last part is not written yet and
should wait until the Kaggle Handler confirms the hardware.

## 7. Open decisions (for Srimon)

1. GPU (T4×2) vs TPU: waiting on the Kaggle Handler.
2. Spend about 3-6 GPU-h on the ModernBERT-base control in week 1? (Recommended: it is the only clean A/B.)
3. Jev rows with `license_use: unspecified` (284k): keep them out (default, commercial-safe) or add them for a research-only build?
4. Use gliclass-v2.0 despite its undocumented upstream provenance? (It is the official GLiClass training set.)
5. Single LR (1e-5) or a 3-point mini-sweep (5e-6 / 1e-5 / 2e-5) on 10% pilots first (about 2 GPU-h total)?
6. Add the v4-style RL + calibrator as a stage 4 later?
7. Uploading the code as a private Kaggle Dataset (needed by the notebook) is an action that needs your OK.

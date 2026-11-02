# S2 pilot investigation (23 Oct 2026, CPU on box only; no Kaggle/GPU)
Scope: (1) overlap/contamination, (2) typed-decisions drop, (3) VRAM/speed.

## Step 0 – setup facts
- Eval typed-decisions set = `LocalLLaMA/typed-decisions` (eval.py `typed_rows`), NOT Jev. Training Jev = `tasksource/tasksource-jev-typed-decisions`; procedural = `tasksource/procedural-typed-decisions`.
- Pilot train data (96,035 rows) is not on the box (only smoke data); plan = rebuild from configs/data_stage2_pilot.yaml (seed 43) on CPU and verify against manifest counts.

## Step 1 – data rebuilt locally
- Rebuilt pilot mix on CPU: `prepare_data.py --recipe configs/data_stage2_pilot.yaml --out_dir data/stage2_pilot_rebuild` (seed 43). Per-source train rows and kinds match the Kaggle manifest exactly (96,035 train / 1,850 val). Row-level identity with the Kaggle copy is UNVERIFIED (Kaggle copy not on box), but counts+seed+code are identical.
- Eval samples regenerated with eval.py's `load_eval_set` + `subsample(500, seed 0)` (same code path as Kaggle quick eval); full eval sets also checked. Cached in <build-box>/s2inv/eval_sets.json.

## Step 2 – overlap results so far (pilot mix rebuilt vs eval samples)
Script: `scripts/overlap_check.py` (normalized exact match + word-3-gram Jaccard; eval = the 500-per-set samples, plus exact match against the full eval sets). Raw: <build-box>/s2inv/overlap_pilot.json
- Exact-match against full eval sets: HWU64 train ↔ massive test 204 utterances (e.g. "will it be raining on thursday"), HWU64 ↔ clinc150 6, gliclass-v2.0 (47k sampled rows) ↔ clinc150 1. Everything else 0 (banking77, financial_phrasebank, snips, sst5, 20NG, ... = 0).
- Within the 500-samples actually evaluated: exact = 36 HWU64 *rows* (= 12 distinct utterances x3 repeat) hit massive; near-dup (J>=0.5) distinct eval items: massive 16/500 (3.2%), sst2 3, sst5 1, clinc150 1. Everything else 0.
- financial_phrasebank: **0 exact, 0 near-dup**. Max bigram-Jaccard between the 660 unique fin-news-sentiment train tweets and all 2,264 phrasebank sentences is 0.15 (no pair >= 0.3). So the 0.255→0.92 jump is NOT text leakage.
- S1's 0.255 phrasebank F1 was a degenerate collapse: S1 accuracy 0.62 == share of "neutral" in the sample, flip rate 0.000 (same answer for every order) → predicts "neutral" for every sentence. Pilot arms: acc 0.93, F1 0.92. Cause = pilot adds twitter-financial-news-sentiment with label schemes {negative,neutral,positive} (exactly phrasebank's label names), same finance domain. This is legitimate in-domain/label-name transfer, not leakage, but it means phrasebank is no longer a zero-shot number for the pilot arms.
- Label-set overlap (normalized label strings of eval classes found in a train pool): massive 56/59 labels are exact HWU64 labels (HWU64 and MASSIVE share the same ontology) → massive is NOT zero-shot for pilot arms (text overlap only 3%, but label taxonomy identical). banking77 0/77, clinc150 1/150 from HWU64 ('play music'; 13/150 clinc label names, e.g. weather, time, taxes, recipe, also occur as label strings in the 47k sampled v2.0 rows) → banking/clinc gains are not label leakage. emotion 6/6 labels in GoEmotions (task-type overlap; but emotion F1 did not rise: 0.447→0.410/0.437/0.468).

### 2b – gliclass-v2.0 (all 1,196,218 rows) scanned against the 500-per-set eval samples
Scripts: `scripts/overlap_full_v2.py` (containment >=0.8 of the eval item's word-3-grams, any row length) and `overlap_full_v2_short.py` (same, row <= 2x eval length + 200 chars, i.e. excludes eval phrases that merely occur inside long web documents). Raw: <build-box>/s2inv/overlap_v2_full*.json
- Short-row scan (real instance-level hits, eval items >= 5 words): SetFit/sst2 60/487 (12.3 %), sst5 51/475 (10.7 %), rotten_tomatoes 48/485 (9.9 %), CR 1/363, clinc150 2/461; all other sets 0. (Unrestricted scan adds only phrase-in-long-document noise: cap_sotu 5, banking77 1, massive 1, phrasebank 1, clinc 7.)
- What these rows are: the SST/RT movie-review sentences appear in v2.0 with the instruction "Define whether subjective or objective the statement is:" and true label `subjective` (they come from the Pang&Lee subjectivity data, which reuses RT snippets). So the TEXT of ~10-12 % of sst2/sst5/RT eval items was seen in v2.0 but under a different task/label (subjectivity, not polarity). Sentiment-label leakage NOT established; text exposure is. This applies equally to ModernBERT GliClass v3 / S1 / pilot (all trained on v2.0 - v3: UNVERIFIED), and ~25 % of v2.0 was used by S1 vs ~4 % by pilot.
- Training-mix level (near-dup J>=0.5, 500-sample eval items): S1 mix (410k rows) -> sst2 12, sst5 15, RT 11, clinc 8 distinct, massive 3, banking77 3, snips 2 (jev short-phrase matches such as "Can you help me?" are generic); pilot mix -> massive 16 (all from HWU64), sst2 3, sst5 1, clinc 1. Exact full-set matches: S1 clinc 10, sst5 2, RT 1, massive 1; pilot massive 204 (HWU64), clinc 7. So apart from HWU64<->massive, the pilot has LESS incidental overlap than S1; text overlap cannot explain pilot>S1 gains on sst/banking/clinc.

## Step 3 – typed-decisions train/eval overlap
- Eval = LocalLLaMA/typed-decisions test: 400 cases / 2,000 decisions (choice 600, noul 600, score 800; 4 workflows x 100 cases), apache-2.0, synthetic customer_service / invoice_processing / security_incidents / agent_trace_observability. We do not train on it (neither its train split).
- Compared all 400 eval states (4-gram containment >= 0.8 and exact) vs ALL 999,832 Jev train rows and procedural-typed-decisions train (360,000) / validation (18,000) / test (18,000): **0 exact, 0 near-dup**. (Caveat: states compared as normalized text; template-style similarity — same System-One question schema, key: value states — is expected and NOT measured.)

## Step 4 – typed-decisions: data facts (before per-item CPU re-scoring)
Eval (LocalLLaMA/typed-decisions test, all 2,000 decisions: choice 600 [K=4:400, K=5:200], noul 600 [K=2], score 800 [K=4:700, K=5:100]; states ~750 chars; natural-language options such as "Halt the agent now.", "Critical; requires action within the same day.").
Per-type results (acc, n=600/600/800; binomial SE ≈ 2.0 / 2.0 / 1.7 pts):

| model | choice | noul | score | overall |
|---|---|---|---|---|
| ModernBERT v3 | .297 | .590 | .313 | .391 |
| S1 | .485 | .645 | .410 | .503 |
| A | .423 | .575 | .391 | .456 |
| B | .418 | .545 | .378 | .440 |
| C | .412 | .650 | .384 | .472 |

- Drop vs S1: choice −6.2/−6.7/−7.3 pts (A/B/C, ≈3σ each), noul −7.0/−10.0/+0.5, score −1.9/−3.2/−2.6 (≈1-2σ). The choice drop is the robust part, shared by all three arms; the noul drop is arm-specific (A, B only; C recovers fully).
- Training data. S1 mix (410,422 rows): 75.6 % gliclass (310k), 24.4 % Jev (100k: choice 77.6k / noul 14.0k / score 8.5k); NO procedural. Pilot mix (96,035): gliclass 49.7 %, Jev 24.5 % (23.5k: choice 18.4k / noul 3.1k / score 2.0k — SAME proportions as S1), procedural 12.2 % (11.75k: choice 5.1k / noul 3.8k / score 2.8k), taxonomy 13.6 %. So the typed SHARE did not fall (24 % → 37 %); the Jev choice share is 19.2 % vs 18.9 %. The pilot starts from S1 and therefore keeps S1's 100k-row Jev exposure on top. "Less typed data" is therefore NOT the cause.
- Domain: neither source resembles the eval. Jev = NLI/legal/multiple-choice style (options like 'entailment', 'holding that ...'; noul 'yes/no'; score '0..4' or similarity prose); procedural = synthetic key:value puzzles (options 'N10', 'Ivy', digits '0'-'9', states ~1,700 chars, 17 % hit the 1024-token cap). The eval is agent-trace / customer-service / invoice / security-incident decisions with prose options (K 2-5). The pilot ADDS procedural rows (non-semantic option strings, long states) that S1 never saw. Hypothesis (UNVERIFIED, needs an ablation): they pull the model toward positional/structure heuristics, hurting semantic-option "choice" (−6 to −7 pts in every arm).

## Step 5 – VRAM / speed (reasoning + CPU probe; no GPU run)
Config actually used (configs/base.yaml + stage2_pilot_A/B/C.yaml; Kaggle log confirms `backbone UltrabertModel, attn=sdpa, params=153.8M`, fp16, world_size 2):
- per-device batch 8, grad accumulation 4, 2 GPUs → effective 64 rows/step; max_length 1024; fp16 autocast (Trainer fp16=True, fp32 master weights); gradient_checkpointing TRUE (base.yaml); AdamW (fp32 states); dataloader workers 2, no length bucketing (random batches, dynamic padding to the longest row in each micro-batch of 8); sdpa attention. Encoder: 24 layers, hidden 768, MLP 1024 (gated SiLU), 6 heads x 128, sliding window 128 with a full-attention layer every 3rd, 32k RoPE.
- Measured on CPU (script <build-box>/s2inv/memcalc.py; autograd saved_tensors_hooks, bf16 autocast = same 2 bytes/elt as fp16): activations saved for backward per 1,024-token sample: **78 MiB with checkpointing (confirmed enabled: every encoder layer reports gradient_checkpointing=True after `gradient_checkpointing_enable()`), 1,061 MiB without**. At 512 tokens: 39 / 630 MiB. CPU kernels differ from CUDA sdpa and some tensors may be fp32 on GPU, so treat ±20 %.
- Static memory: 153.8M params → fp32 weights 0.57 GiB + grads 0.57 + Adam m,v 1.14 = 2.3 GiB, + autocast fp16 weight copies ≈ 0.3 → ≈ 2.6 GiB. Observed peak 3.93-3.97 GiB (A, B, S1), 4.27 (C, extra paired forward). 2.6 + 8 x 78 MiB (0.6) + one layer's recompute working set 8 x 44 MiB (0.35) + head/logits ≈ 3.9 → the model of memory matches the log, i.e. the peak IS the 8 x 1,024 worst case.
- Token lengths (6,000 random pilot rows, tokenizer + label prompt, cap 1,024): mean 240, median 134, p90 641, 2.3 % of rows hit the cap (procedural s1-* rows: 15-17 % capped, mean 530-600 tokens; gliclass rows mean 208, taxonomy 84). P(a micro-batch of 8 contains a capped row) ≈ 17 %, so the worst case occurs within the first few steps → the flat 3.93 from step 50 is credible.
- Caveats on peak_mem: (a) `torch.cuda.max_memory_allocated()` = allocated tensors only, NOT reserved/nvidia-smi (CUDA context ≈ 0.3-0.5 GB + allocator slack add roughly 1-2 GB; real footprint maybe 5-6 GB; UNMEASURED); (b) logged by rank 0 only, every 50 steps, as a process-lifetime cumulative max (so it does cover all steps up to each log, but rank 1 is not reported, and eval-time peaks after the last log are included only if > train peak); (c) first log line shows 2.72-2.9 GB at step 1 (before long batches) → the later 3.93 is the true plateau. I do not think the peak was measured on part of the run only, but rank 1 / reserved are unverified.
- Padding waste: padded length per micro-batch averages **679 tokens vs 240 real** → ~65 % of the token-slots computed are padding. Effective throughput: ≈ 8 x ~116M non-embedding params x 5.4k padded tokens ≈ 5 TFLOP per micro-batch (fwd+recompute+bwd), 4 micro-batches / 3.67 s → ≈ 5.5 TFLOP/s per T4 (peak fp16 tensor 65, practical ~20-30) → GPU is far from saturated, mostly because of padding and small matmuls (hidden 768).

Fit/speedup estimates (UNMEASURED on GPU):
| change | peak allocated (est.) | speed vs now | verdict |
|---|---|---|---|
| ckpt OFF, B=8, ga=4 | 2.6 + 8 x 1.04 (worst 1,024-token batch) + 0.4 ≈ **11.3 GiB allocated, ~12.5-13.5 reserved** | ≈ 1.2-1.3x (removes recompute: 3 vs 4 forward-equivalents; ceiling 1.33x) | fits 14.5 GiB only barely; C arm (+0.3) and a fragmenting allocator could OOM → not the first thing to run |
| ckpt OFF, B=4, ga=8 | 2.6 + 4.2 + 0.4 ≈ **7.2 GiB** (~8-9 reserved) | ≈ 1.2x (slightly less from smaller kernels) | safe, recommended |
| ckpt ON, B=16, ga=2 | ≈ 2.6 + 1.2 + 0.7 + 0.5 ≈ 5.0 GiB | ≈ 1.0-1.1x (kernels already large; 5.4k tokens/micro-batch) | safe but little gain |
| length-bucketed batches (needs code: lengths are only known after tokenisation in the Dataset; use a seeded, resume-deterministic mega-batch sampler or a pre-pass of token lengths) | lower | **≈ 1.8-2.4x** (cuts 679→~270 padded tokens) | biggest lever; changes batch composition so not a like-for-like with pilot arms |
Combined best case ≈ 2.3-3x (3.7 s/step → ~1.3-1.6 s; the 5.1 GPU-h pilot trio → ~2 h). Raising VRAM use per se does nothing: the GPU is limited by compute efficiency, not memory.

Proposed quick Kaggle test (ONLY after Srimon approves; ~25 min wall on 2xT4, ≈ 0.4-0.5 GPU-h of the session counter): configs/speedtest_T0_baseline / T1_nockpt_b8 / T2_nockpt_b4 / T3_ckpt_b16 (created, parse-checked, NOT run): 40 steps each from the S1 checkpoint on arm-A data, effective batch 64 kept, no save/eval; record sec_per_step and peak_mem (add `torch.cuda.max_memory_reserved()` and rank-1 to the log line - 2-line patch to ProgressCallback, not applied). Run T2 first; run T1 only if T2's peak_reserved < 8 GB; stop on any OOM (max_oom_skips=0 fails loudly).

## Step 6 – per-item CPU re-scoring (scripts/peritem_cpu.py, analyze_peritem.py; canonical order, fp32; S1 and A done, C/B queued)
CPU reproduces GPU: S1 typed 0.5025 (GPU 0.503), A 0.455 (GPU 0.456).
- Typed, S1 → A paired (2,000 decisions, 400 cases, case-clustered SE): Δ −4.75 pts (gained 102, lost 197), z ≈ −5 → real, not noise. By type: choice −6.2 (SE 2.3), noul −6.8 (SE 1.0), score −2.1 (SE 1.3).
- By workflow (S1→A): agent_trace .408→.390, customer_service .490→.530 (+), invoice .474→.406, security_incidents .638→.494 (−14 pts). By K: K2 .643→.575, K4 .422→.366, K5 .517→.540. So the loss is concentrated in security_incidents and invoice_processing, not uniform.
- Choice predicted-position distribution (K up to 5): gold positions are skewed (idx2 38 %); S1 predicts idx 0/2/4 = 188/143/123, A predicts 197/127/76/119/81 — no collapse onto one position, so no simple positional-heuristic signature (my earlier procedural-puzzle hypothesis is therefore NOT supported by this check, remains unproven).
- massive (500 sample): S1 acc 0.416 → A 0.582. The 12 eval utterances that exactly appear in HWU64 train: S1 0.417, A 0.667; the other 488: S1 0.416, A 0.580. Removing the overlapping items changes A by ~0.002 → **the massive jump is not text leakage**; it is label-taxonomy transfer (56/59 massive labels = HWU64 labels).

## Step 7 – typed-decisions ablation package (24 Oct 2026, CPU only; nothing run on Kaggle)
Goal: test the (unproven) hypothesis that the procedural rows hurt semantic `choice`. Work log, saved as I go:
- 7.1 `prepare_data.py`: new optional source key `answers_col` (LocalLLaMA/typed-decisions keeps gold in `gold`), plus a hard guard: any enabled
  `LocalLLaMA/typed-decisions` source must be `split: train` with no `val_split` (its `test` split is our eval set; recipe aborts otherwise).
- 7.2 **Correction to Step 4 ("S1 had NO procedural").** Only the *dedicated* `tasksource/procedural-typed-decisions` dataset was absent from S1. The
  Jev source itself carries procedural-style rows (row source tag `procedural-typed-decisions/<task>`; 4,728 of the pilot's 23,500 Jev rows = 20.1 %;
  same synthetic key:value puzzles, S1..S5 / yes-no / digit options, mean ~1,500 chars). So S1's ~100k Jev rows held roughly 20k procedural-style rows
  (~4.9 % of 410k), and the pilot has 4,728 + 11,750 = 16,478 (**17.2 %**). The pilot therefore ~3.5x'd the procedural-style share, it did not introduce it.
  This is why the package has TWO removal arms (E1: back to ~S1's share, 4.9 %; E1z: 0 %).
- 7.3 Rebuild verification (resolves the "row-level identity UNVERIFIED" caveat of Step 1): the Kaggle pilot data is on the box
  (`<artifacts>/S2pilot/ultragliclass/data/stage2_pilot/`). `label_pools.json` md5-identical to my rebuild; train.jsonl (96,035) and
  val.jsonl (1,850): identical rows in identical order (text, label sets, source, kind); only ~1.3 % of rows differ in the *order/expansion of the label list*.
  So the rebuild is row-faithful. Consequence: the ablation arms are evaluated on the Kaggle pilot's actual val file, shipped as `calib/stage2_pilot_val.jsonl`.
- 7.4 Semantic-prose typed data: **LocalLLaMA/typed-decisions train split is usable**: apache-2.0 (card), 1,200 cases / 6,000 decisions, separate generation run and seed from test, 0 exact state matches with our 400 eval states. Not off limits, **but in-schema**: same 4 workflows, same question schemas, all 57 option strings
  identical to the eval's. Training on it makes our typed number a "fitted on train" specialist score (the benchmark lists such models separately). Plan therefore uses only 2 of 4 workflows (customer_service, agent_trace_observability; 3,000 rows = 3.1 %) so invoice_processing + security_incidents stay unseen, and needs Srimon's decision **D10** before E2/E3
  (notebook gate `SEMANTIC_DATA_APPROVED = False`; `prepare_data.py` aborts if that source uses `test`). No other licensed prose-option typed source found (search not exhaustive: unverified).
- 7.5 **Ablation plan** (all arms = pilot arm C config; 1,500 steps x 64, seed 42, init S1; 96,035 rows; shared val file `calib/stage2_pilot_val.jsonl`; full detail `STAGE2_ABLATION_JOB_SPEC.md`):
  | arm | data vs pilot C | procedural-style share | est. s/step | est. wall h |
  |---|---|---|---|---|
  | (pilot C, existing result) | - | 17.2 % | 4.008 measured | 1.79 measured |
  | E1 | dedicated procedural (11,750) -> gliclass-v2.0 | 4.9 % | 3.44 | ~1.55 |
  | E1z | E1 and Jev's procedural-style rows excluded | 0 % | 3.18 | ~1.45 |
  | E2 | E1 + 3,000 semantic rows (LocalLLaMA train, 2 workflows) | 4.9 % | 3.37 | ~1.52 |
  | E3 | C + the same 3,000 semantic rows | 17.2 % | 3.92 | ~1.75 |
  | C0 (optional) | rerun of pilot C (noise) | 17.2 % | 4.01 | ~1.79 |
  Estimates scaled from the MEASURED 4.008 s/step of pilot C via `scripts/estimate_ablation_cost.py` on each arm's real data (+/-15 %). Sessions: P1 E1 ~1.8 h; P2 E1+E2 ~3.3 h (recommended once D10 approved; E1+E1z ~3.2 h otherwise); P3 E1,E1z,E2 ~4.7 h; all five ~8.5 h in two sessions.
  Decision rules (set before running): hypothesis supported if E1 or E1z gets choice >= 0.455 and typed >= 0.490 with avg F1 >= 0.650 and many-label flips <= 0.30; not supported if both <= 0.435; grey zone between -> C0 / E1z. E2 helps only if the unseen workflows gain >= 3 pts over E1.
  `eval.py` typed eval now also writes `accuracy_by_workflow` (notebook records it).
- 7.6 Status: configs, recipes, data check, cost estimate, notebook `kaggle_stage2_ablation.ipynb`, job spec and bundle v4 are prepared locally. NOTHING was uploaded or run on Kaggle; no GPU spend; nothing published or pushed.

## Step 8 – length-grouped batching + fast configs (24 Oct 2026, CPU only; nothing run on Kaggle)
Built the seeded, resume-deterministic, DDP-safe mega-batch length sampler recommended in Step 5 (flag `data.length_group`, default off), with planning mode for many-label expansion and a fixed-denominator loss so label-slot weights stay as in random batches. Estimate on 6,000 pilot rows (real tokenizer, micro-batch 4): padding 50.8 % -> 12.9 %, 1.77x fewer tokens (1.94x as DDP cost), implied ~1.2-1.6 s/step vs 3.57 (ESTIMATE). Details: notes/LENGTH_GROUP.md; configs `stage2_fast_{A,B,C,C_nogroup}`; Kaggle speed test spec: STAGE2_FAST_SPEEDTEST_SPEC.md (needs Srimon's approval).

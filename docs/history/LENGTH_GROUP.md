# Length-grouped batching for stage 2 (v3.3, 24 Oct 2026). CPU-built and CPU-tested; NOT run on a GPU or on Kaggle.

## Why
Measured on T4x2 (speed test of 23 Oct): checkpointing off with per-device batch 4 x accum 8 takes 2.40 s/step vs 3.57 s for the pilot setting (batch 8, checkpointing on).
About 65 % of the computed tokens are padding (mean padded length 679 vs mean real length 240) because batches are random. Length grouping attacks that.
The goal is to do it **without changing what the model learns**, and it is **off by default** (`data.length_group` absent or `enabled: false` = old behaviour; tested for the pilot, speed-test and scale configs).

## What it does (`ugc/length_group.py`)
`LengthGroupedSampler(lengths, micro_batch, world_size, mega_batch_mult=50, seed)`; the Trainer is handed it through `StrictTrainer(train_sampler=...)`.
1. Per epoch, from (seed, epoch) only: shuffle all rows, cut into mega-batches of `mega_batch_mult x micro_batch x world` rows, sort each by length.
2. Cut every sorted mega-batch into **groups of `micro_batch x world`** neighbours. Rank r takes rows `[r*mb:(r+1)*mb]` of a group (this is accelerate's own rule: micro-batch i goes to rank i % world). So **both ranks of a DDP step hold similar-length micro-batches** (a step waits for the slower rank).
3. Shuffle the order of the groups (so a step does not see only short or only long rows).
4. The largest multiple of `mb x world` rows is used per epoch; the random tail (< mb x world rows, e.g. < 8) is dropped and differs every epoch (every row is still used about equally often; tested).
Properties (all unit-tested): deterministic in (seed, epoch); no row lost or duplicated within an epoch; identical number of micro-batches on every rank; resume is exact because the order is a pure function of (seed, epoch) and HF's resume only skips whole batches (`accelerate.skip_first_batches`, skipped batches are not fetched); `set_epoch` is called by `StrictTrainer._run_epoch`.
Pairs (consistency loss) are one dataset item that PairedCollator stacks into the same forward pass, so grouping can never separate them (tested).

## Predicting the length when many-label expansion is on
Expansion happens at fetch time with random k (log-uniform 20-150, probability 0.7 for taxonomy pools, 0.1 for glic-v2), so the true length is not known from the stored row. Fix = **planning mode** (`ManyLabelDataset.enable_planning()`):
the sampler yields `epoch * N + row`, the dataset seeds its expansion RNG from (seed, epoch, row), and the sampler replays the first two draws of `expand_row` (expand? then k, see `planned_expansion`) to add `(k - n_labels) x avg_label_tokens[pool]` to the exact tokenised base length (capped at max_length).
Without planning, base-length-only grouping padded 21 %; with planning 12.9 %. The row stream is still random per (seed, epoch, row) and different in every epoch (tested). Because the RNG is now seeded per row instead of per worker, expansions differ from the pilots' draws (same distribution).
Lengths: the whole train file is tokenised once (about 0.4 s per 6,000 rows, so about 6 s for 96k rows) and cached to `<output_dir>/length_cache/lengths_<hash>.npy` by rank 0 (`dist_barrier("length-cache")`); the hash covers n, max_length, tokenizer and ~1,000 evenly spaced rows.

## Loss weighting (`data.length_group.loss_norm`, default true when grouping is on)
gliclass divides the masked loss sum by the number of valid label slots **in the micro-batch**. Random micro-batches have about the same slot count; grouped ones do not (short rows = few labels), which would up-weight rows with few labels.
With `loss_norm`, training uses `loss = model_loss * n_valid / label_norm`, where `label_norm` = expected slots of a random micro-batch, calibrated from 1,536 real fetches (mean slots per row incl. paired second views x micro-batch). Then every label slot has the same weight as in the random-batch runs (to first order). Evaluation is untouched.
**Residual differences (documented, not removed):** (a) the consistency (pair) loss is a per-micro-batch mean over pairs and is not re-normalised; (b) the exact rows in a step differ from the pilots (same data, same seed, same steps, same effective batch 64, not bit-identical); (c) batch statistics are more homogeneous inside a micro-batch (similar lengths): an inherent property of grouping; (d) with mega_batch_mult 50 the within-mega-batch order is not random, so the loss curve can be a bit noisier step to step.

## Upstream alternatives (not used)
`gliclass.training` offers `train_sampling_strategy`: `group_by_length` (HF LengthGroupedSampler: resume order not controlled by us, no DDP-aware neighbour assignment, unaware of expansion) and `batch_rebalance` (variable batch sizes, which would change the effective batch; `GLiClassDataset.estimate_lengths` ignores many-label expansion). Ours keeps fixed micro-batches and effective batch 64.

## Padding estimate (ESTIMATE, not a GPU measurement; `scripts/estimate_padding.py`, `results/padding_estimate_C.json`)
6,000 sampled pilot rows, config C (many-label + pairs 0.25 + augmentation), real tokenizer, micro-batch 4, mega-batch x50, 5 seeds. Mean real length 274 tokens, 10.1 labels per row, 8.4 % of items paired.
| case | padded tokens / micro-batch | padding share |
|---|---|---|
| random batches (pilot) | 2,511 | 50.8 % |
| grouped, planned lengths (what the code does) | 1,420 | 12.9 % |
| grouped, base length only | 1,564 | 21.0 % |
| oracle (true lengths, same mega-batch) | 1,270 | 2.7 % |
Token reduction **1.77x** (1 GPU), **1.94x** counted as DDP cost (max over the 2 ranks: 3,237 -> 1,671 per step-micro-batch). Correlation of predicted vs true length 0.968 (planned) vs 0.942 (base only).
Implied speed (assumes step time proportional to computed tokens, no fixed overhead, no attention-quadratic term; so optimistic): on top of the 2.40 s/step of checkpointing off + b4, about **1.2-1.6 s/step** (2.2-2.9x vs the 3.57 s pilot baseline). Real value to be measured by STAGE2_FAST_SPEEDTEST_SPEC.md.

## Caveats
- Memory: ckpt-off, b4 peaked at 6.7 GB allocated / 7.4 GB reserved (arm A). Arm C adds paired second views, and a grouped micro-batch can be 4 x 1024 tokens (x2 with a pair), so worst case approaches the T1 footprint (b8, no ckpt: 12.2 GB reserved of 15 GB). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (shapes now vary a lot). The speed test measures this before any real run.
- Resume order is identical only for the **same world size**: a 2-GPU to 1-GPU fallback after a checkpoint changes the order of the remaining batches (the data is still valid, nothing is lost or repeated within the epoch's stream).
- `max_oom_skips` / fp16 behaviour unchanged. Not verified on GPU: fp16, NCCL DDP with the sampler, memory, real speed.

## Tests
`scripts/test_length_group.py` (44 checks, ALL PASS), `scripts/test_stage2.py` (existing, ALL PASS), `scripts/test_length_group_e2e.sh` (real `train.py`: mid-epoch resume across 3 epochs gives the same fetched rows as the uninterrupted run; 2-process gloo DDP completes), 6-step CPU smoke train (`configs/stage2_smoke_lg.yaml`). Logs in `smoke_logs/`.

# Stage-2 FAST speed test: spec for the Kaggle Handler

**Status: PREPARED ONLY. Nothing has been uploaded or run on Kaggle, no GPU spent, nothing published or pushed. Needs Srimon's explicit approval before ANY Kaggle use (new code-dataset version, kernel run).**

## 1. Question
Does length-grouped batching (`ugc/length_group.py`) give the speedup the CPU estimate promises (1.77x fewer computed tokens, 1.94x as DDP cost), on real T4x2, **with a stable loss and no OOM**, for the worst-case arm (**C**: many-label + paired views + consistency)? Everything else is already measured (23 Oct): ckpt off + b4 x accum 8 = 2.40 s/step vs 3.57 s baseline.

## 2. The two runs (same session, in this order)
| name | config | what differs |
|---|---|---|
| `fast_C_nogroup` | `configs/speedtest_fast_C_nogroup.yaml` (base `stage2_fast_C_nogroup`) | random batches (the pilot sampler), b4 x accum 8, ckpt off |
| `fast_C_grouped` | `configs/speedtest_fast_C_grouped.yaml` (base `stage2_fast_C`) | the same + `data.length_group {enabled, mega_batch_mult 50, loss_norm}` |
Both: arm-C data (`data/stage2_pilot`, prepared exactly as the pilot / earlier speed test), init = S1 `final_model`, seed 42, 50 steps x 64 rows (effective batch 64 = 4 x 8 x 2 GPUs), LR 1e-5/3e-5 with the pilot schedule (note: max_steps 50, so warmup and decay are compressed; this is a probe, not a quality run), no save, no eval (`data.val_files=null`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `max_oom_skips=0` (OOM fails loudly).
The pilot's own arms A/B (`stage2_fast_A/B.yaml`) are not tested here: A has no expansion (lengths are exact, so grouping can only be better), B is C without pairs (less memory).

## 3. How to run
Use `kaggle_speedtest_fast.ipynb` (built by `scripts/build_speedtest_fast_notebook.py` from the notebook that ran T0-T3; same pre-flight, watchdog, nvidia-smi sampler). GPU T4 x2, Internet on, attach: code dataset **new version** (needs `ugc/length_group.py`, the new configs; `python scripts/make_code_bundle.py <build-box>/ultragliclass_code_v5` builds it, already built, **not uploaded**) + output of kernel `ultragliclass-s1`.
Handler steps (each only after Srimon approves): (1) upload the bundle as a new version of `srimondanguria/ultragliclass-code`; (2) push/run the notebook; (3) fetch `results/speedtest_summary.json` and the logs.

## 4. What the notebook records (per config)
sec/step for every log interval (logging every 5 steps) and the **mean of steps 21-50** (`sec_per_step_steps21_50_mean`); peak allocated/reserved GB on rank 0 AND rank 1 (+ nvidia-smi peak); loss, grad_norm, cons_loss per interval; the `length_group ON ...` and `label-slot normaliser ...` log lines; STATUS state, rc, OOM flag.

## 5. Pass / fail
All must hold, else do **not** start a real run with `stage2_fast_*` and report back:
1. Both runs `state=complete`, rc 0, no OOM, **peak reserved < 13 GB on both ranks** for the grouped run (T4 = 15 GB; worst case b4 x 1024 tokens with pairs).
2. **Speedup** = nogroup / grouped (steps 21-50 mean s/step) **>= 1.4x** (expected 1.6-1.9x; estimate ceiling 1.94x). Between 1.2 and 1.4: still usable, but say so; below 1.2: stop, investigate (rank imbalance? tokenisation of lengths? data loader workers CPU-bound?).
3. **Loss behaves**: no NaN/inf after the first 2 intervals (an early `nan`/`inf` grad_norm from the fp16 GradScaler warm-up also appeared in the earlier speed test and is not a failure); grouped mean loss over steps 25-50 within +-15 % (relative) of nogroup, and not trending up; grad_norm of the same order (within about 2x); `cons_loss` > 0 in both and `pairs_since_last_log` of similar size.
4. The log shows `length_group ON: ... expansion planning=True` and a plausible `label-slot normaliser` (tens to a few hundred slots per micro-batch).
5. Calibration + length caching cost (log timestamps) stays small (< ~2 min). Expected: tokenising 96k rows takes about 6 s on one CPU core, calibration (1,536 fetches) well under a minute.

## 6. Budget
Earlier 4-config speed test used 0.216 session-hours in total (data prep ~0.05 h once). This one: prep ~0.05 h + 2 x (startup + 50 steps at 1.3-2.9 s + load/save ~0.05 h) = **about 0.15-0.2 GPU-h, hard cap 0.3** (`TOTAL_MAX_HOURS = 0.45` wall stop in the notebook, per-config `time_budget_hours` 0.12; no new config starts after 0.25 h).

## 7. What a pass unlocks (still needs a separate approval)
Pilot-sized runs (1,500 steps) with `configs/stage2_fast_{A,B,C}.yaml` at about 1.3-1.6 s/step instead of 3.57 (estimated 0.55-0.7 h per arm instead of ~1.5 h). They are **not bit-identical** to the pilot arms (same data/seed/steps/effective batch, different micro-batch composition and expansion draws), so any later comparison with the pilots must say so; for a clean A/B of grouping alone use `stage2_fast_C_nogroup.yaml` vs `stage2_fast_C.yaml`.

## 8. Known limits (see notes/LENGTH_GROUP.md)
Estimate assumes time ~ tokens; resume order only reproducible at the same world size; pair loss not re-normalised; fp16/NCCL/memory unverified on GPU (CPU: 44 unit checks, e2e resume, 2-process gloo DDP, 6-step smoke all pass).

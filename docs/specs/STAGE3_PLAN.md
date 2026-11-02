# Stage 3 plan: long context (8,192 tokens) + many options (up to 256), 25 Oct 2026

**Why:** S2 full (arm C) ends at suite F1 0.683 and typed 0.616 (eval_loss 0.0775), but it was trained at 1,024 tokens. To show that the
UltraBERT backbone's 32K RoPE and its sliding + global attention are worth having, stage 3 has to beat S2 **on long inputs**
without losing the short-input skills. Srimon (25 Oct): "we still need stage 3 to prove anything right so let's run stage 3".

**What:** one epoch of S2-final -> stage 3 on **61,184 rows** (box dry run, 25 Oct; md5 `d35eb55c…`):
* **21,184 long Jev rows.** These are Jev `full` train rows with states of at least 6,000 chars, `license_use == commercial`, and no CC-BY-SA, GPL or ODbL rows. 300 rows are carved out as val, and train rows that share a state with val are dropped (774 rows).
* **40,000 replay rows.** This is a seeded sample of the S2 full mix, which guards against regressions on the suite and CLINC. The label pools are copied from S2.

Run settings: max_length 8192, max 256 options, many-label k ≤ 250, consistency on (0.25 / 0.1), batch 1 × accum 32 × 2 GPUs = 64 rows per step,
gradient checkpointing on, encoder LR 5e-6 / head LR 1.5e-5, **956 steps**. Configs: `configs/stage3_longctx.yaml`, `configs/data_stage3.yaml`.

**Go / no-go** (decided before training; protocol in `STAGE3_JOB_SPEC.md` §6):
1. Long-Jev slice (Jev test + validation, states ≥ 6k chars, 431 rows / 344 groups, never trained on): stage 3 must be **≥ +3.0 pts** over S2 at 8,192 (paired).
2. CLINC-150 accuracy must be **≥ S2 (0.650)**, using the same quick protocol (500 rows, 1,024 tokens).
3. Suite avg macro-F1 must be **≥ 0.673** (S2 0.683 − 0.01). Typed accuracy is reported as well; the typed LocalLLaMA TEST split stays held out.
A GO needs all three. Otherwise S2 stays the release candidate and stage 3 is reported as a negative or partial result.

**Schedule (Srimon, 25 Oct):**
* **Now, on the current account:** the Kaggle **smoke**, about **1.5 GPU-h** (hard cap 2 h).
  * It uses the REAL data and REAL config; the S2 baseline is computed on the long-Jev slice.
  * It trains for 1 h, about 250 of 956 steps at 8,192 (UNVERIFIED), and checks s/step, memory and NaN.
  * It ends with a checkpoint (`needs_resume`) and a long-Jev eval of that checkpoint, which is a signal only.
* **Later, on Kaggle account #2** (fresh quota, shared by Srimon): the **full** run. It resumes the smoke output to step 956, about 3.3 h EST, or about 4.5 h fresh. Then the go/no-go evals run.
* **No full train on the current account.** About 10 h of quota stays unused after the smoke.
* **Blocker for the full run:** account #2 needs access to code v8, the s2full v4 output and the smoke output (sharing needs Srimon's OK).

**Biggest unknowns:**
* s/step at 8,192 on a T4. The EST is about 14 s, with a range of 9–24 s.
* SDPA memory with the sliding mask.
* Whether 21k long rows are enough for +3 pts.
* The procedural-style share is 13.9 % (S2 full: about 4.9 %). See the job spec, risks.

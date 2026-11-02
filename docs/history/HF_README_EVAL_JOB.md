# HF-README benchmark job: Kaggle Handler instructions (lean, publishable)

Written 2 Nov 2026, 00:05 IST. **Prepared on the box only: nothing uploaded, no Kaggle run started, nothing published.**
Srimon (t48u): "Just go with a proper benchmark that we can paste in the HF readme. Don't make it harder." So this job covers only the
three tables below. BTZSC, QuALITY, ContractNLI, length curves, GoEmotions, ECE tables and larger models are **out of scope**.
Estimates marked UNVERIFIED have not been measured.

## 1. What gets measured (README tables)
| Table | Sets | Metric | Models |
|---|---|---|---|
| **A. Zero-shot suite** | The 14 GLiClass-v3 card sets on **FULL test splits**: CR, sst2, sst5, imdb, 20NG, enron_spam, massive, banking77, financial_phrasebank, ag_news, emotion, cap_sotu, rotten_tomatoes, snips (~82k rows) | macro-F1, avg over 14 sets (headline) + avg over 12 sets (excluding massive and financial_phrasebank, which are in-domain for us, footnoted) + mean over 3 label orders | v0.1, gliclass-modern-base-v3.0, gliclass-base-v3.0 |
| **B. Intent** | Banking77 (77 labels, 3,076) + CLINC-150 in-scope (150 labels, 4,500) | accuracy, paired McNemar v0.1 vs each peer | same |
| **C. Long context** | long-Jev slice `calib/jev_long_test.jsonl` (431 rows, md5 cc664b35…), max_length 8192 | accuracy + cluster CI; paired Δ v0.1 − peer, cluster CI, McNemar | same (base-v3.0 DeBERTa: if it runs out of memory at 8192, it is recorded as OOM and the table says so) |

Also produced: per-set bootstrap CIs, a paired bootstrap CI on the 14- and 12-set average gap (v0.1 − peer), flip rate across label orders, ECE before and after the global temperature (JSON only), and a **reproduction check** (our average for each peer vs its published card average: 0.5577 modern-base, 0.6764 base).

**Protocol** (frozen in `results/hf_readme/protocol.json`). This is eval.py unchanged apart from:
- full test splits;
- batches sorted by length (padding="longest", so inputs are the same);
- OOM handling that halves the batch;
- resumable per (model, set, label order).

The settings:
- 1024 tokens, fp16, single-label softmax argmax, all labels in one pass.
- Label order 0 = canonical (the headline); orders 1–2 are seeded shuffles.
- Tokenizer: ours uses `add_prefix_space=False` (as trained). The Knowledgator peers use `True`, as in their `test_gliclass.py` (knob `PEER_ADD_PREFIX_SPACE`).

Set details:
- cap_sotu has no test split, so its only split is used (23,040 rows), as the official script does.
- financial_phrasebank is the full 2,264-row allagree set.
- massive and banking77 come from mteb parquet mirrors (en test 2,974; banking 3,076 vs the original 3,080).

## 2. Files (box)
- Code bundle v9: `<build-box>/ultragliclass_code_v9.zip` (directory `<build-box>/ultragliclass_code_v9/`; `dataset-metadata.json` id `srimondanguria/ultragliclass-code`). New in v9:
  - `scripts/hf_readme_eval.py` (prep / worker / aggregate)
  - `scripts/build_hf_readme_eval_notebook.py`
  - `kaggle_eval_readme.ipynb`
  - `scripts/eval_long_jev.py` now accepts `--add_prefix_space` (default unchanged)
- Kernel folder ready to push: `<artifacts>/evalreadme_kernel/` (`kernel-metadata.json` + `kaggle_eval_readme.ipynb`).

## 3. Handler steps
1. Upload code **v9** as a new version of the private dataset `srimondanguria/ultragliclass-code` (from `<build-box>/ultragliclass_code_v9/`). Wait until it is ready and confirm that `scripts/hf_readme_eval.py` is listed.
2. Push the kernel **`srimondanguria/ultragliclass-evalreadme`** from `<artifacts>/evalreadme_kernel/`. Settings: private, **GPU T4 ×2**, **internet ON**. Attachments:
   - dataset `srimondanguria/ultragliclass-code` (v9);
   - kernel output `srimondanguria/ultragliclass-s3full`, which holds `runs/stage3_full/final_model` = v0.1.
   - s2full is **not** needed.
3. Peers download from the HF Hub inside the kernel; the commit sha is recorded in `models.json`:
   - `knowledgator/gliclass-modern-base-v3.0`
   - `knowledgator/gliclass-base-v3.0`
4. Do not edit the knobs: `HARD_STOP_HOURS = 5.5`, `PERMUTATIONS = 3`, `SMOKE_ROWS = None`, `PEER_ADD_PREFIX_SPACE = True`.
5. Watch the log lines `progress N/141 by tier {...}`, written every 5 minutes. The headline cells (tiers 0–2) should be done about 1.5–2 h in (UNVERIFIED).
6. When it finishes, download the output to `<artifacts>/EvalReadme/` and report `results/hf_readme/TABLES.md` + `STATUS.json` + the hours used.
7. **Do not publish** anything: no HF push, no public kernel or dataset. Kaggle outputs stay private.

Stop rules:
- Hard stop at 5.5 h session time. Workers stop taking new cells 10 min before; a running cell aborts 1 min before; the aggregate always runs.
- If `STATUS.state == "partial"`, report the `missing` list. A second session with the same output attached is **not** wired up (rerun = start again), so ask before rerunning.

Failure triage:
- `error:` lines are printed from `worker*.jsonl` at the end. One failed cell does not stop the others.
- If more than ±0.03 of the 14-set average for a peer is off vs its card (the reproduction check), report it. Do not rerun.

## 4. Cost (Kaggle quota = session hours; 7.27/30 used, resets 7 Nov 05:30 IST)
**~3.5 h expected, range 2.5–5 h, hard cap 5.5 h. UNVERIFIED.** Basis:
- S3 T4 timings: 500 rows × 3 orders took 4–6 s on short sets and ~50 s on imdb/20NG/enron. CLINC took 45 s. Long-Jev 431 rows @8192 took 145 s.
- One label order over all sets for v0.1 is ≈ 0.45 h on one T4, so ≈ 1.4 h for 3 orders. modern-base is about the same.
- DeBERTa base is assumed 2× slower (≈ 3 h).
- Calibration and long-Jev add ≈ 0.3 h.
- Total ≈ 6.3 single-T4 hours across 2 GPUs (work-stealing queue) ≈ 3.2 h, plus ≈ 0.3 h of setup and downloads.

## 5. Checks done on the box (2 Nov, CPU)
- `hf_readme_eval.py` prep → 2 parallel workers → aggregate. Settings: 6 rows per set, 2 label orders, all 3 models, long-Jev 4 rows @1024. Result: 96/96 cells ok, TABLES.md and summary.json written. These numbers are meaningless; that run only checked the plumbing.
- Notebook glue: all cells run on a fake `/kaggle/input` (code bundle + S3full output), with 4 rows, v0.1 only, and calibration stubbed. Auto-discovery, the long-Jev md5, prep, the worker, the monitor, the aggregate and STATUS all worked.
- Not tested on GPU: fp16 speed, the DeBERTa 8192 memory, and `add_prefix_space=True` on GPU. All three already worked on CPU.

## 6. README footnotes to carry (already printed in TABLES.md)
- massive and financial_phrasebank are in-domain for v0.1 (HWU64 and fin-tweets in training), so the 12-set average is the fair zero-shot number.
- long-Jev is **not zero-shot for v0.1**: other Jev rows are in its training data, with 0 shared states or groups. It is zero-shot for the peers.
- The peers' training mix (gliclass-v2.0 upstream) is undocumented, so no GLiClass model's zero-shot status is fully verified.
- The Knowledgator card summary-table averages differ from their own per-set tables. We compare against the per-set means.
- Licences: eval-only use. Only metrics are published; outputs hold label ids and logits, never texts.

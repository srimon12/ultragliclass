# Proofs index

Every public claim → the file that backs it. Files under `proofs/` are copied verbatim from the Kaggle outputs
(small JSON/TSV/MD only; no dataset text, no logits, no weights). The full artifacts stay off-repo; their box paths
are listed so they can be archived (e.g. as a Kaggle dataset or GitHub Release asset) before launch.

## README benchmark (headline numbers)

| Claim | File in repo | Full artifact (box) |
|---|---|---|
| 14-set suite, 12-set zero-shot average, per-set macro-F1 | `proofs/hf_readme_eval/TABLES.md`, `summary.json` | `/workspace/kaggle/ultragliclass/EvalReadme/ultragliclass/results/hf_readme/` |
| Banking77 / CLINC-150 all-label accuracy + McNemar | `summary.json`, `TABLES.md` | same, `raw/` (29 MB per-row npz logits) |
| Long-Jev@8192: 0.664 vs 0.434 / 0.415, CIs, by kind / length / licence | `proofs/hf_readme_eval/longjev/*.json` (per-row results, no text) | same, `longjev/` |
| Paired bootstrap CIs on avg gaps | `summary.json` | |
| Peer models pinned by Hub commit (`ac369222…`, `77a70e6c…`) | `proofs/hf_readme_eval/models.json` | |
| Protocol (max_length 1024 / 8192, fp16, 3 label orders, seed 0, calib 2,000 rows) | `protocol.json`, `evalsets_manifest.json`, `jobs.json` | |
| Software + GPUs (torch 2.11.0+cu128, transformers 5.20.2, gliclass 0.1.21, 2× T4) | `env.json` | |
| Run complete, nothing missing, 1.217 h | `STATUS.json` | Kaggle kernel `srimondanguria/ultragliclass-evalreadme` v1 |
| Harness reproduces peer card averages (our run 0.5577 / 0.6728 vs published card 0.5577 / 0.6764) | `TABLES.md` (last table) | |

## Training stages

| Claim | File in repo |
|---|---|
| S1: 410,422 rows, 6,413 steps, val loss 0.0305, quick suite 0.586 | `proofs/stage_summaries/s1/{stage1_data_manifest,STATUS,stage1_ultrabert150m_quick}.json` |
| S2: 1,731,505 rows, 865 deduped, 27,000 steps in 2 sessions, val loss 0.0775, quick suite 0.683, CLINC 0.650, typed 0.616 | `proofs/stage_summaries/s2/*.json` |
| S2 data md5 `d82991a3…` (Kaggle = box rebuild) | `proofs/stage_summaries/s2/stage2_full_data_md5.json` |
| S3: 61,184 rows, 0 eval-slice leakage, 956 steps, long-Jev 0.552 → 0.664 (p = 1.9e-7), quick suite 0.689, CLINC 0.640 | `proofs/stage_summaries/s3/*.json` |
| S3 data md5 `d35eb55c…` (Kaggle = box rebuild) | `proofs/stage_summaries/s3/stage3_data_md5.json`, `stage3_data_check.json` |
| GPU-hours per job (41.69 total; 31.18 for the weight lineage) | `proofs/stage_summaries/gpu_hours_log.tsv` |
| ModernBERT (gliclass-modern-base-v3.0) training-time CPU baseline | `proofs/stage_summaries/baselines/stage1_fair_modernbert_v3_cpu.json` |
| Code that produced each stage | `docs/PROVENANCE.md`, `proofs/code_snapshots/` |

## Off-repo artifacts to archive before launch (box paths)

| What | Path |
|---|---|
| v0.1 weights (615 MB) | `/workspace/kaggle/ultragliclass/S3full/final/ultragliclass/runs/stage3_full/final_model/` (sha256 in `docs/TRAINING_RECIPE.md` §3) |
| S2 / S1 final weights | `…/S2full/session2/output/ultragliclass/runs/stage2_full_C/final_model/`, `…/S1/ultragliclass/runs/stage1_ultrabert150m/final_model/` |
| README-eval raw logits (29 MB) | `/workspace/kaggle/ultragliclass/EvalReadme/ultragliclass/results/hf_readme/raw/` |
| Kaggle notebook logs | `…/S1/logs/`, `…/S2full/session2/`, `…/S3full/final/logs/`, `…/EvalReadme/logs/` |

Kaggle kernels (private): `srimondanguria/ultragliclass-s0a`, `-s0b`, `-s1`, `-s2pilot`, `-s2ablation`, `-speedtest`, `-s2full` (v3 + v4),
`-s3smoke`, `-s3full`, `-evalreadme`.

## Known erratum in a proof file

`proofs/hf_readme_eval/TABLES.md` footnote 3 says the long-Jev rows are from the "test split". The slice manifest
(`calib/jev_long_test_manifest.json`) shows **test 177 + validation 254**. The proof file is kept verbatim; public copy
should use the manifest wording.

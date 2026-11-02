# proofs/

Verbatim copies of the small result files written by the Kaggle runs. Nothing here is hand-edited. Index of which
file backs which claim: [`docs/PROOFS.md`](../docs/PROOFS.md).

- `hf_readme_eval/` — the full-split README benchmark (v0.1 vs `gliclass-modern-base-v3.0` and `gliclass-base-v3.0`).
  Erratum: `TABLES.md` footnote 3 says "test split"; the long-Jev slice is test 177 + validation 254 rows
  (`calib/jev_long_test_manifest.json`).
- `stage_summaries/s1|s2|s3/` — data manifests, data md5 checks, run `STATUS.json`, quick-eval and long-Jev results per stage.
- `stage_summaries/gpu_hours_log.tsv` — every Kaggle GPU session with hours and outcome.
- `stage_summaries/baselines/` — training-time CPU baseline of the ModernBERT peer.
- `code_snapshots/` — files of older code versions that differ from this repo (S1 code; S2's v7 `prepare_data.py`; S3's v8 `eval_long_jev.py`).

Not included (size or dataset text): raw logits, gold files, checkpoints, weights. See `docs/PROOFS.md` for their box paths.

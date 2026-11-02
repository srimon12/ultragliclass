# Data

**No dataset rows are committed to this repository.** Training and eval data are rebuilt from the Hugging Face Hub
with the scripts below. Reasons: size (S2 train is ~1.7 M rows), and licences — the Jev "commercial" bucket contains
CC-BY-SA text, some eval sets are non-commercial or unlicensed, and we do not redistribute any of them.

Everything is written under `data/<stage>/` and `calib/`, both git-ignored (except the manifests in `calib/`).

## Build order (run from the repo root, after `pip install -e ".[train]"`)

| Step | Command | Output | Expected (from the v0.1 run) |
|---|---|---|---|
| S1 data | `python scripts/prepare_data.py --recipe configs/data_stage1.yaml` | `data/stage1/{train,val}.jsonl`, `manifest.json` | 410,422 train / 4,300 val; train md5 `a9de950386ab7c99bdd6f1646254f30b` ¹ |
| S1 calib file | `cp data/stage1/val.jsonl calib/stage1_val.jsonl` (or `python scripts/make_calib_val.py --recipe configs/data_stage1.yaml --out_dir data/stage1`) | `calib/stage1_val.jsonl` | md5 `cacbb98152c871dc57d602a35c0c5cd8` (= S1 val) |
| S2 val | `python scripts/make_calib_val.py --recipe configs/data_stage2_pilot.yaml --out_dir data/stage2_pilot && cp data/stage2_pilot/val.jsonl calib/stage2_pilot_val.jsonl` | `calib/stage2_pilot_val.jsonl` | 1,850 rows, md5 `8e5e644ea992bd05503652d507d5591d` ² |
| S2 data | `python scripts/prepare_data.py --recipe configs/data_stage2_full.yaml` | `data/stage2_full/…` (+ `label_pools.json`) | 1,731,505 train; md5 `d82991a305052e585050954f1ac6778f`; ~9 min, ~4.7 GB RAM on CPU |
| Long-Jev eval slice | `python scripts/build_long_jev_slice.py --out calib/jev_long_test.jsonl` ³ | `calib/jev_long_test.jsonl` | 431 rows / 344 groups; md5 `cc664b35683a6703d4a8be15983fd5cb`; manifest in `calib/jev_long_test_manifest.json` |
| S3 data | `python scripts/prepare_data.py --recipe configs/data_stage3.yaml` | `data/stage3/…` | 61,184 train / 300 val; md5 `d35eb55c3fc957d1c951aa6ab57c1028` |

¹ S1 was built by an earlier `prepare_data.py` (snapshot in `proofs/code_snapshots/s1/`). Later versions are designed to
build older recipes byte-identically, but an S1 rebuild with the current code has **not** been checked against this md5. [UNVERIFIED]
² The md5 is of the file the S2 run used. That this exact command reproduces it is **[UNVERIFIED]**; the pilot data
rebuild matched the Kaggle manifest counts (`docs/history/S2_PILOT_INVESTIGATION.md`).
³ The script's default `--split` is `test validation`, which matches the committed manifest (test 177 + validation 254);
its docstring still says "TEST slice" (stale wording). A byte-identical rebuild of the md5 above is [UNVERIFIED].

Hub dataset revisions can drift; if an md5 differs, compare the `manifest.json` per-source counts first.

## Licences in one line each

Full table: [`docs/DATASETS.md`](../docs/DATASETS.md).

- gliclass-v2.0, gliclass-v3-logic: apache-2.0 tags (v2.0 upstream sources undocumented).
- Jev typed decisions: per-row licence; we keep `license_use == commercial` only — **includes CC-BY-SA rows in S1/S2**.
- LocalLLaMA/typed-decisions: apache-2.0, synthetic, **train split only**.
- HWU64, LEDGAR: CC-BY-4.0 (attribution). go_emotions: apache-2.0. fin-news tweets: MIT. (Platform-terms question for
  Reddit/Twitter text is open decision D3 in `docs/PLAN.md`.)
- Bitext, DBpedia: **not used** (share-alike).
- Eval sets: evaluation only; several are non-commercial or unlicensed.

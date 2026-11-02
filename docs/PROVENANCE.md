# Code provenance

Which code produced which artifact. `**same**` = byte-identical (sha256) to the file in this repo.
Sources: S1 = Kaggle `ultragliclass-s1` output; S2 = code bundle v7 (uploaded as the Kaggle `ultragliclass-code` dataset for both S2 full sessions, per `gpu_hours_log.tsv`); S3 = Kaggle `ultragliclass-s3full` output (code v8); README eval = code bundle v9.

| repo file | sha256[:12] in repo | S1 (Kaggle ultragliclass-s1 output) | S2 full (code bundle v7) | S3 (Kaggle ultragliclass-s3full output, code v8) | README eval (code bundle v9) |
|---|---|---|---|---|---|
| `scripts/prepare_data.py` | `2a315fc0e94a` | 2de1d2df29f7 | b44b7c5afdfc | **same** | **same** |
| `scripts/train.py` | `0390066e7cee` | 08fb23ba0a11 | **same** | **same** | **same** |
| `scripts/eval.py` | `d2765f715071` | 2d3e7b9c963c | **same** | **same** | **same** |
| `scripts/eval_long_jev.py` | `ff583b44427f` | — | — | ca23fd708723 | **same** |
| `scripts/hf_readme_eval.py` | `b423d7228d1a` | — | — | — | **same** |
| `src/ugc/__init__.py` | `7621be1a9ebc` | **same** | **same** | **same** | **same** |
| `src/ugc/config.py` | `40393b448c00` | **same** | **same** | **same** | **same** |
| `src/ugc/convert.py` | `b3258862d73b` | 6da48fa7009d | **same** | **same** | **same** |
| `src/ugc/length_group.py` | `28cad6f16797` | — | **same** | **same** | **same** |
| `src/ugc/many_label.py` | `9a82a1864019` | — | **same** | **same** | **same** |
| `src/ugc/metrics.py` | `d11d53aab8fd` | **same** | **same** | **same** | **same** |
| `src/ugc/model_init.py` | `cf33eb12c8d6` | **same** | **same** | **same** | **same** |
| `src/ugc/trainer.py` | `4ee2e40725ee` | 55fd11e41907 | **same** | **same** | **same** |

What this means:

- **Stage 3 (= v0.1 weights) and the README eval ran exactly the code in this repo** (`scripts/eval_long_jev.py` gained an
  `--add_prefix_space` flag in v9 for the peer models; the v8 version is in `proofs/code_snapshots/s3_v8/`).
- **Stage 2** ran the same trainer, model init and `ugc` modules; only `prepare_data.py` differs (v8 added the Stage 3
  options: `exclude_license_contains`, `jsonl` replay, `pools_file_from`). The v7 file is in
  `proofs/code_snapshots/s2_v7/`. The S2 data md5 from a box rebuild matches the Kaggle run.
- **Stage 1** ran an earlier version of `train.py`, `prepare_data.py`, `eval.py`, `ugc/trainer.py`, `ugc/convert.py`
  (before many-label expansion, length grouping and the consistency loss existed). Those five files are in
  `proofs/code_snapshots/s1/`. Configs `base.yaml`, `stage1_pilot.yaml`, `data_stage1.yaml` are byte-identical to the S1 run.

## Changes made while restructuring into this repo

- Layout only: `ugc/` → `src/ugc/` (installable package), top-level `prepare_data.py` / `train.py` / `eval.py` →
  `scripts/`, helper scripts split into `scripts/analysis/`, `scripts/kaggle/`, `tests/`. **No Python file in
  `src/ugc/` or `scripts/` was edited.**
- `tests/test_train_val_dedupe.py`: one `sys.path` line changed to find `scripts/prepare_data.py`.
- Added: `src/ultragliclass/` (inference wrapper + CLI), `pyproject.toml`, docs, proofs.
- The Kaggle notebooks in `notebooks/` are unchanged and expect the original flat bundle layout (see `notebooks/README.md`).

## Configs: byte-identical to the runs

Each lineage config in `configs/` was compared (byte for byte) with the copy inside the Kaggle output of the run that used it:

| config | S1 output | S2 full output | S3 full output |
|---|---|---|---|
| `base.yaml`, `stage1_pilot.yaml`, `data_stage1.yaml` | same | same | same |
| `stage2_pilot_A/B/C.yaml`, `stage2_full_C.yaml`, `data_stage2_full.yaml` | — | same | same |
| `stage3_longctx.yaml`, `data_stage3.yaml` | — | — | same |

(The S1 and S2 outputs also contain earlier drafts of the Stage 3 configs; those were not used.)

## Notebooks

`notebooks/as_pushed/` holds the notebook file pushed to each Kaggle kernel, with its kernel metadata. Where these
differ from the templates in `notebooks/` is listed in `notebooks/README.md`.

## Path placeholders in the docs

The planning and history notes were written on the build machine. Absolute paths there were replaced with placeholders:
`<artifacts>` = downloaded Kaggle kernel outputs, `<dev-tree>` = the original flat working copy of this code,
`<build-box>` = other scratch folders. Python files and configs were not edited, so a few of them still contain
absolute paths as defaults (see `LAUNCH_CHECKLIST.md`).

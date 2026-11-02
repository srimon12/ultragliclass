# S0b rerun: instructions for the Kaggle Handler (v2 notebook, 22 Oct 2026)

Goal: resume Stage 0 from S0a's checkpoint-75 and finish to step 150 (STATUS `complete`). If anything hangs, the
session must end by itself with logs that show where.

## 0. Before the run (needs Srimon's OK: it is an upload)
Re-upload the code dataset `srimondanguria/ultragliclass-code` as a **new version** from
`/workspace/ultrabert/ultragliclass/`. The new code is in `train.py`, `ugc/trainer.py`, `scripts/dist_check.py`,
`scripts/build_notebook.py` and `kaggle_stage1.ipynb`. Exclude `runs/ data/ results/ smoke_logs/ __pycache__/`.

## 1. Notebook
Use the **v2 `kaggle_stage1.ipynb`** from the dataset, not the patched S0a/S0b notebook. Your mount-path fix is now
built in (`CODE_INPUT="auto"`).
- Accelerator: **GPU T4 x2**. Internet: **on**.
- Inputs:
  - `srimondanguria/ultragliclass-code` (new version);
  - the **output of the completed S0a version** (the notebook output that contains
    `ultragliclass/runs/stage0_kaggle/checkpoint-75` and `STATUS.json`).
- Do **not** attach the S0b output (it has none). Do not attach any other old output either: `auto` picks the
  shortest matching path.

## 2. Cell 1 knobs (exact)
```python
STAGE_CONFIG  = "configs/stage0_kaggle.yaml"
DATA_RECIPE   = "configs/data_stage0_kaggle.yaml"
DATA_DIR      = "data/stage0k"
PREP_ARGS     = "--streaming"
CODE_INPUT    = "auto"
PREV_OUTPUT   = "auto"
USE_SINGLE_GPU = False
AUTO_FALLBACK_SINGLE_GPU = True
DIST_CHECK    = True
NCCL_P2P_DISABLE = False
STALL_MINUTES = 15
SESSION_HOURS = 12.0
RESERVE_HOURS = 0.75
QUICK_EVAL    = False
EXTRA_SET     = []
```
Run with **Save & Run All (commit)**, so the output and logs are kept even if training fails.

## 3. What to expect (~0.3 GPU-h)
- Cell 1: `CODE_INPUT = /kaggle/input/.../ultragliclass-code | PREV_OUTPUT = /kaggle/input/.../ultragliclass`.
- Cell 3: `restore: copied .../model.safetensors (587 MB)` and `optimizer.pt (1174 MB)`, then
  `found runs/stage0_kaggle/STATUS.json needs_resume step 75`.
- Cell 5:
  - `DIST_CHECK_OK` within ~1 min.
  - Then the rank-tagged lines `rank0/2` and `rank1/2`: `init_process_group(nccl, timeout=600s) ... done`,
    `barrier 'before-train': passed`, `resume: load model weights / optimizer + scheduler / RNG state ... done`,
    `first step of this session begins at global_step=75`, then `... done: global_step=76`.
  - About 75 steps at ~3.1 s/step, then `STATUS {"state": "complete", "global_step": 150, ...}`.
- If the 2-GPU run hangs, any one of these ends it, and you should report which one fired:
  - the pre-flight fails, retries with P2P off, then drops to 1 GPU;
  - a collective times out after 10 min and torchrun exits with an error;
  - the watchdog kills it after 15 min with no output.

  The notebook then resumes on 1 GPU by itself (log `train_1gpu_fallback_*.log`, ~6 s/step, ~10 min extra).
- If even that fails: rerun once with `USE_SINGLE_GPU = True` (other knobs the same).

## 4. Send back
- Everything in `/kaggle/working/logs/` (`notebook_*.log`, `dist_check_*.log`, `train_ddp_*.log`,
  `train_1gpu*.log`).
- `/kaggle/working/ultragliclass/runs/stage0_kaggle/STATUS.json` and `HEARTBEAT.json`.
- Session GPU hours, and which path finished the run (2-GPU / P2P-off / 1-GPU fallback).
- If it hung: the last 50 lines of the newest `train_*.log`. The last rank-tagged line shows where it hung.

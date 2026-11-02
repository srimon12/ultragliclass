#!/usr/bin/env python
"""CPU smoke of kaggle_stage2_full.ipynb through the REAL notebook code path with Kaggle paths remapped:
  /kaggle/input -> --kin (read-only mock: code dataset v6 + S1 output [+ previous full-run output]),  /kaggle/working -> --kw.
Knobs shrunk: smoke recipe, tiny steps/eval. Session 1 is forced to stop on the time cap (needs_resume) with --budget_h; session 2 resumes.
Usage: python scripts/run_full_notebook_cpu.py --kin /tmp/kin_full --kw /tmp/kw_full --steps 12 [--budget_h 0.01] --out executed.ipynb"""
import argparse, json, nbformat
from nbclient import NotebookClient
ap = argparse.ArgumentParser()
ap.add_argument("--kin", required=True); ap.add_argument("--kw", required=True)
ap.add_argument("--steps", type=int, default=12); ap.add_argument("--budget_h", type=float, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
nb = nbformat.read("kaggle_stage2_full.ipynb", as_version=4)
extra = [f"train.max_steps={a.steps}", "train.per_device_train_batch_size=2", "train.gradient_accumulation_steps=1", "data.max_length=384",
         "data.max_eval_samples=16", "train.save_steps=2", "train.eval_steps=4", "train.logging_steps=1", "train.save_total_limit=2"]
if a.budget_h:
    extra.append(f"train.time_budget_hours={a.budget_h}")      # applied after the notebook's own time budget -> forces needs_resume
rep = {'DATA_RECIPE   = "configs/data_stage2_full.yaml"': 'DATA_RECIPE = "configs/data_stage2_full_smoke.yaml"',
       'PREP_ARGS     = ""': 'PREP_ARGS = "--streaming"',
       "EVAL_MAX_PER_DATASET = 500": "EVAL_MAX_PER_DATASET = 6", "EVAL_PERMS    = 3": "EVAL_PERMS = 2",
       "EVAL_TYPED_MAX_CASES = None": "EVAL_TYPED_MAX_CASES = 4", "EVAL_CALIB_N  = 2000": "EVAL_CALIB_N = 30",
       "EXTRA_SET     = []": f"EXTRA_SET = {json.dumps(extra)}"}
for c in nb.cells:
    if c.cell_type != "code":
        continue
    src = c.source
    for k, v in rep.items():
        src = src.replace(k, v)
    c.source = src.replace("/kaggle/input", a.kin).replace("/kaggle/working", a.kw)
NotebookClient(nb, timeout=7200, kernel_name="python3").execute()
nbformat.write(nb, a.out)
print("notebook executed OK ->", a.out)

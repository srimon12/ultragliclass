#!/usr/bin/env python
"""CPU smoke of kaggle_stage3.ipynb through the REAL notebook code path, Kaggle paths remapped:
  /kaggle/input -> --kin (mock: code bundle copy + the S2 full output),  /kaggle/working -> --kw.
Shrunk: max_length 512 (train + long-Jev eval), accum 2. smoke: real recipe, tiny --budget_h forces needs_resume;
full: attach the smoke output under --kin, resumes, stops at --steps.
Usage: python scripts/run_stage3_notebook_cpu.py --kin /tmp/kin_s3 --kw /tmp/kw_s3 --mode smoke --out /tmp/s3_nb_smoke.ipynb"""
import argparse, json, nbformat
from nbclient import NotebookClient
ap = argparse.ArgumentParser()
ap.add_argument("--kin", required=True); ap.add_argument("--kw", required=True)
ap.add_argument("--mode", default="smoke"); ap.add_argument("--steps", type=int, default=2); ap.add_argument("--out", required=True)
ap.add_argument("--budget_h", type=float, default=0.004)   # smoke: forces needs_resume after a few CPU steps
a = ap.parse_args()
nb = nbformat.read("kaggle_stage3.ipynb", as_version=4)
extra = ([f"train.max_steps={a.steps}"] if a.mode == "full" else [f"train.time_budget_hours={a.budget_h}"]) + ["train.gradient_accumulation_steps=2", "data.max_length=512", "data.max_eval_samples=8", "train.logging_steps=1"]
rep = {'MODE          = "smoke"': f'MODE = "{a.mode}"', 
       "LONGJEV_MAXLEN = 8192": "LONGJEV_MAXLEN = 512", "EVAL_MAX_PER_DATASET = 500": "EVAL_MAX_PER_DATASET = 6",
       "EVAL_PERMS    = 3": "EVAL_PERMS = 2", "EVAL_TYPED_MAX_CASES = None": "EVAL_TYPED_MAX_CASES = 4",
       "EVAL_CALIB_N  = 2000": "EVAL_CALIB_N = 30", "EXTRA_SET     = []": f"EXTRA_SET = {json.dumps(extra)}",
       "--batch_size 1 --dtype fp16": "--batch_size 4 --dtype fp32"}
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

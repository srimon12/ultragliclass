#!/usr/bin/env python
"""Run kaggle_stage2_pilot.ipynb end to end on CPU with Kaggle paths remapped (smoke test of the notebook logic).
  /kaggle/input -> /tmp/kin   (read-only mock: code dataset + the S1 kernel output, built by hand, see SMOKE_TEST.md)
  /kaggle/working -> /tmp/kw
Knobs are shrunk (tiny streamed recipe, 4 steps per arm, tiny eval). Usage: python scripts/run_notebook_cpu.py [--arms A B C]"""
import argparse, json, re, nbformat
from nbclient import NotebookClient
ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
ap.add_argument("--nb", default="kaggle_stage2_pilot.ipynb")
ap.add_argument("--out", default="/tmp/kw/executed_pilot.ipynb")
ap.add_argument("--extra", default=None, help="python list literal replacing EXTRA_SET")
ap.add_argument("--prev", default=None, help="PREV_OUTPUT value")
ap.add_argument("--steps", type=int, default=4)
a = ap.parse_args()
nb = nbformat.read(a.nb, as_version=4)
extra = a.extra or json.dumps([f"train.max_steps={a.steps}", "train.per_device_train_batch_size=2", "train.gradient_accumulation_steps=2", "data.max_length=512",
                               "data.max_eval_samples=24", f"train.save_steps={max(a.steps // 2, 1)}", f"train.eval_steps={max(a.steps // 2, 1)}", "train.logging_steps=1",
                               "train.gradient_checkpointing=false", "train.num_workers=2"])
rep = {
    'ARMS          = ["A", "B", "C"]': f"ARMS = {json.dumps(a.arms)}",
    'DATA_RECIPE   = "configs/data_stage2_pilot.yaml"': 'DATA_RECIPE = "configs/data_stage2_pilot_smoke.yaml"',
    'PREP_ARGS     = ""': 'PREP_ARGS = "--streaming --out_dir data/stage2_pilot"',
    "EVAL_MAX_PER_DATASET = 500": "EVAL_MAX_PER_DATASET = 6", "EVAL_PERMS    = 3": "EVAL_PERMS = 2",
    "EVAL_TYPED_MAX_CASES = None": "EVAL_TYPED_MAX_CASES = 4", "EVAL_CALIB_N  = 2000": "EVAL_CALIB_N = 30",
    "EXTRA_SET     = []": f"EXTRA_SET = {extra}",
}
if a.prev:
    rep["PREV_OUTPUT   = None"] = f"PREV_OUTPUT = {a.prev!r}"
for c in nb.cells:
    if c.cell_type != "code":
        continue
    src = c.source
    for k, v in rep.items():
        src = src.replace(k, v)
    c.source = src.replace("/kaggle/input", "/tmp/kin").replace("/kaggle/working", "/tmp/kw")
NotebookClient(nb, timeout=7200, kernel_name="python3").execute()
nbformat.write(nb, a.out)
print("notebook executed OK ->", a.out)

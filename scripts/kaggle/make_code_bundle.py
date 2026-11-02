#!/usr/bin/env python
"""Build the Kaggle code-dataset bundle (private dataset srimondanguria/ultragliclass-code, version 9 = + HF-README benchmark: scripts/hf_readme_eval.py, kaggle_eval_readme.ipynb).
Everything EXCEPT runs/ data/ results/ smoke_logs/ logs_fair/ __pycache__ (and *.pyc, *.zip). Run from the project root.
Output: <out> (a directory: its root = the dataset root) and <out>.zip   (default /workspace/ultrabert/ultragliclass_code_v3)"""
import json, os, shutil, sys, zipfile
out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ultrabert/ultragliclass_code_v9"
skip_dirs = {"runs", "data", "results", "smoke_logs", "logs_fair", "__pycache__"}
shutil.rmtree(out, ignore_errors=True)
if os.path.exists(out + ".zip"):
    os.remove(out + ".zip")
n = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in skip_dirs] if True else dirs
    rel = os.path.relpath(root, ".")
    for f in files:
        if f.endswith((".pyc", ".zip")) or f == ".write_probe":
            continue
        dst = os.path.join(out, rel, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(root, f), dst); n += 1
json.dump({"title": "ultragliclass-code", "id": "srimondanguria/ultragliclass-code", "licenses": [{"name": "other"}]},
          open(os.path.join(out, "dataset-metadata.json"), "w"))
with zipfile.ZipFile(out + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(out):
        for f in files:
            p = os.path.join(root, f); z.write(p, os.path.relpath(p, out))
print(f"bundle {out}: {n + 1} files; zip {out}.zip {os.path.getsize(out + '.zip') / 1e6:.2f} MB")

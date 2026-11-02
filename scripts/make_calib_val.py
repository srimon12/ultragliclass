#!/usr/bin/env python
"""Rebuild ONLY val.jsonl of a data recipe (the global-temperature calibration file) without the huge train
downloads. Identical to prepare_data.py for the validation rows: the carved-out val rows are the first n_val of the
seeded shuffle and do not depend on n_train; for sources with a separate val_split only that split is read.

    python scripts/make_calib_val.py --recipe configs/data_stage1.yaml --out_dir data/stage1
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_data import take  # noqa: E402
from ugc.config import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
recipe = load_config(a.recipe, [])
seed = recipe.get("seed", 42)
defaults = {"max_options": recipe.get("max_options", 50), "max_chars": recipe.get("max_chars", 8000)}
val, seen = [], set()
for src in recipe["sources"]:
    n_val = src.get("n_val", 0)
    n_train = src.get("n_train")
    if src.get("val_split"):
        val += take(src, src["val_split"], n_val, seed, False, defaults, seen, src.get("max_scan")) if n_val else []
    else:  # must also consume the train rows, so `seen` (dedupe) matches the full prepare_data run
        need = None if n_train in (None, "all") else int(n_train) + n_val
        rows = take(src, src.get("split", "train"), need, seed, False, defaults, seen, src.get("max_scan"))
        val += rows[:n_val]
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "val.jsonl"), "w") as f:
    for ex in val:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print("val rows", len(val))

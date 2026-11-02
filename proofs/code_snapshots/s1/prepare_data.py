#!/usr/bin/env python
"""Build a stage's training mix in GLiClass JSONL format from a data recipe YAML.

    python prepare_data.py --recipe configs/data_stage1.yaml
    python prepare_data.py --recipe configs/data_stage0.yaml --streaming     # tiny, no full downloads

Writes <out_dir>/train.jsonl, <out_dir>/val.jsonl and <out_dir>/manifest.json (counts, licenses, seed).
Deterministic for a given recipe + seed, so a resumed Kaggle session rebuilds the identical file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugc.config import load_config  # noqa: E402
from ugc.convert import from_gliclass, from_jev, from_system_one, text_hash  # noqa: E402


def iter_source(src: dict, split: str, seed: int, streaming: bool):
    kwargs = {"split": split}
    if src.get("config"):
        kwargs["name"] = src["config"]
    if streaming:
        ds = load_dataset(src["id"], streaming=True, **kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=src.get("shuffle_buffer", 2000))
    else:
        ds = load_dataset(src["id"], **kwargs)
        ds = ds.shuffle(seed=seed)
    return iter(ds)


def convert(src: dict, row: dict, defaults: dict):
    max_options = src.get("max_options", defaults["max_options"])
    max_chars = src.get("max_chars", defaults["max_chars"])
    if src["format"] == "gliclass":
        return from_gliclass(row, src["id"], max_options, max_chars)
    min_state = src.get("min_state_chars")
    if min_state and len(str(row.get("state") or row.get("text") or "")) < min_state:
        return None  # stage 3: keep only long contexts
    if src["format"] == "jev":
        allowed = src.get("license_use")
        if allowed and row.get("license_use") not in allowed:
            return None
        kinds = src.get("kinds")
        if kinds and row.get("kind") not in kinds:
            return None
        return from_jev(row, src["id"], max_options, max_chars, src.get("max_state_chars"))
    if src["format"] == "system_one":  # state + questions + answers (procedural-typed-decisions)
        if src.get("max_state_chars") and len(row.get("state") or "") > src["max_state_chars"]:
            return None
        return from_system_one(row["state"], row["questions"], row["answers"], src["id"], max_options, max_chars)
    raise ValueError(f"unknown format {src['format']}")


def take(src: dict, split: str, n: int | None, seed: int, streaming: bool, defaults: dict,
         exclude: set[str], max_scan: int | None) -> list[dict]:
    out, scanned = [], 0
    for row in iter_source(src, split, seed, streaming):
        scanned += 1
        if max_scan and scanned > max_scan:
            break
        converted = convert(src, row, defaults)
        if converted is None:
            continue
        for ex in converted if isinstance(converted, list) else [converted]:
            key = text_hash(ex["text"] + ex.get("prompt", ""), ex["all_labels"])
            if key in exclude:
                continue
            exclude.add(key)
            out.append(ex)
        if n is not None and len(out) >= n:
            out = out[:n]
            break
    print(f"  {src['id']} [{split}]: kept {len(out)} of {scanned} scanned", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--streaming", action="store_true", help="stream sources (smoke tests / low disk)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. sources.0.n_train=100")
    args = ap.parse_args()

    recipe = load_config(args.recipe, [s for s in args.set if not s.startswith("sources.")])
    for item in args.set:  # sources.<i>.<key>=value overrides
        if item.startswith("sources."):
            key, _, val = item.partition("=")
            _, idx, field = key.split(".", 2)
            recipe["sources"][int(idx)][field] = int(val) if val.isdigit() else val
    out_dir = args.out_dir or recipe["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = recipe.get("seed", 42)
    defaults = {"max_options": recipe.get("max_options", 50), "max_chars": recipe.get("max_chars", 8000)}
    streaming = args.streaming or recipe.get("streaming", False)

    t0 = time.time()
    train, val, seen = [], [], set()
    per_source = {}
    for src in recipe["sources"]:
        print(f"source {src['id']}", flush=True)
        n_val = src.get("n_val", 0)
        n_train = src.get("n_train")
        n_train = None if n_train in (None, "all") else int(n_train)
        max_scan = src.get("max_scan")
        if src.get("val_split"):
            v = take(src, src["val_split"], n_val, seed, streaming, defaults, seen, max_scan) if n_val else []
            t = take(src, src.get("split", "train"), n_train, seed, streaming, defaults, seen, max_scan)
        else:  # carve validation out of the same shuffled split
            need = None if n_train is None else n_train + n_val
            rows = take(src, src.get("split", "train"), need, seed, streaming, defaults, seen, max_scan)
            v, t = rows[:n_val], rows[n_val:]
        repeat = int(src.get("repeat", 1))
        train.extend(t * repeat)
        val.extend(v)
        per_source[f"{src['id']}:{src.get('config') or 'default'}"] = {
            "train_unique": len(t), "repeat": repeat, "train_rows": len(t) * repeat, "val_rows": len(v),
            "license": src.get("license"), "kinds": dict(Counter(x["kind"] for x in t)),
        }

    rng = random.Random(seed)
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    n_labels = [len(x["all_labels"]) for x in train] or [0]
    manifest = {
        "recipe": os.path.abspath(args.recipe), "name": recipe.get("name"), "seed": seed, "streaming": streaming,
        "train_rows": len(train), "val_rows": len(val), "sources": per_source,
        "kinds": dict(Counter(x["kind"] for x in train)),
        "labels_per_row": {"max": max(n_labels), "mean": round(sum(n_labels) / len(n_labels), 2)},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

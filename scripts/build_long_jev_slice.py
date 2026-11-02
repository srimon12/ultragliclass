#!/usr/bin/env python
"""Stage-3 go/no-go bar (1): the long-state Jev TEST slice.

Rows of tasksource/tasksource-jev-typed-decisions, config `full`, split `test` (15,000 rows, ~9.8 MB download), whose `state`
has >= --min_state_chars characters (default 6,000), converted with the SAME converter as training (ugc.convert.from_jev):
soft targets, question as prompt, options as labels (<= 256), state capped at --max_chars (30,000 = the stage-3 recipe).
EVAL ONLY: never a training source (the stage-3 recipe trains on `train`, validates on `validation`).
All license_use buckets are kept by default (evaluating on non-commercial rows is fine; see DATASETS.md); --license_use
commercial restricts it. Rows whose gold argmax is tied are dropped (accuracy = argmax vs gold argmax is undefined there).

    python scripts/build_long_jev_slice.py --out calib/jev_long_test.jsonl
    python scripts/build_long_jev_slice.py --parquet /path/test.parquet ...      # offline (box)
    python scripts/build_long_jev_slice.py --check_train data/stage3/train.jsonl  # + leak check vs a train file
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ugc.convert import from_jev  # noqa: E402


def norm_hash(s: str) -> str:
    return hashlib.md5(" ".join(str(s).split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib/jev_long_test.jsonl")
    ap.add_argument("--parquet", nargs="*", default=None, help="local parquet(s) of the full splits (else load_dataset)")
    ap.add_argument("--split", nargs="+", default=["test", "validation"],
                    help="default: test + validation (~430 long rows; test alone is ~175, too few for a 3-pt bar)")
    ap.add_argument("--min_state_chars", type=int, default=6000)
    ap.add_argument("--max_chars", type=int, default=30000)
    ap.add_argument("--max_options", type=int, default=256)
    ap.add_argument("--license_use", nargs="*", default=None, help="restrict, e.g. commercial (default: all)")
    ap.add_argument("--check_train", nargs="*", default=[], help="train jsonl(s): drop + count slice rows whose state appears there")
    args = ap.parse_args()

    rows = []
    if args.parquet:
        import pyarrow.parquet as pq
        for p in args.parquet:
            sp = os.path.basename(p).split("_")[0].split("-")[0]
            rows += [{**r, "_split": sp} for r in pq.read_table(p).to_pylist()]
    else:
        from datasets import load_dataset
        for sp in args.split:
            rows += [{**r, "_split": sp} for r in load_dataset("tasksource/tasksource-jev-typed-decisions", "full", split=sp)]
    n_split = len(rows)
    train_states = set()
    for p in args.check_train:
        with open(p) as f:
            for line in f:
                ex = json.loads(line)
                if str(ex.get("kind", "")).startswith("jev") or len(ex["text"]) >= args.min_state_chars // 2:
                    train_states.add(norm_hash(ex["text"]))
    out, why = [], Counter()
    for r in rows:
        st = r.get("state") or ""
        if len(st) < args.min_state_chars:
            why["short"] += 1; continue
        if args.license_use and r.get("license_use") not in args.license_use:
            why["license"] += 1; continue
        ex = from_jev(r, "tasksource/tasksource-jev-typed-decisions:full", args.max_options, args.max_chars)
        if ex is None:
            why["unconvertible"] += 1; continue
        gold = [float(ex["true_labels"].get(l, 0.0)) for l in ex["all_labels"]]
        if sorted(gold, reverse=True)[:2].count(max(gold)) > 1:
            why["gold_tie"] += 1; continue
        if train_states and norm_hash(ex["text"]) in train_states:
            why["state_in_train"] += 1; continue
        ex.update({"gold_probs": gold, "state_chars": len(st), "id": r.get("id"), "group_id": r.get("group_id"),
                   "jev_source": r.get("source"), "jev_split": r["_split"], "kind_raw": r.get("kind"), "license_use_row": r.get("license_use")})
        out.append(ex)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    man = {"split": args.split, "split_rows": n_split, "min_state_chars": args.min_state_chars, "kept": len(out),
           "dropped": dict(why), "by_kind": dict(Counter(x["kind_raw"] for x in out)), "by_split": dict(Counter(x["jev_split"] for x in out)),
           "by_license_use": dict(Counter(x["license_use_row"] for x in out)),
           "distinct_groups": len({x["group_id"] for x in out}), "distinct_states": len({norm_hash(x["text"]) for x in out}),
           "n_options": dict(Counter(min(len(x["all_labels"]), 999) for x in out).most_common(10)),
           "top_sources": dict(Counter(x["jev_source"] for x in out).most_common(15)),
           "leak_check_train_files": args.check_train}
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Side-by-side table of two eval.py result files (S1 vs fair baseline). Marks cells that are not comparable
(different n, different dtype/device handled in the header note). Usage:
  python scripts/compare_results.py A.json B.json [--names UltraBERT-S1 ModernBERT-v3] [--md out.md] [--json out.json]"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--names", nargs=2, default=["A", "B"])
ap.add_argument("--md"); ap.add_argument("--json")
args = ap.parse_args()
A, B = json.load(open(args.a)), json.load(open(args.b))
na, nb = args.names
rows, md = [], []
f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
md.append(f"| dataset | C | n | F1 {na} | F1 {nb} | acc {na} | acc {nb} | flip {na} | flip {nb} | ECE@T {na} | ECE@T {nb} |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
sa = sb = fa = fb = 0; k = 0
for name, da in A["datasets"].items():
    db = B["datasets"].get(name)
    if db is None:
        continue
    note = "" if da["n"] == db["n"] else f" (n differs: {da['n']} vs {db['n']}; NOT COMPARABLE)"
    rows.append({"dataset": name, "C": da["n_classes"], "n": da["n"], na: da, nb: db, "note": note})
    md.append(f"| {name}{note} | {da['n_classes']} | {da['n']} | {f(da['macro_f1'])} | {f(db['macro_f1'])} | {f(da['accuracy'])} | "
              f"{f(db['accuracy'])} | {f(da['perm_flip_rate'])} | {f(db['perm_flip_rate'])} | {f(da['ece_global_T'])} | {f(db['ece_global_T'])} |")
    if not note:
        sa += da["macro_f1"]; sb += db["macro_f1"]; fa += da["perm_flip_rate"]; fb += db["perm_flip_rate"]; k += 1
md.append(f"| **mean over {k} comparable sets** | | | {f(sa / k)} | {f(sb / k)} | | | {f(fa / k)} | {f(fb / k)} | | |")
md.append("")
ta, tb = A["typed_decisions"], B["typed_decisions"]
md.append(f"| typed decisions (n={ta['T1']['decisions']}) | {na} | {nb} |")
md.append("|---|---|---|")
for lab, key in (("accuracy", "accuracy"), ("KL from gold", "kl_from_gold"), ("Brier", "brier"), ("ECE", "ece")):
    md.append(f"| {lab}, T=1 | {f(ta['T1'][key])} | {f(tb['T1'][key])} |")
    md.append(f"| {lab}, global T | {f(ta['global_T'][key])} | {f(tb['global_T'][key])} |")
for t in ("choice", "noul", "score"):
    md.append(f"| accuracy[{t}] | {f(ta['T1']['accuracy_by_type'].get(t))} | {f(tb['T1']['accuracy_by_type'].get(t))} |")
md.append(f"| global T fitted | {f(A['global_T'])} | {f(B['global_T'])} |")
md.append(f"| dtype / device | {A['dtype']} / {A['device']} | {B['dtype']} / {B['device']} |")
out = "\n".join(md)
print(out)
if args.md:
    open(args.md, "w").write(out + "\n")
if args.json:
    json.dump({"rows": rows, "mean_f1": {na: sa / k, nb: sb / k}, "mean_flip": {na: fa / k, nb: fb / k}, "typed": {na: ta, nb: tb}},
              open(args.json, "w"), indent=2)

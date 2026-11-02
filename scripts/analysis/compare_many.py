#!/usr/bin/env python
"""N-way table of eval.py result files. Usage:
  python scripts/compare_many.py --md notes/V4_COMPARISON_TABLE.md NAME=path.json[:params_millions] ...
Mean F1 / flip are over the 14 'gliclass' suite sets (same as eval.py's gliclass_avg_macro_f1); intent-only sets listed separately."""
import argparse, json
SUITE = ["SetFit/CR","SetFit/sst2","SetFit/sst5","stanfordnlp/imdb","SetFit/20_newsgroups","SetFit/enron_spam","massive",
         "banking77","financial_phrasebank","ag_news","dair-ai/emotion","MoritzLaurer/cap_sotu","rotten_tomatoes","snips"]
ap = argparse.ArgumentParser(); ap.add_argument("items", nargs="+"); ap.add_argument("--md"); a = ap.parse_args()
names, R, P = [], {}, {}
for it in a.items:
    n, rest = it.split("=", 1); p = None
    if ":" in rest: rest, p = rest.rsplit(":", 1)
    names.append(n); R[n] = json.load(open(rest)); P[n] = p
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
m = lambda xs: sum(xs) / len(xs) if xs else None
out = []
out.append("| metric | " + " | ".join(names) + " |"); out.append("|---|" + "---|" * len(names))
def row(label, fn): out.append(f"| {label} | " + " | ".join(fn(R[n], n) for n in names) + " |")
row("params (M)", lambda r, n: P[n] or "?")
row("avg macro-F1 (14 sets)", lambda r, n: f(m([r["datasets"][d]["macro_f1"] for d in SUITE if d in r["datasets"]])))
row("avg accuracy (14 sets)", lambda r, n: f(m([r["datasets"][d]["accuracy"] for d in SUITE if d in r["datasets"]])))
row("avg label-order flip rate (14 sets)", lambda r, n: f(m([r["datasets"][d]["perm_flip_rate"] for d in SUITE if d in r["datasets"]])))
row("avg ECE, global T (14 sets)", lambda r, n: f(m([r["datasets"][d]["ece_global_T"] for d in SUITE if d in r["datasets"]])))
row("avg ECE, T=1 (14 sets)", lambda r, n: f(m([r["datasets"][d]["ece"] for d in SUITE if d in r["datasets"]])))
row("typed-decision accuracy", lambda r, n: f(r["typed_decisions"]["T1"]["accuracy"]))
for t in ("choice", "noul", "score"):
    row(f"  typed acc [{t}]", lambda r, n, t=t: f(r["typed_decisions"]["T1"]["accuracy_by_type"].get(t)))
row("typed ECE, T=1", lambda r, n: f(r["typed_decisions"]["T1"]["ece"]))
row("typed ECE, global T", lambda r, n: f(r["typed_decisions"]["global_T"]["ece"]))
row("typed KL from gold, global T", lambda r, n: f(r["typed_decisions"]["global_T"]["kl_from_gold"]))
row("typed Brier, global T", lambda r, n: f(r["typed_decisions"]["global_T"]["brier"]))
row("global T fitted", lambda r, n: f(r.get("global_T")))
row("dtype / device", lambda r, n: f"{r['dtype'].replace('torch.','')} / {r['device']}")
row("latency p50 ms/example", lambda r, n: f(r.get("latency_ms_per_example_p50"), 1))
out.append("")
out.append("Per-dataset macro-F1 (flip rate in brackets), n=500 each, 3 label orders:")
out.append(""); out.append("| dataset | C | " + " | ".join(names) + " |"); out.append("|---|---|" + "---|" * len(names))
allsets = SUITE + [d for d in R[names[0]]["datasets"] if d not in SUITE]
for d in allsets:
    c = next((R[n]["datasets"][d]["n_classes"] for n in names if d in R[n]["datasets"]), "?")
    cells = []
    for n in names:
        x = R[n]["datasets"].get(d)
        cells.append("-" if x is None else f"{x['macro_f1']:.3f} ({x['perm_flip_rate']:.2f})")
    tag = "" if d in SUITE else " *(intent suite only, not in avg)*"
    out.append(f"| {d}{tag} | {c} | " + " | ".join(cells) + " |")
txt = "\n".join(out); print(txt)
if a.md: open(a.md, "w").write(txt + "\n")

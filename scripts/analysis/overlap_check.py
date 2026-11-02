"""Overlap check: pilot train mix vs eval sets. CPU only.  python scripts/overlap_check.py <train.jsonl> <eval_sets.json> <out.json>"""
import json, re, sys, collections, ast
train_f, eval_f, out_f = sys.argv[1:4]
ev = json.load(open(eval_f))
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
def shingles(n, k=3):
    w = n.split()
    if len(w) < k: return {hash(n)}
    return {hash(" ".join(w[i:i+k])) for i in range(len(w)-k+1)}
# eval indexes
full_norm = {}      # norm text -> set of datasets (full eval sets)
sub_norm = {}
sub_items = []      # (dataset, norm, shingles)
for d, v in ev.items():
    for t in v["full"]: full_norm.setdefault(norm(t), set()).add(d)
    for t in v["sub500"]:
        n = norm(t); sub_norm.setdefault(n, set()).add(d)
        sub_items.append((d, n, shingles(n)))
inv = collections.defaultdict(list)
for i, (_, _, sh) in enumerate(sub_items):
    for h in sh: inv[h].append(i)
inv = {h: l for h, l in inv.items() if len(l) <= 30}
rows = [json.loads(l) for l in open(train_f)]
res = {"exact_full": collections.Counter(), "exact_sub500": collections.Counter(), "near_sub500_j>=0.5": collections.Counter(),
       "near_sub500_j>=0.8": collections.Counter(), "train_rows_by_src": collections.Counter()}
examples = collections.defaultdict(list)
matched_eval = collections.defaultdict(set)   # dataset -> set of sub500 norm texts hit (>=0.5)
for r in rows:
    src = r["source"].split(":")[0]; res["train_rows_by_src"][src] += 1
    n = norm(r["text"])
    if len(n) < 8: continue
    for d in full_norm.get(n, ()):
        res["exact_full"][(src, d)] += 1
    for d in sub_norm.get(n, ()):
        res["exact_sub500"][(src, d)] += 1
    if len(n) > 4000: n = n[:4000]
    sh = shingles(n)
    cnt = collections.Counter()
    for h in sh:
        for i in inv.get(h, ()): cnt[i] += 1
    best = {}
    for i, c in cnt.most_common(5):
        d, en, esh = sub_items[i]
        j = c / (len(sh) + len(esh) - c)
        if j >= 0.5 and (d not in best or j > best[d][0]): best[d] = (j, en)
    for d, (j, en) in best.items():
        res["near_sub500_j>=0.5"][(src, d)] += 1; matched_eval[d].add(en)
        if j >= 0.8: res["near_sub500_j>=0.8"][(src, d)] += 1
        if len(examples[(src, d)]) < 3: examples[(src, d)].append((round(j, 2), r["text"][:160], en[:160]))
out = {k: {f"{a}|{b}": c for (a, b), c in v.items()} if k != "train_rows_by_src" else dict(v) for k, v in res.items()}
out["distinct_sub500_hit_by_near>=0.5"] = {d: len(s) for d, s in matched_eval.items()}
out["examples"] = {f"{a}|{b}": e for (a, b), e in examples.items()}
json.dump(out, open(out_f, "w"), indent=1)
for k in ("exact_full", "exact_sub500", "near_sub500_j>=0.5", "near_sub500_j>=0.8"):
    print(k, json.dumps(out[k], indent=0)[:1500])
print("distinct", out["distinct_sub500_hit_by_near>=0.5"])

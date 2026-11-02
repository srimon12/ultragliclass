"""Containment scan of ALL of knowledgator/gliclass-v2.0 (1.196M rows) against the 500-sample eval items. CPU only."""
import json, re, sys, collections
from datasets import load_dataset
ev = json.load(open("/workspace/ultrabert/s2inv/eval_sets.json"))
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
def sh(n, k=3):
    w = n.split()
    return {hash(" ".join(w[i:i+k])) for i in range(len(w)-k+1)} if len(w) >= k else None
items = []
for d, v in ev.items():
    for t in v["sub500"]:
        n = norm(t); s = sh(n)
        if s and len(s) >= 3: items.append((d, n, s))     # eval texts with >= 5 words
inv = collections.defaultdict(list)
for i, (_, _, s) in enumerate(items):
    for h in s: inv[h].append(i)
inv = {h: l for h, l in inv.items() if len(l) <= 20}
ds = load_dataset("knowledgator/gliclass-v2.0", split="train")
hit = collections.defaultdict(dict)   # dataset -> {eval idx: (containment,row idx)}
for ri, r in enumerate(ds):
    n = norm(r["text"]); s = sh(n[:6000])
    if not s: continue
    cnt = collections.Counter()
    for h in s:
        for i in inv.get(h, ()): cnt[i] += 1
    for i, c in cnt.items():
        cont = c / len(items[i][2])
        if cont >= 0.8 and len(n) <= 2*len(items[i][1]) + 200:
            d = items[i][0]
            if i not in hit[d] or cont > hit[d][i][0]: hit[d][i] = (cont, ri)
    if ri % 200000 == 0: print(ri, {d: len(v) for d, v in hit.items()}, flush=True)
out = {d: len(v) for d, v in hit.items()}
ex = {}
for d, v in hit.items():
    ex[d] = []
    for i, (c, ri) in list(v.items())[:3]:
        ex[d].append((round(c, 2), items[i][1][:120], ds[ri]["text"][:200].replace("\n", " | "), ds[ri]["true_labels"][:80]))
json.dump({"distinct_eval_items_contained_ge0.8": out, "n_eval_items_ge5words": collections.Counter(i[0] for i in items), "examples": ex},
          open("/workspace/ultrabert/s2inv/overlap_v2_full_short.json", "w"), indent=1)
print("DONE", out)

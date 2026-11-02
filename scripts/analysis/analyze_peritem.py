import json, os, re, numpy as np, collections, itertools, math
D = "/workspace/ultrabert/s2inv/"
R = {t: json.load(open(f"{D}peritem_{t}.json")) for t in ["S1", "A", "B", "C"] if os.path.exists(f"{D}peritem_{t}.json")}
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
print("== typed: accuracy by workflow x type (n per cell), canonical order, CPU fp32")
base = R["S1"]["typed"]
def acc(items, f): 
    v = [i["ok"] for i in items if f(i)]; return np.mean(v), len(v)
for t, r in R.items():
    row = []
    for wf in ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]:
        row.append(f"{wf[:8]} {acc(r['typed'], lambda i: i['wf']==wf)[0]:.3f}")
    kk = [f"K{k} {acc(r['typed'], lambda i: i['k']==k)[0]:.3f}" for k in (2, 4, 5)]
    print(t, " ".join(row), "|", " ".join(kk), "| overall", round(acc(r['typed'], lambda i: True)[0], 4))
print("\n== paired vs S1 (per decision): net change, McNemar-ish z; case-clustered SE")
for t, r in R.items():
    if t == "S1": continue
    a = np.array([i["ok"] for i in base]); b = np.array([i["ok"] for i in r["typed"]])
    d = b - a; n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    cases = collections.defaultdict(list)
    for i, x in zip(base, d): cases[i["case"]].append(x)
    cm = np.array([np.mean(v) for v in cases.values()]); se = cm.std(ddof=1) / math.sqrt(len(cm))
    print(f"{t}: delta {d.mean():+.4f} (gained {n01}, lost {n10}); case-clustered SE {se:.4f} -> z {d.mean()/se:+.1f}")
    for qt in ("choice", "noul", "score"):
        dd = np.array([x for x, i in zip(d, base) if i["t"] == qt]); 
        cc = collections.defaultdict(list)
        for x, i in zip(d, base):
            if i["t"] == qt: cc[i["case"]].append(x)
        cm = np.array([np.mean(v) for v in cc.values()]); se = cm.std(ddof=1) / math.sqrt(len(cm))
        print(f"    {qt}: {dd.mean():+.3f} clustered SE {se:.3f}")
    # position bias: how often pred index equals each index; share of predictions = option 0 / last
    for tt, rr in (("S1", base), (t, r["typed"])):
        pc = collections.Counter(i["pred"] for i in rr if i["t"] == "choice"); gc = collections.Counter(i["gold"] for i in rr if i["t"] == "choice")
        print(f"    choice pred-position dist {tt}: {dict(sorted(pc.items()))}")
    print(f"    choice GOLD-position dist: {dict(sorted(gc.items()))}")
print("\n== massive (500): accuracy on items whose text appears in HWU64 train (exact normalized) vs rest")
hw = set(norm(json.loads(l)['text']) for l in open('/workspace/ultrabert/ultragliclass/data/stage2_pilot_rebuild/train.jsonl') if '"DeepPavlov/hwu64"' in l)
for t, r in R.items():
    m = r.get("massive")
    if not m: continue
    ov = [norm(x) in hw for x in m["texts"]]
    ok = np.array([p == g for p, g in zip(m["pred"], m["gold"])])
    ovm = np.array(ov)
    print(f"{t}: acc all {ok.mean():.3f}  overlap-items n={ovm.sum()} acc {ok[ovm].mean() if ovm.any() else float('nan'):.3f}  non-overlap n={(~ovm).sum()} acc {ok[~ovm].mean():.3f}")

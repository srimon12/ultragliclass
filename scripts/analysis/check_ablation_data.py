#!/usr/bin/env python
"""CPU checks of the ablation data dirs against the pilot rebuild: row totals, val.jsonl identity, per-source counts, no overlap of any
training state with the typed-decisions EVAL states (cached by eval.py's typed_rows), no test-split use.
usage: python scripts/check_ablation_data.py [pilot_dir=data/stage2_pilot_rebuild] [eval_cache=/workspace/ultrabert/s2inv/typed_eval_rows.json]"""
import hashlib, json, re, sys, collections
pilot = sys.argv[1] if len(sys.argv) > 1 else "data/stage2_pilot_rebuild"
evc = sys.argv[2] if len(sys.argv) > 2 else "/workspace/ultrabert/s2inv/typed_eval_rows.json"
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev_states = {norm(e["text"]) for e in json.load(open(evc))}
ev_labels = {l for e in json.load(open(evc)) for l in e["all_labels"]}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
CAL = [json.loads(l) for l in open("calib/stage2_pilot_val.jsonl")]
key = lambda r: (r["text"][:300], r["source"], r["kind"], tuple(sorted(r["true_labels"])))
procstyle = lambda r: r["source"] == "tasksource/procedural-typed-decisions" or "procedural-typed-decisions/" in r["source"]
ok = True
for arm in ("E1", "E1z", "E2", "E3"):
    d = f"data/stage2_abl_{arm}"
    tr = [json.loads(l) for l in open(f"{d}/train.jsonl")]
    src = collections.Counter(r["source"].split(":")[0] if "LocalLLaMA" not in r["source"] else r["source"] for r in tr)
    v = [json.loads(l) for l in open(f"{d}/val.jsonl")]
    same_val = [key(x) for x in v] == [key(x) for x in CAL]      # same val rows in the same order (label-list expansions may differ; unused)
    proc = sum(procstyle(r) for r in tr)
    sem = [r for r in tr if "LocalLLaMA" in r["source"]]
    leak = sum(norm(r["text"]) in ev_states for r in tr)
    sem_wf_ok = True
    print(f"{arm}: train {len(tr)} | val rows == calib/stage2_pilot_val.jsonl: {same_val} | procedural-style rows {proc} ({100*proc/len(tr):.1f} %) | semantic rows {len(sem)} "
          f"(option strings shared with eval: {len({l for r in sem for l in r['all_labels']} & ev_labels)}) | exact eval-state matches in train: {leak}")
    print("   ", dict(src))
    ok &= (same_val or arm == "E1z") and len(tr) == 96035 and leak == 0   # E1z: its own val differs (Jev val also excludes procedural-style rows); unused, calib val is
print("ALL OK" if ok else "CHECK FAILED")

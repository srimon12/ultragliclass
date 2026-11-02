#!/usr/bin/env python
"""Stage-3 bar (1): accuracy on the long-state Jev test slice (scripts/build_long_jev_slice.py), and a PAIRED comparison
of two models on the same rows (stage 2 final = baseline, stage 3 final = candidate).

accuracy = argmax(model softmax over the row's options) == argmax(gold soft target), as in eval.py's typed suite.
Same formatting as eval.py (official GLiClass pipeline: labels -> <<SEP>> -> prompt -> text), label order = the row's
order (run 0) plus --permutations-1 seeded shuffles (flip rate). Truncation is reported: rows whose labels+prompt+text
exceed --max_length tokens are cut from the END of the text by the pipeline.

    python scripts/eval_long_jev.py --model runs/stage2_full_C/final_model --slice calib/jev_long_test.jsonl \
        --max_length 8192 --batch_size 2 --dtype fp16 --out results/stage3_baseline_s2_longjev_8192.json
    python scripts/eval_long_jev.py --compare results/stage3_baseline_s2_longjev_8192.json results/stage3_final_longjev_8192.json
"""
from __future__ import annotations

import argparse, json, os, random, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def softmax(x):
    x = np.asarray(x, dtype=np.float64); e = np.exp(x - x.max()); return e / e.sum()


def bootstrap_ci(v, n=2000, seed=0, groups=None):
    """95 % CI of the mean; cluster bootstrap over `groups` (shared states) when given."""
    rng = np.random.default_rng(seed); v = np.asarray(v, dtype=float)
    if groups is None:
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    else:
        g = np.asarray(groups); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
        bs = []
        for _ in range(n):
            pick = rng.choice(ug, len(ug)); sel = np.concatenate([idx[u] for u in pick]); bs.append(v[sel].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarize(rows, correct, key):
    out = {}
    for r, c in zip(rows, correct):
        out.setdefault(str(r.get(key)), []).append(c)
    return {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))}


def bucket(n):
    for b in (8000, 12000, 16000, 24000):
        if n < b:
            return f"<{b}"
    return ">=24000"


def run_eval(a):
    import torch
    from eval import GLiClassScorer
    rows = [json.loads(l) for l in open(a.slice)]
    if a.max_rows and len(rows) > a.max_rows:
        rows = random.Random(a.seed).sample(rows, a.max_rows)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype] if a.dtype != "auto" else (
        torch.float16 if dev.type == "cuda" else torch.float32)
    sc = GLiClassScorer(a.model, dev, dt, a.max_length, a.batch_size)
    # token accounting (label prompt + prompt + text), to report how much the slice is truncated at this max_length
    ntok = []
    for r in rows:
        lab = "".join(f"<<LABEL>>{l}" for l in r["all_labels"]) + "<<SEP>>"
        ntok.append(len(sc.tok(lab + (r.get("prompt") or "") + r["text"])["input_ids"]))
    # long rows first so an OOM shows up immediately; batches are sorted by length to cut padding
    order = sorted(range(len(rows)), key=lambda i: -ntok[i])
    t0 = time.time()
    preds_runs = []
    for k in range(a.permutations):
        lg = [None] * len(rows)
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            labs, perms = [], []
            for i in idx:
                p = list(range(len(rows[i]["all_labels"])))
                if k > 0:
                    random.Random(a.seed + 1000 * k + i).shuffle(p)
                perms.append(p); labs.append([rows[i]["all_labels"][j] for j in p])
            out = sc.logits([rows[i]["text"] for i in idx], labs, [rows[i].get("prompt") or "" for i in idx])
            for i, p, x in zip(idx, perms, out):
                lg[i] = np.asarray(x)[np.argsort(p)]          # back to the row's canonical option order
        preds_runs.append(lg)
        print(f"perm {k}: {time.time() - t0:.0f}s", flush=True)
    L = preds_runs[0]
    gold = [int(np.argmax(r["gold_probs"])) for r in rows]
    correct = [int(np.argmax(x) == g) for x, g in zip(L, gold)]
    flips = [int(any(int(np.argmax(run[i])) != int(np.argmax(L[i])) for run in preds_runs[1:]))
             for i in range(len(rows))] if a.permutations > 1 else [0] * len(rows)
    for r, n in zip(rows, ntok):
        r["_len_bucket"] = bucket(r.get("state_chars", len(r["text"]))); r["_trunc"] = int(n > a.max_length)
    groups = [r.get("group_id") or r["text"][:200] for r in rows]
    res = {"model": a.model, "slice": a.slice, "max_length": a.max_length, "dtype": a.dtype, "n": len(rows),
           "accuracy": float(np.mean(correct)), "accuracy_ci95_cluster": bootstrap_ci(correct, groups=groups),
           "flip_rate": float(np.mean(flips)), "perm_runs": a.permutations,
           "truncated_frac": float(np.mean([r["_trunc"] for r in rows])),
           "tokens": {"median": float(np.median(ntok)), "p90": float(np.percentile(ntok, 90)), "max": int(max(ntok))},
           "by_kind": summarize(rows, correct, "kind_raw"), "by_state_chars": summarize(rows, correct, "_len_bucket"),
           "by_truncated": summarize(rows, correct, "_trunc"), "by_license_use": summarize(rows, correct, "license_use_row"),
           "seconds": round(time.time() - t0, 1),
           "per_row": [{"id": r.get("id"), "group": g, "correct": c, "flip": f, "pred": int(np.argmax(x)),
                        "p_max": float(softmax(x).max())} for r, g, c, f, x in zip(rows, groups, correct, flips, L)]}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_row"}, indent=1))


def compare(base_p, cand_p, bar=0.03):
    b, c = json.load(open(base_p)), json.load(open(cand_p))
    bm = {r["id"]: r for r in b["per_row"]}; pairs = [(bm[r["id"]], r) for r in c["per_row"] if r["id"] in bm]
    d = np.array([y["correct"] - x["correct"] for x, y in pairs], dtype=float)
    groups = [y["group"] for _, y in pairs]
    n01 = int(sum(1 for x, y in pairs if x["correct"] == 0 and y["correct"] == 1))
    n10 = int(sum(1 for x, y in pairs if x["correct"] == 1 and y["correct"] == 0))
    from math import comb
    m = n01 + n10; k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    out = {"paired_n": len(pairs), "base_acc": float(np.mean([x["correct"] for x, _ in pairs])),
           "cand_acc": float(np.mean([y["correct"] for _, y in pairs])), "delta": float(d.mean()),
           "delta_ci95_cluster": bootstrap_ci(d, groups=groups), "mcnemar_exact_p": p, "wins": n01, "losses": n10,
           "bar": f"+{bar:.2f} accuracy", "bar_point_estimate_met": bool(d.mean() >= bar),
           "bar_ci_lower_above_zero": bool(bootstrap_ci(d, groups=groups)[0] > 0)}
    print(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--slice", default="calib/jev_long_test.jsonl")
    ap.add_argument("--max_length", type=int, default=8192); ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--permutations", type=int, default=1); ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "CAND_JSON"))
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run_eval(a)

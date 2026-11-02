#!/usr/bin/env python
"""Label-order diagnostics for a GLiClass checkpoint (CPU ok, small n).

For each dataset: K label orders (canonical + seeded shuffles), raw logits kept, then
  - accuracy / flip rate (as eval.py), flip rate split by whether the canonical answer was right
  - position bias: histogram of the *position* of the predicted label over all orders (uniform = none),
    accuracy by position of the gold label (primacy / recency), mean z-scored logit by position
  - margin: top1-top2 logit gap; flip rate for low-margin vs high-margin examples
  - cheap fixes at inference: permutation-averaged logits (accuracy), majority vote
Usage: python scripts/analyze_perm.py --model <dir|hf id> --datasets banking77 clinc150 --n 100 --k 4 --out x.json
"""
import argparse, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import GLiClassScorer, load_eval_set, subsample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--max_length", type=int, default=1024)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
sc = GLiClassScorer(a.model, "cpu", torch.float32, a.max_length, a.batch_size)
res = {}
for name in a.datasets:
    t0 = time.time()
    texts, classes, gold = load_eval_set(name)
    texts, gold = subsample(texts, gold, a.n, a.seed)
    C = len(classes); y = np.array([classes.index(g) for g in gold])
    perms, L, pos_of = [], [], []   # L[k]: (N, C) logits in canonical class order
    for k in range(a.k):
        perm = list(range(C))
        if k:
            random.Random(a.seed + k).shuffle(perm)
        lg = sc.logits(texts, [classes[i] for i in perm])
        inv = np.argsort(perm)                   # inv[c] = position of canonical class c in this order
        L.append(np.stack([x[inv] for x in lg])); pos_of.append(inv)
    L = np.stack(L)                              # (K, N, C)
    preds = L.argmax(-1)                         # (K, N) canonical class ids
    acc = [(p == y).mean() for p in preds]
    flip = np.array([len(set(preds[:, i])) > 1 for i in range(len(y))])
    right0 = preds[0] == y
    # position bias
    pred_pos = np.concatenate([pos_of[k][preds[k]] for k in range(a.k)]) / max(C - 1, 1)
    gold_pos = np.concatenate([pos_of[k][y] for k in range(a.k)]) / max(C - 1, 1)
    correct = np.concatenate([preds[k] == y for k in range(a.k)])
    bins = np.minimum((pred_pos * 5).astype(int), 4)
    gbins = np.minimum((gold_pos * 5).astype(int), 4)
    z = (L - L.mean(-1, keepdims=True)) / (L.std(-1, keepdims=True) + 1e-6)
    zpos = np.zeros(5); zcnt = np.zeros(5)
    for k in range(a.k):
        pb = np.minimum((pos_of[k] / max(C - 1, 1) * 5).astype(int), 4)        # (C,)
        for b in range(5):
            m = pb == b
            zpos[b] += z[k][:, m].mean() * m.sum(); zcnt[b] += m.sum()
    # margin
    srt = np.sort(L[0], axis=-1)
    margin = srt[:, -1] - srt[:, -2]
    lo = margin < np.median(margin)
    # inference-side fixes
    avgL = (L - L.mean(-1, keepdims=True)).mean(0)
    votes = np.array([np.bincount(preds[:, i], minlength=C).argmax() for i in range(len(y))])
    res[name] = {
        "n": len(y), "C": C, "K": a.k, "acc_per_perm": [round(float(x), 4) for x in acc],
        "flip_rate": round(float(flip.mean()), 4),
        "flip_rate_if_canonical_right": round(float(flip[right0].mean()), 4) if right0.any() else None,
        "flip_rate_if_canonical_wrong": round(float(flip[~right0].mean()), 4) if (~right0).any() else None,
        "pred_position_hist_quintiles": [round(float((bins == b).mean()), 3) for b in range(5)],
        "acc_by_gold_position_quintile": [round(float(correct[gbins == b].mean()), 3) if (gbins == b).any() else None for b in range(5)],
        "mean_z_logit_by_position_quintile": [round(float(zpos[b] / zcnt[b]), 3) for b in range(5)],
        "margin_median": round(float(np.median(margin)), 3),
        "flip_rate_low_margin_half": round(float(flip[lo].mean()), 4),
        "flip_rate_high_margin_half": round(float(flip[~lo].mean()), 4),
        "acc_perm_averaged_logits": round(float((avgL.argmax(-1) == y).mean()), 4),
        "acc_majority_vote": round(float((votes == y).mean()), 4),
        "seconds": round(time.time() - t0, 1),
    }
    print(name, json.dumps(res[name]), flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.model, **res}, open(a.out, "w"), indent=1)

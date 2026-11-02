#!/usr/bin/env python
"""ESTIMATE of the padding reduction from length-grouped batching, on sampled pilot rows with the real tokenizer.

Takes N random pilot-train rows, FETCHES each through the real (augmenting, many-label-expanding, pairing) dataset to get the
true padded input_ids length (and 2nd view for pairs), then simulates micro-batches of `--mb` rows two ways:
  random   : a random permutation cut into micro-batches (what the pilots did)
  grouped  : ugc.length_group.LengthGroupedSampler driven by the CHEAP length estimate (what training will use)
  oracle   : sorted by the true fetched length (upper bound of what any sampler could do)
Cost model: tokens computed per micro-batch = n_sequences * max_len (n_sequences counts paired second views). Speed-up is the ratio of
tokens computed, i.e. it ASSUMES step time ~ proportional to computed tokens (no fixed overhead, no attention-quadratic term).
It is an estimate, not a GPU measurement. With world>1 a DDP step waits for the slower rank, so we also report the cost
as the per-group MAX over ranks.
  python scripts/estimate_padding.py --config configs/stage2_pilot_C.yaml --n 6000 --out results/padding_estimate_C.json
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from transformers import AutoTokenizer
from ugc.config import load_config
from ugc.length_group import LengthGroupedSampler, LengthModel, base_lengths
from ugc.many_label import LabelPools, ManyLabelDataset

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage2_pilot_C.yaml")
ap.add_argument("--data_dir", default="data/stage2_pilot_rebuild")
ap.add_argument("--tokenizer", default="/workspace/kaggle/ultragliclass/S1/ultragliclass/runs/stage1_ultrabert150m/final_model")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--mb", type=int, default=4)
ap.add_argument("--mult", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--plan", type=int, default=1, help="1 = planning mode (what length_group uses)")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()

cfg = load_config(a.config, a.set)
d, m = cfg["data"], cfg["model"]
tok = AutoTokenizer.from_pretrained(a.tokenizer)
rows_all = [json.loads(l) for l in open(os.path.join(a.data_dir, "train.jsonl"))]
rng = random.Random(a.seed)
rows = rng.sample(rows_all, a.n)
ml, cons = d.get("many_label") or {}, d.get("consistency") or {}
use_ml = bool(ml.get("enabled")); pair_prob = float(cons.get("pair_prob", 0)) if cons.get("enabled") else 0.0
aug = d.get("augmentation") or {}
ac = AugmentationConfig(enabled=aug.get("enabled", False), **{k: v for k, v in aug.items() if k != "enabled" and k.startswith("random_") or k == "max_num_examples"})
common = dict(max_length=d["max_length"], problem_type="multi_label_classification", architecture_type="uni-encoder", prompt_first=True,
              labels_tokenizer=None, shuffle_labels=True, max_labels=int(m.get("max_num_classes", 50)))
pools = LabelPools.from_file(os.path.join(a.data_dir, "label_pools.json"), top_m=int(ml.get("top_m", 48))) if use_ml else None
if pools:
    for p in pools.pools: pools.neighbours(p)
random.seed(a.seed); torch.manual_seed(a.seed)
if use_ml or pair_prob > 0:
    ds = ManyLabelDataset(rows, tok, ac, {}, pools=pools, many_label=ml.get("params"), pair_prob=pair_prob, seed=a.seed, **common)
else:
    ds = GLiClassDataset(rows, tok, ac, {}, **common)
if a.plan and hasattr(ds, "enable_planning"):
    ds.enable_planning()   # fetches are seeded from (seed, epoch=0, row) so LengthModel can replay the expansion draws
t0 = time.time()
true_len = np.zeros(len(rows), int); n_seq = np.ones(len(rows), int); n_lab = np.zeros(len(rows), int)
for i in range(len(rows)):
    it = ds[i]
    true_len[i] = len(it["input_ids"]); n_lab[i] = len(it["labels_text"])
    if "pair" in it:
        n_seq[i] = 2; true_len[i] = max(true_len[i], len(it["pair"]["input_ids"]))
print(f"fetched {len(rows)} rows in {time.time()-t0:.0f}s; mean real len {true_len.mean():.0f}, mean labels {n_lab.mean():.1f}, paired {np.mean(n_seq==2):.3f}")

t0 = time.time()
base = base_lengths(rows, tok, d["max_length"])
est = LengthModel(base, rows, tok, pools if use_ml else None, ml.get("params") or {}, d["max_length"], seed=a.seed).lengths(0) if (use_ml and a.plan) else base
print(f"cheap length estimate: {time.time()-t0:.1f}s for {len(rows)} rows; corr(est, true)={np.corrcoef(est, true_len)[0,1]:.3f}, corr(base,true)={np.corrcoef(base, true_len)[0,1]:.3f}")

def cost(order, world):
    """order: row indices in stream order (micro-batch = consecutive mb rows; micro-batch j goes to rank j%world).
    Returns (padded tokens per micro-batch, real tokens per micro-batch, DDP cost = sum over groups of max over ranks of padded tokens)."""
    mbs = [order[i:i + a.mb] for i in range(0, len(order) - a.mb + 1, a.mb)]
    pad = [max(true_len[b]) * int(n_seq[b].sum()) if False else int(max(true_len[b]) * n_seq[b].sum()) for b in mbs]  # n_seq rows share the padded length
    real = [int((true_len[b] * n_seq[b]).sum()) for b in mbs]
    ddp = sum(max(pad[j:j + world]) for j in range(0, len(pad) - world + 1, world)) * world
    return float(np.mean(pad)), float(np.mean(real)), ddp / max(1, (len(pad) // world * world))

res = {"config": a.config, "n_rows": a.n, "mb": a.mb, "mega_batch_mult": a.mult,
       "mean_real_len": float(true_len.mean()), "mean_labels_per_row": float(n_lab.mean()), "paired_frac": float(np.mean(n_seq == 2)),
       "corr_est_true": float(np.corrcoef(est, true_len)[0, 1]), "corr_base_true": float(np.corrcoef(base, true_len)[0, 1]), "worlds": {}}
for world in (1, 2):
    out = {}
    reps = 5
    rnd = []; grp = []; grp_base = []; orc = []
    for rep in range(reps):
        g = np.random.default_rng(rep)
        n_use = len(rows) // (a.mb * world) * a.mb * world
        rnd.append(cost(g.permutation(len(rows))[:n_use], world))
        for name, lens, acc in (("est", est, grp), ("base", base, grp_base)):
            s = LengthGroupedSampler(lens, a.mb, world, a.mult, seed=rep); acc.append(cost(s.epoch_order(0), world))
        oo = LengthGroupedSampler(true_len, a.mb, world, a.mult, seed=rep); orc.append(cost(oo.epoch_order(0), world))
        # oracle with a mega-batch of everything
        # (kept separate: global sort of the whole sample)
    mean = lambda L, k: float(np.mean([x[k] for x in L]))
    for name, L in (("random", rnd), ("grouped_est", grp), ("grouped_baseLenOnly", grp_base), ("oracle_trueLen_same_megabatch", orc)):
        out[name] = {"padded_tokens_per_microbatch": mean(L, 0), "real_tokens_per_microbatch": mean(L, 1),
                     "padding_fraction": 1 - mean(L, 1) / mean(L, 0), "ddp_cost_per_microbatch": mean(L, 2)}
    out["token_reduction_x_grouped_est"] = out["random"]["padded_tokens_per_microbatch"] / out["grouped_est"]["padded_tokens_per_microbatch"]
    out["token_reduction_x_ddp_grouped_est"] = out["random"]["ddp_cost_per_microbatch"] / out["grouped_est"]["ddp_cost_per_microbatch"]
    res["worlds"][str(world)] = out
    print(f"\nworld={world}")
    for k, v in out.items():
        print("  ", k, {kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3))
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=2); print("wrote", a.out)

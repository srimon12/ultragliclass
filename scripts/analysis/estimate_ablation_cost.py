#!/usr/bin/env python
"""Relative step-cost of the ablation arms vs pilot arm C, from the ACTUAL built train.jsonl of each arm (tokenised through ManyLabelDataset with
the C settings: many-label expansion + pair_prob 0.25), same cost model as scripts/estimate_pilot_cost.py (R * L * (1 + L / (6 d)), max over 2 GPUs of 4
micro-batches of 8). The ratio to the pilot-C data is applied to the MEASURED pilot-C 4.008 s/step (Kaggle STATUS, T4x2, fp16, ckpt).
usage: python scripts/estimate_ablation_cost.py [--rows 3000] [--steps 500]   (CPU, ~minutes)"""
import argparse, json, os, random, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gliclass.data_processing import AugmentationConfig
from transformers import AutoTokenizer
from ugc.many_label import LabelPools, ManyLabelDataset

ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=3000); ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--out", default="results/stage2_ablation_cost_estimate.json"); args = ap.parse_args()
D, MEASURED_C = 768, 4.008
tok = AutoTokenizer.from_pretrained("/workspace/kaggle/ultragliclass/S1/ultragliclass/runs/stage1_ultrabert150m/final_model")
common = dict(max_length=1024, problem_type="multi_label_classification", architecture_type="uni-encoder", prompt_first=True,
              labels_tokenizer=None, shuffle_labels=True, max_labels=150)
ML = {"expand_prob": 0.7, "k_min": 20, "k_max": 150, "hard_frac": 0.5, "dup_sim": 0.8, "text_keep_tokens": 256, "pair_min_labels": 8,
      "pool_overrides": {"glic-v2": {"expand_prob": 0.10, "k_min": 12, "k_max": 60, "hard_frac": 0.2, "dup_sim": 0.6}}}


def arm_cost(d, seed=0):
    rows = [json.loads(l) for l in open(f"{d}/train.jsonl")]
    rng = random.Random(seed); rng.shuffle(rows); rows = rows[:args.rows]
    pools = LabelPools.from_file(f"{d}/label_pools.json")
    ds = ManyLabelDataset(rows, tok, AugmentationConfig(enabled=False), {}, pools=pools, many_label=ML, pair_prob=0.25, seed=seed, **common)
    items = []
    for i in range(len(rows)):
        it = ds[i]; items.append((len(it["input_ids"]), 1 + int("pair" in it)))
    costs, pad = [], []
    for _ in range(args.steps):
        gpu = []
        for g in range(2):
            c = 0.0
            for _mb in range(4):
                b = [rng.choice(items) for _ in range(8)]
                L = max(x[0] for x in b); R = sum(x[1] for x in b)
                c += R * L * (1 + L / (6 * D)); pad.append(L)
            gpu.append(c)
        costs.append(max(gpu))
    return statistics.mean(costs), statistics.mean(pad), statistics.mean(x[0] for x in items)


ref_dir = "data/stage2_pilot_rebuild"
ref, ref_pad, ref_len = arm_cost(ref_dir)
out = {"measured_pilot_C_s_per_step": MEASURED_C, "ref": {"padded_len": ref_pad, "mean_row_len": ref_len}}
print(f"pilot C data: padded {ref_pad:.0f}, mean row {ref_len:.0f}", flush=True)
for arm in ("E1", "E1z", "E2", "E3"):
    c, pad, ml = arm_cost(f"data/stage2_abl_{arm}")
    s = MEASURED_C * c / ref
    out[arm] = {"ratio_vs_pilot_C": c / ref, "sec_per_step": s, "train_hours_1500": 1500 * s / 3600, "padded_len": pad, "mean_row_len": ml}
    print(f"{arm}: ratio {c / ref:.3f} -> {s:.2f} s/step -> {1500 * s / 3600:.2f} h train; padded {pad:.0f}, mean row {ml:.0f}", flush=True)
out["C0"] = {"ratio_vs_pilot_C": 1.0, "sec_per_step": MEASURED_C, "train_hours_1500": 1500 * MEASURED_C / 3600}
json.dump(out, open(args.out, "w"), indent=2)

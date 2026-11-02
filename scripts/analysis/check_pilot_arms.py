#!/usr/bin/env python
"""Are the three pilot arms comparable? Builds the A / B / C training datasets exactly as train.py does (same recipe data,
same seeds, same sampler) and reports, per arm and per optimizer step (2 GPUs x 4 micro-batches x 8 rows):
  * steps and rows (identical by construction: train.max_steps=1500 in all three configs; asserted below),
  * forward rows (C adds the second view of paired rows), label slots, share of slots that are positives,
  * the GLiClass loss normaliser: loss = sum(valid slots) / (number of valid slots) PER MICRO-BATCH, so a row's weight in the
    gradient is proportional to its label count and every slot of a micro-batch is down-weighted when a many-label row is in it.
    We report the mean valid slots per micro-batch and the share of the loss weight carried by rows with >= 20 labels / <= 7 labels.
Usage: python scripts/check_pilot_arms.py [--data data/stage2_pilot_smoke] [--steps 200] [--model <final_model dir for the tokenizer>]"""
import argparse, json, os, sys, random
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from ugc.config import load_config
from ugc.many_label import LabelPools, ManyLabelDataset, PairedCollator

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/stage2_pilot_smoke")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--model", default="/workspace/kaggle/ultragliclass/S1/ultragliclass/runs/stage1_ultrabert150m/final_model")
ap.add_argument("--out", default=None)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(os.path.join(a.data, "train.jsonl"))]
res = {}
orders = {}
for arm in "ABC":
    cfg = load_config(f"configs/stage2_pilot_{arm}.yaml")
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    assert t["max_steps"] == 1500 and t["per_device_train_batch_size"] == 8 and t["gradient_accumulation_steps"] == 4
    aug = d["augmentation"]
    ac = AugmentationConfig(enabled=aug["enabled"], random_label_removal_prob=aug["random_label_removal_prob"],
                            random_label_addition_prob=aug["random_label_addition_prob"],
                            random_text_addition_prob=aug["random_text_addition_prob"], random_add_description_prob=0.0,
                            random_add_synonyms_prob=0.0, random_add_examples_prob=0.0)
    common = dict(max_length=d["max_length"], problem_type="multi_label_classification", architecture_type="uni-encoder",
                  prompt_first=True, labels_tokenizer=None, shuffle_labels=True, max_labels=int(m["max_num_classes"]))
    ml, cons = d["many_label"], d["consistency"]
    pair_prob = float(cons["pair_prob"]) if cons["enabled"] else 0.0
    random.seed(0); torch.manual_seed(0)
    if ml["enabled"] or pair_prob > 0:
        pools = LabelPools.from_file(os.path.join(a.data, "label_pools.json"), top_m=ml["top_m"]) if ml["enabled"] else None
        ds = ManyLabelDataset(rows, tok, ac, {}, pools=pools, many_label=ml["params"], pair_prob=pair_prob, seed=int(t["seed"]), **common)
    else:
        ds = GLiClassDataset(rows, tok, ac, {}, **common)
    coll = PairedCollator(DataCollatorWithPadding()) if pair_prob > 0 else DataCollatorWithPadding()
    g = torch.Generator().manual_seed(int(t["seed"]))          # same data_seed => same row order in every arm
    sampler = torch.utils.data.RandomSampler(ds, generator=g)
    order = []
    while len(order) < a.steps * 64:                            # several passes over the (small) smoke set
        order += list(sampler)
    order = order[: a.steps * 64]
    orders[arm] = order
    st = dict(fwd_rows=0, slots=0, pos=0, mb=0, slots_many=0, slots_few=0, rows_many=0, rows_all=0, mb_slots=[], steps_with_many=0)
    for s in range(a.steps):
        many_in_step = False
        for mb in range(4 * 2):
            idx = order[(s * 8 + mb) * 8:(s * 8 + mb + 1) * 8]
            batch = coll([ds[i] for i in idx])
            n_primary = len(idx)
            labs = batch["labels_text"]
            lab = batch["labels"]
            valid = [len(x) for x in labs]
            tot = sum(valid)
            st["fwd_rows"] += len(valid); st["slots"] += tot; st["pos"] += float((lab > 0).sum()); st["mb"] += 1
            st["mb_slots"].append(tot)
            for v in valid[:n_primary] if False else valid:
                st["rows_all"] += 1
                if v >= 20: st["slots_many"] += v; st["rows_many"] += 1; many_in_step = True
                elif v <= 7: st["slots_few"] += v
        st["steps_with_many"] += int(many_in_step)
    n = a.steps
    res[arm] = {
        "rows_per_step_primary": 64, "forward_rows_per_step": st["fwd_rows"] / n, "label_slots_per_step": st["slots"] / n,
        "mean_valid_slots_per_microbatch": st["slots"] / st["mb"], "positive_slot_share": st["pos"] / st["slots"],
        "loss_weight_share_rows_ge20_labels": st["slots_many"] / st["slots"], "loss_weight_share_rows_le7_labels": st["slots_few"] / st["slots"],
        "share_forward_rows_ge20_labels": st["rows_many"] / st["rows_all"], "share_steps_with_a_ge20_row": st["steps_with_many"] / n,
        "max_steps": 1500, "optimizer_steps_per_arm": t["max_steps"],
    }
assert orders["A"] == orders["B"] == orders["C"], "row order differs between arms"
print(f"row order identical across arms for the first {a.steps * 64} draws (same sampler seed)")
print(f"{'':45s} {'A':>10s} {'B':>10s} {'C':>10s}")
for k in res["A"]:
    print(f"{k:45s} " + " ".join(f"{res[x][k]:10.3f}" for x in "ABC"))
if a.out:
    json.dump(res, open(a.out, "w"), indent=2)

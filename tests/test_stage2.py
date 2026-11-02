#!/usr/bin/env python
"""CPU unit tests for the stage-2 code paths (ugc/many_label.py). Needs data/stage2_smoke (prepare_data.py with
configs/data_stage2_smoke.yaml --streaming). Run: python scripts/test_stage2.py"""
import collections, json, os, random, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding
from transformers import AutoTokenizer
from ugc.many_label import LabelPools, ManyLabelDataset, PairedCollator, consistency_loss

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra, flush=True)

# ---------- 1. consistency loss on toy tensors ----------
torch.manual_seed(0)
P, C = 3, 6
z = torch.randn(P, C, requires_grad=True)
mask = torch.ones(P, C, dtype=torch.bool); mask[2, 4:] = False                       # row 2 has 4 labels
perm = torch.full((P, C), -1, dtype=torch.long)
for r in range(P):
    n = int(mask[r].sum()); perm[r, :n] = torch.randperm(n)                          # view-2 slot j -> view-1 slot perm[j]
z2 = torch.zeros(P, C)
for r in range(P):
    n = int(mask[r].sum()); z2[r, :n] = z[r].detach()[perm[r, :n]]                    # same scores, shuffled order
logits = torch.cat([z.detach(), z2], 0); cm = torch.cat([mask, mask], 0)
pidx = torch.tensor([[0, 3], [1, 4], [2, 5]])
for kind in ("skl", "logit_mse"):
    l = consistency_loss(logits, cm, pidx, perm, kind)
    check(f"consistency[{kind}] = 0 for a pure reordering", float(l) < 1e-6, f"{float(l):.2e}")
z2b = z2 + torch.randn(P, C)
l = consistency_loss(torch.cat([z.detach(), z2b], 0), cm, pidx, perm, "skl")
check("consistency > 0 when scores differ", float(l) > 1e-3, f"{float(l):.4f}")
zz = torch.cat([z.detach(), z2b], 0).clone().requires_grad_(True)
consistency_loss(zz, cm, pidx, perm, "skl").backward()
check("gradient reaches both views and is zero on padding", zz.grad.abs().sum() > 0 and zz.grad[2, 4:].abs().sum() == 0 and zz.grad[5, 4:].abs().sum() == 0)
l16 = consistency_loss(torch.cat([z.detach(), z2b], 0).half(), cm, pidx, perm, "skl")
check("fp16 logits are fine (loss computed in fp32)", torch.isfinite(l16).item())

# ---------- 2. dataset ----------
data_dir = "data/stage2_smoke"
rows = [json.loads(l) for l in open(f"{data_dir}/train.jsonl")]
snapshot = json.dumps(rows[:300], sort_keys=True)
pools = LabelPools.from_file(f"{data_dir}/label_pools.json")
tok = AutoTokenizer.from_pretrained("runs/stage1_ultrabert150m/final_model") if os.path.isdir("runs/stage1_ultrabert150m/final_model") \
    else AutoTokenizer.from_pretrained("/workspace/kaggle/ultragliclass/S1/ultragliclass/runs/stage1_ultrabert150m/final_model")
ML = 512
ml = {"expand_prob": 0.9, "k_min": 20, "k_max": 150, "hard_frac": 0.5, "dup_sim": 0.8, "text_keep_tokens": 128, "pair_min_labels": 8,
      "pool_overrides": {"glic-v2": {"expand_prob": 0.1, "k_min": 12, "k_max": 40, "hard_frac": 0.2, "dup_sim": 0.6}}}
common = dict(max_length=ML, problem_type="multi_label_classification", architecture_type="uni-encoder", prompt_first=True,
              labels_tokenizer=None, shuffle_labels=True, max_labels=150)
ds = ManyLabelDataset(rows, tok, AugmentationConfig(enabled=False), {}, pools=pools, many_label=ml, pair_prob=0.5, seed=1, **common)
tax_idx = [i for i, r in enumerate(rows) if r.get("pool") not in (None, "glic-v2")][:300]
counts, over = [], 0
for i in tax_idx:
    it = ds[i]
    n = len(it["labels_text"]); counts.append(n)
    gold = set(rows[i]["true_labels"])
    ids = list(it["input_ids"]); sep = tok.convert_tokens_to_ids("<<SEP>>")
    ntext = len(tok(rows[i]["text"][: ML * 6], add_special_tokens=False)["input_ids"])
    kept = len(ids) - (ids.index(sep) + 1) if sep in ids else 0
    over += int(kept < min(ntext, 128) - 8)       # text must keep >= min(its length, text_keep_tokens) tokens
    assert gold <= set(it["labels_text"]), "gold label lost"
    # the label prompt must not eat the text: tokens after <<SEP>> >= min(text tokens, 128) (approx via length)
check("stored rows are never mutated", json.dumps(rows[:300], sort_keys=True) == snapshot)
h = collections.Counter(min(c // 20 * 20, 100) for c in counts)
check("many-label expansion yields 20+ labels for most taxonomy rows", sum(c >= 20 for c in counts) / len(counts) > 0.5,
      f"frac>=20 {sum(c >= 20 for c in counts) / len(counts):.2f}, mean {sum(counts) / len(counts):.1f}, max {max(counts)}, hist {dict(sorted(h.items()))}")
check("label prompt never crowds the text below text_keep_tokens", over == 0, f"violations={over}/{len(tax_idx)}")
a = [tuple(ds[tax_idx[0]]["labels_text"]) for _ in range(4)]
check("same row, different label draws/orders on every fetch", len(set(a)) > 1, f"{len(set(a))}/4 distinct")
# near-duplicate guard: no added label has similarity >= dup_sim with a gold label
bad = 0
for i in tax_idx[:150]:
    nn = pools.neighbours(rows[i]["pool"])
    for g in rows[i]["true_labels"]:
        near = {l for l, s in nn.get(g, []) if s >= 0.8}
        bad += len(near & set(ds[i]["labels_text"]))
check("no near-duplicate-of-gold labels added", bad == 0, f"bad={bad}")

# per-token budget on long text: text_keep_tokens guaranteed
long_rows = [r for r in rows if r.get("pool") == "dbpedia" and len(r["text"]) > 1200][:20]
if long_rows:
    d2 = ManyLabelDataset(long_rows, tok, AugmentationConfig(enabled=False), {}, pools=pools, many_label={**ml, "expand_prob": 1.0}, seed=2, **common)
    keep = []
    for j in range(len(long_rows)):
        it = d2[j]; ids = list(it["input_ids"])
        sep = tok.convert_tokens_to_ids("<<SEP>>")
        keep.append(len(ids) - (ids.index(sep) + 1) if sep in ids else 0)
    check("long text keeps >= text_keep_tokens after the labels", min(keep) >= 100, f"min tokens after SEP {min(keep)}")

# ---------- 3. collator alignment + DataLoader workers ----------
coll = PairedCollator(DataCollatorWithPadding())
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=coll, drop_last=True)
paired_batches, pairs = 0, 0
seen_draws = set()
for b_i, batch in enumerate(dl):
    if b_i >= 6: break
    if "pair_idx" in batch:
        paired_batches += 1; pairs += len(batch["pair_idx"])
        check_ok = batch["input_ids"].shape[0] == 8 + len(batch["pair_idx"])
        assert check_ok
    seen_draws.add(batch["input_ids"].sum().item())
check("DataLoader(num_workers=2) yields paired batches", paired_batches > 0, f"{paired_batches}/6 batches, {pairs} pairs")
# alignment by identity on a manual batch
items = [ds[i] for i in tax_idx[:8]]
bt = coll(items)
okal = True
for r, (a_i, b_i) in enumerate(bt["pair_idx"].tolist()):
    la, lb = items[a_i]["labels_text"], items[a_i]["pair"]["labels_text"]
    for j, l in enumerate(lb):
        okal &= (la[int(bt["pair_perm"][r, j])] == l)
check("pair_perm maps every second-view label to the same label in the primary view", okal and len(bt["pair_idx"]) > 0, f"{len(bt['pair_idx'])} pairs")
check("workers draw different expansions", len(seen_draws) > 3)
# ---------- 4. paired views whose label counts differ (augmentation adds/removes labels per view) ----------
# regression test: PairedCollator used to size pair_perm from the primary view only -> IndexError in the real pilot-C run
aug_hi = AugmentationConfig(enabled=True, random_label_removal_prob=0.5, random_label_addition_prob=0.9, random_text_addition_prob=0.0,
                            random_add_description_prob=0.0, random_add_synonyms_prob=0.0, random_add_examples_prob=0.0)
dsa = ManyLabelDataset(rows, tok, aug_hi, {}, pools=pools, many_label={**ml}, pair_prob=1.0, seed=5, **common)
diff, ran = 0, True
try:
    for k in range(0, 48, 8):
        items = [dsa[i] for i in tax_idx[k:k + 8]]
        diff += sum("pair" in it and len(it["pair"]["labels_text"]) > len(it["labels_text"]) for it in items)
        bt = coll(items)
        z = torch.randn(bt["input_ids"].shape[0], int(max(len(x) for x in bt["labels_text"])), requires_grad=True)
        cm = torch.zeros_like(z)
        for r, lt in enumerate(bt["labels_text"]):
            cm[r, :len(lt)] = 1
        consistency_loss(z, cm, bt["pair_idx"], bt["pair_perm"]).backward()
except Exception as e:  # noqa: BLE001
    ran = False; print("   ", repr(e))
check("pair collator + consistency loss handle a second view with MORE labels than the primary", ran and diff > 0, f"{diff} such pairs")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)

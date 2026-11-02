#!/usr/bin/env python
"""CPU unit tests for length-grouped batching (ugc/length_group.py, planning mode of ugc/many_label.py, label_norm in ugc/trainer.py).
Needs data/stage2_pilot_smoke (1.6k pilot-like rows with label pools) and the S1 tokenizer. Run: python scripts/test_length_group.py"""
import json, math, os, random, sys, tempfile, types
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accelerate.data_loader import BatchSamplerShard, SkipBatchSampler
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from torch.utils.data import BatchSampler, DataLoader
from transformers import AutoTokenizer
from ugc.config import load_config
from ugc.length_group import (LengthGroupedSampler, LengthModel, base_lengths, calibrate_label_norm, estimate_lengths,
                              parse_length_group)
from ugc.many_label import LabelPools, ManyLabelDataset, PairedCollator, expand_row, plan_seed, planned_expansion

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra, flush=True)

TOK = "/workspace/kaggle/ultragliclass/S1/ultragliclass/runs/stage1_ultrabert150m/final_model"
tok = AutoTokenizer.from_pretrained(TOK)
D = "data/stage2_pilot_smoke"
rows = [json.loads(l) for l in open(f"{D}/train.jsonl")]
snapshot = json.dumps(rows[:200], sort_keys=True)
pools = LabelPools.from_file(f"{D}/label_pools.json")
ML = 1024
ml = load_config("configs/stage2_pilot_C.yaml")["data"]["many_label"]["params"]
common = dict(max_length=ML, problem_type="multi_label_classification", architecture_type="uni-encoder", prompt_first=True,
              labels_tokenizer=None, shuffle_labels=True, max_labels=150)
N = len(rows)
rng = np.random.default_rng(0)
fake_len = rng.integers(20, 1024, size=N)          # synthetic lengths for the pure-sampler tests

# ---------- 1. determinism ----------
s1 = LengthGroupedSampler(fake_len, 4, 2, 50, seed=7); s2 = LengthGroupedSampler(fake_len, 4, 2, 50, seed=7)
a, b = list(s1), list(s2)
check("same (seed, epoch) -> identical stream, also from a fresh instance", a == b)
s2.set_epoch(1)
check("different epoch -> different stream", list(s2) != a)
check("different seed -> different stream", list(LengthGroupedSampler(fake_len, 4, 2, 50, seed=8)) != a)
s2.set_epoch(0)
check("set_epoch back to 0 reproduces epoch 0", list(s2) == a)
check("a second pass over the same epoch gives the same stream (no hidden state)", list(s1) == a)

# ---------- 2. no row lost / duplicated ----------
for world in (1, 2, 4):
    for mb in (1, 4):
        s = LengthGroupedSampler(fake_len, mb, world, 50, seed=3)
        o = list(s)
        G = mb * world
        uniq = len(set(o)) == len(o)
        check(f"world={world} mb={mb}: no duplicates, in range, len = largest multiple of {G} <= {N}",
              uniq and min(o) >= 0 and max(o) < N and len(o) == (N // G) * G == len(s), f"{len(o)} of {N}")
        d0 = set(range(N)) - set(o)
        s.set_epoch(1); d1 = set(range(N)) - set(s.epoch_order(1).tolist())
        check(f"world={world} mb={mb}: dropped tail is < {G} rows and differs between epochs", len(d0) < G and (len(d0) == 0 or d0 != d1))
# every row is seen about equally often over several epochs (the dropped tail rotates)
cnt = np.zeros(N, int)
s = LengthGroupedSampler(fake_len, 4, 2, 50, seed=1)
for e in range(40):
    cnt[s.epoch_order(e)] += 1
check("over 40 epochs every row is used 35..40 times (tail drop is random)", cnt.min() >= 34 and cnt.max() <= 40, f"min {cnt.min()} max {cnt.max()}")

# ---------- 3. grouping quality + DDP shard balance (through accelerate's real BatchSamplerShard) ----------
mb, world = 4, 2
s = LengthGroupedSampler(fake_len, mb, world, 50, seed=5)
shards = []
for r in range(world):
    bs = BatchSamplerShard(BatchSampler(s, batch_size=mb, drop_last=True), num_processes=world, process_index=r, split_batches=False, even_batches=True)
    shards.append([list(x) for x in bs])
check("every rank gets the same number of micro-batches", len({len(x) for x in shards}) == 1, f"{[len(x) for x in shards]}")
flat = [i for sh in shards for b in sh for i in b]
check("ranks get disjoint rows and together exactly the epoch's rows", len(flat) == len(set(flat)) == len(s) and set(flat) == set(s.epoch_order().tolist()))
check("every micro-batch has exactly mb rows", all(len(b) == mb for sh in shards for b in sh))
def pad_cost(batches): return np.mean([max(fake_len[b]) * len(b) for b in batches])
rand_batches = [list(g.permutation(N)[: (N // 8) * 8][i:i + mb]) for g in [np.random.default_rng(9)] for i in range(0, (N // 8) * 8, mb)]
real_mean = fake_len.mean() * mb
check("grouped micro-batches are nearly pad-free (<= 1.15x the real tokens) and far below random", pad_cost([b for sh in shards for b in sh]) < 1.15 * real_mean and pad_cost([b for sh in shards for b in sh]) < 0.75 * pad_cost(rand_batches),
      f"{pad_cost([b for sh in shards for b in sh]):.0f} vs {pad_cost(rand_batches):.0f} padded tokens per micro-batch")
step_cost = lambda sh: np.mean([max(max(fake_len[sh[0][j]]) * mb, max(fake_len[sh[1][j]]) * mb) for j in range(len(sh[0]))])
rand_sh = [[rand_batches[j] for j in range(r, len(rand_batches), 2)] for r in range(2)]
n = min(len(rand_sh[0]), len(rand_sh[1]))
rand_step = np.mean([max(max(fake_len[rand_sh[0][j]]) * mb, max(fake_len[rand_sh[1][j]]) * mb) for j in range(n)])
check("ranks work on similar-length micro-batches at the same time (DDP step cost = slower rank)", step_cost(shards) < 1.2 * real_mean and step_cost(shards) < 0.75 * rand_step,
      f"{step_cost(shards):.0f} vs {rand_step:.0f}")
# micro-batch order is shuffled: the first 200 micro-batches are not sorted by length
lens = [max(fake_len[b]) for b in shards[0][:200]]
check("micro-batch order is shuffled (not a sorted stream)", np.corrcoef(lens, range(len(lens)))[0, 1] < 0.5 if len(lens) > 20 else True)

# ---------- 4. pairs stay in the same micro-batch / forward pass (real dataset + PairedCollator + sampler) ----------
aug = AugmentationConfig(enabled=False)
ds = ManyLabelDataset(rows, tok, aug, {}, pools=pools, many_label={**ml}, pair_prob=0.5, seed=11, **common)
ds.enable_planning()
lm = LengthModel(base_lengths(rows, tok, ML), rows, tok, pools, ml, ML, seed=11)
samp = LengthGroupedSampler(lm.lengths, 4, 1, 50, seed=11, encode_epoch=True, n_total=N)
dl = DataLoader(ds, batch_size=4, sampler=samp, collate_fn=PairedCollator(DataCollatorWithPadding()), drop_last=True, num_workers=0)
nb = npair = bad = 0
seen_rows = []
for bi, batch in enumerate(dl):
    nb += 1
    if bi >= 120: break
    n = 4
    if "pair_idx" in batch:
        for r, (i, j) in enumerate(batch["pair_idx"].tolist()):
            npair += 1
            bad += int(not (i < n <= j < batch["input_ids"].shape[0] and batch["input_texts"][i] == batch["input_texts"][j]))
            la, lb = batch["labels_text"][i], batch["labels_text"][j]
            for jj, l in enumerate(lb):
                bad += int(int(batch["pair_perm"][r, jj]) >= 0 and la[int(batch["pair_perm"][r, jj])] != l)
        bad += int(batch["input_ids"].shape[0] != n + len(batch["pair_idx"]))
    else:
        bad += int(batch["input_ids"].shape[0] != n)
check("both views of a pair are rows of the same batch (same text, aligned label perm, row count = mb + pairs)", bad == 0 and npair > 10, f"{npair} pairs in {min(nb, 121)} batches, bad={bad}")

# ---------- 5. resume: skipping the first k micro-batches gives exactly the tail, in every epoch ----------
resume_ok, n_cmp = True, 0
for world in (1, 2):
    s = LengthGroupedSampler(fake_len, 4, world, 50, seed=2)
    for epoch in (0, 1):
        s.set_epoch(epoch)
        for r in range(world):
            full = [list(x) for x in BatchSamplerShard(BatchSampler(s, batch_size=4, drop_last=True), world, r, False, True)]
            for k in (0, 1, 7, len(full) - 1):
                fresh = LengthGroupedSampler(fake_len, 4, world, 50, seed=2); fresh.set_epoch(epoch)   # "new process after a crash"
                skipped = [list(x) for x in SkipBatchSampler(BatchSamplerShard(BatchSampler(fresh, 4, True), world, r, False, True), skip_batches=k)]
                n_cmp += 1
                resume_ok &= (skipped == full[k:])
check("resume: fresh sampler + skip k batches == tail of the uninterrupted stream (world 1 and 2, epochs 0 and 1, every rank)", resume_ok, f"{n_cmp} comparisons")
# end-to-end through the same objects the Trainer uses: Accelerator.prepare(DataLoader) + skip_first_batches + set_epoch
from accelerate import Accelerator, skip_first_batches
acc = Accelerator(cpu=True)
class Idx(torch.utils.data.Dataset):
    def __len__(self): return N
    def __getitem__(self, i): return torch.tensor([i])
ss = LengthGroupedSampler(fake_len, 4, 1, 50, seed=4)
dlp = acc.prepare(DataLoader(Idx(), batch_size=4, sampler=ss, drop_last=True))
ss.set_epoch(1)
full = [b.flatten().tolist() for b in dlp]
ss2 = LengthGroupedSampler(fake_len, 4, 1, 50, seed=4)
dlp2 = acc.prepare(DataLoader(Idx(), batch_size=4, sampler=ss2, drop_last=True))
sk = skip_first_batches(dlp2, 10)
ss2.set_epoch(1)
tail = [b.flatten().tolist() for b in sk]
check("Accelerator.prepare + skip_first_batches + set_epoch(1) reproduces the tail of epoch 1", tail == full[10:], f"{len(tail)} batches")
check("len(dataloader) = micro-batches per rank (Trainer derives steps per epoch from it)", len(dlp) == len(ss) // 4)

# ---------- 6. planning mode: the sampler's replay of the expansion draws mirrors expand_row; fetches are deterministic ----------
ds_p = ManyLabelDataset(rows, tok, aug, {}, pools=pools, many_label={**ml}, pair_prob=0.0, seed=11, **common)
ds_p.enable_planning()
tax = [i for i, r in enumerate(rows) if r.get("pool")][:400]
mism = exact = 0
for i in tax:
    r = rows[i]
    pc = {**ml, **ml.get("pool_overrides", {}).get(r["pool"], {})}
    k = planned_expansion(11, 0, i, len(r["all_labels"]), len(pools.pools[r["pool"]]), pc)
    got = len(ds_p[i]["labels_text"])
    if k is None:
        mism += int(got != len(r["all_labels"]))
    else:
        mism += int(got > k or got < len(r["all_labels"])); exact += int(got == k)
check("planned expansion decision matches the real fetch (expand or not; label count <= planned k)", mism == 0, f"{len(tax)} rows, label count == planned k for {exact}")
chk = [i for i in tax[:60]]
same = all(set(ds_p[i]["labels_text"]) == set(ds_p[i]["labels_text"]) for i in chk)
diff_epoch = sum(set(ds_p[i]["labels_text"]) != set(ds_p[i + N]["labels_text"]) for i in chk)
check("planning mode: same (epoch,row) -> same expansion; next epoch -> different draws", same and diff_epoch > 20, f"{diff_epoch}/{len(chk)} rows differ in epoch 1")
check("stored rows are never mutated", json.dumps(rows[:200], sort_keys=True) == snapshot)
# planning off = original behaviour (index == row, rng from torch seed): ds[i] must equal ds[i] semantics, nothing encoded
ds_off = ManyLabelDataset(rows, tok, aug, {}, pools=pools, many_label={**ml}, pair_prob=0.0, seed=11, **common)
check("planning OFF by default: plain indices only", ds_off._plan is False and len(ds_off[0]["input_ids"]) > 0)

# ---------- 7. length estimate ----------
plain = GLiClassDataset(rows, tok, aug, {}, **common)
bl = base_lengths(rows, tok, ML)
sample = random.Random(0).sample(range(N), 300)
true = np.array([len(plain[i]["input_ids"]) for i in sample])
check("cached/cheap base length == the dataset's real tokenised length (no augmentation/expansion)", np.array_equal(bl[sample], true), f"max |diff| {np.abs(bl[sample] - true).max()}")
with tempfile.TemporaryDirectory() as td:
    a1 = estimate_lengths(rows, tok, ML, td, log=lambda *_: None)
    t = [f for f in os.listdir(td) if f.startswith("lengths_")]
    a2 = estimate_lengths(rows, tok, ML, td, log=lambda *_: None)
    check("length cache written once and reloaded identically", len(t) == 1 and np.array_equal(a1, a2) and np.array_equal(a1, bl))
    a3 = estimate_lengths(rows[:-5], tok, ML, td, log=lambda *_: None)
    check("cache key changes with the row set (no stale cache)", len(a3) == N - 5 and len(os.listdir(td)) == 2)
lens_e0, lens_e1 = lm.lengths(0), lm.lengths(1)
check("LengthModel: pool rows get longer (expansion), other rows keep the base length; epochs differ", (lens_e0 >= bl).all() and (lens_e0 > bl).sum() > 50 and not np.array_equal(lens_e0, lens_e1)
      and all(lens_e0[i] == bl[i] for i, r in enumerate(rows) if not r.get("pool")), f"{(lens_e0 > bl).sum()} rows lengthened of {N}")
# the predictor really helps: padded tokens when grouping by the model vs by base length, measured on real fetches (epoch 0, planning mode)
real = np.array([len(ds_p[i]["input_ids"]) for i in range(N)])
def padded(lengths):
    o = LengthGroupedSampler(lengths, 4, 1, 50, seed=0).epoch_order(0)
    return np.mean([real[o[i:i + 4]].max() * 4 for i in range(0, len(o) - 3, 4)])
rnd = np.mean([real[p[i:i + 4]].max() * 4 for p in [np.random.default_rng(0).permutation(N)] for i in range(0, N - 3, 4)])
check("grouping by the planned length pads less than random AND less than grouping by the unexpanded length",
      padded(lens_e0) < 0.75 * rnd and padded(lens_e0) <= padded(bl), f"random {rnd:.0f}, base-length {padded(bl):.0f}, planned {padded(lens_e0):.0f} padded tokens/micro-batch")

# ---------- 8. config parsing: default OFF ----------
check("data.length_group absent / false / disabled -> OFF", parse_length_group({}) is None and parse_length_group({"length_group": False}) is None
      and parse_length_group({"length_group": {"enabled": False}}) is None)
lg = parse_length_group({"length_group": True, "mega_batch_mult": 30})
check("data.length_group: true (+ data.mega_batch_mult) -> ON with defaults", lg == {"mega_batch_mult": 30, "loss_norm": True, "cache_dir": None}, str(lg))
check("existing pilot / speedtest configs do not enable it", all(not parse_length_group(load_config(f"configs/{c}.yaml")["data"]) for c in
      ("stage2_pilot_A", "stage2_pilot_B", "stage2_pilot_C", "speedtest_T2_nockpt_b4", "stage2_scale")))

# ---------- 9. loss normalisation: slots / FIXED norm instead of slots / (slots in this micro-batch) ----------
import gliclass.training as gt
from ugc.trainer import StrictTrainer
def fake_parent(self, model, inputs, return_outputs=False, **kw):
    labels = inputs["labels"]; valid = (labels != -100)
    per = torch.where(valid, labels.float() * 0 + 1.0, torch.zeros(()))        # every valid slot has loss 1
    loss = (per * valid).sum() / valid.sum().clamp_min(1)                        # model-style: mean over valid slots of this micro-batch
    return loss, types.SimpleNamespace(class_mask=valid.long(), logits=torch.zeros_like(labels, dtype=torch.float))
orig = gt.Trainer.compute_loss; gt.Trainer.compute_loss = fake_parent
try:
    t = object.__new__(StrictTrainer); t.label_norm = 40.0; t.consistency_weight = 0.0
    few = {"labels": torch.tensor([[1, 0, -100, -100], [1, 1, -100, -100]])}      # 4 valid slots
    many = {"labels": torch.tensor([[1, 0, 1, 0]] * 2)}                          # 8 valid slots
    m = types.SimpleNamespace(training=True)
    lf, lm_ = float(t.compute_loss(m, dict(few))), float(t.compute_loss(m, dict(many)))
    check("label_norm: micro-batch loss = masked sum / fixed norm (4 slots -> 4/40, 8 slots -> 8/40; was 1.0 for both)", abs(lf - 0.1) < 1e-6 and abs(lm_ - 0.2) < 1e-6, f"{lf:.3f} {lm_:.3f}")
    m.training = False
    check("label_norm is NOT applied at eval (eval_loss stays comparable)", abs(float(t.compute_loss(m, dict(many))) - 1.0) < 1e-6)
    t.label_norm = None
    m.training = True
    check("label_norm=None (default) leaves the loss untouched", abs(float(t.compute_loss(m, dict(many))) - 1.0) < 1e-6)
finally:
    gt.Trainer.compute_loss = orig

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)

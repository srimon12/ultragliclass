#!/usr/bin/env python
"""GPU-hour estimate for the three stage-2 PILOT arms (A/B/C, 1,500 steps, pilot data mix without DBpedia/Bitext), calibrated on the measured stage-1 run (3.0 s/step, T4 x2, 64 rows/step as 2 GPUs x 4
accumulation x 8 rows, gradient checkpointing, no length bucketing).

Cost model: a micro-batch of R rows padded to L tokens costs  R * L * (1 + L / (6 * d_model))  units  (linear layers plus
the attention term; d_model 768). A step costs max over the 2 GPUs of the sum of their 4 micro-batches. Row token lengths
come from tokenising real rows with the same dataset class the trainer uses (ManyLabelDataset incl. expansion and
pairing). seconds-per-unit is fitted so that the stage-1 mix reproduces 3.0 s/step.
Sources of rows: S1 val.jsonl (gliclass / logic / jev proxies for the natural sources, incl. procedural-style rows),
data/stage2_smoke (taxonomy rows), re-weighted by the true row counts of each recipe."""
import argparse, json, os, random, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gliclass.data_processing import AugmentationConfig
from transformers import AutoTokenizer
from ugc.many_label import LabelPools, ManyLabelDataset

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600, help="simulated steps per estimate")
ap.add_argument("--out", default="results/stage2_pilot_cost_estimate.json")
args = ap.parse_args()
D, ML = 768, 1024
tok = AutoTokenizer.from_pretrained("/workspace/kaggle/ultragliclass/S1/ultragliclass/runs/stage1_ultrabert150m/final_model")
common = dict(max_length=ML, problem_type="multi_label_classification", architecture_type="uni-encoder", prompt_first=True,
              labels_tokenizer=None, shuffle_labels=True, max_labels=150)
val = [json.loads(l) for l in open("data/stage1/val.jsonl")]
gl = [r for r in val if r["source"] == "knowledgator/gliclass-v2.0"]
logic = [r for r in val if "logic" in r["source"]]
jev_all = [r for r in val if r["source"].startswith("tasksource/tasksource-jev")]
proc = [r for r in jev_all if "procedural-typed-decisions/" in r["source"]]
jev = [r for r in jev_all if "procedural-typed-decisions/" not in r["source"]]
tax = [json.loads(l) for l in open("data/stage2_pilot_smoke/train.jsonl")]
tax = [r for r in tax if r["kind"] == "taxonomy"]
by_pool = {}
for r in tax:
    by_pool.setdefault(r.get("pool") or "fin-sent", []).append(r)
pools = LabelPools.from_file("data/stage2_pilot_smoke/label_pools.json")
# vocabulary pool for gliclass rows (labels of the val rows themselves; the real one has 20k labels)
vocab = sorted({l for r in gl for l in r["all_labels"]})
pools.pools["glic-v2"] = vocab
pools._nn.pop("glic-v2", None)
for r in gl:
    r["pool"] = "glic-v2"
print({k: len(v) for k, v in by_pool.items()}, "gl", len(gl), "logic", len(logic), "jev", len(jev), "proc", len(proc), flush=True)

ML_CFG = {"expand_prob": 0.7, "k_min": 20, "k_max": 150, "hard_frac": 0.5, "dup_sim": 0.8, "text_keep_tokens": 256,
          "pair_min_labels": 8, "pool_overrides": {"glic-v2": {"expand_prob": 0.10, "k_min": 12, "k_max": 60, "hard_frac": 0.2, "dup_sim": 0.6}}}


def lengths(rows, many_label, pair_prob, n=1200, seed=0):
    ds = ManyLabelDataset(rows, tok, AugmentationConfig(enabled=False), {}, pools=pools if many_label else None,
                          many_label=ML_CFG, pair_prob=pair_prob, seed=seed, **common)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        it = ds[rng.randrange(len(rows))]
        k = 1 + int("pair" in it)
        out.append((len(it["input_ids"]), k, len(it["labels_text"])))
    return out


def simulate(mix, steps, seed=0):
    """mix = [(weight, [(len, nrows, nlabels), ...])] -> (mean step cost units, mean padded len, mean rows/micro-batch, mean labels)."""
    rng = random.Random(seed)
    w = [m[0] for m in mix]
    costs, padded, rows_mb, labs = [], [], [], []
    for _ in range(steps):
        gpu = []
        for g in range(2):
            c = 0.0
            for _mb in range(4):
                items = []
                for _r in range(8):
                    src = rng.choices(range(len(mix)), weights=w)[0]
                    items.append(rng.choice(mix[src][1]))
                L = max(i[0] for i in items); R = sum(i[1] for i in items)   # pairs add rows of the same length
                c += R * L * (1 + L / (6 * D))
                padded.append(L); rows_mb.append(R); labs.extend(i[2] for i in items)
            gpu.append(c)
        costs.append(max(gpu))
    return statistics.mean(costs), statistics.mean(padded), statistics.mean(rows_mb), statistics.mean(labs)


L_gl0, L_logic, L_jev, L_proc = (lengths(x, False, 0) for x in (gl, logic, jev, proc))
# ---- S1 calibration: 300k gl / 10.4k logic / 100k jev ----
s1 = [(300_000, L_gl0), (10_422, L_logic), (100_000, L_jev + L_proc[:0])]
c1, p1, r1, k1 = simulate(s1, args.steps)
sec_per_unit = 3.0 / c1
print(f"S1 sim: padded len {p1:.0f}, labels/row {k1:.1f}; 3.0 s/step -> {sec_per_unit:.3e} s/unit", flush=True)

# pilot recipe row counts (configs/data_stage2_pilot.yaml)
PILOT_W = {"gl": 47_100, "logic": 205 * 3, "jev": 23_500, "proc": 11_750,
           "intents": 600 * 3, "legal-clauses": 4_160, "fin-topics": 1_180 * 2, "emotions": 2_770, "fin-sent": 660 * 3}


def stage2(many_label, pair_prob, taxonomy=True):
    mix = [(PILOT_W["gl"], lengths(gl, many_label, 0)), (PILOT_W["logic"], L_logic), (PILOT_W["jev"], L_jev), (PILOT_W["proc"], L_proc)]
    if taxonomy:
        for pool in ("intents", "legal-clauses", "fin-topics", "emotions", "fin-sent"):
            rows = by_pool.get(pool)
            if rows:
                mix.append((PILOT_W[pool], lengths(rows, many_label, pair_prob)))
    total = sum(m[0] for m in mix)
    c, p, r, k = simulate(mix, args.steps)
    sec = c * sec_per_unit
    return {"rows": total, "steps": 1500, "sec_per_step": sec, "padded_len": p, "rows_per_microbatch": r, "labels_per_row": k,
            "t4x2_hours": 1500 * sec / 3600}


res = {
    "calibration": {"s1_sim_padded_len": p1, "s1_labels_per_row": k1, "sec_per_unit": sec_per_unit},
    "A0_natural_rows_only(no taxonomy, no expansion)": stage2(False, 0.0, taxonomy=False),
    "A_pilot_no_expansion_no_consistency": stage2(False, 0.0),
    "B_pilot_expansion": stage2(True, 0.0),
    "C_pilot_expansion+consistency(pair_prob 0.25)": stage2(True, 0.25),
}
for k, v in res.items():
    if isinstance(v, dict) and "t4x2_hours" in v:
        print(f"{k:50s} rows {v['rows']}  {v['sec_per_step']:.2f} s/step  padded {v['padded_len']:.0f} labels/row {v['labels_per_row']:.1f}  "
              f"-> {v['t4x2_hours']:.2f} T4x2 h train (1500 steps)", flush=True)
json.dump(res, open(args.out, "w"), indent=2)

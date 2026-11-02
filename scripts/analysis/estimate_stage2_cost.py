#!/usr/bin/env python
"""GPU-hour estimate for stage 2. `--plan_full --steps N` (24 Oct 2026): session plan of the full arm-C run from the MEASURED fast-setting
s/step (2.4 s/step x 1.05 pairs; see plan_full). Without it: the original token simulation, calibrated on the measured stage-1 run (3.0 s/step, T4 x2, 64 rows/step as 2 GPUs x 4
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
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--out", default="results/stage2_cost_estimate.json")
ap.add_argument("--plan_full", action="store_true", help="session plan of the FULL arm-C run from MEASURED s/step (no simulation)")
ap.add_argument("--sec_per_step", type=float, default=2.4, help="measured fast-setting s/step (speed test T2 2.40; ablation arm C at b4 2.24-2.37)")
ap.add_argument("--pair_factor", type=float, default=1.05, help="arm-C paired views (pilot C/A 4.008/3.669 = 1.09; at b4 the ablation already includes them)")
ap.add_argument("--session_cap_h", type=float, default=10.8, help="notebook TOTAL_MAX_HOURS (+ ~0.2 h output upload => <= 11 h)")
args = ap.parse_args()


def plan_full(a):
    """24 Oct 2026 update: per-step cost from the Kaggle MEASUREMENTS (fast settings, T4 x2), not the token simulation below.
    Overheads (h) are measured where possible (ablation/S1 notebook logs) and marked EST otherwise."""
    sps = a.sec_per_step * a.pair_factor
    periodic_s = 18 + 25 + 0.0      # per 500 steps: val eval 18 s (measured) + checkpoint save ~25 s (EST, 1.8 GB)
    per_step = sps + periodic_s / 500
    ov1 = {"setup_pip_init_distcheck": 0.05,            # measured 1.2 min (ablation)
           "data_prep_full_mix": 0.20,                  # EST: S1 410k rows 1.5 min; ~1.74M rows + share/md5/overlap scans
           "train_startup_load_1.7M_rows": 0.10,        # EST: 2 ranks x json load + label collection + 20k-label TF-IDF neighbours
           "mid_eval": 0.12,                            # measured 6.5-7 min (quick eval, ablation)
           "train_budget_reserve_unused": 0.0}
    ov2 = {"setup_pip_init_distcheck": 0.05, "restore_prev_output_~4GB": 0.05, "train_startup_load_1.7M_rows": 0.10,
           "resume_skip_batches": 0.03, "final_eval": 0.12}
    upload = 0.2                                         # EST: Kaggle output commit (~4 GB), outside the notebook clock
    train1_h = a.session_cap_h - 0.2 - sum(v for k, v in ov1.items() if k != "mid_eval") - 0.15   # eval reserve 0.2, startup slack
    steps1 = int(train1_h * 3600 / per_step)
    steps2 = a.steps - steps1
    train2_h = steps2 * per_step / 3600
    s1 = sum(ov1.values()) + train1_h + upload
    s2 = sum(ov2.values()) + train2_h + upload
    res = {"steps": a.steps, "rows": a.steps * 64, "sec_per_step_used": round(sps, 3), "sec_per_step_incl_periodic": round(per_step, 3),
           "train_hours_total": round(a.steps * per_step / 3600, 2),
           "session1": {"steps": steps1, "train_h": round(train1_h, 2), "overheads_h": ov1, "upload_h": upload, "total_h": round(s1, 2)},
           "session2": {"steps": steps2, "train_h": round(train2_h, 2), "overheads_h": ov2, "upload_h": upload, "total_h": round(s2, 2)},
           "total_gpu_h": round(s1 + s2, 2),
           "if_measured_2.37_s_per_step_total_h": round(s1 + s2 - a.steps * (sps - 2.37) / 3600, 2),
           "if_10pct_slower_total_h": round(s1 + s2 + 0.1 * a.steps * sps / 3600, 2)}
    print(json.dumps(res, indent=1))
    return res


if args.plan_full:
    json.dump(plan_full(args), open(args.out, "w"), indent=2)
    sys.exit(0)
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
tax = [json.loads(l) for l in open("data/stage2_smoke/train.jsonl")]
tax = [r for r in tax if r["kind"] == "taxonomy"]
by_pool = {}
for r in tax:
    by_pool.setdefault(r.get("pool") or "fin-sent", []).append(r)
pools = LabelPools.from_file("data/stage2_smoke/label_pools.json")
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

def stage2(many_label, pair_prob, taxonomy=True):
    mix = [(1_200_000, lengths(gl, many_label, 0)), (15_600, L_logic), (600_000, L_jev), (300_000, L_proc)]
    if taxonomy:
        tw = {"intents": 8_600 * 3 + 26_000, "legal-clauses": 60_000, "dbpedia": 120_000, "fin-topics": 17_000 * 2, "emotions": 40_000, "fin-sent": 9_500 * 3}
        for pool, n in tw.items():
            rows = by_pool.get(pool)
            if rows:
                mix.append((n, lengths(rows, many_label, pair_prob)))
    total = sum(m[0] for m in mix)
    c, p, r, k = simulate(mix, args.steps)
    steps_full = total / 64
    sec = c * sec_per_unit
    return {"rows": total, "steps": steps_full, "sec_per_step": sec, "padded_len": p, "rows_per_microbatch": r, "labels_per_row": k,
            "t4x2_hours": steps_full * sec / 3600}

res = {
    "calibration": {"s1_sim_padded_len": p1, "s1_labels_per_row": k1, "sec_per_unit": sec_per_unit},
    "A_stage2_no_taxonomy_no_expansion(first draft)": stage2(False, 0.0, taxonomy=False),
    "B_taxonomy_static_(no expansion)": stage2(False, 0.0),
    "C_taxonomy+expansion": stage2(True, 0.0),
    "D_taxonomy+expansion+consistency(pair_prob 0.25)": stage2(True, 0.25),
}
for k, v in res.items():
    if isinstance(v, dict) and "t4x2_hours" in v:
        print(f"{k:55s} rows {v['rows']/1e6:.2f}M  {v['sec_per_step']:.2f} s/step  steps {v['steps']:.0f}  "
              f"padded {v['padded_len']:.0f} labels/row {v['labels_per_row']:.1f}  -> {v['t4x2_hours']:.0f} T4x2 h", flush=True)
json.dump(res, open(args.out, "w"), indent=2)

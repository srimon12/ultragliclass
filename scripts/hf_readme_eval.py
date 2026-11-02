#!/usr/bin/env python
"""Lean, publishable benchmark for the HF README (code v9). Thin wrapper around eval.py (GLiClassScorer, load_eval_set,
global_temperature) and scripts/eval_long_jev.py (run_eval, compare). No protocol changes vs eval.py except:
FULL test splits (no 500-row subsample), length-sorted batches (padding="longest" -> same inputs, faster), per-chunk
OOM halving, and per-(model, dataset, permutation) resumable outputs so a timeout loses nothing.

Protocol (frozen, written to results/hf_readme/protocol.json):
  * suite  = the 14 GLiClass-v3 card sets (eval.SUITES["gliclass"]), FULL test split (cap_sotu: its only split, train;
             financial_phrasebank: the full 2,264-row allagree set), single-label softmax argmax, macro-F1.
  * intent = Banking77 (77 labels, = the suite run) + CLINC-150 in-scope (150 labels, 4,500 test rows), accuracy.
  * max_length 1024, fp16, uni-encoder pipeline formatting (labels -> <<SEP>> -> text), all labels in one pass.
  * label order: perm 0 = canonical (dataset feature order) = HEADLINE; perms 1..K-1 = seeded shuffles (robustness).
  * tokenizer: each model with its authors' setting (ours: add_prefix_space=False as trained; Knowledgator peers:
    add_prefix_space=True as in their test_gliclass.py). Knob in the notebook.
  * calibration: one global temperature per model on calib/stage1_val.jsonl (2,000 rows) -> ECE@T only (argmax unchanged).
  * long-Jev: calib/jev_long_test.jsonl (431 rows, states >= 6k chars), max_length 8192, perm 0, paired vs v0.1.

    python scripts/hf_readme_eval.py prep      --out results/hf_readme --models_json M.json
    CUDA_VISIBLE_DEVICES=0 python scripts/hf_readme_eval.py worker --out results/hf_readme --worker 0 --hard_stop_unix T
    python scripts/hf_readme_eval.py aggregate --out results/hf_readme
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, sys, time, traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

SUITE = ["SetFit/CR", "SetFit/sst2", "SetFit/sst5", "stanfordnlp/imdb", "SetFit/20_newsgroups", "SetFit/enron_spam",
         "massive", "banking77", "financial_phrasebank", "ag_news", "dair-ai/emotion", "MoritzLaurer/cap_sotu",
         "rotten_tomatoes", "snips"]
IN_DOMAIN_FOR_US = ["massive", "financial_phrasebank"]      # HWU64 / fin-tweets sentiment in our S2 train mix
INTENT = ["banking77", "clinc150"]
SHORT = {"SetFit/CR": "CR", "SetFit/sst2": "sst2", "SetFit/sst5": "sst5", "stanfordnlp/imdb": "imdb",
         "SetFit/20_newsgroups": "20_newsgroups", "SetFit/enron_spam": "spam", "massive": "massive",
         "banking77": "banking", "financial_phrasebank": "financial_phrasebank", "ag_news": "ag_news",
         "dair-ai/emotion": "emotion", "MoritzLaurer/cap_sotu": "cap_sotu", "rotten_tomatoes": "rotten_tomatoes",
         "snips": "snips", "clinc150": "clinc150"}
# Published per-set macro-F1 from the gliclass-*-v3.0 model cards (benchmark table, fetched 1 Nov 2026; notes/card_*.md)
CARD = {
    "gliclass-base-v3.0": {"CR": .9127, "sst2": .8959, "sst5": .3376, "20_newsgroups": .4759, "spam": .6760,
                           "financial_phrasebank": .8971, "imdb": .9251, "ag_news": .7279, "emotion": .4447,
                           "cap_sotu": .4614, "rotten_tomatoes": .7943, "massive": .5040, "banking": .4698, "snips": .9474},
    "gliclass-modern-base-v3.0": {"CR": .8902, "sst2": .8959, "sst5": .2756, "20_newsgroups": .3433, "spam": .6398,
                                  "financial_phrasebank": .4200, "imdb": .9158, "ag_news": .6663, "emotion": .4254,
                                  "cap_sotu": .3625, "rotten_tomatoes": .7070, "massive": .3442, "banking": .3561, "snips": .5663},
}
OURS = "ultragliclass-v0.1"


def slug(s):
    return s.replace("/", "__")


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ prep
def cmd_prep(a):
    """Download / parse every eval set once (texts stay in a private dir, never in the outputs) + pre-cache peer models."""
    from eval import load_eval_set
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.sets_dir, exist_ok=True)
    models = json.load(open(a.models_json))
    man = {}
    for name in SUITE + ["clinc150"]:
        t0 = time.time()
        texts, classes, gold = load_eval_set(name, clinc_oos=False)
        if a.max_rows:            # smoke only
            idx = sorted(random.Random(0).sample(range(len(texts)), min(a.max_rows, len(texts))))
            texts, gold = [texts[i] for i in idx], [gold[i] for i in idx]
        assert all(g in set(classes) for g in gold), name
        json.dump({"texts": texts, "classes": classes, "gold": gold}, open(os.path.join(a.sets_dir, slug(name) + ".json"), "w"))
        os.makedirs(os.path.join(a.out, "gold"), exist_ok=True)     # label ids only (no texts) so aggregate can re-run anywhere
        c2i = {c: i for i, c in enumerate(classes)}
        np.save(os.path.join(a.out, "gold", slug(name) + ".npy"), np.array([c2i[g] for g in gold], dtype=np.int32))
        h = hashlib.md5(json.dumps([texts, gold]).encode()).hexdigest()
        man[name] = {"n": len(texts), "n_classes": len(classes), "classes": classes, "md5_texts_gold": h,
                     "chars_median": float(np.median([len(t) for t in texts]))}
        say(f"set {name:28s} n={len(texts):6d} C={len(classes):3d} {time.time() - t0:.0f}s")
    json.dump(man, open(os.path.join(a.out, "evalsets_manifest.json"), "w"), indent=1)
    from huggingface_hub import HfApi, snapshot_download
    for m, spec in models.items():
        if not os.path.isdir(spec["path"]):
            local = snapshot_download(spec["path"])
            spec["hf_id"], spec["path"] = spec["path"], local
            try:
                spec["hf_sha"] = HfApi().model_info(spec["hf_id"]).sha
            except Exception as e:  # noqa: BLE001
                spec["hf_sha"] = f"unknown ({type(e).__name__})"
        cfg = json.load(open(os.path.join(spec["path"], "config.json")))
        spec["encoder"] = (cfg.get("encoder_config") or {}).get("_name_or_path") or cfg.get("encoder_model_name")
        say("model", m, spec)
    json.dump(models, open(os.path.join(a.out, "models.json"), "w"), indent=1)
    # job list: tier 0 calib, 1 headline perm 0, 2 long-Jev, 3 robustness perms
    jobs = []
    for m in models:
        jobs.append({"tier": 0, "kind": "calib", "model": m})
        jobs += [{"tier": 1, "kind": "cls", "model": m, "dataset": d, "perm": 0} for d in SUITE + ["clinc150"]]
        jobs.append({"tier": 2, "kind": "longjev", "model": m})
        jobs += [{"tier": 3, "kind": "cls", "model": m, "dataset": d, "perm": k}
                 for k in range(1, a.permutations) for d in SUITE + ["clinc150"]]
    for j in jobs:
        j["id"] = f"{j['model']}__{j['kind']}" + (f"__{slug(j['dataset'])}__p{j['perm']}" if j["kind"] == "cls" else "")
    json.dump(jobs, open(os.path.join(a.out, "jobs.json"), "w"), indent=1)
    import torch, transformers, gliclass, datasets
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "gliclass": getattr(gliclass, "__version__", "?"),
           "datasets": datasets.__version__, "cuda": torch.cuda.is_available(),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    json.dump(env, open(os.path.join(a.out, "env.json"), "w"), indent=1)
    json.dump({"max_length": a.max_length, "longjev_max_length": a.longjev_max_length, "permutations": a.permutations,
               "dtype": "fp16" if env["cuda"] else "fp32", "calib_file": a.calib_file, "calib_n": a.calib_n,
               "longjev_slice": a.longjev_slice, "max_rows_smoke": a.max_rows, "seed": 0,
               "suite": SUITE, "intent": INTENT, "in_domain_for_us": IN_DOMAIN_FOR_US}, open(os.path.join(a.out, "protocol.json"), "w"), indent=1)
    say(f"prep done: {len(jobs)} jobs")


# ------------------------------------------------------------------------------------------------ worker
class Stop(Exception):
    pass


def chunked_logits(sc, texts, labels, bs, deadline):
    """scorer.logits on length-sorted chunks; halves the chunk on CUDA OOM (down to 1). Returns list in input order."""
    import torch
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = [None] * len(texts)
    s = 0
    cur = bs
    while s < len(order):
        if deadline and time.time() > deadline:
            raise Stop("hard stop reached mid-job")
        idx = order[s:s + cur]
        try:
            sc.batch_size = len(idx)
            res = sc.logits([texts[i] for i in idx], labels)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur == 1:
                raise
            cur = max(1, cur // 2); say(f"  OOM -> batch {cur}"); continue
        for i, x in zip(idx, res):
            out[i] = np.asarray(x, dtype=np.float32)
        s += len(idx)
    return out, cur


def claim(out, job):
    os.makedirs(os.path.join(out, "claims"), exist_ok=True)
    try:
        fd = os.open(os.path.join(out, "claims", job["id"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); return True
    except FileExistsError:
        return False


def job_out(out, job):
    if job["kind"] == "cls":
        return os.path.join(out, "raw", job["model"], f"{slug(job['dataset'])}__p{job['perm']}.npz")
    if job["kind"] == "calib":
        return os.path.join(out, "raw", job["model"], "calib.json")
    return os.path.join(out, "longjev", f"{job['model']}.json")


def cmd_worker(a):
    import torch
    from eval import GLiClassScorer, global_temperature
    models = json.load(open(os.path.join(a.out, "models.json")))
    jobs = json.load(open(os.path.join(a.out, "jobs.json")))
    pref = a.affinity.split(",") if a.affinity else list(models)
    rank = {m: (pref.index(m) if m in pref else 99) for m in models}
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev != "cpu" else torch.float32
    reserve = a.reserve_min * 60
    sc, sc_model = None, None
    log = open(os.path.join(a.out, f"worker{a.worker}.jsonl"), "a", buffering=1)
    while True:
        if a.hard_stop_unix and time.time() > a.hard_stop_unix - reserve:
            say("worker", a.worker, "stopping: hard-stop reserve reached"); break
        todo = [j for j in jobs if not os.path.exists(job_out(a.out, j)) and
                not os.path.exists(os.path.join(a.out, "claims", j["id"]))]
        if not todo:
            say("worker", a.worker, "no jobs left"); break
        todo.sort(key=lambda j: (j["tier"], 0 if j["model"] == sc_model else 1, rank[j["model"]]))
        job = next((j for j in todo if claim(a.out, j)), None)
        if job is None:
            continue
        spec = models[job["model"]]
        t0 = time.time(); rec = {"job": job["id"], "worker": a.worker, "start": t0}
        try:
            if job["kind"] == "longjev":
                if sc is not None:
                    del sc; sc, sc_model = None, None; torch.cuda.empty_cache()
                import eval_long_jev as elj
                ns = argparse.Namespace(model=spec["path"], slice=a.longjev_slice, max_length=a.longjev_max_length,
                                        batch_size=spec.get("longjev_batch", 1), dtype="auto", permutations=1,
                                        max_rows=a.longjev_max_rows, seed=0, out=job_out(a.out, job),
                                        add_prefix_space=spec.get("add_prefix_space", False))
                try:
                    elj.run_eval(ns)
                    r = json.load(open(ns.out)); r["model_name"] = job["model"]; json.dump(r, open(ns.out, "w"), indent=1)
                    rec["status"] = "ok"
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    json.dump({"model_name": job["model"], "status": "oom", "max_length": a.longjev_max_length,
                               "batch_size": ns.batch_size, "error": str(e)[:500]}, open(ns.out, "w"))
                    rec["status"] = "oom"
            else:
                if sc_model != job["model"]:
                    sc = None; torch.cuda.empty_cache()
                    sc = GLiClassScorer(spec["path"], dev, dt, a.max_length, spec.get("batch", 16),
                                        spec.get("add_prefix_space", False)); sc_model = job["model"]
                os.makedirs(os.path.dirname(job_out(a.out, job)), exist_ok=True)
                if job["kind"] == "calib":
                    sc.batch_size = spec.get("batch", 16)
                    T = global_temperature(sc, a.calib_file, a.calib_n, 0)
                    json.dump({"global_T": T}, open(job_out(a.out, job), "w")); rec["status"] = "ok"
                else:
                    d = json.load(open(os.path.join(a.sets_dir, slug(job["dataset"]) + ".json")))
                    classes = d["classes"]; perm = list(range(len(classes)))
                    if job["perm"] > 0:
                        random.Random(0 + job["perm"]).shuffle(perm)       # = eval.py (seed + k)
                    order = [classes[i] for i in perm]
                    lg, bs_used = chunked_logits(sc, d["texts"], order, spec.get("batch", 16),
                                                 a.hard_stop_unix - 60 if a.hard_stop_unix else None)
                    inv = np.argsort(perm)
                    L = np.stack([x[inv] for x in lg]).astype(np.float16)
                    np.savez_compressed(job_out(a.out, job) + ".tmp.npz", logits=L)
                    os.replace(job_out(a.out, job) + ".tmp.npz", job_out(a.out, job))
                    rec.update(status="ok", n=len(lg), batch_final=bs_used)
        except Stop as e:
            rec["status"] = f"stopped: {e}"; os.remove(os.path.join(a.out, "claims", job["id"]))
            log.write(json.dumps(rec) + "\n"); break
        except Exception as e:  # noqa: BLE001 - log and move on, the aggregate reports missing cells
            rec["status"] = f"error: {type(e).__name__}: {str(e)[:300]}"; traceback.print_exc()
        rec["seconds"] = round(time.time() - t0, 1)
        say("worker", a.worker, job["id"], rec["status"], f"{rec['seconds']}s")
        log.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------------------------------------ aggregate
def macro_f1(y, p, C):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    present = (cm.sum(1) + cm.sum(0)) > 0           # sklearn: labels present in y_true or y_pred
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[present].mean())


def boot_idx(n, B, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def mcnemar(a_ok, b_ok):
    from math import comb
    n01 = int(((~a_ok) & b_ok).sum()); n10 = int((a_ok & ~b_ok).sum()); m = n01 + n10; k = min(n01, n10)
    if m == 0:
        return 1.0, n01, n10
    if m > 2000:   # normal approx with continuity correction for big m
        from math import erf, sqrt
        z = (abs(n01 - n10) - 1) / sqrt(m); return float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))), n01, n10
    return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m), n01, n10


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def cmd_aggregate(a):
    from ugc.metrics import ece, softmax
    models = json.load(open(os.path.join(a.out, "models.json")))
    man = json.load(open(os.path.join(a.out, "evalsets_manifest.json")))
    proto = json.load(open(os.path.join(a.out, "protocol.json")))
    B = a.bootstrap
    S = {"protocol": proto, "models": {}, "paired": {}, "longjev": {}, "missing": []}
    preds, golds = {}, {}
    for m in models:
        cal = os.path.join(a.out, "raw", m, "calib.json")
        T = json.load(open(cal))["global_T"] if os.path.exists(cal) else None
        mm = {"global_T": T, "sets": {}}
        for d in SUITE + ["clinc150"]:
            info = man[d]; C = info["n_classes"]
            if d not in golds:
                gf = os.path.join(a.out, "gold", slug(d) + ".npy")
                golds[d] = np.load(gf).astype(np.int64) if os.path.exists(gf) else None
            y = golds[d]
            runs = []
            for k in range(proto["permutations"]):
                f = os.path.join(a.out, "raw", m, f"{slug(d)}__p{k}.npz")
                if os.path.exists(f):
                    runs.append(np.load(f)["logits"].astype(np.float32))
                elif k == 0:
                    S["missing"].append(f"{m}/{d}/p0")
            if not runs or y is None:
                continue
            L = runs[0]; p0 = L.argmax(-1); preds[(m, d)] = p0
            r = {"n": int(len(y)), "n_classes": C, "macro_f1": macro_f1(y, p0, C), "accuracy": float((p0 == y).mean()),
                 "perms_done": len(runs)}
            bs = [macro_f1(y[i], p0[i], C) for i in boot_idx(len(y), B, 0)]
            r["macro_f1_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            ok = (p0 == y).astype(float); ba = [ok[i].mean() for i in boot_idx(len(y), B, 1)]
            r["accuracy_ci95"] = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]
            if len(runs) > 1:
                ps = [x.argmax(-1) for x in runs]
                r["perm_macro_f1_mean"] = float(np.mean([macro_f1(y, p, C) for p in ps]))
                r["perm_macro_f1_std"] = float(np.std([macro_f1(y, p, C) for p in ps]))
                r["perm_flip_rate"] = float(np.mean(np.any(np.stack(ps) != ps[0], axis=0)))
            P = softmax(L); r["ece"] = ece(P.max(-1), P.argmax(-1) == y)
            if T:
                Pt = softmax(L, T); r["ece_global_T"] = ece(Pt.max(-1), Pt.argmax(-1) == y)
            mm["sets"][d] = r
        for tag, sets in (("avg14", SUITE), ("avg12_zs", [d for d in SUITE if d not in IN_DOMAIN_FOR_US])):
            v = [mm["sets"][d]["macro_f1"] for d in sets if d in mm["sets"]]
            mm[tag] = float(np.mean(v)) if len(v) == len(sets) else None
            pv = [mm["sets"][d].get("perm_macro_f1_mean") for d in sets if d in mm["sets"]]
            mm[tag + "_perm_mean"] = float(np.mean(pv)) if len(pv) == len(sets) and None not in pv else None
        if m in CARD:
            card = CARD[m]
            mm["card_avg14"] = float(np.mean([card[SHORT[d]] for d in SUITE]))
            mm["card_avg12_zs"] = float(np.mean([card[SHORT[d]] for d in SUITE if d not in IN_DOMAIN_FOR_US]))
            mm["repro_delta_avg14_vs_card"] = (mm["avg14"] - mm["card_avg14"]) if mm["avg14"] is not None else None
        lj = os.path.join(a.out, "longjev", f"{m}.json")
        if os.path.exists(lj):
            r = json.load(open(lj))
            S["longjev"][m] = {k: v for k, v in r.items() if k != "per_row"}
        S["models"][m] = mm
    # paired v0.1 - peer
    import eval_long_jev as elj
    for peer in [m for m in models if m != OURS]:
        pr = {}
        for tag, sets in (("avg14", SUITE), ("avg12_zs", [d for d in SUITE if d not in IN_DOMAIN_FOR_US])):
            if not all((OURS, d) in preds and (peer, d) in preds for d in sets):
                continue
            ds = []
            for b in range(B):
                tot = 0.0
                for j, d in enumerate(sets):
                    y = golds[d]; C = man[d]["n_classes"]; i = np.random.default_rng(1000 * b + j).integers(0, len(y), len(y))
                    tot += macro_f1(y[i], preds[(OURS, d)][i], C) - macro_f1(y[i], preds[(peer, d)][i], C)
                ds.append(tot / len(sets))
            pt = S["models"][OURS][tag] - S["models"][peer][tag]
            pr[tag] = {"delta": pt, "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                       "p_boot_le0": float(np.mean(np.array(ds) <= 0))}
        for d in INTENT:
            if (OURS, d) in preds and (peer, d) in preds:
                y = golds[d]; a_ok = preds[(peer, d)] == y; b_ok = preds[(OURS, d)] == y
                p, w, l = mcnemar(a_ok, b_ok)
                pr[d + "_acc"] = {"delta": float(b_ok.mean() - a_ok.mean()), "mcnemar_p": p, "wins": w, "losses": l}
        f_o, f_p = (os.path.join(a.out, "longjev", f"{x}.json") for x in (OURS, peer))
        if all(os.path.exists(f) and "per_row" in json.load(open(f)) for f in (f_o, f_p)):
            pr["longjev"] = elj.compare(f_p, f_o)
        S["paired"][peer] = pr
    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    open(os.path.join(a.out, "TABLES.md"), "w").write(tables(S, models, man))
    say("wrote", os.path.join(a.out, "summary.json"), "and TABLES.md; missing:", S["missing"][:20])


def tables(S, models, man):
    order = [OURS] + [m for m in models if m != OURS]
    M = S["models"]; P = S["paired"]; proto = S["protocol"]
    L = ["# UltraGliClass v0.1: zero-shot benchmark (README tables)", "",
         f"Protocol: full test splits, max_length {proto['max_length']}, {proto['dtype']}, single-label softmax argmax, canonical label order "
         f"(headline), {proto['permutations']} label orders for robustness. Same harness and GPU for every model. "
         "Each model uses its authors' tokenizer setting.", ""]
    if proto.get("max_rows_smoke"):
        L += [f"**SMOKE RUN ({proto['max_rows_smoke']} rows per set): NOT publishable.**", ""]
    hdr = "| Dataset | n | " + " | ".join(order) + " |"
    L += ["## Zero-shot text classification (macro-F1)", "", hdr, "|" + "---|" * (len(order) + 2)]
    for d in SUITE:
        cells = []
        for m in order:
            r = M.get(m, {}).get("sets", {}).get(d)
            cells.append(fmt(r["macro_f1"]) if r else "—")
        star = "¹" if d in IN_DOMAIN_FOR_US else ""
        L.append(f"| {SHORT[d]}{star} | {man[d]['n']:,} | " + " | ".join(cells) + " |")
    L.append("| **Average (14)** | | " + " | ".join(f"**{fmt(M[m].get('avg14'))}**" for m in order) + " |")
    L.append("| **Average (12, zero-shot for all)²** | | " + " | ".join(f"**{fmt(M[m].get('avg12_zs'))}**" for m in order) + " |")
    L.append("| Average (14), mean of label orders | | " + " | ".join(fmt(M[m].get('avg14_perm_mean')) for m in order) + " |")
    L += ["", "¹ In-domain for UltraGliClass: related data (HWU64 intents, financial-tweet sentiment) is in our training mix. "
          "² Excludes massive and financial_phrasebank.", ""]
    L += ["## Intent detection, all labels in one pass (accuracy)", "", "| Dataset | labels | n | " + " | ".join(order) + " |",
          "|" + "---|" * (len(order) + 3)]
    for d, lab in (("banking77", 77), ("clinc150", 150)):
        cells = [fmt(M.get(m, {}).get("sets", {}).get(d, {}).get("accuracy")) for m in order]
        L.append(f"| {'Banking77' if d == 'banking77' else 'CLINC-150 (in-scope)'} | {lab} | {man[d]['n']:,} | " + " | ".join(cells) + " |")
    L += ["", f"## Long-context decisions: long-Jev @ {proto['longjev_max_length']:,} tokens (accuracy)³", "",
          "| Model | accuracy [95% CI] | truncated | Δ v0.1 − model [95% CI] | McNemar p |", "|---|---|---|---|---|"]
    for m in order:
        r = S["longjev"].get(m)
        if not r:
            L.append(f"| {m} | not run | | | |"); continue
        if r.get("status") == "oom":
            L.append(f"| {m} | OOM at {r['max_length']} (T4 16 GB, batch {r['batch_size']}) | | | |"); continue
        c = P.get(m, {}).get("longjev")
        dl = f"{c['delta']:+.3f} [{c['delta_ci95_cluster'][0]:+.3f}, {c['delta_ci95_cluster'][1]:+.3f}]" if c else ""
        pv = f"{c['mcnemar_exact_p']:.1e}" if c else ""
        ci = r["accuracy_ci95_cluster"]
        L.append(f"| {m} | {r['accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['truncated_frac'] * 100:.1f}% | {dl} | {pv} |")
    L += ["", "³ 431 held-out decision rows with states ≥ 6k characters from tasksource/jev-typed-decisions (test split; no state "
          "or group shared with training). **Not zero-shot for UltraGliClass** (other Jev rows are in our training data); "
          "zero-shot for the peers. Cluster bootstrap by state.", ""]
    L += ["## Significance (paired, same rows): v0.1 − peer", "", "| Peer | Δ avg14 [95% CI] | Δ avg12 [95% CI] | Δ Banking77 acc (p) | Δ CLINC acc (p) |",
          "|---|---|---|---|---|"]
    for peer, pr in P.items():
        def ci(t):
            x = pr.get(t); return f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" if x else "—"
        def ac(t):
            x = pr.get(t); return f"{x['delta']:+.3f} ({x['mcnemar_p']:.1e})" if x else "—"
        L.append(f"| {peer} | {ci('avg14')} | {ci('avg12_zs')} | {ac('banking77_acc')} | {ac('clinc150_acc')} |")
    L += ["", "## Reproduction check vs published card numbers (same 14 sets)", "", "| Model | our run avg14 | card avg14 | Δ |", "|---|---|---|---|"]
    for m in order:
        if "card_avg14" in M.get(m, {}):
            L.append(f"| {m} | {fmt(M[m].get('avg14'), 4)} | {M[m]['card_avg14']:.4f} | {fmt(M[m].get('repro_delta_avg14_vs_card'), 4)} |")
    if S["missing"]:
        L += ["", f"**Missing headline cells:** {', '.join(S['missing'])}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "worker", "aggregate"])
    ap.add_argument("--out", default="results/hf_readme")
    ap.add_argument("--sets_dir", default="/tmp/ugc_evalsets")
    ap.add_argument("--models_json")
    ap.add_argument("--max_rows", type=int, default=None, help="SMOKE only: subsample every set")
    ap.add_argument("--permutations", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--longjev_max_length", type=int, default=8192)
    ap.add_argument("--longjev_slice", default="calib/jev_long_test.jsonl")
    ap.add_argument("--longjev_max_rows", type=int, default=None)
    ap.add_argument("--calib_file", default="calib/stage1_val.jsonl")
    ap.add_argument("--calib_n", type=int, default=2000)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--affinity", default="", help="comma list of model names this worker prefers")
    ap.add_argument("--hard_stop_unix", type=float, default=0)
    ap.add_argument("--reserve_min", type=float, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    {"prep": cmd_prep, "worker": cmd_worker, "aggregate": cmd_aggregate}[a.cmd](a)

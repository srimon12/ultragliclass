#!/usr/bin/env python
"""UltraGliClass evaluation suite.

Suites
  gliclass   the zero-shot sets of knowledgator/GLiClass test_gliclass.py (+ snips from the model cards),
             single-label, macro-F1, softmax over the label logits (official pipeline behaviour)
  intent     Banking77 with all 77 labels, CLINC-150 with all 150 in-scope labels (--clinc_oos adds 'oos')
  typed      LocalLLaMA/typed-decisions test (400 cases / 2,000 decisions) via the System One -> GLiClass
             mapping; accuracy (argmax vs gold argmax), KL from gold, Brier, ECE (10 bins), as in the
             dataset's bench/score_with_jobs.py
Every classification set also reports: option-order robustness (K label permutations: mean/std accuracy and
flip rate), ECE before / after temperature scaling (global T fitted on --calib_file, plus a 2-fold
per-dataset T), and latency.

    python eval.py --model runs/stage1_ultrabert150m/final_model --suites gliclass intent typed \
        --calib_file data/stage1/val.jsonl --out results/stage1.json
    python eval.py --model knowledgator/gliclass-modern-base-v3.0 ...     # published control
    python eval.py --backend laya --suites typed                          # Laya control (adapter unverified)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugc.convert import from_system_one  # noqa: E402
from ugc.metrics import brier, ece, fit_temperature, kl, softmax  # noqa: E402


# ----------------------------------------------------------------------------- datasets
def _split(ds, prefer=("test",)):
    for s in prefer:
        if s in ds:
            return ds[s]
    return ds["train"]


def _from_columns(ds, text_col, label_col, classes=None):
    texts = ds[text_col]
    gold = ds[label_col]
    feat = ds.features[label_col]
    if classes is None:
        classes = list(feat.names) if hasattr(feat, "names") else sorted(set(gold))
    if isinstance(gold[0], int):
        gold = [classes[g] for g in gold]
    return list(texts), list(classes), list(gold)


def load_eval_set(name: str, clinc_oos: bool = False):
    """Return (texts, classes, gold). Parquet mirrors replace script-based sets that datasets>=4 can't load."""
    if name == "SetFit/sst5":
        ds = _split(load_dataset("SetFit/sst5"))
        return _from_columns(ds, "text", "label", ["very negative", "negative", "neutral", "positive", "very positive"])
    if name in {"SetFit/CR", "SetFit/sst2", "SetFit/20_newsgroups", "SetFit/enron_spam"}:
        return _from_columns(_split(load_dataset(name)), "text", "label_text")
    if name == "stanfordnlp/imdb":
        return _from_columns(_split(load_dataset(name)), "text", "label", ["negative", "positive"])
    if name == "massive":  # official: AmazonScience/massive en-US intent (script) -> mteb parquet mirror
        return _from_columns(_split(load_dataset("mteb/amazon_massive_intent", "en")), "text", "label_text")
    if name == "banking77":  # official: PolyAI/banking77 (script) -> mteb parquet mirror, same 77 names
        return _from_columns(_split(load_dataset("mteb/banking77")), "text", "label_text")
    if name == "financial_phrasebank":  # official: takala/financial_phrasebank sentences_allagree (script, 2,264)
        d = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
        ds = concatenate_datasets([d["train"], d["test"]])  # = the full 2,264-sentence allagree set
        return _from_columns(ds, "sentence", "label", ["negative", "neutral", "positive"])
    if name == "ag_news":
        return _from_columns(_split(load_dataset("fancyzhx/ag_news")), "text", "label")
    if name == "dair-ai/emotion":
        return _from_columns(_split(load_dataset("dair-ai/emotion", "split")), "text", "label")
    if name == "MoritzLaurer/cap_sotu":
        return _from_columns(_split(load_dataset(name)), "text", "labels")
    if name == "rotten_tomatoes":
        return _from_columns(_split(load_dataset("cornell-movie-review-data/rotten_tomatoes")), "text", "label")
    if name == "snips":
        return _from_columns(_split(load_dataset("benayas/snips")), "text", "category")
    if name == "clinc150":
        ds = _split(load_dataset("clinc/clinc_oos", "plus"))
        names = ds.features["intent"].names
        texts, gold = [], []
        for t, g in zip(ds["text"], ds["intent"]):
            if names[g] == "oos" and not clinc_oos:
                continue
            texts.append(t)
            gold.append(names[g])
        classes = [n for n in names if clinc_oos or n != "oos"]
        return texts, classes, gold
    raise KeyError(name)


SUITES = {
    "gliclass": ["SetFit/CR", "SetFit/sst2", "SetFit/sst5", "stanfordnlp/imdb", "SetFit/20_newsgroups",
                 "SetFit/enron_spam", "massive", "banking77", "financial_phrasebank", "ag_news",
                 "dair-ai/emotion", "MoritzLaurer/cap_sotu", "rotten_tomatoes", "snips"],
    "intent": ["banking77", "clinc150"],
}


# ----------------------------------------------------------------------------- backends
class GLiClassScorer:
    """Raw per-label logits from a GLiClass checkpoint, formatted exactly like the official
    UniEncoderZeroShotClassificationPipeline (labels -> <<SEP>> -> prompt -> text)."""

    def __init__(self, path, device, dtype, max_length, batch_size, add_prefix_space=False):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        self.model = GLiClassModel.from_pretrained(path).to(dtype=dtype)
        # official test_gliclass.py passes add_prefix_space=True (DeBERTa/ModernBERT); our trainer tokenizes
        # without it, so the default here matches training. --add_prefix_space reproduces the official script.
        self.tok = AutoTokenizer.from_pretrained(path, add_prefix_space=add_prefix_space)
        self.pipe = ZeroShotClassificationPipeline(self.model, self.tok, classification_type="single-label",
                                                   device=str(device), max_length=max_length,
                                                   progress_bar=False).pipe
        self.batch_size = batch_size
        self.latency_ms = []

    @torch.no_grad()
    def logits(self, texts, labels, prompts=None):
        """labels: one shared list, or one list per text. Returns a list of 1-D numpy logit arrays."""
        shared = isinstance(labels[0], str)
        out = []
        for i in range(0, len(texts), self.batch_size):
            bt = texts[i:i + self.batch_size]
            bl = labels if shared else labels[i:i + self.batch_size]
            bp = prompts[i:i + self.batch_size] if prompts else None
            t0 = time.perf_counter()
            enc = self.pipe.prepare_inputs(bt, bl, shared, prompt=bp)
            n_cls = len(bl) if shared else max(len(x) for x in bl)
            lg = self.model(**enc, max_num_classes=n_cls).logits.float().cpu().numpy()
            self.latency_ms.append((time.perf_counter() - t0) * 1000 / len(bt))
            for j in range(len(bt)):
                k = len(bl) if shared else len(bl[j])
                out.append(lg[j, :k])
        return out


class LayaScorer:
    """Control backend: Laya (pip `laya`). Its predict(state, questions) API is taken from the README and
    is UNVERIFIED here (no smoke test was run). It returns probabilities, so log-probs stand in for logits."""

    def __init__(self, model_name=None):
        try:
            from laya import Router
        except ImportError as e:
            raise SystemExit("pip install laya==0.4.13 to use --backend laya") from e
        self.router = Router(model_name) if model_name else Router()
        self.latency_ms = []

    def logits(self, texts, labels, prompts=None):
        shared = isinstance(labels[0], str)
        out = []
        for i, text in enumerate(texts):
            labs = labels if shared else labels[i]
            q = {"q": {"type": "choice", "instructions": (prompts[i] if prompts else "") or "Classify the text.",
                       "criteria": {lab: lab for lab in labs}}}
            t0 = time.perf_counter()
            res = self.router.predict(text, q)
            self.latency_ms.append((time.perf_counter() - t0) * 1000)
            ans = res["answers"]["q"]
            probs = ans.get("probabilities") or {ans.get("label"): 1.0}
            out.append(np.log(np.array([float(probs.get(lab, 0.0)) for lab in labs]) + 1e-9))
        return out


# ----------------------------------------------------------------------------- evaluation
def subsample(texts, gold, n, seed):
    if not n or len(texts) <= n:
        return texts, gold
    idx = sorted(random.Random(seed).sample(range(len(texts)), n))
    return [texts[i] for i in idx], [gold[i] for i in idx]


def label_prompt_tokens(scorer, classes):
    tok = getattr(scorer, "tok", None)
    if tok is None:
        return None
    return len(tok("".join(f"<<LABEL>>{c}" for c in classes) + "<<SEP>>")["input_ids"])


def eval_classification(scorer, name, texts, classes, gold, k_perm, global_t, seed, max_length=1024):
    n_lab_tok = label_prompt_tokens(scorer, classes)
    if n_lab_tok and n_lab_tok > max_length - 32:
        print(f"WARNING {name}: the label prompt alone is {n_lab_tok} tokens (max_length {max_length}); "
              f"labels/text get truncated and scores are meaningless. Raise --max_length.", flush=True)
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[g] for g in gold])
    runs = []
    for k in range(k_perm):
        perm = list(range(len(classes)))
        if k > 0:  # run 0 = canonical order, runs 1..K-1 = seeded shuffles
            random.Random(seed + k).shuffle(perm)
        order = [classes[i] for i in perm]
        lg = scorer.logits(texts, order)
        inv = np.argsort(perm)  # back to canonical class order
        runs.append(np.stack([x[inv] for x in lg]))
    L = runs[0]
    preds = [r.argmax(-1) for r in runs]
    accs = [float((p == y).mean()) for p in preds]
    f1s = [float(f1_score(y, p, average="macro")) for p in preds]
    flip = float(np.mean([np.any(np.stack(preds)[:, i] != preds[0][i]) for i in range(len(y))])) if k_perm > 1 else 0.0
    P = softmax(L)
    res = {"n": len(y), "n_classes": len(classes), "label_prompt_tokens": n_lab_tok,
           "macro_f1": f1s[0], "accuracy": accs[0],
           "perm_accuracy_mean": float(np.mean(accs)), "perm_accuracy_std": float(np.std(accs)),
           "perm_macro_f1_mean": float(np.mean(f1s)), "perm_flip_rate": flip,
           "ece": ece(P.max(-1), P.argmax(-1) == y)}
    onehot = np.eye(len(classes))[y]
    if global_t:
        Pg = softmax(L, global_t)
        res["ece_global_T"] = ece(Pg.max(-1), Pg.argmax(-1) == y)
    # 2-fold per-dataset temperature (fit on one half, score the other) - an optimistic in-domain bound
    idx = np.arange(len(y))
    folds = [idx[idx % 2 == 0], idx[idx % 2 == 1]]
    conf, corr, ts = np.zeros(len(y)), np.zeros(len(y)), []
    for a, b in [(0, 1), (1, 0)]:
        if len(folds[a]) == 0 or len(folds[b]) == 0:
            continue
        t = fit_temperature(list(L[folds[a]]), list(onehot[folds[a]]))
        ts.append(t)
        Pb = softmax(L[folds[b]], t)
        conf[folds[b]], corr[folds[b]] = Pb.max(-1), Pb.argmax(-1) == y[folds[b]]
    if ts:
        res["ece_dataset_T"] = ece(conf, corr)
        res["dataset_T"] = float(np.mean(ts))
    return res


def typed_rows(split="test", workflow="all"):
    ds = load_dataset("LocalLLaMA/typed-decisions", workflow, split=split)
    rows = []
    for r in ds:
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        for q_rows, (qname, q) in zip(
                [from_system_one(r["state"], {qn: qq}, gold, "typed-decisions", 10_000, 100_000)
                 for qn, qq in questions.items()], questions.items()):
            for row in q_rows:
                row.update({"case": r["id"], "workflow": r["workflow"], "question": qname, "qtype": q["type"],
                            "gold_probs": [float(row["true_labels"].get(lab, 0.0)) for lab in row["all_labels"]]})
                rows.append(row)
    return rows


def eval_typed(scorer, global_t, max_cases=None, seed=0):
    rows = typed_rows()
    if max_cases:
        keep = set(random.Random(seed).sample(sorted({r["case"] for r in rows}), max_cases))
        rows = [r for r in rows if r["case"] in keep]
    lg = scorer.logits([r["text"] for r in rows], [r["all_labels"] for r in rows], [r["prompt"] for r in rows])
    out = {}
    for tag, t in [("T1", 1.0)] + ([("global_T", global_t)] if global_t else []):
        acc, kls, brs, conf = [], [], [], []
        per_type, per_wf = {}, {}
        for r, x in zip(rows, lg):
            p = softmax(x, t)
            g = np.array(r["gold_probs"])
            g = g / g.sum()
            ok = int(p.argmax() == g.argmax())
            acc.append(ok); kls.append(kl(g, p)); brs.append(brier(g, p)); conf.append(p.max())
            per_type.setdefault(r["qtype"], []).append(ok)
            per_wf.setdefault(r["workflow"], []).append(ok)
        out[tag] = {"decisions": len(rows), "accuracy": float(np.mean(acc)), "kl_from_gold": float(np.mean(kls)),
                    "brier": float(np.mean(brs)), "ece": ece(np.array(conf), np.array(acc), n_bins=10),
                    "accuracy_by_type": {k: float(np.mean(v)) for k, v in per_type.items()},
                    "accuracy_by_workflow": {k: float(np.mean(v)) for k, v in per_wf.items()}}
    return out


def global_temperature(scorer, calib_file, n, seed):
    rows = [json.loads(l) for l in open(calib_file)]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    lg = scorer.logits([r["text"] for r in rows], [r["all_labels"] for r in rows],
                       [r.get("prompt", "") for r in rows])
    targets = []
    for r in rows:
        tl = r["true_labels"]
        v = np.array([float(tl.get(l, 0.0)) if isinstance(tl, dict) else float(l in tl) for l in r["all_labels"]])
        targets.append(v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v)))
    return fit_temperature(lg, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="GLiClass checkpoint dir or HF id")
    ap.add_argument("--backend", default="gliclass", choices=["gliclass", "laya"])
    ap.add_argument("--suites", nargs="+", default=["gliclass", "intent", "typed"])
    ap.add_argument("--datasets", nargs="*", help="override: explicit dataset names from the suites")
    ap.add_argument("--max_per_dataset", type=int, default=None, help="subsample (quick runs); omit = full sets")
    ap.add_argument("--typed_max_cases", type=int, default=None)
    ap.add_argument("--permutations", type=int, default=3)
    ap.add_argument("--calib_file", default=None, help="jsonl (GLiClass format) used to fit the global T")
    ap.add_argument("--calib_n", type=int, default=2000)
    ap.add_argument("--clinc_oos", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add_prefix_space", action="store_true", help="tokenizer flag used by official test_gliclass.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.dtype, torch.float16 if device != "cpu" else torch.float32)
    if args.backend == "gliclass":
        scorer = GLiClassScorer(args.model, device, dtype, args.max_length, args.batch_size, args.add_prefix_space)
    else:
        scorer = LayaScorer(args.model)

    results = {"model": args.model, "backend": args.backend, "device": device, "dtype": str(dtype),
               "max_per_dataset": args.max_per_dataset, "permutations": args.permutations, "datasets": {}}
    global_t = None
    if args.calib_file:
        global_t = global_temperature(scorer, args.calib_file, args.calib_n, args.seed)
        results["global_T"] = global_t
        print(f"global temperature from {args.calib_file}: {global_t:.3f}", flush=True)

    names = args.datasets or [d for s in args.suites if s in SUITES for d in SUITES[s]]
    for name in dict.fromkeys(names):
        t0 = time.time()
        texts, classes, gold = load_eval_set(name, args.clinc_oos)
        texts, gold = subsample(texts, gold, args.max_per_dataset, args.seed)
        r = eval_classification(scorer, name, texts, classes, gold, args.permutations, global_t, args.seed,
                                args.max_length)
        r["seconds"] = round(time.time() - t0, 1)
        results["datasets"][name] = r
        print(f"{name:28s} n={r['n']:5d} C={r['n_classes']:3d} macroF1={r['macro_f1']:.4f} "
              f"acc={r['accuracy']:.4f} perm_std={r['perm_accuracy_std']:.4f} flip={r['perm_flip_rate']:.3f} "
              f"ECE={r['ece']:.3f}" + (f" ECE@T={r['ece_global_T']:.3f}" if global_t else ""), flush=True)
    gl = [results["datasets"][d]["macro_f1"] for d in SUITES["gliclass"] if d in results["datasets"]]
    if gl:
        results["gliclass_avg_macro_f1"] = float(np.mean(gl))
        results["gliclass_avg_n_sets"] = len(gl)
        print(f"GLiClass-suite average macro-F1 over {len(gl)} sets: {np.mean(gl):.4f}")
    if "typed" in args.suites:
        results["typed_decisions"] = eval_typed(scorer, global_t, args.typed_max_cases, args.seed)
        print("typed-decisions:", json.dumps(results["typed_decisions"]))
    if scorer.latency_ms:
        results["latency_ms_per_example_p50"] = statistics.median(scorer.latency_ms)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

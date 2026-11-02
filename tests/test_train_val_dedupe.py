#!/usr/bin/env python
"""CPU test of prepare_data.dedupe_train_vs_val (v7): every train row whose text equals a val text (own val + dedupe_val_files)
is dropped, repeated copies too; per-source manifest counts follow; untouched rows keep their order; the val file is not modified;
the notebook's leak check (exact `text` equality against calib/stage2_pilot_val.jsonl) then reads 0.
    python scripts/test_train_val_dedupe.py"""
import hashlib, json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))  # repo layout: prepare_data.py lives in scripts/
from prepare_data import dedupe_train_vs_val  # noqa: E402

VAL = "calib/stage2_pilot_val.jsonl"
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); ok or fails.append(name)

md5_before = hashlib.md5(open(VAL, "rb").read()).hexdigest()
pv = [json.loads(l) for l in open(VAL)]
g = next(r for r in pv if r["source"] == "knowledgator/gliclass-v2.0")
lg = next(r for r in pv if r["source"] == "knowledgator/gliclass-v3-logic-dataset")
own_val = [{"text": "own val text", "source": "DeepPavlov/hwu64"}]
def row(text, skey, labels=("x",)):
    return {"text": text, "all_labels": list(labels), "true_labels": [labels[0]], "source": skey.split(":")[0], "_skey": skey}
logic = row(lg["text"], "knowledgator/gliclass-v3-logic-dataset:default", ("other", "labels"))
train = ([row(f"clean {i}", "knowledgator/gliclass-v2.0:default") for i in range(50)]
         + [row(g["text"], "knowledgator/gliclass-v2.0:default", ("a different", "label set"))]   # same text, other labels
         + [logic] * 3                                                                            # repeat 3 -> 3 copies
         + [row("own val text", "DeepPavlov/hwu64:default")] * 3
         + [row(" " + g["text"], "knowledgator/gliclass-v2.0:default")])                         # not exact -> kept (= notebook rule)
per_source = {"knowledgator/gliclass-v2.0:default": {"train_rows": 52}, "knowledgator/gliclass-v3-logic-dataset:default": {"train_rows": 3},
              "DeepPavlov/hwu64:default": {"train_rows": 3}, "LocalLLaMA/typed-decisions:customer_service": {"train_rows": 0}}
order = [r["text"] for r in train if r["text"] not in {g["text"], lg["text"], "own val text"}]
info = dedupe_train_vs_val(train, own_val, {"dedupe_val_files": [VAL]}, per_source)
check("dropped 7 rows (1 gliclass + 3 logic copies + 3 own-val copies)", info["dropped"] == 7 and len(train) == 51)
check("per-source counts", info["dropped_by_train_source"] == {"knowledgator/gliclass-v3-logic-dataset:default": 3,
      "DeepPavlov/hwu64:default": 3, "knowledgator/gliclass-v2.0:default": 1})
check("manifest train_rows updated", per_source["knowledgator/gliclass-v2.0:default"]["train_rows"] == 51
      and per_source["knowledgator/gliclass-v3-logic-dataset:default"]["train_rows"] == 0
      and per_source["DeepPavlov/hwu64:default"]["train_rows"] == 0
      and "dropped_val_text" not in per_source["LocalLLaMA/typed-decisions:customer_service"])
check("kept rows keep their order", [r["text"] for r in train] == order)
vt = {r["text"] for r in pv} | {"own val text"}
check("notebook leak rule reads 0", sum(r["text"] in vt for r in train) == 0)
check("val file unchanged", hashlib.md5(open(VAL, "rb").read()).hexdigest() == md5_before)
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)

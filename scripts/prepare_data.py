#!/usr/bin/env python
"""Build a stage's training mix in GLiClass JSONL format from a data recipe YAML.

    python prepare_data.py --recipe configs/data_stage1.yaml
    python prepare_data.py --recipe configs/data_stage0.yaml --streaming     # tiny, no full downloads

Writes <out_dir>/train.jsonl, <out_dir>/val.jsonl and <out_dir>/manifest.json (counts, licenses, seed).
Deterministic for a given recipe + seed, so a resumed Kaggle session rebuilds the identical file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugc.config import load_config  # noqa: E402
from ugc.convert import from_gliclass, from_jev, from_system_one, from_taxonomy, humanize, text_hash  # noqa: E402


TAX_RNG = random.Random(0)  # re-seeded in main() from the recipe seed


def iter_source(src: dict, split: str, seed: int, streaming: bool):
    kwargs = {"split": split}
    if src.get("config"):
        kwargs["name"] = src["config"]
    if streaming:
        ds = load_dataset(src["id"], streaming=True, **kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=src.get("shuffle_buffer", 2000))
    else:
        ds = load_dataset(src["id"], **kwargs)
        if src.get("min_state_chars"):   # stage 3: vectorised pre-filter (~1 in 215 Jev `full` rows survive), then shuffle
            m = int(src["min_state_chars"])
            col = "state" if "state" in ds.column_names else "text"
            ds = ds.filter(lambda b: [len(str(x or "")) >= m for x in b[col]], batched=True, batch_size=2000,
                           input_columns=None)
        ds = ds.shuffle(seed=seed)
    return iter(ds)


def resolve_names(src: dict) -> list[str] | None:
    """Class names of a taxonomy source: `names: {from: features}` (ClassLabel names of label_col),
    `{from: dataset, id, config, split, col}` (a names table), `{list: [...]}`, or None when labels are strings."""
    spec = src.get("names")
    if not spec:
        return None
    if "list" in spec:
        names = list(spec["list"])
    elif spec["from"] == "features":
        from datasets import load_dataset_builder
        feat = load_dataset_builder(src["id"], src.get("config")).info.features[src.get("label_col", "label")]
        feat = getattr(feat, "feature", feat)  # Sequence(ClassLabel) -> ClassLabel
        names = list(feat.names)
    else:
        kw = {"name": spec["config"]} if spec.get("config") else {}
        names = [r[spec.get("col", "name")] for r in load_dataset(spec["id"], split=spec["split"], **kw)]
    return names


def taxonomy_labels(src: dict, row: dict) -> list[str]:
    raw = row[src.get("label_col", "label")]
    raws = raw if isinstance(raw, list) else [raw]
    names = src["_names"]
    out = [names[int(r)] if names is not None else str(r) for r in raws]
    ren = src.get("rename") or {}
    return [ren.get(o, o) if src.get("humanize", True) is False else humanize(ren.get(o, o)) for o in out]


def taxonomy_row(src: dict, row: dict, defaults: dict, rng: random.Random):
    gold = taxonomy_labels(src, row)
    if src.get("schemes"):  # no-pool sources with several label phrasings (e.g. bearish/bullish vs negative/positive)
        scheme = rng.choice(src["schemes"])
        gold = [scheme[g] for g in gold if g in scheme]
        out = from_taxonomy(row[src.get("text_col", "text")], gold, src["id"], None,
                            src.get("max_chars", defaults["max_chars"]), all_labels=list(dict.fromkeys(scheme.values())))
        if out:
            out["all_labels"] = list(out["all_labels"])
        return out
    return from_taxonomy(row[src.get("text_col", "text")], gold, src["id"], src.get("pool"),
                         src.get("max_chars", defaults["max_chars"]),
                         n_base=tuple(src.get("n_base", (2, 8))))


def finalize_taxonomy(train: list[dict], val: list[dict], pool_labels: dict, pool_groups: dict, rng: random.Random,
                      ml: dict, max_length: int) -> dict:
    """Fill in base negatives from each pool, and statically expand a fraction of the val rows (seeded) so the
    val loss/monitor sees many-label rows. Returns the label_pools.json payload."""
    from ugc.many_label import LabelPools, expand_row
    pools_json = {p: ({"labels": labs, "groups": pool_groups[p]} if p in pool_groups else labs)
                  for p, labs in pool_labels.items()}
    pools = LabelPools(pools_json)
    ml = {"expand_prob": 1.0, "k_min": 20, "k_max": 150, "hard_frac": 0.5, "dup_sim": 0.8, "text_keep_tokens": 256, **ml}
    n_val_expanded = 0
    for rows, is_val in ((train, False), (val, True)):
        for ex in rows:
            nb = ex.pop("_nb", None)
            if nb is None:
                continue
            gold = set(ex["true_labels"])
            others = [l for l in pools.pools[ex["pool"]] if l not in gold]
            ex["all_labels"] = list(dict.fromkeys(ex["all_labels"] + rng.sample(others, min(len(others), rng.randint(*nb)))))
            if is_val and rng.random() < ml.get("val_expand_frac", 0.5):
                k = int(round(math.exp(rng.uniform(math.log(ml["k_min"]), math.log(ml["k_max"])))))
                new = expand_row(ex, pools, rng, ml, lambda l: len(l.split()) * 2 + 2,
                                 lambda t: len(t) // 4, max_length, k=k)
                n_val_expanded += int(new is not ex)
                ex["all_labels"] = new["all_labels"]
    return {"pools": pools_json, "n_val_expanded": n_val_expanded}


def convert(src: dict, row: dict, defaults: dict):
    max_options = src.get("max_options", defaults["max_options"])
    max_chars = src.get("max_chars", defaults["max_chars"])
    if src["format"] == "gliclass":
        return from_gliclass(row, src["id"], max_options, max_chars)
    if src["format"] == "taxonomy":
        return taxonomy_row(src, row, defaults, TAX_RNG)
    min_state = src.get("min_state_chars")
    if min_state and len(str(row.get("state") or row.get("text") or "")) < min_state:
        return None  # stage 3: keep only long contexts
    if src["format"] == "jev":
        allowed = src.get("license_use")
        if allowed and row.get("license_use") not in allowed:
            return None
        lic_ex = src.get("exclude_license_contains")   # stage 3: e.g. ["-sa", "by-sa", "gpl", "odbl"] = no share-alike / copyleft rows
        if lic_ex and any(x.lower() in str(row.get("license", "")).lower() for x in lic_ex):
            return None
        excl = src.get("exclude_source_contains")   # e.g. "procedural-typed-decisions/": Jev carries procedural-style subsets
        if excl and excl in str(row.get("source", "")):
            return None
        kinds = src.get("kinds")
        if kinds and row.get("kind") not in kinds:
            return None
        return from_jev(row, src["id"], max_options, max_chars, src.get("max_state_chars"))
    if src["format"] == "system_one":  # state + questions + answers (procedural-typed-decisions)
        if src.get("max_state_chars") and len(row.get("state") or "") > src["max_state_chars"]:
            return None
        # answers_col: LocalLLaMA/typed-decisions stores the gold distributions in `gold` (procedural: `answers`)
        return from_system_one(row["state"], row["questions"], row[src.get("answers_col", "answers")], src["id"], max_options, max_chars)
    raise ValueError(f"unknown format {src['format']}")


def take(src: dict, split: str, n: int | None, seed: int, streaming: bool, defaults: dict,
         exclude: set[str], max_scan: int | None) -> list[dict]:
    out, scanned = [], 0
    for row in iter_source(src, split, seed, streaming):
        scanned += 1
        if max_scan and scanned > max_scan:
            break
        converted = convert(src, row, defaults)
        if converted is None:
            continue
        for ex in converted if isinstance(converted, list) else [converted]:
            key = text_hash(ex["text"] + ex.get("prompt", ""), ex["all_labels"])
            if key in exclude:
                continue
            exclude.add(key)
            out.append(ex)
        if n is not None and len(out) >= n:
            out = out[:n]
            break
    print(f"  {src['id']} [{split}]: kept {len(out)} of {scanned} scanned", flush=True)
    return out


def take_jsonl(src: dict, n: int | None, seed: int, exclude: set[str]) -> list[dict]:
    """`format: jsonl` (stage 3 replay): sample n rows (seeded, without replacement) from an already-converted GLiClass
    JSONL file, e.g. the stage-2 train mix data/stage2_full/train.jsonl, so replay keeps that exact distribution (pool keys
    for many-label expansion, typed rows, taxonomy rows). Two passes, never holds the whole file in RAM.
    `path` may be a glob; the first match wins (Kaggle input mounts differ from the box)."""
    import glob as _glob
    hits = sorted(_glob.glob(src["path"], recursive=True))
    if not hits:
        raise SystemExit(f"jsonl source not found: {src['path']}")
    path = hits[0]
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    rng = random.Random(seed)
    want = n_lines if n is None else min(n_lines, int(n * 1.02) + 50)   # small surplus for duplicates/exclusions
    pick = set(rng.sample(range(n_lines), want))
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i not in pick:
                continue
            ex = json.loads(line)
            key = text_hash(ex["text"] + ex.get("prompt", ""), ex["all_labels"])
            if key in exclude:
                continue
            exclude.add(key)
            ex["source"] = f"replay:{ex.get('source', '?')}"
            out.append(ex)
    rng.shuffle(out)
    out = out if n is None else out[:n]
    print(f"  {src['id']} [jsonl {path}]: kept {len(out)} of {n_lines} lines", flush=True)
    src["_resolved_path"] = path
    return out


def dedupe_train_vs_val(train: list[dict], val: list[dict], recipe: dict, per_source: dict) -> dict:
    """Drop (in place) every train row whose `text` equals the text of a val row: this run's own val rows plus the rows of
    the recipe's `dedupe_val_files` (e.g. calib/stage2_pilot_val.jsonl). Exact `text` equality = the notebook's leak check.
    The carve-out in take() only dedupes on (text + prompt, labels), so the same text with another label set (gliclass-v2.0
    re-labels, logic rows x repeat, HWU64 train/test repeats) could reach train. Val files are never modified."""
    val_src: dict[str, str] = {}
    for ex in val:
        val_src.setdefault(ex["text"], ex.get("source", "?"))
    files = recipe.get("dedupe_val_files") or []
    for p in files:
        with open(p) as f:
            for line in f:
                ex = json.loads(line)
                val_src.setdefault(ex["text"], ex.get("source", "?"))
    by_train_src, by_val_src, hit_texts = Counter(), Counter(), set()
    keep = []
    for ex in train:
        vs = val_src.get(ex["text"])
        if vs is None:
            keep.append(ex)
            continue
        by_train_src[ex["_skey"]] += 1      # per recipe source (id:config); repeated rows count once per copy
        by_val_src[vs] += 1
        hit_texts.add(ex["text"])
    n_before = len(train)
    train[:] = keep
    for key, n in by_train_src.items():     # keep the manifest's per-source train_rows truthful (the notebook asserts on them)
        info = per_source[key]
        info["train_rows_before_dedupe"] = info["train_rows"]
        info["train_rows"] -= n
        info["dropped_val_text"] = n
    out = {"val_texts": len(val_src), "val_files": files, "train_rows_before": n_before, "train_rows_after": len(train),
           "dropped": n_before - len(train), "distinct_val_texts_hit": len(hit_texts),
           "dropped_by_train_source": dict(by_train_src.most_common()), "dropped_by_val_source": dict(by_val_src.most_common())}
    print(f"train/val dedupe: dropped {out['dropped']} train rows ({len(hit_texts)} distinct val texts) "
          f"by train source {out['dropped_by_train_source']} | by val source {out['dropped_by_val_source']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--streaming", action="store_true", help="stream sources (smoke tests / low disk)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. sources.0.n_train=100")
    args = ap.parse_args()

    recipe = load_config(args.recipe, [s for s in args.set if not s.startswith("sources.")])
    for item in args.set:  # sources.<i>.<key>=value overrides
        if item.startswith("sources."):
            key, _, val = item.partition("=")
            _, idx, field = key.split(".", 2)
            recipe["sources"][int(idx)][field] = int(val) if val.isdigit() else val
    out_dir = args.out_dir or recipe["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = recipe.get("seed", 42)
    defaults = {"max_options": recipe.get("max_options", 50), "max_chars": recipe.get("max_chars", 8000)}
    streaming = args.streaming or recipe.get("streaming", False)

    # eval-set guard (decision D1): none of these may be an ENABLED training source (recipe key `forbid_ids`)
    bad = [x["id"] for x in recipe["sources"] if x.get("enabled", True) is not False
           and any(f.lower() in x["id"].lower() for f in recipe.get("forbid_ids", []))]
    if bad:
        raise SystemExit(f"recipe enables eval-set / forbidden sources: {bad}")
    # LocalLLaMA/typed-decisions TEST split is our typed-decisions EVAL set: it may never be read as training data.
    # Only the `train` split (separate run, different seed, per its dataset card) is allowed, and only via an explicit recipe entry.
    td = [x for x in recipe["sources"] if x.get("enabled", True) is not False and "localllama/typed-decisions" in x["id"].lower()]
    bad_td = [x for x in td if x.get("split", "train") != "train" or x.get("val_split")]
    if bad_td:
        raise SystemExit(f"LocalLLaMA/typed-decisions: only split=train, no val_split (the test split is the eval set): {bad_td}")
    disabled = [{"id": x["id"], "config": x.get("config"), "reason": x.get("disabled_reason") or x.get("eval_overlap")}
                for x in recipe["sources"] if x.get("enabled", True) is False]

    t0 = time.time()
    TAX_RNG.seed(seed)
    train, val, seen = [], [], set()
    per_source = {}
    pool_labels: dict[str, list[str]] = {}
    pool_groups: dict[str, dict[str, str]] = {}
    pool_counts: dict[str, Counter] = {}
    for src in recipe["sources"]:
        if src.get("enabled", True) is False:
            print(f"source {src['id']} disabled in recipe, skipped", flush=True)
            continue
        print(f"source {src['id']}", flush=True)
        if src["format"] == "taxonomy":
            src["_names"] = resolve_names(src)
            if src.get("pool"):
                labs = [humanize(src.get("rename", {}).get(n, n)) if src.get("humanize", True) else n
                        for n in (src["_names"] or [])]
                pool_labels.setdefault(src["pool"], [])
                pool_labels[src["pool"]].extend(l for l in labs if l not in pool_labels[src["pool"]])
        n_val = src.get("n_val", 0)
        n_train = src.get("n_train")
        n_train = None if n_train in (None, "all") else int(n_train)
        max_scan = src.get("max_scan")
        if src["format"] == "jsonl":            # stage-3 replay of an earlier stage's converted mix (train only)
            v, t = [], take_jsonl(src, n_train, seed, seen)
        elif src.get("val_split"):
            v = take(src, src["val_split"], n_val, seed, streaming, defaults, seen, max_scan) if n_val else []
            t = take(src, src.get("split", "train"), n_train, seed, streaming, defaults, seen, max_scan)
        else:  # carve validation out of the same shuffled split
            need = None if n_train is None else n_train + n_val
            rows = take(src, src.get("split", "train"), need, seed, streaming, defaults, seen, max_scan)
            v, t = rows[:n_val], rows[n_val:]
        if src["format"] == "taxonomy" and src.get("pool"):
            for ex in t + v:  # string-labelled sources: the universe is what we saw
                if ex["pool"] and ex["true_labels"]:
                    for g in ex["true_labels"]:
                        if g not in pool_labels.setdefault(src["pool"], []):
                            pool_labels[src["pool"]].append(g)
            if src.get("group_col") and (t or v):
                raw = {}
                for r in iter_source(src, src.get("split", "train"), seed, streaming):
                    raw[taxonomy_labels(src, r)[0]] = humanize(r[src["group_col"]])
                    if len(raw) >= len(pool_labels[src["pool"]]) or len(raw) > 5000:
                        break
                pool_groups.setdefault(src["pool"], {}).update(raw)
        if src["format"] == "gliclass" and src.get("pool"):
            # label universe of a big natural-label source (e.g. gliclass-v2.0): rows can be expanded at train time with
            # neighbour labels drawn from the most frequent labels of the whole source
            cnt = pool_counts.setdefault(src["pool"], Counter())
            for ex in t + v:
                ex["pool"] = src["pool"]
                cnt.update(ex["all_labels"])
        repeat = int(src.get("repeat", 1))
        for ex in t:
            ex["_skey"] = f"{src['id']}:{src.get('config') or 'default'}"   # internal, stripped before writing
        train.extend(t * repeat)
        val.extend(v)
        per_source[f"{src['id']}:{src.get('config') or 'default'}"] = {
            "train_unique": len(t), "repeat": repeat, "train_rows": len(t) * repeat, "val_rows": len(v),
            "license": src.get("license"), "kinds": dict(Counter(x["kind"] for x in t)),
        }
        if src.get("license_note"):
            per_source[f"{src['id']}:{src.get('config') or 'default'}"]["license_note"] = src["license_note"]
        if src["format"] == "taxonomy":
            per_source[f"{src['id']}:{src.get('config') or 'default'}"].update(
                {"pool": src.get("pool"), "eval_overlap": src.get("eval_overlap"), "license_note": src.get("license_note")})

    for pid, cnt in pool_counts.items():
        cfg = next(x for x in recipe["sources"] if x.get("pool") == pid and x["format"] == "gliclass")
        top = [l for l, c in cnt.most_common(int(cfg.get("pool_max_labels", 20000))) if c >= int(cfg.get("pool_min_count", 5))]
        pool_labels[pid] = top
    rng = random.Random(seed)
    tax_info = {}
    if recipe.get("pools_file_from") and not pool_labels:   # stage 3: reuse the stage-2 label pools for replay rows
        import glob as _glob, shutil as _shutil
        hits = sorted(_glob.glob(recipe["pools_file_from"], recursive=True))
        if not hits:
            raise SystemExit(f"pools_file_from not found: {recipe['pools_file_from']}")
        _shutil.copyfile(hits[0], os.path.join(out_dir, "label_pools.json"))
        tax_info = {"pools_copied_from": hits[0]}
    if pool_labels:
        fin = finalize_taxonomy(train, val, pool_labels, pool_groups, random.Random(seed + 7), recipe.get("many_label", {}),
                                int(recipe.get("max_length", 1024)))
        with open(os.path.join(out_dir, "label_pools.json"), "w") as f:
            json.dump(fin["pools"], f, ensure_ascii=False)
        tax_info = {"pool_sizes": {p: len(v) for p, v in pool_labels.items()}, "val_rows_statically_expanded": fin["n_val_expanded"]}
    dedupe_info = dedupe_train_vs_val(train, val, recipe, per_source) if recipe.get("dedupe_train_vs_val") else None
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for ex in rows:
                ex.pop("_skey", None)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    n_labels = [len(x["all_labels"]) for x in train] or [0]
    manifest = {
        "recipe": os.path.abspath(args.recipe), "name": recipe.get("name"), "seed": seed, "streaming": streaming,
        "train_rows": len(train), "val_rows": len(val), "sources": per_source,
        "kinds": dict(Counter(x["kind"] for x in train)),
        "labels_per_row": {"max": max(n_labels), "mean": round(sum(n_labels) / len(n_labels), 2)},
        "taxonomy": tax_info, "disabled_sources": disabled, "train_val_dedupe": dedupe_info, "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

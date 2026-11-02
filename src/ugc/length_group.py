"""Length-grouped batching for UltraGliClass training (stage-2 speed-up; OFF by default, see STAGE2_FAST_SPEEDTEST_SPEC.md).

Why: batches are random and padded to the longest row, so ~65 % of the computed tokens in the stage-2 pilots were padding
(mean padded length 679 vs mean real 240). This module groups rows of similar token length into the same micro-batch.

Pieces
  * estimate_lengths(...)      one cheap tokenisation pass over the train rows (exact base lengths), cached on disk.
  * LengthModel                per-epoch sorting keys: base length + the tokens the many-label expansion WILL add (the dataset runs in
                               "planning mode", seeding every fetch from (seed, epoch, row), so the sampler can replay the draws).
  * LengthGroupedSampler       seeded, resume-deterministic, DDP-safe mega-batch sampler (flat index stream; the DataLoader cuts it
                               into micro-batches of `micro_batch`).
  * calibrate_label_norm(...)  mean number of label slots per micro-batch under RANDOM batching (used to keep the loss weighting
                               like-for-like; see StrictTrainer.label_norm).

Sampler algorithm for one epoch (all ranks compute the identical sequence from (seed, epoch); no communication):
  1. permute all rows (PCG64 seeded by seed and epoch), drop the tail so that the epoch is a whole number of
     G = micro_batch * world_size rows (a different tail every epoch);
  2. cut the permutation into mega-batches of ~ mega_batch_mult * micro_batch rows (rounded up to a multiple of G);
  3. sort each mega-batch by estimated length (random tie-break) and cut it into groups of G consecutive rows;
     one group = one micro-batch for every rank, taken from neighbours in the sort order, so the ranks do similar work at the
     same time (a DDP step waits for its slowest rank);
  4. shuffle all groups of the epoch (so an optimizer step is a random mixture of lengths, not 8 neighbours);
  5. emit group by group: rows [r*mb:(r+1)*mb] of a group are the micro-batch of rank r. accelerate's BatchSamplerShard hands
     micro-batch number i to rank i % world_size, which is exactly this layout.
Every row appears at most once per epoch; every rank gets the same number of micro-batches. Paired views (consistency) live inside the dataset item and
are stacked by PairedCollator into the same forward pass as their primary row, so grouping cannot separate them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch

LENGTH_VERSION = 2


def parse_length_group(dcfg: dict) -> dict | None:
    """data.length_group: false/absent -> None (original random batching). true, or a dict:
         enabled        (default true when a dict is given)
         mega_batch_mult  chunk = mega_batch_mult x micro-batch rows are sorted together (default 50; also accepted as data.mega_batch_mult)
         loss_norm      keep every label slot's loss weight like-for-like with random batches (default true; see StrictTrainer)
         cache_dir      where the token-length cache lives (default <output_dir>/length_cache)"""
    lg = dcfg.get("length_group")
    if not lg:
        return None
    if lg is True:
        lg = {}
    if not lg.get("enabled", True):
        return None
    return {"mega_batch_mult": int(lg.get("mega_batch_mult", dcfg.get("mega_batch_mult", 50))),
            "loss_norm": bool(lg.get("loss_norm", True)), "cache_dir": lg.get("cache_dir")}


# ---------------------------------------------------------------------------------------------------------------
# length estimation
# ---------------------------------------------------------------------------------------------------------------
def _prompt_text(row: dict, label_token: str = "<<LABEL>>", sep_token: str = "<<SEP>>") -> str:
    """The string GLiClassDataset tokenises for a row (prompt_first=True, no examples), in the stored label order."""
    labels = "".join(f"{label_token}{l}" for l in row["all_labels"])
    return f"{labels}{sep_token}{row.get('prompt', '')}{row['text']}"


def base_lengths(rows: list[dict], tokenizer, max_length: int, chunk: int = 2000) -> np.ndarray:
    """Exact token length of every row as the dataset will tokenise it (no augmentation, no expansion)."""
    out = np.zeros(len(rows), dtype=np.int32)
    cap = max_length * 12  # chars; more than max_length tokens always remain, so the length saturates at max_length
    for s in range(0, len(rows), chunk):
        texts = []
        for r in rows[s:s + chunk]:
            t = _prompt_text(r)
            texts.append(t if len(t) <= cap else t[:cap])
        ids = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
        out[s:s + len(ids)] = [len(x) for x in ids]
    return out


class LengthModel:
    """Per-epoch sorting keys for the sampler.

    base[i]  exact token length of row i as the dataset tokenises it (cached on disk; no augmentation, no expansion).
    With many-label expansion on (planning mode, see ManyLabelDataset.enable_planning) the dataset seeds each fetch from
    (seed, epoch, row), so we replay the expansion draws and add the tokens of the k - n negatives it will add:
        length(i, epoch) = min(max_length, base[i] + (k - n_labels[i]) * avg_label_tokens[pool])      (k from planned_expansion)
    Not modelled (small, random): which negatives are drawn (token cost), label augmentation, paired second views."""

    def __init__(self, base: np.ndarray, rows=None, tokenizer=None, pools=None, ml=None, max_length=1024, seed=0):
        self.base = np.asarray(base, dtype=np.int32)
        self.max_length, self.seed = int(max_length), int(seed)
        self.plan = None
        if pools is not None and rows is not None:
            ml = ml or {}
            avg_tok: dict[str, float] = {}
            for pid, labs in pools.pools.items():
                sample = labs if len(labs) <= 400 else random.Random(0).sample(labs, 400)
                ids = tokenizer(list(sample), add_special_tokens=False)["input_ids"]
                avg_tok[pid] = float(np.mean([len(x) + 1 for x in ids])) if ids else 4.0    # +1 for <<LABEL>>
            idx, n_lab, psize, pcfgs, toks = [], [], [], [], []
            for i, r in enumerate(rows):
                pool = r.get("pool")
                if pool and pool in pools.pools:
                    idx.append(i); n_lab.append(len(r["all_labels"])); psize.append(len(pools.pools[pool]))
                    pcfgs.append({**ml, **(ml.get("pool_overrides", {}).get(pool, {}))}); toks.append(avg_tok[pool])
            self.plan = (np.array(idx, dtype=np.int64), n_lab, psize, pcfgs, toks)

    def lengths(self, epoch: int = 0) -> np.ndarray:
        out = self.base.astype(np.int64)
        if self.plan is None:
            return out
        from ugc.many_label import planned_expansion
        idx, n_lab, psize, pcfgs, toks = self.plan
        for j in range(len(idx)):
            k = planned_expansion(self.seed, epoch, int(idx[j]), n_lab[j], psize[j], pcfgs[j])
            if k is not None and k > n_lab[j]:
                out[idx[j]] = min(self.max_length, int(self.base[idx[j]] + (k - n_lab[j]) * toks[j]))
        return out


def _cache_key(rows, tokenizer, max_length) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"v": LENGTH_VERSION, "n": len(rows), "max_length": max_length, "vocab": len(tokenizer),
                         "tok": getattr(tokenizer, "name_or_path", "")}, sort_keys=True).encode())
    for r in rows[:: max(1, len(rows) // 997)]:   # ~1000 evenly spaced rows: catches a different/reordered file
        h.update(str(r.get("text", ""))[:200].encode())
        h.update(("|".join(map(str, r.get("all_labels", [])[:8]))).encode())
    return h.hexdigest()[:16]


def estimate_lengths(rows, tokenizer, max_length, cache_dir=None, log=print) -> np.ndarray:
    """Base token lengths (int32) of every train row; cached in cache_dir/lengths_<key>.npy."""
    key = _cache_key(rows, tokenizer, max_length)
    path = os.path.join(cache_dir, f"lengths_{key}.npy") if cache_dir else None
    if path and os.path.isfile(path):
        arr = np.load(path)
        if len(arr) == len(rows):
            log(f"length cache hit: {path}")
            return arr
    t0 = time.time()
    arr = base_lengths(rows, tokenizer, max_length)
    log(f"estimated token lengths of {len(rows)} rows in {time.time() - t0:.0f}s "
        f"(mean {arr.mean():.0f}, p90 {np.percentile(arr, 90):.0f}, max {arr.max()})")
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
    return arr


# ---------------------------------------------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------------------------------------------
class LengthGroupedSampler(torch.utils.data.Sampler):
    """Flat index stream; a DataLoader with batch_size=micro_batch (and accelerate's BatchSamplerShard) turns it into
    length-homogeneous micro-batches. Deterministic in (seed, epoch); set_epoch() selects the epoch."""

    def __init__(self, lengths, micro_batch: int, world_size: int = 1, mega_batch_mult: int = 50, seed: int = 0,
                 epoch: int = 0, encode_epoch: bool = False, n_total: int | None = None):
        """lengths: array of sorting keys, or a callable epoch -> array (LengthModel.lengths). encode_epoch: yield
        epoch * n_rows_total + row (planning mode of ManyLabelDataset) instead of the plain row index."""
        assert micro_batch >= 1 and world_size >= 1 and mega_batch_mult >= 1
        self._lengths_fn = lengths if callable(lengths) else None
        self._lengths0 = None if callable(lengths) else np.asarray(lengths, dtype=np.int64)
        self.encode_epoch = bool(encode_epoch)
        self.n_total = int(n_total) if n_total is not None else (len(self._lengths0) if self._lengths_fn is None else len(self._lengths_fn(0)))
        self.mb, self.world = int(micro_batch), int(world_size)
        self.G = self.mb * self.world                      # rows per group = one micro-batch for every rank
        self.mult = int(mega_batch_mult)
        self.seed, self.epoch = int(seed), int(epoch)
        n = self.n_total
        self.n_rows = (n // self.G) * self.G
        if self.n_rows == 0:
            raise ValueError(f"{n} rows cannot fill one group of {self.G}")
        self.chunk = max(self.G, int(math.ceil(self.mb * self.mult / self.G)) * self.G)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_rows

    def lengths_for(self, epoch: int) -> np.ndarray:
        return self._lengths0 if self._lengths_fn is None else np.asarray(self._lengths_fn(epoch), dtype=np.int64)

    def epoch_order(self, epoch: int | None = None) -> np.ndarray:
        """Row indices (0..n_total) in stream order for one epoch (plain indices, no epoch encoding)."""
        e = self.epoch if epoch is None else int(epoch)
        lengths = self.lengths_for(e)
        g = np.random.Generator(np.random.PCG64([self.seed, e, 0x5EED]))
        perm = g.permutation(self.n_total)[: self.n_rows]
        groups = []
        for s in range(0, self.n_rows, self.chunk):
            c = perm[s:s + self.chunk]
            tie = g.random(len(c))
            c = c[np.lexsort((tie, lengths[c]))]      # by length, random tie-break
            groups.append(c.reshape(-1, self.G))
        allg = np.concatenate(groups, axis=0)
        return allg[g.permutation(len(allg))].reshape(-1)

    def __iter__(self):
        order = self.epoch_order()
        if self.encode_epoch:
            order = order + self.epoch * self.n_total
        return iter(order.tolist())


# ---------------------------------------------------------------------------------------------------------------
# loss-weighting calibration
# ---------------------------------------------------------------------------------------------------------------
def calibrate_label_norm(train_ds, micro_batch: int, n_probe: int = 1536, seed: int = 0, log=print) -> float:
    """Mean number of label slots (primary + paired second views) in a micro-batch of RANDOM rows, measured by really fetching
    n_probe random rows through the (expanding, pairing) dataset. Dataset RNG and Python's global RNG are restored afterwards."""
    state = random.getstate()
    stats = dict(getattr(train_ds, "stats", {}) or {})
    try:
        random.seed(seed)
        idx = np.random.Generator(np.random.PCG64([seed, 77])).choice(len(train_ds), size=min(n_probe, len(train_ds)),
                                                                      replace=False)
        slots = []
        for i in idx:
            it = train_ds[int(i)]
            n = len(it["labels_text"]) + (len(it["pair"]["labels_text"]) if "pair" in it else 0)
            slots.append(n)
    finally:
        random.setstate(state)
        if hasattr(train_ds, "stats"):
            train_ds.stats.clear()
            train_ds.stats.update(stats)
    norm = float(np.mean(slots)) * micro_batch
    log(f"label-slot normaliser: {norm:.1f} label slots per micro-batch of {micro_batch} random rows "
        f"({np.mean(slots):.2f} per row, {len(slots)} probe rows)")
    return norm

"""Stage-2 label-order robustness tools (all optional, all off unless the config switches them on).

1. Many-label expansion with hard negatives (dynamic, re-sampled on every fetch = every epoch).
   Rows that carry a "pool" key (a taxonomy id, see prepare_data.py `taxonomy` sources) get extra negative labels
   drawn from the same label pool: part nearest neighbours of the gold labels (char n-gram TF-IDF over the label
   strings, so siblings like "card_arrival" / "card_delivery_estimate" are preferred), part uniform. The result
   has 20-150 labels, within a token budget so the label prompt never crowds out the text.
2. Paired views: the same example rendered with two independent label orders; the collator stacks both views into
   ONE forward pass (DDP-safe) and `ugc.trainer.StrictTrainer` adds a symmetric-KL consistency term between the
   per-label scores of the two orders.
GLiClassDataset already re-shuffles all_labels on every fetch (shuffle_labels=True), so order randomisation happens
every epoch with no extra code; the weakness found in stage 1 is the *label count*, not the lack of shuffling.
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import torch
from gliclass.data_processing import GLiClassDataset


class LabelPools:
    """{pool_id: [labels]} plus lazily built nearest-neighbour tables."""

    def __init__(self, pools: dict, top_m: int = 48):
        # a pool is either a list of labels or {"labels": [...], "groups": {label: group}}; labels sharing a group
        # (e.g. DBpedia classes under one parent) are preferred hard negatives
        self.pools = {k: list(dict.fromkeys(v["labels"] if isinstance(v, dict) else v)) for k, v in pools.items()}
        self._groups: dict[str, dict[str, list[str]]] = {}
        for k, v in pools.items():
            if isinstance(v, dict) and v.get("groups"):
                by_group: dict[str, list[str]] = {}
                for lab, grp in v["groups"].items():
                    by_group.setdefault(grp, []).append(lab)
                self._groups[k] = {lab: by_group[grp] for lab, grp in v["groups"].items()}
        self.top_m = top_m
        self._nn: dict[str, dict[str, list[tuple[str, float]]]] = {}

    @classmethod
    def from_file(cls, path: str, **kw) -> "LabelPools":
        with open(path) as f:
            return cls(json.load(f), **kw)

    def siblings(self, pool: str, label: str) -> list[str]:
        return [l for l in self._groups.get(pool, {}).get(label, []) if l != label]

    def neighbours(self, pool: str) -> dict[str, list[tuple[str, float]]]:
        if pool in self._nn:
            return self._nn[pool]
        labels = self.pools[pool]
        table: dict[str, list[tuple[str, float]]] = {}
        if len(labels) > 1:
            from sklearn.feature_extraction.text import TfidfVectorizer
            X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(
                [l.lower() for l in labels])
            for s in range(0, len(labels), 512):  # chunked: pools up to ~50k labels stay in memory
                sim = (X[s:s + 512] @ X.T).toarray()
                for r in range(sim.shape[0]):
                    i = s + r
                    sim[r, i] = -1.0
                    idx = np.argpartition(-sim[r], min(self.top_m, len(labels) - 1))[: self.top_m]
                    idx = idx[np.argsort(-sim[r, idx])]
                    table[labels[i]] = [(labels[j], float(sim[r, j])) for j in idx]
        self._nn[pool] = table
        return table


def plan_seed(seed: int, epoch: int, idx: int) -> int:
    """Per-(epoch, row) seed of the expansion RNG in planning mode (injective for idx < 1e6+3, epoch < 1e6+3)."""
    return (int(seed) * 1_000_003 + int(epoch)) * 1_000_003 + int(idx)


def planned_expansion(seed: int, epoch: int, idx: int, n_labels: int, pool_size: int, pcfg: dict) -> int | None:
    """Replays the first draws of expand_row() for a (epoch, row): returns the target label count k, or None if the row is not
    expanded. MUST mirror expand_row (checked by scripts/test_length_group.py): draw 1 = random() vs expand_prob, draw 2 = uniform(log k_lo,
    log k_hi) for k. Does not know the final label set (token costs of the drawn negatives) or the pair/augmentation draws."""
    rng = random.Random(plan_seed(seed, epoch, idx))
    if rng.random() >= pcfg["expand_prob"]:
        return None
    k_hi = min(int(pcfg["k_max"]), pool_size)
    if k_hi <= max(n_labels, 1) + 1:
        return None
    k_lo = min(max(int(pcfg["k_min"]), n_labels + 1), k_hi)
    return min(int(round(math.exp(rng.uniform(math.log(k_lo), math.log(k_hi))))), k_hi)


def expand_row(ex: dict, pools: "LabelPools | None", rng: random.Random, ml: dict, label_tokens, text_tokens,
               max_length: int, k: int | None = None) -> dict:
    """Return a copy of ex with extra negative labels from its pool, or ex itself when nothing applies.
    label_tokens(label) -> int and text_tokens(text) -> int give the token costs (the dataset passes tokenizer-based
    counters, prepare_data a cheap word-based estimate). k forces the target label count (static val expansion)."""
    pool = ex.get("pool")
    if not (pools and pool and pool in pools.pools):
        return ex
    pcfg = {**ml, **(ml.get("pool_overrides", {}).get(pool, {}))}
    if k is None and rng.random() >= pcfg["expand_prob"]:
        return ex
    labels = list(ex["all_labels"])
    true = ex["true_labels"]
    gold = set(true) if not isinstance(true, dict) else {a for a, v in true.items() if v > 0}
    k_hi = min(int(pcfg["k_max"]), len(pools.pools[pool]))
    if k is None and k_hi <= max(len(labels), 1) + 1:
        return ex
    if k is None:
        k_lo = min(max(int(pcfg["k_min"]), len(labels) + 1), k_hi)
        k = int(round(math.exp(rng.uniform(math.log(k_lo), math.log(k_hi)))))  # log-uniform: many mid sizes too
    k = min(k, k_hi)
    # token budget: keep at least text_keep_tokens of the text (labels come first with prompt_first=True)
    text_tok = min(text_tokens(str(ex["text"])), int(pcfg["text_keep_tokens"]))
    budget = max_length - text_tok - 24
    used = sum(label_tokens(l) for l in labels)
    have = set(labels)
    nn = pools.neighbours(pool)
    banned = set(gold)
    near_gold: list[str] = []
    for g in gold:
        for lab in pools.siblings(pool, g):
            near_gold.append(lab)
        for lab, sim in nn.get(g, []):
            if sim >= pcfg["dup_sim"]:
                banned.add(lab)          # near-duplicate / synonym of a gold label: could be a false negative
            else:
                near_gold.append(lab)
    rng.shuffle(near_gold)
    universe = pools.pools[pool]
    added: list[str] = []

    def take(lab: str) -> bool:
        nonlocal used
        if lab in have or lab in banned:
            return False
        t = label_tokens(lab)
        if used + t > budget:
            return False
        have.add(lab); added.append(lab); used += t
        return True

    n_hard = int(round((k - len(labels)) * pcfg["hard_frac"]))
    for lab in near_gold:
        if len(added) >= n_hard:
            break
        take(lab)
    tries = 0
    while len(labels) + len(added) < k and tries < 6 * k:
        tries += 1
        take(universe[rng.randrange(len(universe))])
    if not added:
        return ex
    out = dict(ex)
    out["all_labels"] = labels + added
    return out


class ManyLabelDataset(GLiClassDataset):
    """GLiClassDataset + dynamic many-label expansion + optional paired (two label orders) views."""

    def __init__(self, *args, pools: LabelPools | None = None, many_label: dict | None = None,
                 pair_prob: float = 0.0, seed: int = 0, **kw):
        super().__init__(*args, **kw)
        self.pools = pools
        self.ml = {"expand_prob": 0.7, "k_min": 20, "k_max": 150, "hard_frac": 0.5, "dup_sim": 0.80,
                   "text_keep_tokens": 256, **(many_label or {})}
        self.pair_prob = pair_prob
        self._seed = seed
        self._rng = random.Random(seed)
        self._rng_key = None
        self._ltok: dict[str, int] = {}
        self.stats = {"rows": 0, "expanded": 0, "paired": 0, "labels_sum": 0}
        self._plan = False                      # length-grouped training: see enable_planning()
        self._plan_rng: random.Random | None = None

    # -- helpers --------------------------------------------------------------------------------------------
    def enable_planning(self) -> None:
        """Length-grouped batching (ugc.length_group): the sampler yields `epoch * len(self) + row` instead of `row`, and every
        fetch seeds its expansion RNG from (seed, epoch, row). The sampler can then replay the first two draws of expand_row (expand?
        and k) to predict how long the expanded row will be, so rows of similar FINAL length share a micro-batch. Distributions are
        unchanged (same draws, same code path); only the seeding differs. Off unless a LengthGroupedSampler is used."""
        self._plan = True

    def __getitem__(self, i):
        if not self._plan:
            return super().__getitem__(i)
        epoch, idx = divmod(int(i), len(self._data))
        self._plan_rng = random.Random(plan_seed(self._seed, epoch, idx))
        try:
            return super().__getitem__(idx)
        finally:
            self._plan_rng = None

    @property
    def rng(self) -> random.Random:
        """Re-seeded from torch's per-worker / per-epoch seed, so DataLoader workers (copies of this object) and
        successive epochs draw different negatives instead of replaying the same stream."""
        if self._plan_rng is not None:
            return self._plan_rng
        key = torch.initial_seed()
        if key != self._rng_key:
            self._rng_key = key
            self._rng = random.Random(self._seed * 1_000_003 + key)
        return self._rng

    def _label_tokens(self, label: str) -> int:
        n = self._ltok.get(label)
        if n is None:
            n = len(self.tokenizer(label, add_special_tokens=False)["input_ids"]) + 1  # +1 for <<LABEL>>
            self._ltok[label] = n
        return n

    def _text_tokens(self, text: str) -> int:
        return len(self.tokenizer(text[: self.max_length * 6], add_special_tokens=False,
                                  truncation=True, max_length=self.max_length)["input_ids"])

    def expand(self, ex: dict) -> dict:
        return expand_row(ex, self.pools, self.rng, self.ml, self._label_tokens, self._text_tokens, self.max_length)

    # -- GLiClassDataset hook -------------------------------------------------------------------------------
    def _prepare(self, example):
        ex = dict(example)
        ex["all_labels"] = list(example["all_labels"])  # super() shuffles in place: never touch the stored row
        ex = self.expand(ex)
        self.stats["rows"] += 1
        self.stats["labels_sum"] += len(ex["all_labels"])
        self.stats["expanded"] += int(len(ex["all_labels"]) > len(example["all_labels"]))
        pair = (self.pair_prob > 0 and len(ex["all_labels"]) >= int(self.ml.get("pair_min_labels", 8))
                and self.rng.random() < self.pair_prob)
        a = super()._prepare({**ex, "all_labels": list(ex["all_labels"])})
        if pair:
            self.stats["paired"] += 1
            a["pair"] = super()._prepare({**ex, "all_labels": list(ex["all_labels"])})
        return a


class PairedCollator:
    """Wraps the GLiClass collator. Items may carry a second view under "pair"; both views go into one batch:
    rows [0, N) are the primary views, [N, N+P) the second views of the P paired items. Adds
      pair_idx  (P, 2) long: row indices (primary, second)
      pair_perm (P, C) long: second-view label slot j -> primary-view slot (-1 = padding)"""

    def __init__(self, base):
        self.base = base

    def __call__(self, batch):
        paired = [i for i, b in enumerate(batch) if "pair" in b]
        seconds = [batch[i]["pair"] for i in paired]
        items = [{k: v for k, v in b.items() if k != "pair"} for b in batch]
        full = self.base(items + seconds)
        if paired:
            n = len(items)
            # the second view can have MORE labels than the primary (augmentation adds/removes labels per view)
            c = max(max(len(batch[i]["labels_text"]), len(batch[i]["pair"]["labels_text"])) for i in paired)
            perm = torch.full((len(paired), c), -1, dtype=torch.long)
            for r, i in enumerate(paired):
                pos1 = {l: j for j, l in enumerate(batch[i]["labels_text"])}
                for j, l in enumerate(batch[i]["pair"]["labels_text"]):
                    perm[r, j] = pos1.get(l, -1)   # augmentation may drop/add labels independently per view
            full["pair_idx"] = torch.tensor([[i, n + r] for r, i in enumerate(paired)], dtype=torch.long)
            full["pair_perm"] = perm
        return full


def consistency_loss(logits: torch.Tensor, class_mask: torch.Tensor, pair_idx: torch.Tensor,
                     pair_perm: torch.Tensor, kind: str = "skl", temperature: float = 1.0) -> torch.Tensor:
    """Penalise score differences between two label orders of the same example.
    skl: symmetric KL between softmax(z/T) over the valid labels of the two views (per-label scores aligned by label
    identity through pair_perm). logit_mse: mean squared difference of mean-centred logits."""
    a, b = pair_idx[:, 0], pair_idx[:, 1]
    c = pair_perm.shape[1]
    z1 = logits[a][:, :c].float()
    z2 = logits[b][:, :c].float()
    valid2 = pair_perm >= 0
    idx = torch.where(valid2, pair_perm, torch.full_like(pair_perm, c))      # padding -> dump column c
    z2a = torch.zeros(z1.shape[0], c + 1, device=z1.device, dtype=z1.dtype).scatter(
        1, idx, z2.masked_fill(~valid2, 0.0))[:, :c]                          # second view in primary slot order
    valid = torch.zeros(z1.shape[0], c + 1, device=z1.device, dtype=torch.bool).scatter(1, idx, valid2)[:, :c]
    valid = valid & class_mask[a][:, :c].bool()
    neg = torch.finfo(z1.dtype).min / 4
    if kind == "logit_mse":
        n = valid.sum(1, keepdim=True).clamp_min(1)
        m1 = (z1 * valid).sum(1, keepdim=True) / n
        m2 = (z2a * valid).sum(1, keepdim=True) / n
        d = ((z1 - m1) - (z2a - m2)) * valid
        return (d.pow(2).sum(1) / n.squeeze(1)).mean()
    l1 = torch.log_softmax((z1 / temperature).masked_fill(~valid, neg), dim=1)
    l2 = torch.log_softmax((z2a / temperature).masked_fill(~valid, neg), dim=1)
    p1, p2 = l1.exp(), l2.exp()
    kl12 = (p1 * (l1 - l2)).masked_fill(~valid, 0.0).sum(1)
    kl21 = (p2 * (l2 - l1)).masked_fill(~valid, 0.0).sum(1)
    return (0.5 * (kl12 + kl21)).mean()

# UltraGliClass v0.1: final benchmark numbers and peer comparison (PARTIAL)

Written 1 Nov 2026, 23:30 IST. **Status: PARTIAL.** Local CPU evals were stopped on request at 23:25 IST. The full comparison suite will be planned and run on Kaggle.
Every number comes from a box JSON or a live page fetched on 1 Nov 2026. Anything else is marked UNVERIFIED. Nothing was published and no money was spent.

## Sources
- S3 (v0.1): `/workspace/kaggle/ultragliclass/S3full/final/ultragliclass/results/stage3_final_quick.json`, `stage3_final_longjev_8192.json`, `stage3_longjev_compare.json`
- S2: `/workspace/kaggle/ultragliclass/S2full/session2/output/ultragliclass/results/stage2_full_C_final_quick.json`, plus S3 `stage3_baseline_s2_longjev_8192.json`
- Our ModernBERT re-run (suite): `results/stage1_fair_modernbert_v3_cpu.json`
- **New today**, ModernBERT long-Jev run: `results/compare/longjev_modernbase_v3_8192_cpu.json`, with paired compares in `results/compare/longjev_compare_modernbase_vs_s2.json` and `..._vs_s3.json`. Log: `logs_compare/longjev_modernbase_v3_8192.log`
- Published numbers:
  - https://huggingface.co/knowledgator/gliclass-modern-base-v3.0 and https://huggingface.co/knowledgator/gliclass-base-v3.0 (same benchmark table on both cards)
  - arXiv 2508.07662 (https://arxiv.org/abs/2508.07662), Table 3
  - v4 cards, from the notes/V4_MODELS.md snapshot taken 23 Oct

## Our eval protocol (eval.py)
| Setting | Our protocol |
|---|---|
| Sample size | ≤500 rows per set (random subsample, seed 0) |
| Labels | single-label softmax, canonical label order |
| Permutations | 3 label orders run; the mean is reported separately |
| Context | max_length **1024** |
| Tokenizer | `add_prefix_space=False` |
| Metric | macro-F1 |
| Calibration | global T fit on calib/stage1_val.jsonl. It affects ECE only, not F1 or accuracy |
| Hardware | S2/S3: fp16 on T4. ModernBERT: fp32 on CPU |

The official `test_gliclass.py` (local copy in `../gliclass_repo`) differs in a few ways:
- It uses the **full test split** and the pipeline default max_length of 1024.
- It uses `add_prefix_space=True`, fp16, and 1 label order.
- Its dataset list has no snips, although the card table does include snips.

## 1. Final numbers: v0.1 = Stage 3 final_model (Stage 2 final for context)
| Metric (eval context) | S2 final | **v0.1 (S3)** |
|---|---|---|
| Suite avg macro-F1, 14 sets (1024) | 0.6829 | **0.6892** |
| Suite avg, 12 truly zero-shot sets (excludes massive and financial_phrasebank, which are in-domain) | 0.6520 | **0.6575** |
| Suite avg, mean over 3 label orders (14 / 12) | 0.6811 / 0.6512 | 0.6845 / 0.6542 |
| Banking77 acc (77 labels, 1024) | 0.554 | **0.548** (macro-F1 0.515) |
| CLINC-150 acc (150 in-scope labels, 1024) | 0.650 | **0.640** (macro-F1 0.596) |
| Typed-decisions acc (LocalLLaMA test, 2,000 decisions, 1024). **Partly in-domain**: the customer_service and agent_trace workflows were in S2 train | 0.6155 | **0.574** |
| Long-Jev acc (431 rows, states ≥6k chars, **8192**, 5.6% truncated) | 0.552 [0.498, 0.605] | **0.664** [0.614, 0.713] |
| Long-Jev paired Δ vs S2 | | **+0.111**, cluster CI [+0.068, +0.157], McNemar p = 1.94e-7 (67 wins / 19 losses) |
| Latency p50 (T4 fp16, 1024) | 5.45 ms | 5.82 ms |

Per-set macro-F1 for v0.1 (S2 in brackets):

| Set | v0.1 (S2) | Set | v0.1 (S2) |
|---|---|---|---|
| CR | 0.911 (0.917) | sst2 | 0.906 (0.912) |
| sst5 | 0.238 (0.224) | imdb | 0.940 (0.944) |
| 20NG | 0.420 (0.412) | enron_spam | 0.567 (0.541) |
| massive¹ | 0.837 (0.839) | banking77 | 0.515 (0.523) |
| fin_phrasebank¹ | 0.921 (0.898) | ag_news | 0.737 (0.733) |
| emotion | 0.468 (0.476) | cap_sotu | 0.474 (0.455) |
| rotten_tomatoes | 0.819 (0.792) | snips | 0.897 (0.894) |

¹ In-domain: HWU64 and the fin-tweets sentiment set were in S2 training.

## 2. Side by side with peers
Labels:
- **FAIR**: same code, protocol, and sample.
- **PARTIAL**: same datasets and metric but a different protocol, or a domain advantage on one side.
- **INTERNAL-ONLY**: no public peer number exists.

### 2a. 14-set GLiClass suite, macro-F1
| Model | Params | 14-set avg | 12 zero-shot avg | Source | Label |
|---|---|---|---|---|---|
| **UltraGliClass v0.1** | ~150M | **0.689** | **0.658** | box JSON | (ours) |
| gliclass-modern-base-v3.0, our re-run | 151M | 0.556 | 0.582 | box JSON | **FAIR** |
| gliclass-modern-base-v3.0, published | 151M | 0.5577 | 0.587 | card / arXiv T3 | PARTIAL |
| gliclass-base-v3.0 (DeBERTa-v3-base), published | 187M | **0.6764** | **0.672** | card / arXiv T3 | PARTIAL |
| gliclass-modern-large-v3.0, published | 399M | 0.6197 | 0.641 | card | PARTIAL |
| gliclass-large-v3.0, published | 439M | **0.7193** | **0.717** | card | PARTIAL |
| gliclass-edge-v3.0, published | 32.7M | 0.4900 | 0.510 | card | PARTIAL |
| deberta-v3-large-zeroshot-v2.0, published (13 sets, no 20NG) | 435M (UNVERIFIED) | 0.6821 (13 sets) | n/a | card | PARTIAL, different set list |

I computed the 12-set averages from the card's per-set cells. The cards' summary table lists 0.5571 / 0.6556 / 0.7001 / 0.6082 / 0.4873, which differs from their own per-set table averages. I used the per-set table.

**Honest read:**
- v0.1 clearly beats ModernBERT-base-v3 under the same protocol: +0.133 on 14 sets, +0.075 on the 12 zero-shot sets.
- Against **gliclass-base-v3.0** (DeBERTa, 187M), v0.1 leads on 14 sets (0.689 vs 0.676). That lead comes from our two in-domain sets.
- On the **12 zero-shot sets, v0.1 is below base-v3.0 (0.658 vs 0.672, about −0.015)**. That comparison is PARTIAL: their protocol is full test sets, ours is 500 rows.
- gliclass-large-v3.0 is ahead on both averages.
- Protocol noise: our ModernBERT re-run matches the card average within 0.002, but individual sets differ by up to ±0.056 (enron_spam −0.056, massive +0.034, imdb −0.024). Per-set gaps under about 0.03 are not meaningful.
- Not fixed yet: a FAIR re-run of gliclass-base-v3.0 under eval.py. I started it on CPU and killed it at the steer while it was still fitting calibration, so it produced **no numbers**.

### 2b. Intent sets with all labels (accuracy)
| Model | Banking77 acc | CLINC-150 acc | Label |
|---|---|---|---|
| v0.1 | 0.548 | 0.640 | |
| modern-base-v3, our re-run | 0.302 | 0.486 | FAIR |
| v3 cards | not reported (Banking appears only as macro-F1 inside the suite) | not reported | n/a |
| v4 cards (edge / 2B / 9B / 27B) | 90.7 / 82.6 / 80.1 / 80.0 "skill" | 92.2 / 82.6 / 85.0 / 88.9 "skill", **with OOS** | **NOT comparable**: chance-corrected skill score, Decision Index harness, OOS included, instructions used |

### 2c. Typed decisions
- Our re-run of modern-base-v3 scores 0.391. That is FAIR on protocol, but v0.1 is partly in-domain, so it is **not a zero-shot comparison**.
- v4 cards report "typed decisions 49.3–68.5" on their own 28-row suite. That is a different suite and metric, and v4 trains on Jev-style data, so it is **not comparable**.

### 2d. Long-context decisions (long-Jev ≥6k chars, 8192 tokens). New today
| Model | Acc [cluster 95% CI] | Truncated | Paired vs ModernBERT | Label |
|---|---|---|---|---|
| Chance (mean 1/#options) | 0.331 | | | |
| gliclass-modern-base-v3.0 (CPU fp32, 8192, 23 min) | **0.434** [0.382, 0.483] | 1.9% (its tokenizer: max 11.5k tokens) | n/a | PARTIAL |
| UltraGliClass S2 final | 0.552 | 5.6% | +0.118, CI [+0.054, +0.183], p = 1.4e-4 (113 W / 62 L) | PARTIAL |
| **UltraGliClass v0.1** | **0.664** | 5.6% | **+0.230**, CI [+0.167, +0.300], p = 7.1e-13 (147 W / 48 L) | PARTIAL |

ModernBERT-v3 breakdown:
- By kind: choice 0.428, noul 0.571, score 0.308.
- By state length: <8k 0.427, 8–12k 0.477, 12–16k 0.321, 16–24k 0.638, ≥24k 0.381.

Why this is PARTIAL and not FAIR:
1. **Domain:** v0.1 trained on Jev typed decisions (different rows, 0 eval overlap). ModernBERT-v3 never saw Jev, and the paper says it was trained at 1024 tokens.
2. **Tokenizers differ**, so truncation differs (1.9% vs 5.6%).
3. **Precision and hardware differ:** fp32 CPU vs fp16 T4.
4. ModernBERT was given the Jev `prompt` field, which v3 was not trained to use.
5. **No public peer reports this slice, or any eval at 6k+ chars / 8192 tokens.** The v3 paper cites a context of about 1024. v4 cards report Decision Index at 16k/32k tokens, a different harness.

The S2→S3 gain (+0.111) remains an **internal paired gain**, not a leaderboard result.

## 3. What I fixed or ran
- Pulled and checked all final JSONs. Recomputed the 14-set, 12-set and permutation-mean averages. Diffed our ModernBERT re-run against the card per set.
- Live-checked the modern-base-v3 card, the base-v3 card, and arXiv 2508.07662.
  - The card's summary table and per-set table disagree (0.5571 vs 0.5577).
  - The GLiClass author list for the BibTeX is now verified from arXiv: Stepanov, Shtopko, Vodianytskyi, Lukashov, Yavorskyi, Yaroshenko.
- **Ran gliclass-modern-base-v3.0 on long-Jev@8192 on the box CPU** (exit 0, 1,370 s), plus paired compares against S2 and S3.
- Started a FAIR suite re-run of gliclass-base-v3.0 on CPU and **killed it per the steer** with no output. Partial log: `logs_compare/fair_gliclass_base_v3_cpu.log`.

## 4. Gaps needing Srimon's OK (Kaggle)
1. **FAIR suite re-run of the v3 peers** under eval.py at the same 500 rows, 3 permutations and 1024 tokens: gliclass-base-v3.0, gliclass-large-v3.0, gliclass-modern-large-v3.0, and optionally deberta-v3-large-zeroshot-v2.0 (needs an NLI backend that eval.py does not have). This is the decisive check for whether v0.1 beats base-v3.0 on the 12 zero-shot sets.
2. **Full test splits under the official test_gliclass.py protocol** for v0.1, so its number sits directly next to the card numbers. Also report an `add_prefix_space=True` variant for the DeBERTa and ModernBERT peers.
3. **Long-Jev@8192 on GPU** for gliclass-v4-edge (DeBERTa-large, trained on Jev-style data, so a closer fair peer) and gliclass-base-v3.0 (DeBERTa, O(n²) attention at 8192, too big for 3 GB of free box RAM). Also re-run modern-base-v3 in fp16 on T4 to match precision.
4. Optionally, a v4-style eval with instructions for v0.1. The v4 Decision Index skill and CLINC+OOS numbers cannot be compared otherwise.
5. Card fix: the draft card's sentence "Our average of 0.556 reproduces its published card average of 0.5577" is fine. The card should also state the 12-set gap to gliclass-base-v3.0.

Exact Kaggle one-shot command for each peer (code v8, T4):
`python -u eval.py --model <peer> --suites gliclass intent typed --max_per_dataset 500 --permutations 3 --calib_file calib/stage1_val.jsonl --calib_n 2000 --max_length 1024 --dtype fp16 --out results/compare/fair_<peer>.json`
`python -u scripts/eval_long_jev.py --model <peer> --slice calib/jev_long_test.jsonl --max_length 8192 --batch_size 1 --dtype fp16 --out results/compare/longjev_<peer>_8192.json`

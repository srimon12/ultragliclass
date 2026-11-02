# Stage 2 plan: label-order robustness, many-label training, calibration (23 Oct 2026)

Status: code, configs and CPU smoke tests are done (`scripts/test_stage2.py` ALL PASS; `configs/stage2_smoke.yaml` trained 6 steps).
Nothing has been pushed, uploaded or run on Kaggle. Stage 1 numbers are the UltraBERT-150M pilot (`S1`), eval run on Kaggle (fp16, T4).
The fair ModernBERT baseline (`knowledgator/gliclass-modern-base-v3.0`) runs on the box CPU (fp32) with the same eval code. Tables: `results/compare_s1_vs_modernbert.md`.

## 1. Fair baseline, in one paragraph
Same `eval.py`, same suites, 500 examples per dataset (`random.Random(0).sample`), 3 label orders (canonical + 2 seeded shuffles), same calibration file
(`data/stage1/val.jsonl`, rebuilt on the box; T is fitted on its 2000-row sample), same `max_length` 1024. Only the model, device (CPU vs T4) and dtype (fp32 vs fp16) differ.
On the 14 GLiClass sets ModernBERT averages **macro-F1 0.556**, which reproduces its published 0.5577. S1 averages **0.586** (+0.030). S1 also wins on accuracy
(0.627 vs 0.579) and on mean flip rate (0.179 vs 0.215), but is worse on mean ECE after temperature (0.173 vs 0.143).
S1 loses clearly on **sst5** (F1 0.107 vs 0.278), **financial_phrasebank** (0.255 vs 0.416) and **enron_spam** (0.506 vs 0.584).
(Per-dataset table, CLINC and typed decisions: see `results/compare_s1_vs_modernbert.md`, filled in when the baseline finishes.)

## 2. Diagnosis: why S1 flips on label order
Flip rate = share of examples whose predicted label changes across 3 label orders.
S1: clinc150 0.61, banking77 0.54, massive 0.47, 20ng 0.34, cap_sotu 0.31, emotion 0.26. ModernBERT-v3 has similar flips on the same sets
(banking 0.40, massive 0.48, 20ng 0.38, cap_sotu 0.33, emotion 0.21). **So most of the flipping is not S1-specific**: it is a property of zero-shot
classification with many close labels at 30-50% accuracy. Where S1 is clearly worse is banking77 (0.54 vs 0.40) and clinc150 (see table). Evidence from
`scripts/analyze_perm.py` (n=100, 4 orders, `results/perm_analysis_{s1,modernbert}.json`; small n, treat differences below about 0.07 as noise):

1. **Flips live in near-ties.** On S1, the low-margin half of examples flips at 0.70-0.94 on the many-label sets, the high-margin half at 0.10-0.50.
   Flip rate is bounded below by the error rate: 78-93% of examples that the canonical order gets wrong flip, as do 22-43% of those it gets right.
2. **Averaging the logits over orders recovers accuracy for S1** (banking77 0.425 -> 0.52, clinc 0.39 -> 0.45, canonical order 0.44 / 0.42), not for ModernBERT
   (banking 0.265 -> 0.25). So S1's scores contain signal that the order destroys; this is what a consistency term should train in. Majority vote does not help (0.42).
3. **Training never showed many labels.** gliclass-v2.0 (1.196M rows): 46.9% single-label rows (half of those positive-only; 23.7% of all rows have every label true),
   31% with 2-9 labels, 21.7% with exactly 26 labels (from only 7 distinct label sets, one fixed topic taxonomy), and <0.01% with 10-25 or 27+ (max 34). The S1 sample
   (300k rows, mean 6.9 labels, max 50) never has 59-150 labels at test-set quality. Banking77 (77), CLINC (150), massive (59) are all out of distribution in label count.
   (`results/label_count_stats_gliclass_v2.json`.)
4. **No cross-label coupling in the loss.** Per-label BCE/focal on independent logits plus a softmax at eval means near-tie ordering is unconstrained.
   Loss normaliser is `sum over valid label slots / number of slots`, so adding many-label rows will change the effective weighting (about 43% of label slots in our sim).
5. **Shuffling is not the missing piece.** `shuffle_labels` is already on and reshuffles on every fetch in gliclass 0.1.21. `get_negatives` is a dead parameter, `contrastive_loss_coef`
   is a contrastive loss over class embeddings (not about order), and there is no pairwise, consistency or hard-negative loss in the package.
6. **Truncation is not the cause.** `prompt_first=True` puts labels first; the banking (560 tokens) and CLINC (667 tokens) label prompts fit in 1024 with room for the text
   (utterances are about 15 tokens). Truncation does hurt long texts with many labels; see max_length below.
7. **Position bias exists but is mild, except on sst5.** Predicted-position histograms are roughly flat on the many-label sets (clinc has a bump in the last quintile, 0.35).
   On sst5, S1 predicts the first 3 label positions 96% of the time and gets 0.49 accuracy when the gold label is in the first fifth and 0.00 when it is in the last fifth (n=100).
8. **sst5 and financial_phrasebank are a different failure: collapse, not flipping.** FPB flip is 0.0 and accuracy 0.62 = the neutral share (always neutral); sst5 accuracy 0.20.
   The training mix has no ordinal / fine-grained sentiment or financial-text sentiment. ModernBERT-v3 is better there (sst5 0.278 F1 / 0.46 acc).

Take-away for stage 2: (a) teach label-count diversity (20-150 hard labels), (b) make the order irrelevant via a consistency term (with an ablation), (c) add licensed fine-grained
sentiment and financial data, (d) cheap inference-time fallback: average over 2-4 label orders (4x cost) if training does not close the gap.

## 3. Stage-2 changes (all implemented; CPU-tested)

| # | Change | Where | Default |
|---|---|---|---|
| a | **Many-label hard-negative examples (20-150 labels).** Taxonomy sources (HWU64 + Bitext = 91-label intent pool, LEDGAR 100 legal clauses, DBPedia l3 206 classes, financial-news topics 20, GoEmotions 28) carry a label pool. At fetch time a row gets K ~ log-uniform in [20, min(150, pool)] labels: gold labels + 50% hard negatives (TF-IDF char-n-gram neighbours and same-parent siblings) + random pool labels. Near-duplicates of gold (sim >= 0.80) are never added (false-negative guard). The label prompt is trimmed so >= 256 text tokens survive. gliclass-v2.0 rows get a mild version (prob 0.10, K 12-60, sim 0.6). | `ugc/many_label.py` (`expand_row`, `ManyLabelDataset`), `prepare_data.py` (`taxonomy` format, `label_pools.json`) | `data.many_label.enabled: true`, `expand_prob 0.7` (taxonomy rows) |
| b | **Label order re-randomised on every fetch (so every epoch)**, plus fresh label draws; DataLoader workers reseed from `torch.initial_seed()` so workers do not repeat. Stored rows are never mutated. | `ManyLabelDataset`, `gliclass shuffle_labels` | on |
| c | **Permutation-consistency regulariser.** For 25% of expanded rows the same example is fetched with a second label order and both views are stacked into ONE forward pass (so DDP-safe, +1 row in that batch). Loss: symmetric KL between the per-label softmaxes of the two views, aligned by label identity (labels present in only one view after augmentation are masked), weight 0.1. | `PairedCollator`, `consistency_loss`, `StrictTrainer.compute_loss` | `data.consistency.enabled: true`; **ablate** |
| d | **Fine-grained sentiment / financial data**, licensed only: `zeroshot/twitter-financial-news-sentiment` (MIT, 3 classes, x3 with 3 label phrasings), `zeroshot/twitter-financial-news-topic` (MIT, 20 labels, x2), `go_emotions` (Apache-2.0, 28 labels, 40k). **Not used:** financial_phrasebank (it is an eval set, and CC-BY-NC-SA), SST-5 train (the eval set; licence unclear), Yelp-5 / Amazon-5 (no clean commercial licence confirmed). | `configs/data_stage2.yaml` | on |
| e | **max_length stays 1024.** 150 short intent labels = about 700-800 tokens, which leaves >= 256 tokens of text; the expansion is token-budgeted so a 150-label row cannot crowd the text out. 1536 would help only long text with >100 labels (not in our eval); cost would be about +50% per row on the expanded rows. Revisit at stage 3 (8192 for state tracking). | `data.max_length`, `text_keep_tokens` | 1024 |
| f | `model.max_num_classes: 150` (config hygiene; allocation per batch stays dynamic). | `configs/stage2_scale.yaml` | |

Eval-overlap rule: banking77, CLINC-OOS (and massive, snips) stay out of training; their sources exist in the recipe with `enabled: false` (decision D1). HWU64 and Bitext cover
intent-style labels without being the eval sets (HWU64/BANKING-style domains overlap in topic, which we note as a caveat). The 91-label intent pool cannot reach CLINC's 150 labels; K is capped at pool size.

### Data (about 2.45M rows, see `configs/data_stage2.yaml`)
Natural backbone: gliclass-v2.0 (all, 1.196M), logic x3, jev (commercial licence only), procedural 300k. Taxonomy: HWU64 x3, Bitext 26k, LEDGAR 60k, DBPedia-l3 120k, fin-topic x2, GoEmotions 40k, fin-sentiment x3.
Licence flags (amber = needs your OK): DBPedia (card CC0, upstream Wikipedia abstracts are CC-BY-SA-3.0), Bitext (CDLA-Sharing-1.0), tweet text in fin-topic / fin-sentiment (MIT card; platform terms for the text), HWU64 CC-BY-4.0, LEDGAR CC-BY-4.0, GoEmotions Apache-2.0.

## 4. GPU-hours (measured S1 + cost model)
Measured on S1 (T4 x2, fp16, 64 rows/step as 2 GPUs x 4 accumulation x 8 rows, grad checkpointing): **3.0 s/step**, **3.95 GB peak** per GPU, mean padded length about 551, 6.75 labels/row.
`scripts/estimate_stage2_cost.py` tokenises real rows of each source with the same dataset class (cost per micro-batch = R·L·(1 + L/(6·768)), fitted to 3.0 s/step):

| variant | rows | steps | s/step | T4x2 hours |
|---|---|---|---|---|
| first draft (no taxonomy, no expansion) | 2.12M | 33.1k | 3.67 | 33.7 |
| + taxonomy, static labels | 2.45M | 38.3k | 3.52 | 37.4 |
| + hard-negative expansion | 2.45M | 38.3k | 3.65 | 38.8 |
| **+ consistency (pair_prob 0.25) = the spec** | 2.45M | 38.3k | 3.73 | **39.7** |

Memory: 3.95 GB measured at mean length 551. Worst case 8 x 1024 tokens (+1 paired row) extrapolates to about 6 GB (weights + Adam about 2.4 GB fixed, activations scale with tokens); this is an extrapolation, not measured. Plenty of headroom on 15 GB.
Plan: 3 pilots of 1,500 steps on the stage-2 mix starting from S1 (about 1.6 h each = 4.7 h): **A** S1 continue (no expansion, no consistency), **B** + expansion, **C** + expansion + consistency. Then the full run: about 40 h = 4 sessions of 10 h.
**Total about 46 GPU-h = about 1.5 weeks of the ~30 h/week quota** (+ about 1 h of evals). Caveat: the procedural rows are proxied by the S1 val rows, and real throughput with the new collator is unmeasured on GPU.

## 5. Go / no-go bar (set from the fair baseline; protocol = `eval.py --max_per_dataset 500 --permutations 3`, 14 GLiClass sets)
Pilot gate (after A/B/C, 1,500 steps, quick eval): proceed to the full run with the arm that is best on the many-label sets and does not lose more than 0.01 suite average to A.
Full-run gate (after stage 2):

| # | Bar | Reference |
|---|---|---|
| 1 | Suite avg macro-F1 >= **0.606** (ModernBERT 0.556 + 0.05, = S1 0.586 + 0.02) | S1 0.586; ModernBERT 0.556; published 0.5577 |
| 2 | No set more than 0.03 below ModernBERT-v3 (today S1 fails on sst5, FPB, enron_spam) | per-set table |
| 3 | Mean flip rate on banking77 / clinc150 / massive / 20ng <= **0.40**, and below ModernBERT's mean on those sets | S1 about 0.49 |
| 4 | banking77 and clinc150 accuracy >= S1 (0.454 / 0.398) | S1 |
| 5 | Typed-decisions accuracy >= max(S1 0.503, ModernBERT) and ECE after global T <= 0.10 | S1 0.503 / 0.155 |
| 6 | Mean ECE after global T over the 14 sets <= ModernBERT's (0.143) (the old 0.05 bar is unreachable for either model) | |
| 7 | Flip-rate cell is not worse by >0.03 on sets with <= 7 labels | S1 |

If 1-2 pass but 3 fails: ship, and use 3-order logit averaging at inference. If 1 fails: ablate the data mix (drop DBPedia/LEDGAR first), not the consistency term.
Pilot ablation also answers whether the consistency term helps (C vs B) and whether the expansion is worth its cost (B vs A).

## 6. Smoke tests (CPU, done)
`python scripts/test_stage2.py`: ALL PASS (see SMOKE_TEST.md v3). `configs/stage2_smoke.yaml` trains 6 steps through the real `train.py` with expansion + pairs (cons_loss about 0.007-0.011). The full `data/stage2` build was not run (needs about 2.45M-row download and prep).
Bug found and fixed on the way: `UGC_DEBUG_HANG_RANK` hook slept forever in non-torchrun runs (`None == None`); stage 1 on Kaggle (torchrun) was not affected.

## 7. Risks
1. Many-label rows dominate the loss normaliser (about 43% of label slots): watch positive/negative balance and calibration in the pilots.
2. False negatives from the noisy glic-v2 pool (mild settings) and near-synonym classes in DBPedia/LEDGAR (similarity guard 0.80).
3. The consistency term can flatten scores (minimise KL by being uninformative): keep the weight at 0.1 and check ECE/accuracy in pilot C.
4. Held-in vs held-out for intent evals (D1): training on banking77 / clinc would lift the numbers but destroys the zero-shot claim.
5. fp16 behaviour of the new loss on T4 is untested on GPU (tested fp16 on CPU only).
6. Quota: 46 GPU-h needs two weeks if anything else runs.

## 8. Open decisions (Srimon)
D1 held-in vs held-out intent sets (default: held out). D2 DBPedia (CC-BY-SA upstream) and Bitext (CDLA-Sharing) amber licences: keep or drop. D3 tweet text (platform terms) in fin-topic/sentiment. D4 consistency weight 0.1 and pair_prob 0.25 vs ablating first (recommended). D5 max_length 1024 vs 1536. D6 SST-5 train: excluded. D7 financial_phrasebank: excluded (NC + eval set); this leaves FPB to be fixed by the twitter-financial-news data only. D8 3-order logit averaging as an inference default. D9 Laya: not run (see baseline status).

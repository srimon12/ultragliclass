# Datasets (licenses checked live via the Hugging Face Hub API on 22 Oct 2026)

> **Note (restructure, 2 Nov 2026):** the "Mix ratios" section is the pre-run plan. What actually ran: no dedicated procedural source in S2/S3 (0 train rows), S3 label addition 0.05 (not 0.5) and text addition 0.15 (not 0.3); S3 = 21,184 long Jev rows + 40,000 S2 replay (61,184), not ~90k. "Procedural" in the run manifests means the procedural-style rows that arrive inside other sources (e.g. Jev), not the dedicated source. Exact per-stage counts are in `docs/TRAINING_RECIPE.md` and `proofs/stage_summaries/*/…_data_manifest.json`. Paths below that say `prepare_data.py` / `ugc/` now live at `scripts/prepare_data.py` / `src/ugc/`.

Legend: ✅ commercially usable as tagged · ⚠️ usable with a caveat · ⛔ non-commercial or unclear, **eval only**.
Every training row is converted to the GLiClass format by `prepare_data.py` / `ugc/convert.py`:
`{"text", "all_labels", "true_labels" (list, or {label: prob} soft targets), "prompt"?, "source", "kind", "license_use"?}`.

## Training sources

| Dataset (HF id) | License (Hub tag) | Size | Used in | Preprocessing to GLiClass format |
|---|---|---|---|---|
| `knowledgator/gliclass-v2.0` | ✅⚠️ apache-2.0; upstream sources undocumented | 1,196,218 rows; 7.6 labels avg (max 34); text 750 chars avg | S1 300k, S2 all, S3 30k replay | native format; drop empty or duplicate labels; text capped at 6,000 chars; ≤50 labels |
| `knowledgator/gliclass-v3-logic-dataset` | ✅ apache-2.0 | 7,776 rows; up to 104 labels | S1 ×2, S2 ×3 | native; rows with >50 labels dropped in S1/S2 (kept in S3 via max_options 256) |
| `tasksource/tasksource-jev-typed-decisions` (config `default`) | ⚠️ "other": a per-row `license_use` of commercial 600,555 / unspecified 283,843 / non-commercial 115,434 (train) | train 999,832 / val 15,000 / test 15,000; choice 785k, noul 108k, score 107k | S1 100k, S2 all (commercial) | **filter `license_use == "commercial"`**; choice/score → options as labels, soft target = option probabilities, question as prompt; noul → labels ["yes","no"], target {yes: p, no: 1−p}; JSON states rendered as `key: value` lines; state ≤4,000 chars (S1/S2) |
| same, config `full` | ⚠️ as above | train 2,550,012 | S3 | same filter, plus only states of ≥6,000 chars (about 1 row in 215) |
| `tasksource/procedural-typed-decisions` (configs `all`, `needle_retrieval`) | ✅ apache-2.0 (procedurally generated, Reasoning Core [R22]) | `all`: 360k train / 18k val / 18k test, 18 tasks | S2 300k, S3 50k | System One `state + questions + answers` → one row per question, GLiClass-v4 mapping [R8]: choice = option descriptions, noul = true/false descriptions (yes/no when missing), score = level descriptions in order; instructions as prompt; target = answer probabilities |

Mix ratios (rows after filtering; EST where the filters were not run on the full data):

| Stage | gliclass-v2.0 | logic | jev (commercial) | procedural | Total | Notes |
|---|---|---|---|---|---|---|
| 0 (Kaggle) | 7,200 (72%) | 400 (4%) | 2,400 (24%) | – | 10k | streamed |
| 1 pilot | 300,000 (72%) | 14,952 (4%; 7,476 × 2) | 100,000 (24%) | – | ~415k | val: 2k + 300 + 2k |
| 2 scale | ~1.19M (57%) | ~22k (1%; × 3) | ~570k (27%, EST after the ≤4k-char filter) | 300k (14%) | ~2.1M | + official augmentation (label removal/addition, text addition, p=0.05) |
| 3 long/many | 30k replay (~35%) | – | ~7k long states (EST) | 50k (needle + all) | ~90k (EST) | max_length 8192, ≤256 options, label-addition p=0.5, text-addition p=0.3 |

Candidates not in v1 (checked, kept in reserve):

| Dataset | License | Why not now |
|---|---|---|
| `knowledgator/gliclass-v2.0-RAC` | ✅ apache-2.0 | retrieval-augmented variant; only needed if we train RAC |
| `MoritzLaurer/synthetic_zeroshot_mixtral_v0.1` | ✅ apache-2.0 | a good S2 add-on (synthetic zero-shot); held back to keep S1 identical to the official GLiClass data |
| `tasksource/zero-shot-label-nli` | ⚠️ other (mixed) | overlaps with gliclass-v2.0 NLI parts; license mixed |
| `BioMike/formal-logic-reasoning-gliclass-2k` | ⛔ no license tag | unclear license |
| jev `license_use: unspecified` (284k rows) | ⛔ unclear | open decision for Srimon (research-only build) |
| jev `license_use: non-commercial` (115k rows) | ⛔ NC | excluded |

Caveat on "commercial": the jev `license_use: commercial` bucket includes share-alike sources (cc-by-sa-3.0/4.0).
Whether a model trained on CC-BY-SA text carries share-alike duties is unsettled. It is flagged for a legal check
before any commercial release.

## Evaluation sets (never trained on)

| Set (as used in eval.py) | HF id loaded | License | Test rows / labels | Notes |
|---|---|---|---|---|
| SetFit/CR | `SetFit/CR` | ⛔ no tag | 376 / 2 | official set |
| SetFit/sst2 | `SetFit/sst2` | ⛔ no tag | 1,821 / 2 | official |
| SetFit/sst5 | `SetFit/sst5` | ⛔ no tag | 2,210 / 5 | label names fixed as in the official script |
| stanfordnlp/imdb | `stanfordnlp/imdb` | ⛔ other | 25,000 / 2 | labels negative/positive as in the official script |
| SetFit/20_newsgroups | `SetFit/20_newsgroups` | ⛔ no tag | 7,532 / 20 | official |
| SetFit/enron_spam | `SetFit/enron_spam` | ⛔ no tag | 2,000 / 2 | official |
| massive | `mteb/amazon_massive_intent` (en) | ✅ apache-2.0 tag (upstream AmazonScience/massive is cc-by-4.0) | 2,974 / 59 | parquet mirror of the script-based `AmazonScience/massive` en-US |
| banking77 (all 77 labels) | `mteb/banking77` | ✅ MIT tag (upstream PolyAI/banking77 cc-by-4.0) | 3,076 / 77 | parquet mirror of the script-based `PolyAI/banking77`; same label strings |
| financial_phrasebank | `gtfintechlab/financial_phrasebank_sentences_allagree` (config 5768, train+test = 2,264) | ⚠️ cc-by-sa-3.0 tag; upstream `takala/financial_phrasebank` is ⛔ cc-by-nc-sa-3.0 | 2,264 / 3 | same 2,264 allagree sentences as the official script; eval only |
| ag_news | `fancyzhx/ag_news` | ⛔ unknown | 7,600 / 4 | official |
| dair-ai/emotion | `dair-ai/emotion` (split) | ⛔ other | 2,000 / 6 | official |
| MoritzLaurer/cap_sotu | `MoritzLaurer/cap_sotu` | ⛔ no tag | 23,040 (train only) / 22 | official uses the train split |
| rotten_tomatoes | `cornell-movie-review-data/rotten_tomatoes` | ⛔ unknown | 1,066 / 2 | official label names neg/pos |
| snips | `benayas/snips` | ✅ apache-2.0 | 1,400 / 7 | listed on the GLiClass cards, not in test_gliclass.py |
| clinc150 (all 150 in-scope labels) | `clinc/clinc_oos` (plus) | ✅ cc-by-3.0 | 4,500 in-scope (+1,000 oos with `--clinc_oos`) / 150 | intent names as labels |
| typed-decisions | `LocalLLaMA/typed-decisions` (all, test) | ✅ apache-2.0 | 400 cases / 2,000 decisions | train split (1,200 cases) is not used |

⛔ on an eval set only means "do not train on it". Evaluating and reporting is fine for research.

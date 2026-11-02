---
license: apache-2.0
language:
- en
library_name: gliclass
pipeline_tag: zero-shot-classification
base_model: dmis-lab/UltraBERT-150M
tags:
- GLiClass
- zero-shot-classification
- text-classification
- long-context
- decision-model
- ultrabert
datasets:
- knowledgator/gliclass-v2.0
- knowledgator/gliclass-v3-logic-dataset
- tasksource/tasksource-jev-typed-decisions
- LocalLLaMA/typed-decisions
- DeepPavlov/hwu64
- coastalcph/lex_glue
- zeroshot/twitter-financial-news-topic
- zeroshot/twitter-financial-news-sentiment
- google-research-datasets/go_emotions
metrics:
- f1
- accuracy
---

# UltraGliClass v0.1

UltraGliClass is a zero-shot text classifier with about 150M parameters. It uses the [GLiClass](https://github.com/Knowledgator/GLiClass) uni-encoder head on top of the [`dmis-lab/UltraBERT-150M`](https://huggingface.co/dmis-lab/UltraBERT-150M) encoder. The official GLiClass base models use ModernBERT-base or DeBERTa-v3-base instead.

You pass a text, a list of label strings and, optionally, an instruction. The model scores every label in one forward pass. It works for single-label and multi-label classification, intent detection, and choice / yes-no / score decisions over long inputs.

This release is the final checkpoint of Stage 3. Training ran in three stages. The last stage trained on inputs of up to 8,192 tokens and up to 256 labels.

> **License notice.** The code and the weights are released under Apache-2.0. Stages 1 and 2 trained on rows of `tasksource/tasksource-jev-typed-decisions` that are licensed CC-BY-SA. Stage 3 removed share-alike rows from its new data, but it starts from the Stage 2 weights and replays Stage 2 data. **These weights are therefore not free of the CC-BY-SA question.** Whether share-alike terms apply to model weights is legally unsettled. If you need weights with no share-alike training data, do not use v0.1. Details are in [Licenses and training data](#licenses-and-training-data).

- **Developed by:** Srimon Danguria
- **Model type:** GLiClass uni-encoder (zero-shot sequence classification)
- **Base model:** `dmis-lab/UltraBERT-150M` (Apache-2.0)
- **Language:** English
- **Code, training recipe and result files:** [github.com/srimon12/ultragliclass](https://github.com/srimon12/ultragliclass)

## Results at a glance

All numbers below come from one evaluation run on full test splits, with the same harness and the same GPU (Kaggle 2× T4) for every model, on 2 November 2026.

| | **UltraGliClass v0.1** | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|
| 14-set GLiClass zero-shot suite, average macro-F1 | **0.688** | 0.558 | 0.673 |
| 12 sets that are zero-shot for every model ¹ | 0.663 | 0.586 | **0.668** |
| Banking77, all 77 labels in one pass (accuracy) | **0.584** | 0.318 | 0.454 |
| CLINC-150, all 150 labels in one pass (accuracy) | **0.636** | 0.473 | 0.608 |
| Long Jev decisions at 8,192 tokens (accuracy, n = 431) ² | **0.664** | 0.434 | 0.415 |

¹ Leaves out `massive` and `financial_phrasebank`, which are close to our training data. On these 12 sets v0.1 ties gliclass-base-v3.0 (Δ −0.005, 95% CI [−0.011, +0.000]).
² Held-out Jev rows whose state is at least 6,000 characters. This slice is **not zero-shot for v0.1**, because other Jev rows are in its training data. It is zero-shot for the two other models.

What this means in practice:
- Against gliclass-modern-base-v3.0 (same encoder size), v0.1 is ahead on the suite by +0.130 (95% CI [+0.124, +0.136]).
- Against gliclass-base-v3.0, v0.1 is ahead by +0.015 [+0.009, +0.020] on 14 sets and tied on the 12 strictly zero-shot sets. Its clear gains are on many-label intent detection and long inputs.
- On long decision inputs at 8,192 tokens, v0.1 beats modern-base by +0.230 [+0.167, +0.300] (McNemar p = 7.1e-13) and base-v3 by +0.248 [+0.180, +0.316] (p = 1.3e-14).
- gliclass-base-v3.0 is better on sst5 (0.338 vs 0.239), 20 Newsgroups (0.505 vs 0.420), spam (0.625 vs 0.581), cap_sotu (0.465 vs 0.435) and snips (0.926 vs 0.913).
- CLINC-150 accuracy (0.636) is below the bar we set before Stage 3 (0.650, the Stage 2 value). We accepted this for v0.1.

## Context length

- v0.1 was trained and evaluated at **8,192 tokens**. That is the length behind every long-input number on this card.
- The UltraBERT encoder supports 32,768 positions, and nothing in the model stops you from passing longer inputs.
- **Inputs longer than 8,192 tokens have not been tested with this checkpoint.** If you use 16K or 32K windows, measure accuracy, memory and latency on your own data first.
- The shipped tokenizer config sets `model_max_length` to 8,192 so that the default matches what was trained. Pass a larger `max_length` yourself if you want to try longer inputs.

## How to use

UltraBERT is supported natively in Transformers 5.20 and later, so no `trust_remote_code` is needed. All training and evaluation used `transformers==5.20.2` and `gliclass==0.1.21`.

```bash
pip install "transformers>=5.20.2" "gliclass==0.1.21"
```

```python
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

repo = "Srimon12/ultragliclass-v0.1"
model = GLiClassModel.from_pretrained(repo)
tokenizer = AutoTokenizer.from_pretrained(repo, add_prefix_space=False)  # trained without a prefix space

pipe = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type="multi-label",   # or "single-label"
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    max_length=8192,                     # trained length; gliclass defaults to 1,024
    max_classes=256,                     # trained label count; gliclass defaults to 25
)

text = "The invoice total does not match the purchase order and the vendor is not on the approved list."
labels = ["approve", "reject", "escalate to finance"]
print(pipe(text, labels, threshold=0.5)[0])
```

Notes:
- Always pass `max_length=8192` and `max_classes=256` to the pipeline. The `gliclass` pipeline does not read these from the config, and its defaults (1,024 tokens, 25 labels) are lower than what the model was trained for.
- Load the tokenizer with `add_prefix_space=False`. That is how the model was trained and evaluated.
- Older Transformers versions (before 5.20) have not been tested.
- This is a classifier, not a Sentence Transformers embedding model.
- For fine-tuning, use the GLiClass `train.py` recipe (uni-encoder) with an encoder learning rate of about 5e-6 to 1e-5, a head learning rate about 2.5–3× the encoder rate, and fp16 or bf16.

## Intended use

- A starting point for fine-tuning your own classification, intent, routing or decision tasks, especially when inputs are long.
- Zero-shot classification over label sets you define at run time, single-label or multi-label.
- Typed decisions (choice, yes/no, score) over long inputs such as documents, records or logs, up to the 8,192 tokens it was trained at.

Not recommended:
- High-stakes decisions without human review, such as medical, legal, credit or hiring decisions.
- Non-English text. Training and evaluation were English only.
- Inputs longer than 8,192 tokens without your own testing (see [Context length](#context-length)).
- Reporting the typed-decisions score as zero-shot (see [Limitations](#limitations)).

## Model details

| | |
|---|---|
| Architecture | GLiClass `uni-encoder`: labels and text in one sequence (`<<LABEL>>l1<<LABEL>>l2…<<SEP>>[prompt]text`), prompt first, first-token pooling for text and labels, MLP scorer, dropout 0.3 |
| Encoder | `dmis-lab/UltraBERT-150M`: 24 layers, hidden size 768, 6 heads (head_dim 128), intermediate size 1,024 with SiLU, RMSNorm. Every third layer uses full attention; the others use sliding-window attention (window 128). RoPE θ 2.56M (full) / 10k (sliding). 32,768 max positions |
| Vocabulary | 49,152 BPE tokens (GPT2Tokenizer) plus 3 GLiClass special tokens (`<<LABEL>>`, `<<SEP>>`, `<<EXAMPLE>>`) = 49,155 |
| Weights | `model.safetensors`, 615 MB, fp32 |
| Loss | GLiClass focal BCE (α 0.75, γ −1) over label slots, with soft targets for Jev score/probability rows. Stages 2 and 3 add a label-order consistency term (symmetric KL, weight 0.1, pair probability 0.25) |
| Trained context | 1,024 tokens (Stages 1–2), **8,192 tokens** (Stage 3) |
| Trained label count | up to 50 (S1), 20–150 with hard-negative expansion (S2), up to 256 (S3, expansion k ≤ 250) |
| Training precision | fp16 on 2× T4 (DDP, SDPA attention) |
| Software | `transformers==5.20.2`, `gliclass==0.1.21` |

## Training

All three stages ran on Kaggle with 2× T4, fp16 and an effective batch of 64 rows per step. The optimizer was AdamW (β 0.9/0.98, ε 1e-6) with weight decay 0.1 on the encoder, 0.01 on the heads and none on the RMSNorm gains, with a linear schedule. Each stage starts from the final model of the stage before it.

| Stage | Data (train rows) | Context / labels | Steps | LR (encoder / head) | Result |
|---|---|---|---|---|---|
| 1 pilot | 410,422: gliclass-v2.0 300,000 + gliclass-v3-logic 10,422 (×2) + Jev typed decisions (commercial bucket) 100,000 | 1,024 / up to 50 | 6,413 | 1e-5 / 3e-5, 5% warmup | val loss 0.0305; suite F1 0.586 |
| 2 scale (arm C) | 1,731,505: gliclass-v2.0 1,089,242; logic 12,891; Jev (commercial bucket) 436,985; LEDGAR 59,930; go_emotions 39,998; fin-news topic 33,980; fin-news sentiment 28,626; HWU64 26,853; LocalLLaMA/typed-decisions **train** 3,000 | 1,024 / 20–150 (hard-negative expansion) + consistency loss | 27,000 (2 sessions) | 1e-5 / 3e-5, 2% warmup | val loss 0.0775; suite F1 0.683 |
| 3 long context | 61,184: 21,184 long Jev rows (states ≥ 6,000 chars, commercial, **share-alike and copyleft rows removed**) + 40,000 replay rows sampled from the Stage 2 mix | **8,192** / up to 256 | 956 (smoke 150 + resume 806) | 5e-6 / 1.5e-5 | val loss 0.1070 (different val set) |

Data hygiene:
- Train rows whose text exactly matched a validation or eval text were dropped: 865 in Stage 2 and 774 in Stage 3.
- Stage 3 had 0 train texts in the long-Jev eval slice.
- The LocalLLaMA/typed-decisions **test** split was never read. A code guard stops the data build if it is.
- The suite's eval sets were blocked from training with a `forbid_ids` guard. Two of them still have close in-domain training data; they are marked below.

Stage 3 compute: about 7.27 Kaggle GPU-hours (smoke 1.16 + full 6.11), a median of about 24 s per step at 8,192 tokens, and 5.47 GB peak reserved memory per T4.

The full recipe, with configs, data-build scripts and data checksums, is in [`docs/TRAINING_RECIPE.md`](https://github.com/srimon12/ultragliclass/blob/main/docs/TRAINING_RECIPE.md).

## Evaluation

### Protocol

- **Full test splits** of every set. cap_sotu has no test split, so its only split (23,040 rows) is used, as in the official GLiClass script. About 82k suite rows, plus 3,076 Banking77 rows and 4,500 CLINC-150 rows.
- Single-label softmax argmax. Macro-F1 for the suite, accuracy for the intent sets. Label strings are the ones in the official `test_gliclass.py`. Inputs up to 1,024 tokens (8,192 for long Jev), fp16.
- Each set is scored with **3 label orders** (the canonical order plus 2 seeded shuffles, seed 0). Tables show the canonical-order score; the mean over orders is also given.
- **Same harness and same GPU (Kaggle 2× T4) for all three models.** Each model uses its authors' tokenizer setting: `add_prefix_space=False` for UltraGliClass and `True` for the two Knowledgator models. Calibration uses one global temperature fitted on 2,000 rows of `calib/stage1_val.jsonl`.
- Comparison models, pinned by Hub commit: `knowledgator/gliclass-modern-base-v3.0` (ModernBERT-base, commit `ac369222`) and `knowledgator/gliclass-base-v3.0` (DeBERTa-v3-base, commit `77a70e6c`).
- Software: `torch 2.11.0+cu128`, `transformers 5.20.2`, `gliclass 0.1.21`, `datasets 5.1.0`.
- Paired confidence intervals are bootstrapped over the same rows. Intent and long-Jev p-values are exact McNemar tests.

**Harness check.** Our run reproduces the published averages of both comparison models on the same 14 sets: modern-base **0.5577** (published 0.5577, Δ −0.0000) and base-v3 **0.6728** (published 0.6764, Δ −0.0036).

### Zero-shot text classification (macro-F1, full test splits)

| Dataset | n | **UltraGliClass v0.1** | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|---|
| CR | 376 | **0.911** | 0.896 | 0.913 |
| sst2 | 1,821 | **0.918** | 0.899 | 0.895 |
| sst5 | 2,210 | **0.239** | 0.276 | 0.338 |
| imdb | 25,000 | **0.949** | 0.916 | 0.925 |
| 20_newsgroups | 7,532 | **0.420** | 0.334 | 0.505 |
| spam (enron_spam) | 2,000 | **0.581** | 0.590 | 0.625 |
| massive ¹ | 2,974 | **0.772** | 0.363 | 0.501 |
| banking77 | 3,076 | **0.564** | 0.349 | 0.467 |
| financial_phrasebank ¹ | 2,264 | **0.894** | 0.419 | 0.897 |
| ag_news | 7,600 | **0.747** | 0.707 | 0.721 |
| emotion | 2,000 | **0.481** | 0.425 | 0.447 |
| cap_sotu | 23,040 | **0.435** | 0.361 | 0.465 |
| rotten_tomatoes | 1,066 | **0.804** | 0.707 | 0.794 |
| snips | 1,400 | **0.913** | 0.567 | 0.926 |
| **Average (14)** | | **0.688** | **0.558** | **0.673** |
| **Average (12, zero-shot for all)** ² | | **0.663** | **0.586** | **0.668** |
| Average (14), mean of 3 label orders | | 0.683 | 0.571 | 0.668 |

The v0.1 column is bold for readability, not because it is always highest.

¹ **Not zero-shot for UltraGliClass.** `massive` shares 56 of its 59 intents with HWU64, which was in Stage 2 training. `financial_phrasebank` uses the same 3 sentiment labels as the financial-tweets sentiment set, also in Stage 2 training. ² Leaves out massive and financial_phrasebank.

Where v0.1 is behind:
- vs **modern-base**: sst5 (−0.037) and spam (−0.009).
- vs **base-v3**: sst5 (−0.099), 20_newsgroups (−0.085), spam (−0.044), cap_sotu (−0.030), snips (−0.013), financial_phrasebank (−0.003), CR (−0.002).

### Intent detection, all labels in one pass (accuracy, full test splits)

| Dataset | labels | n | **UltraGliClass v0.1** | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|---|---|
| Banking77 | 77 | 3,076 | **0.584** | 0.318 | 0.454 |
| CLINC-150 (in-scope) | 150 | 4,500 | **0.636** | 0.473 | 0.608 |

**CLINC-150:** before Stage 3 we set the bar "CLINC-150 ≥ Stage 2 (0.650)". On the full 4,500-row test split v0.1 scores **0.636** (95% CI 0.622–0.650), so it misses that bar. The earlier 500-row Stage 3 check gave 0.640. v0.1 is still ahead of both comparison models on CLINC-150 (+0.163 over modern-base, p = 6.1e-93; +0.028 over base-v3, p = 1.1e-4).

### Long-context decisions: long Jev at 8,192 tokens (accuracy)

The eval slice (`calib/jev_long_test.jsonl` in the GitHub repo):
- Rows from Jev typed decisions (config `full`) whose state is at least 6,000 characters, taken from the **test (177) and validation (254) splits**. Training used only the `train` split. No state or group is shared with training, and Stage 3 had 0 train texts in this slice.
- 431 rows in 344 groups: choice 390, yes/no 28, score 13. By `license_use`: commercial 324, unspecified 83, non-commercial 24. These rows were used only for evaluation.
- Length with the UltraBERT tokenizer: median 3,463 tokens, p90 7,648, max 23,508.
- **Not zero-shot for UltraGliClass** (other Jev rows are in its training data); zero-shot for the other two models. Confidence intervals use a cluster bootstrap by state.

| Model | accuracy [95% CI] | truncated at 8,192 | Δ v0.1 − model [95% CI] | McNemar p |
|---|---|---|---|---|
| **UltraGliClass v0.1** | **0.664** [0.614, 0.713] | 5.6% | | |
| gliclass-modern-base-v3.0 | 0.434 [0.382, 0.483] | 1.9% | **+0.230** [+0.167, +0.300] | 7.1e-13 (147 wins / 48 losses) |
| gliclass-base-v3.0 | 0.415 [0.365, 0.465] | 1.2% | **+0.248** [+0.180, +0.316] | 1.3e-14 (153 wins / 46 losses) |

Breakdown (accuracy):

| Slice | n | **v0.1** | modern-base | base-v3 |
|---|---|---|---|---|
| choice | 390 | **0.662** | 0.428 | 0.418 |
| yes/no | 28 | **0.786** | 0.571 | 0.536 |
| score | 13 | **0.462** | 0.308 | 0.077 |
| state < 8k chars | 124 | **0.710** | 0.427 | 0.363 |
| 8k–12k | 86 | **0.744** | 0.477 | 0.453 |
| 12k–16k | 56 | **0.500** | 0.321 | 0.286 |
| 16k–24k | 47 | **0.617** | 0.638 | 0.596 |
| ≥ 24k | 118 | **0.653** | 0.381 | 0.432 |

On the same slice, Stage 2 scored 0.552. Stage 3's long-context training added **+0.111** (cluster 95% CI [+0.068, +0.157], McNemar exact p = 1.9e-7). This Stage 2 vs Stage 3 comparison comes from the training-time eval (`stage3_longjev_compare.json`).

### Paired significance on the suite (same rows): v0.1 − other model

| Model | Δ avg14 [95% CI] | Δ avg12 [95% CI] | Δ Banking77 acc (p) | Δ CLINC-150 acc (p) |
|---|---|---|---|---|
| gliclass-modern-base-v3.0 | +0.130 [+0.124, +0.136] | +0.078 [+0.072, +0.084] | +0.266 (1.2e-169) | +0.163 (6.1e-93) |
| gliclass-base-v3.0 | +0.015 [+0.009, +0.020] | −0.005 [−0.011, +0.000] | +0.129 (5.1e-51) | +0.028 (1.1e-04) |

### Other measurements (same run)

- How often the prediction changes across 3 label orders (v0.1 / modern-base / base-v3): Banking77 0.248 / 0.420 / 0.214, CLINC-150 0.198 / 0.496 / 0.278, massive 0.116 / 0.506 / 0.455, 20 Newsgroups 0.165 / 0.384 / 0.256.
- Calibration: mean ECE over the 14 suite sets after one global temperature is **0.114** for v0.1 (T = 1.81), 0.137 for modern-base (T = 3.19) and 0.161 for base-v3 (T = 3.63).

### Earlier training-time checks (kept for history)

During training, go/no-go decisions used a smaller protocol: **at most 500 examples per set**, the same label strings, 1,024 tokens, fp16 on T4, with the modern-base baseline re-run on CPU in fp32. These numbers only show progress between stages. Use the full-split tables above for any comparison.

| ≤ 500-row check | S1 | S2 | S3 (= v0.1) |
|---|---|---|---|
| Suite average (14), macro-F1 | 0.586 | 0.683 | 0.689 |
| Banking77 accuracy | 0.454 | 0.554 | 0.548 |
| CLINC-150 accuracy | 0.398 | 0.650 | 0.640 |

#### Typed decisions (LocalLLaMA/typed-decisions test, 400 cases / 2,000 decisions; training-time check, not re-run in the full evaluation)

| | gliclass-modern-base-v3.0 (our CPU run) | S1 | S2 | **v0.1** |
|---|---|---|---|---|
| Accuracy | 0.391 | 0.503 | 0.616 | **0.574** |
| ECE (raw) | 0.413 | 0.249 | 0.106 | **0.078** |

v0.1 by workflow: customer_service 0.710 ³, agent_trace_observability 0.612 ³, security_incidents 0.584, invoice_processing 0.390. Stage 2 by workflow: 0.726 / 0.626 / 0.592 / 0.518.

³ **In-domain.** Stage 2 trained on 1,500 rows from each of these two workflows, taken from the dataset's **train** split. The overall typed score is therefore **not zero-shot**. Only security_incidents and invoice_processing are unseen workflows.

v0.1 is **lower than Stage 2 on typed decisions** (−0.042), mostly on invoice_processing (−0.128). One possible cause is the larger share of procedural-style rows in the Stage 3 data (about 13.9%, against about 4.9% in Stage 2). We have not confirmed this.

Latency (training-time check): median 5.8 ms per example, batched, fp16 on a T4, at 1,024 tokens.

## Limitations

- **CLINC-150 misses the Stage 3 bar:** 0.636 on the full test split against a bar of 0.650 (the Stage 2 value on the 500-row protocol). Stage 2 was also slightly better on Banking77 and typed decisions in the training-time checks. v0.1 gives up these small short-input points for the long-context gain.
- **Not better than gliclass-base-v3.0 on strictly zero-shot short text.** On the 12 sets that are zero-shot for all models, v0.1 (0.663) ties base-v3 (0.668; Δ −0.005, 95% CI [−0.011, +0.000]). Part of the 14-set lead (+0.015) comes from the two in-domain sets.
- **Typed decisions are partly in-domain** (footnote ³), and the typed score went down from Stage 2. Do not report it as zero-shot.
- **Two suite sets are not zero-shot** (massive, financial_phrasebank), and the long-Jev slice is not zero-shot for v0.1. The 12-set average is given for this reason.
- Weak on fine-grained 5-way sentiment (sst5 0.239; base-v3 0.338) and on 20 Newsgroups (0.420; base-v3 0.505).
- Many-label tasks are sensitive to label order (prediction changes for 0.20 of CLINC-150 rows and 0.25 of Banking77 rows). Averaging logits over a few label orders helps.
- Long inputs: accuracy is lowest for states of 12k–16k characters (0.500). Inputs over 8,192 tokens are cut off, and cut-off rows score lower (0.583, n = 24). Nothing was tested beyond 8,192 tokens.
- UltraBERT was pre-trained in bf16. All our training and evaluation used fp16 with no NaN losses. bf16 inference with this checkpoint was not tested.
- Possible benchmark overlap: the upstream sources of gliclass-v2.0 are not documented, so overlap with eval domains cannot be ruled out. The official GLiClass cards carry the same caveat.
- English only. Small sets (CR n = 376, rotten_tomatoes n = 1,066) have wide per-set intervals, so a difference of 0.02–0.03 on one small set is within noise.

## Licenses and training data

**Release license:** Apache-2.0 for the code and the weights, with the CC-BY-SA caveat below.

| Component | License | Note |
|---|---|---|
| Base encoder `dmis-lab/UltraBERT-150M` | Apache-2.0 | per its Hub card, checked 1 Nov 2026 |
| GLiClass code and recipe | Apache-2.0 | Knowledgator |
| knowledgator/gliclass-v2.0, gliclass-v3-logic-dataset | Apache-2.0 (Hub tag) | upstream sources of gliclass-v2.0 are not documented |
| tasksource/tasksource-jev-typed-decisions | per-row `license` / `license_use` field | see below |
| LocalLLaMA/typed-decisions (train split only) | Apache-2.0, synthetic | in-domain for the typed eval |
| HWU64, LEDGAR (via LexGLUE) | CC-BY-4.0 | attribution required |
| go_emotions | Apache-2.0 | Reddit text; Reddit's terms also apply |
| twitter-financial-news topic / sentiment | MIT (dataset card) | tweet text; X/Twitter's terms also apply |
| DBpedia, Bitext | not used | excluded because of share-alike licenses |

**The CC-BY-SA caveat, step by step:**
1. **Stages 1 and 2:** Jev rows were filtered to `license_use == "commercial"`. That bucket still includes share-alike sources (CC-BY-SA-3.0 and 4.0). So Stage 1 (100,000 Jev rows) and Stage 2 (436,985 Jev rows) trained on some CC-BY-SA text.
2. **Stage 3:** the 21,184 new long-Jev rows are commercial-only and exclude every row whose license contains `-sa`, `by-sa`, `gpl` or `odbl`. That filter removed about 1.8k rows, so Stage 3 added no new share-alike or copyleft text.
3. The 40,000 Stage 3 replay rows were sampled from the Stage 2 mix, and v0.1 starts from the Stage 2 weights. **So v0.1 is not free of the CC-BY-SA caveat.**
4. Whether weights trained on CC-BY-SA text carry share-alike duties is not settled law. If this matters for your use, get legal advice. Jev rows labeled `unspecified` (284k) or `non-commercial` (115k) were never used for training; rows of every `license_use` value appear only in the long-Jev eval slice.

A later version trained without any share-alike data would remove this caveat. v0.1 does not.

**Typed-decisions overlap:** Stage 2 used 3,000 rows from the LocalLLaMA/typed-decisions **train** split (customer_service 1,500, agent_trace_observability 1,500). They share the test split's schema, workflows and 57 option strings, so typed results must be reported as partly in-domain.

**Eval sets** (SST, CR, IMDB, 20NG, Enron, AG News, emotion, cap_sotu, rotten_tomatoes, financial_phrasebank, CLINC, Banking77, MASSIVE, SNIPS) were used only for evaluation. Several of them have unclear or non-commercial licenses, and none were trained on.

## Files

| file | bytes | sha256 |
|---|---|---|
| `model.safetensors` (fp32) | 615,323,864 | `8964cf620013b4b1307baf7f9b132ca13051b314f7cb34b0bcae557e803751fa` |
| `config.json` | 4,589 | `56ea39a073f7d7e835909d4c49f58a746bbfd04694ab1dba9e6bd9038413eeb7` |
| `tokenizer.json` | 3,524,287 | `b22a0247b08dc8420d1fd950f51bf903b4d579a851f081e3962e8cb1bc53d225` |
| `tokenizer_config.json` | 985 | `3bbeea3e325eafaa07f1bc9888c6a40acb8b86a1723d83af4851b8e4159d5ee1` |

The weights and `tokenizer.json` are byte-identical to the Stage 3 training output. Two config values were changed for this release so the defaults match training: `max_num_classes` 50 → 256 in `config.json`, and `max_length` 1,024 → 8,192 and `model_max_length` 32,768 → 8,192 in `tokenizer_config.json` (a machine-local `is_local` key was also dropped). These changes do not affect the scores: the `gliclass` pipeline sizes label slots from the labels you pass and takes `max_length` as an argument. The original training-output hashes are `config.json` `748a02d905618b1dec94511bdcfc0a743e0f4748aedc16ca3f5af608a684785e` and `tokenizer_config.json` `949f9464da03d1e6a54deb9d50ba9016304ee5c3d8fb0d5b89f52e2a82951387`.

## Where the numbers come from

Every number on this card is taken from a result file in the [GitHub repo](https://github.com/srimon12/ultragliclass) (paths relative to the repo root; index in `docs/PROOFS.md`):

- **Full evaluation (main tables):** `proofs/hf_readme_eval/` — `TABLES.md`, `STATUS.json` (complete, nothing missing, 1.217 GPU-hours), `summary.json` (per-set scores, paired tests, long Jev), `models.json` (comparison-model commits), `protocol.json`, `env.json`, `longjev/*.json` (per-row long-Jev results, no text). Harness: `scripts/hf_readme_eval.py`. (The footnote in `TABLES.md` says the long-Jev rows are from the test split; the slice manifest `calib/jev_long_test_manifest.json` shows test 177 + validation 254, which is what this card states.)
- **Stage 3 training-time checks, typed decisions, latency:** `proofs/stage_summaries/s3/stage3_final_quick.json`
- **Stage 3 vs Stage 2 on long Jev:** `proofs/stage_summaries/s3/stage3_final_longjev_8192.json`, `stage3_baseline_s2_longjev_8192.json`, `stage3_longjev_compare.json`
- **Stage 3 status and data:** `proofs/stage_summaries/s3/STATUS.json`, `stage3_full_summary_s1.json`, `stage3_smoke_summary_s1.json`, `stage3_data_manifest.json`, `stage3_data_check.json`
- **Stage 2:** `proofs/stage_summaries/s2/stage2_full_C_final_quick.json`, `stage2_full_data_manifest.json`, `STATUS.json`, `STATUS_session1.json`
- **Stage 1:** `proofs/stage_summaries/s1/stage1_ultrabert150m_quick.json`, `stage1_data_manifest.json`, `STATUS.json`
- **Compute:** `proofs/stage_summaries/gpu_hours_log.tsv`
- **ModernBERT training-time baseline (CPU):** `proofs/stage_summaries/baselines/stage1_fair_modernbert_v3_cpu.json`
- **Licenses:** `docs/DATASETS.md`, `configs/data_stage3.yaml`
- **Recipe:** `docs/TRAINING_RECIPE.md`

## Citation

```bibtex
@misc{danguria2026ultragliclass,
  title        = {UltraGliClass v0.1: GLiClass on UltraBERT-150M with long-context decision training},
  author       = {Danguria, Srimon},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Srimon12/ultragliclass-v0.1}}
}

@misc{stepanov2025gliclass,
  title         = {GLiClass: Generalist Lightweight Model for Sequence Classification Tasks},
  author        = {Stepanov, Ihor and Shtopko, Mykhailo and Vodianytskyi, Dmytro and Lukashov, Oleksandr and Yavorskyi, Alexander and Yaroshenko, Mykyta},
  year          = {2025},
  eprint        = {2508.07662},
  archivePrefix = {arXiv}
}

@misc{park2026ultrabert,
  title         = {UltraBERT: Beyond the Limit of BERT Pre-Training},
  author        = {Park, Jungwoo and Lee, Taewhoo and Kang, Jaewoo},
  year          = {2026},
  eprint        = {2610.15028},
  archivePrefix = {arXiv}
}
```

If you use results on Jev typed decisions, please also cite the dataset as its card asks: [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions).

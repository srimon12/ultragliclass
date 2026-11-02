---
license: apache-2.0   # REVIEW: proposed. Base model and GLiClass are apache-2.0. See "Licenses and data caveats" (CC-BY-SA question is open).
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

# UltraGliClass v0.1 (UltraBERT-150M + GLiClass uni-encoder)

> **Status: pre-release draft (not published).** Numbers come from the full-split README evaluation (Kaggle kernel `ultragliclass-evalreadme` v1, finished about 01:29 IST on 2 Nov 2026, status `complete`, nothing missing, 1.22 GPU-h) and from the stage training artifacts. Every number cites a result file in this repo's `proofs/` (see the appendix and `docs/PROOFS.md`). **[REVIEW]** = needs a decision before release; **[UNVERIFIED]** = not checked against a live source. Nothing has been pushed to the Hugging Face Hub.

UltraGliClass is a zero-shot text classifier and decision model with about 150M parameters. It puts the [GLiClass](https://github.com/Knowledgator/GLiClass) uni-encoder head on top of the [`dmis-lab/UltraBERT-150M`](https://huggingface.co/dmis-lab/UltraBERT-150M) encoder, where the official GLiClass base models use ModernBERT-base. You give it a text and a set of label strings, plus an optional instruction prompt, and it scores every label in one forward pass.

v0.1 is the **Stage 3 final checkpoint**. It was trained in three stages, and the last stage extended training to 8,192-token inputs and up to 256 options.

**In short** (full test splits, same harness and GPU for every model, 2 Nov 2026 IST):
- **14-set GLiClass zero-shot suite, average macro-F1: 0.688.** `knowledgator/gliclass-modern-base-v3.0` scores **0.558** and `knowledgator/gliclass-base-v3.0` scores **0.673** on the same run. Paired gains: **+0.130** [95% CI +0.124, +0.136] over modern-base and **+0.015** [+0.009, +0.020] over base-v3.
- **12 sets that are zero-shot for every model** (massive and financial_phrasebank removed): v0.1 **0.663**, modern-base **0.586**, base-v3 **0.668**. Against base-v3 this is a tie (−0.005, 95% CI [−0.011, +0.000]).
- **Intent detection with all labels in one pass (accuracy):** Banking77 **0.584** (modern 0.318, base-v3 0.454); CLINC-150 **0.636** (modern 0.473, base-v3 0.608).
- **Long decision inputs** (431 held-out Jev rows with states of at least 6,000 characters, at 8,192 tokens): accuracy **0.664**, against **0.434** for modern-base and **0.415** for base-v3. That is **+0.230** [+0.167, +0.300] (McNemar p = 7.1e-13) and **+0.248** [+0.180, +0.316] (p = 1.3e-14). This slice is not zero-shot for v0.1 (other Jev rows are in its training data); it is zero-shot for the peers.
- **Where base-v3 is better:** sst5 (0.239 vs 0.338), 20 Newsgroups (0.420 vs 0.505), spam (0.581 vs 0.625), cap_sotu (0.435 vs 0.465), snips (0.913 vs 0.926).
- **Known shortfall:** full-split CLINC-150 accuracy is **0.636**, still under the pre-set Stage 3 bar of 0.650 (the Stage 2 value). This was accepted for v0.1.
- Earlier Stage 1/2/3 training-eval numbers (at most 500 examples per set, for example the old 0.689 vs 0.556 suite average) are **superseded for README claims** by this full-split run.

## Intended use

- **Main use: a generalist starting point for further fine-tuning.** It is a strong initialization for your own classification, intent, routing or decision tasks with GLiClass's `train.py`, especially when inputs are long.
- Zero-shot classification over label sets you define at runtime, both single-label and multi-label.
- System One-style typed decisions (choice / yes-no / score) over long states, such as documents, records or logs, up to the 8,192 tokens it was trained at.

**Out of scope / not recommended:**
- High-stakes decisions without human review, such as medical, legal, credit or employment decisions.
- Non-English text. Training and evaluation were English only.
- Inputs longer than 8,192 tokens. The encoder supports 32K positions, but this model was never trained or tested beyond 8,192.
- Treating the typed-decisions score as a zero-shot result (see Limitations).

## Model details

| | |
|---|---|
| Architecture | GLiClass `uni-encoder`: labels and text in one sequence (`<<LABEL>>l1<<LABEL>>l2…<<SEP>>[prompt]text`), prompt first, first-token pooling for text and labels, MLP scorer, dropout 0.3 |
| Encoder | `dmis-lab/UltraBERT-150M`: 24 layers, hidden 768, 6 heads (head_dim 128), intermediate 1,024 SiLU, RMSNorm. Every third layer is full attention and the others use sliding-window attention (window 128). RoPE θ 2.56M (full) / 10k (sliding). 32,768 max positions |
| Vocabulary | 49,152 BPE (GPT2Tokenizer) + 3 GLiClass special tokens (`<<LABEL>>`, `<<SEP>>`, `<<EXAMPLE>>`) = 49,155 |
| Weights | `model.safetensors` 615 MB, fp32 |
| Loss | GLiClass focal BCE (α 0.75, γ −1) over label slots, with soft targets for Jev score/probability rows. Stages 2–3 add a permutation-consistency term (symmetric KL, weight 0.1, pair prob 0.25) |
| Trained context | 1,024 tokens (Stages 1–2), **8,192 tokens** (Stage 3) |
| Trained label count | ≤50 (S1), 20–150 with hard-negative expansion (S2), up to 256 (S3, expansion k ≤ 250) |
| Precision in training | fp16 on 2× T4 (DDP, SDPA attention) |
| Software | `transformers==5.20.2`, `gliclass==0.1.21` |
| Developed by | Srimon Danguria **[REVIEW: author / org name for the card]** |

## Training

All three stages ran on Kaggle with 2× T4, fp16 and an effective batch of 64 rows per step. They used AdamW (β 0.9/0.98, ε 1e-6) with weight decay 0.1 on the encoder, 0.01 on the heads and none on the RMSNorm gains, and a linear schedule. Each stage starts from the final model of the one before it.

| Stage | Data (train rows) | Context / labels | Steps | LR (enc / head) | Result |
|---|---|---|---|---|---|
| 1 pilot | 410,422: gliclass-v2.0 300,000 + gliclass-v3-logic 10,422 (×2) + Jev typed decisions (commercial bucket) 100,000 | 1,024 / ≤50 | 6,413 | 1e-5 / 3e-5, 5% warmup | val loss 0.0305; suite F1 0.586 |
| 2 scale (arm C) | 1,731,505: gliclass-v2.0 1,089,242; logic 12,891; Jev (commercial bucket) 436,985; LEDGAR 59,930; go_emotions 39,998; fin-news topic 33,980; fin-news sentiment 28,626; HWU64 26,853; LocalLLaMA/typed-decisions **train** 3,000 | 1,024 / 20–150 (hard-negative expansion) + consistency loss | 27,000 (2 sessions) | 1e-5 / 3e-5, 2% warmup | val loss 0.0775; suite F1 0.683 |
| 3 long context | 61,184: 21,184 long Jev rows (states ≥6,000 chars, commercial, **share-alike/copyleft rows removed**) + 40,000 replay rows sampled from the Stage 2 mix | **8,192** / up to 256 | 956 (smoke 150 + resume 806) | 5e-6 / 1.5e-5 | val loss 0.1070 (different val set); see below |

Data hygiene:
- Train rows whose text exactly matched a validation or eval-slice text were dropped: 865 in S2 and 774 in S3.
- Stage 3 had 0 train texts in the long-Jev eval slice.
- The LocalLLaMA/typed-decisions **test** split was never read, and a code guard aborts the build if it is.
- The eval sets in the suite were blocked from training through a `forbid_ids` guard. Two sets still have close in-domain training data, flagged below.

Stage 3 compute: about 7.27 Kaggle GPU-h (smoke 1.16 + full 6.11), with a median of about 24 s/step at 8,192 tokens and 5.47 GB peak reserved memory per T4.

## Evaluation

### Protocol (README numbers)

- **Full test splits** of every set (cap_sotu has no test split, so its only split, 23,040 rows, is used, as the official GLiClass script does). About 82k suite rows, plus 3,076 Banking77 and 4,500 CLINC-150 rows.
- Single-label softmax argmax, macro-F1 for the suite and accuracy for the intent sets, with the official `test_gliclass.py` label strings. Inputs up to 1,024 tokens (8,192 for long-Jev), fp16.
- Each set is scored with **3 label orders** (canonical order plus 2 seeded shuffles, seed 0). Tables show the canonical-order score; the mean over orders is reported too.
- **Same harness and the same GPU (Kaggle 2× T4) for all three models.** Each model uses its authors' tokenizer setting (`add_prefix_space=False` for UltraGliClass, `True` for the two Knowledgator models). Calibration uses one global temperature fitted on 2,000 rows of `calib/stage1_val.jsonl`.
- Peers, pinned by Hub commit: `knowledgator/gliclass-modern-base-v3.0` (ModernBERT-base, commit `ac369222`) and `knowledgator/gliclass-base-v3.0` (DeBERTa-v3-base, commit `77a70e6c`).
- Software: `torch 2.11.0+cu128`, `transformers 5.20.2`, `gliclass 0.1.21`, `datasets 5.1.0`.
- Paired confidence intervals are bootstrap over the same rows; intent and long-Jev p-values are exact McNemar tests.

**Reproduction check.** Our run reproduces the published card averages of both peers on the same 14 sets: modern-base **0.5577** (card 0.5577, Δ −0.0000) and base-v3 **0.6728** (card 0.6764, Δ −0.0036).

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

¹ **Not zero-shot for UltraGliClass.** `massive` shares 56 of its 59 intents with HWU64, which was in Stage 2 training. `financial_phrasebank` uses the same 3 sentiment labels as the financial-tweets sentiment set, which was in Stage 2 training. ² Excludes massive and financial_phrasebank.

Where v0.1 trails:
- vs **modern-base**: sst5 (−0.037) and spam (−0.009).
- vs **base-v3**: sst5 (−0.099), 20_newsgroups (−0.085), spam (−0.044), cap_sotu (−0.030), snips (−0.013), financial_phrasebank (−0.003), CR (−0.002).

### Intent detection, all labels in one pass (accuracy, full test splits)

| Dataset | labels | n | **UltraGliClass v0.1** | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|---|---|
| Banking77 | 77 | 3,076 | **0.584** | 0.318 | 0.454 |
| CLINC-150 (in-scope) | 150 | 4,500 | **0.636** | 0.473 | 0.608 |

**CLINC-150 caveat:** the pre-registered Stage 3 bar was "CLINC-150 ≥ Stage 2 (0.650)". On the full 4,500-row test split v0.1 scores **0.636** (95% CI 0.622–0.650), so the bar is **still not met**. The earlier ≤500-row Stage 3 quick eval gave 0.640. v0.1 is still ahead of both peers on CLINC-150 (+0.163 over modern-base, p = 6.1e-93; +0.028 over base-v3, p = 1.1e-4).

### Long-context decisions: long-Jev at 8,192 tokens (accuracy)

Eval slice (`calib/jev_long_test.jsonl`):
- Jev typed decisions (config `full`), held-out rows with states ≥6,000 characters from the **test (177) and validation (254) splits** (`calib/jev_long_test_manifest.json`, `by_split`). Training used only the `train` split; no state or group is shared with training, and Stage 3 had 0 train texts in this slice. **[REVIEW]** `proofs/hf_readme_eval/TABLES.md` (generated by the eval run) says "test split" in its footnote; the manifest shows test + validation. Use the manifest wording in public copy.
- 431 rows / 344 groups: choice 390, noul 28, score 13. By `license_use`: commercial 324, unspecified 83, non-commercial 24. These rows are used for eval only.
- Tokens (UltraBERT tokenizer): median 3,463, p90 7,648, max 23,508.
- **Not zero-shot for UltraGliClass** (other Jev rows are in its training data); zero-shot for the peers. Confidence intervals use a cluster bootstrap by state.

| Model | accuracy [95% CI] | truncated at 8,192 | Δ v0.1 − model [95% CI] | McNemar p |
|---|---|---|---|---|
| **UltraGliClass v0.1** | **0.664** [0.614, 0.713] | 5.6% | | |
| gliclass-modern-base-v3.0 | 0.434 [0.382, 0.483] | 1.9% | **+0.230** [+0.167, +0.300] | 7.1e-13 (147 wins / 48 losses) |
| gliclass-base-v3.0 | 0.415 [0.365, 0.465] | 1.2% | **+0.248** [+0.180, +0.316] | 1.3e-14 (153 wins / 46 losses) |

Breakdown (accuracy):

| Slice | n | **v0.1** | modern-base | base-v3 |
|---|---|---|---|---|
| choice | 390 | **0.662** | 0.428 | 0.418 |
| noul (yes/no) | 28 | **0.786** | 0.571 | 0.536 |
| score | 13 | **0.462** | 0.308 | 0.077 |
| state < 8k chars | 124 | **0.710** | 0.427 | 0.363 |
| 8k–12k | 86 | **0.744** | 0.477 | 0.453 |
| 12k–16k | 56 | **0.500** | 0.321 | 0.286 |
| 16k–24k | 47 | **0.617** | 0.638 | 0.596 |
| ≥ 24k | 118 | **0.653** | 0.381 | 0.432 |

Training history on the same slice: Stage 2 scored 0.552, so Stage 3's long-context training added **+0.111** (cluster 95% CI [+0.068, +0.157], McNemar exact p = 1.9e-7). This Stage 2 vs Stage 3 comparison comes from the training eval (`stage3_longjev_compare.json`).

### Paired significance on the suite (same rows): v0.1 − peer

| Peer | Δ avg14 [95% CI] | Δ avg12 [95% CI] | Δ Banking77 acc (p) | Δ CLINC-150 acc (p) |
|---|---|---|---|---|
| gliclass-modern-base-v3.0 | +0.130 [+0.124, +0.136] | +0.078 [+0.072, +0.084] | +0.266 (1.2e-169) | +0.163 (6.1e-93) |
| gliclass-base-v3.0 | +0.015 [+0.009, +0.020] | −0.005 [−0.011, +0.000] | +0.129 (5.1e-51) | +0.028 (1.1e-04) |

### Other measurements (full-split README run)
- Label-order flip rate across 3 orders (v0.1 / modern-base / base-v3): Banking77 0.248 / 0.420 / 0.214, CLINC-150 0.198 / 0.496 / 0.278, massive 0.116 / 0.506 / 0.455, 20 Newsgroups 0.165 / 0.384 / 0.256.
- Calibration: mean ECE over the 14 suite sets after one global temperature is **0.114** for v0.1 (T = 1.81), 0.137 for modern-base (T = 3.19) and 0.161 for base-v3 (T = 3.63).

### Earlier training-time evaluations (historical, superseded for README claims)

Stage go/no-go decisions during training used a smaller protocol: **at most 500 examples per set**, the same label strings, 1,024 tokens, fp16 on T4, with the modern-base baseline re-run on CPU in fp32. Those numbers are kept here only to show progress between stages. The full-split tables above replace them for any public claim.

| ≤500-row training eval | S1 | S2 | S3 (= v0.1) |
|---|---|---|---|
| Suite average (14), macro-F1 | 0.586 | 0.683 | 0.689 |
| Banking77 accuracy | 0.454 | 0.554 | 0.548 |
| CLINC-150 accuracy | 0.398 | 0.650 | 0.640 |

#### Typed decisions (LocalLLaMA/typed-decisions test, 400 cases / 2,000 decisions; training-time eval, not re-run in the README eval)

| | gliclass-modern-base-v3.0 (ours, CPU) | S1 | S2 | **v0.1** |
|---|---|---|---|---|
| Accuracy | 0.391 | 0.503 | 0.616 | **0.574** |
| ECE (raw) | 0.413 | 0.249 | 0.106 | **0.078** |

v0.1 by workflow: customer_service 0.710 ³, agent_trace_observability 0.612 ³, security_incidents 0.584, invoice_processing 0.390. S2 by workflow: 0.726 / 0.626 / 0.592 / 0.518.

³ **In-domain.** Stage 2 trained on 1,500 rows from each of these two workflows, taken from the dataset's **train** split. The overall typed score is therefore **not zero-shot**. Only security_incidents and invoice_processing are unseen workflows.

v0.1 is **lower than Stage 2 on typed decisions** (−0.042), with most of the loss on invoice_processing (−0.128). One suspected cause is the high procedural-style share of the Stage 3 data, about 13.9% compared with about 4.9% in S2. That cause is not confirmed.

Latency (training-time eval): median 5.8 ms per example, batched, fp16 on a T4, at 1,024 tokens.

## How to use

UltraGliClass loads with the `gliclass` package. UltraBERT has been **native in Transformers since v5.20**, and GLiClass loads the encoder without `trust_remote_code`. All training and evaluation ran this way with `transformers==5.20.2`.

```bash
pip install "transformers>=5.20.2" "gliclass==0.1.21"
# or, from this repository: pip install -e .   ->  from ultragliclass import load_pipeline, predict
```

```python
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

repo = "<hf-user>/ultragliclass-v0.1"   # [REVIEW] placeholder repo id
model = GLiClassModel.from_pretrained(repo)
tokenizer = AutoTokenizer.from_pretrained(repo, add_prefix_space=False)  # trained without prefix space

pipe = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type="multi-label",   # or "single-label"
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    max_length=8192,                     # trained up to 8,192; the tokenizer default is 1,024
    max_classes=256,                     # trained up to 256 labels; gliclass's pipeline default is 25
)

text = "The invoice total does not match the purchase order and the vendor is not on the approved list."
labels = ["approve", "reject", "escalate to finance"]
print(pipe(text, labels, threshold=0.5)[0])
```

Notes:
- `max_length` and the number of labels: the saved config has `max_num_classes: 50` and the tokenizer has `max_length: 1024`. Pass `max_length` (and `max_num_classes` if your GLiClass version needs it) to use long inputs or many labels. **[REVIEW: consider setting these in the released config.]**
- **[UNVERIFIED] Older Transformers (< 5.20):** the encoder config keeps UltraBERT's `auto_map` (`modeling_ultrabert.*`), and the base repo ships custom code (`custom_code` tag on the Hub). Loading through `trust_remote_code=True` on older versions is untested for this model, and GLiClass does not pass that flag itself.
- **[UNVERIFIED] Sentence Transformers:** the brief says the UltraBERT family works in Sentence Transformers with `trust_remote_code`. That applies to the UltraBERT encoder and embedding models. This GLiClass checkpoint is a classifier and is **not** a Sentence Transformers model. It was not tested that way.
- Fine-tuning: use the GLiClass `train.py` recipe (uni-encoder) with encoder LR around 5e-6 to 1e-5, head LR about 2.5–3× the encoder LR, and fp16 or bf16.

## Limitations

- **CLINC-150 is under the pre-set Stage 3 bar:** 0.636 on the full test split against a bar of 0.650 (the Stage 2 value on the ≤500-row protocol). Stage 2 was also marginally better on Banking77 and typed decisions in the training-time evals. v0.1 trades those small short-input losses for the long-context gain.
- **Not better than gliclass-base-v3.0 on purely zero-shot short text.** On the 12 sets that are zero-shot for all models, v0.1 (0.663) ties base-v3 (0.668; Δ −0.005, 95% CI [−0.011, +0.000]). The 14-set lead (+0.015) depends partly on the two in-domain sets.
- **Typed decisions are partly in-domain** (see footnote ³), and the overall typed score went down from S2. Do not report it as zero-shot.
- **Two suite sets are not zero-shot** (massive, financial_phrasebank), and the long-Jev slice is not zero-shot for v0.1. The 12-set average is given above.
- Weak on fine-grained 5-way sentiment (sst5 0.239, base-v3 0.338) and on 20 Newsgroups (0.420, base-v3 0.505).
- Many-label tasks remain sensitive to label order (CLINC-150 flip rate 0.20, Banking77 0.25). Averaging logits over a few label orders helps.
- Long inputs: accuracy is lowest for states of 12k–16k characters (0.500). Inputs over 8,192 tokens are truncated, and truncated rows score worse (0.583, n = 24). Nothing was tested beyond 8,192 tokens.
- fp16 training: UltraBERT was pre-trained in bf16. All our runs used fp16 with no NaN losses, but bf16 inference or training of this checkpoint was not tested.
- Benchmark contamination: gliclass-v2.0's upstream sources are undocumented, so overlap with eval domains cannot be ruled out. The published GLiClass cards carry the same caveat.
- English only. The README tables use full test splits; small sets (CR n = 376, rotten_tomatoes n = 1,066) still have wide per-set intervals, so differences of about 0.02 to 0.03 on one small set are within noise.

## Licenses and data caveats

**[REVIEW] Release license.** The proposed license is Apache-2.0. Before choosing it, Srimon must decide on the CC-BY-SA question below and, if this is a commercial release, get a legal check.

| Component | License | Note |
|---|---|---|
| Base encoder `dmis-lab/UltraBERT-150M` | **Apache-2.0** (Hub card, checked live 1 Nov 2026) | The brief said MIT; the live card says apache-2.0. **[REVIEW]** |
| GLiClass code / recipe | Apache-2.0 | Knowledgator |
| knowledgator/gliclass-v2.0, gliclass-v3-logic-dataset | apache-2.0 (Hub tag) | gliclass-v2.0's upstream sources are undocumented |
| tasksource/tasksource-jev-typed-decisions | "other": a per-row `license` / `license_use` field | see below |
| LocalLLaMA/typed-decisions (train split only) | apache-2.0, synthetic | in-domain for the typed eval |
| HWU64, LEDGAR (via LexGLUE) | CC-BY-4.0 | attribution |
| go_emotions | Apache-2.0 | Reddit text; platform terms (open decision D3) |
| twitter-financial-news topic / sentiment | MIT (card) | tweet text; platform terms (open decision D3) |
| DBpedia, Bitext | not used | excluded for their share-alike licences |

**Jev / CC-BY-SA share-alike caveat (must stay on the card):**
1. **Stages 1 and 2:** Jev rows were filtered to `license_use == "commercial"`. That bucket **still includes share-alike sources (CC-BY-SA-3.0 / 4.0)**. So S1 (100,000 Jev rows) and S2 (436,985 Jev rows) trained on some CC-BY-SA text.
2. **Stage 3:** the 21,184 new long-Jev rows are commercial-only **and** exclude every row whose license contains `-sa`, `by-sa`, `gpl` or `odbl`. That filter removed about 1.8k rows, so Stage 3 added no share-alike or copyleft text.
3. **But** the 40,000 Stage 3 replay rows are sampled from the Stage 2 mix and inherit its caveats. Because v0.1 also starts from the S2 weights, **v0.1 is not free of the CC-BY-SA caveat.**
4. Whether model weights trained on CC-BY-SA text carry share-alike duties is legally unsettled. The project flagged it for a legal check before any commercial release. Jev rows labelled `unspecified` (284k) or `non-commercial` (115k) were **not** used for training. Rows with all `license_use` values appear only in the long-Jev **eval** slice.

**Typed-decisions in-domain caveat:** 3,000 rows from the LocalLLaMA/typed-decisions **train** split were used in Stage 2: customer_service 1,500 and agent_trace_observability 1,500. They are in the same schema as the test split, with the same workflows and the same 57 option strings. Typed results must be reported as partly in-domain.

**Eval sets** (SST, CR, IMDB, 20NG, Enron, AG News, emotion, cap_sotu, rotten_tomatoes, financial_phrasebank, CLINC, Banking77, MASSIVE, SNIPS) were used for evaluation only. Several of them have unclear or non-commercial licences and were never trained on.

## Citation

```bibtex
@misc{ultragliclass2026,
  title  = {UltraGliClass v0.1: GLiClass on UltraBERT-150M with long-context decision training},
  author = {Danguria, Srimon},
  year   = {2026},
  note   = {Model card draft; [REVIEW] repo URL}
}

@misc{stepanov2025gliclass,
  title         = {GLiClass: Generalist Lightweight Model for Sequence Classification Tasks},
  author        = {Stepanov and others},  % [UNVERIFIED] full author list
  year          = {2025},
  eprint        = {2508.07662},
  archivePrefix = {arXiv}
}

@misc{ultrabert2026,
  title         = {UltraBERT: Beyond the Limit of BERT Pre-Training},
  author        = {{DMIS Lab}},
  year          = {2026},
  eprint        = {2610.15028},
  archivePrefix = {arXiv}
}
```
Citation notes:
- **[UNVERIFIED]** The full author lists of both papers are not checked; the BibTeX needs completing from arXiv.
- The arXiv ids come from the project's REFERENCES.md. The Hub tags confirm `arxiv:2610.15028` on UltraBERT-150M and `arxiv:2508.07662` on gliclass-modern-base-v3.0.
- Add the Jev / tasksource and Reasoning Core (arXiv 2608.05148) citations if the dataset authors ask for them. **[UNVERIFIED]**

## Weights (v0.1 = Stage 3 final model)

| file | bytes | sha256 |
|---|---|---|
| `model.safetensors` (fp32) | 615,323,864 | `8964cf620013b4b1307baf7f9b132ca13051b314f7cb34b0bcae557e803751fa` |
| `config.json` | 4,588 | `748a02d905618b1dec94511bdcfc0a743e0f4748aedc16ca3f5af608a684785e` |
| `tokenizer.json` | 3,524,287 | `b22a0247b08dc8420d1fd950f51bf903b4d579a851f081e3962e8cb1bc53d225` |
| `tokenizer_config.json` | 1,006 | `949f9464da03d1e6a54deb9d50ba9016304ee5c3d8fb0d5b89f52e2a82951387` |

**[REVIEW]** Distribution channel (HF Hub repo id vs GitHub Release) and whether to set `max_num_classes: 256` / tokenizer `max_length: 8192` in the released config (would change the config.json hash above).

## Appendix: sources of every number

All paths are relative to the repository root. The full Kaggle artifacts (logits, checkpoints) stay off-repo; their box paths are in `docs/PROOFS.md`.

- **README evaluation (primary):** `proofs/hf_readme_eval/` — `TABLES.md`, `STATUS.json` (`complete`, missing `[]`, 1.217 h), `summary.json` (per-set scores, paired tests, long-Jev), `models.json` (peer Hub commits), `protocol.json`, `env.json`, `longjev/*.json` (per-row long-Jev results). Kaggle kernel `srimondanguria/ultragliclass-evalreadme` v1 (private). Job spec: `docs/history/HF_README_EVAL_JOB.md`. Harness: `scripts/hf_readme_eval.py`.
- **Stage 3 (= v0.1) training eval / typed / latency:** `proofs/stage_summaries/s3/stage3_final_quick.json`
- **Stage 3 long-Jev vs S2:** `proofs/stage_summaries/s3/stage3_final_longjev_8192.json`, `stage3_baseline_s2_longjev_8192.json`, `stage3_longjev_compare.json`
- **Stage 3 status / data:** `proofs/stage_summaries/s3/STATUS.json`, `stage3_full_summary_s1.json`, `stage3_smoke_summary_s1.json`, `stage3_data_manifest.json`, `stage3_data_check.json`
- **Stage 2:** `proofs/stage_summaries/s2/stage2_full_C_final_quick.json`, `stage2_full_data_manifest.json`, `STATUS.json`, `STATUS_session1.json`
- **Stage 1:** `proofs/stage_summaries/s1/stage1_ultrabert150m_quick.json`, `stage1_data_manifest.json`, `STATUS.json`
- **Compute:** `proofs/stage_summaries/gpu_hours_log.tsv`
- **ModernBERT training-time baseline (CPU):** `proofs/stage_summaries/baselines/stage1_fair_modernbert_v3_cpu.json`
- **Licences:** `docs/DATASETS.md`, `docs/specs/STAGE2_FULL_JOB_SPEC.md` §2, `docs/specs/STAGE3_JOB_SPEC.md`, `configs/data_stage3.yaml`
- **Recipe:** `docs/TRAINING_RECIPE.md`

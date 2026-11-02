# UltraGliClass v0.1: zero-shot benchmark (README tables)

Protocol: full test splits, max_length 1024, fp16, single-label softmax argmax, canonical label order (headline), 3 label orders for robustness. Same harness and GPU for every model. Each model uses its authors' tokenizer setting.

## Zero-shot text classification (macro-F1)

| Dataset | n | ultragliclass-v0.1 | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|---|
| CR | 376 | 0.911 | 0.896 | 0.913 |
| sst2 | 1,821 | 0.918 | 0.899 | 0.895 |
| sst5 | 2,210 | 0.239 | 0.276 | 0.338 |
| imdb | 25,000 | 0.949 | 0.916 | 0.925 |
| 20_newsgroups | 7,532 | 0.420 | 0.334 | 0.505 |
| spam | 2,000 | 0.581 | 0.590 | 0.625 |
| massive¹ | 2,974 | 0.772 | 0.363 | 0.501 |
| banking | 3,076 | 0.564 | 0.349 | 0.467 |
| financial_phrasebank¹ | 2,264 | 0.894 | 0.419 | 0.897 |
| ag_news | 7,600 | 0.747 | 0.707 | 0.721 |
| emotion | 2,000 | 0.481 | 0.425 | 0.447 |
| cap_sotu | 23,040 | 0.435 | 0.361 | 0.465 |
| rotten_tomatoes | 1,066 | 0.804 | 0.707 | 0.794 |
| snips | 1,400 | 0.913 | 0.567 | 0.926 |
| **Average (14)** | | **0.688** | **0.558** | **0.673** |
| **Average (12, zero-shot for all)²** | | **0.663** | **0.586** | **0.668** |
| Average (14), mean of label orders | | 0.683 | 0.571 | 0.668 |

¹ In-domain for UltraGliClass: related data (HWU64 intents, financial-tweet sentiment) is in our training mix. ² Excludes massive and financial_phrasebank.

## Intent detection, all labels in one pass (accuracy)

| Dataset | labels | n | ultragliclass-v0.1 | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|---|---|
| Banking77 | 77 | 3,076 | 0.584 | 0.318 | 0.454 |
| CLINC-150 (in-scope) | 150 | 4,500 | 0.636 | 0.473 | 0.608 |

## Long-context decisions: long-Jev @ 8,192 tokens (accuracy)³

| Model | accuracy [95% CI] | truncated | Δ v0.1 − model [95% CI] | McNemar p |
|---|---|---|---|---|
| ultragliclass-v0.1 | 0.664 [0.614, 0.713] | 5.6% |  |  |
| gliclass-modern-base-v3.0 | 0.434 [0.382, 0.483] | 1.9% | +0.230 [+0.167, +0.300] | 7.1e-13 |
| gliclass-base-v3.0 | 0.415 [0.365, 0.465] | 1.2% | +0.248 [+0.180, +0.316] | 1.3e-14 |

³ 431 held-out decision rows with states ≥ 6k characters from tasksource/jev-typed-decisions (test split; no state or group shared with training). **Not zero-shot for UltraGliClass** (other Jev rows are in our training data); zero-shot for the peers. Cluster bootstrap by state.

## Significance (paired, same rows): v0.1 − peer

| Peer | Δ avg14 [95% CI] | Δ avg12 [95% CI] | Δ Banking77 acc (p) | Δ CLINC acc (p) |
|---|---|---|---|---|
| gliclass-modern-base-v3.0 | +0.130 [+0.124, +0.136] | +0.078 [+0.072, +0.084] | +0.266 (1.2e-169) | +0.163 (6.1e-93) |
| gliclass-base-v3.0 | +0.015 [+0.009, +0.020] | -0.005 [-0.011, +0.000] | +0.129 (5.1e-51) | +0.028 (1.1e-04) |

## Reproduction check vs published card numbers (same 14 sets)

| Model | our run avg14 | card avg14 | Δ |
|---|---|---|---|
| gliclass-modern-base-v3.0 | 0.5577 | 0.5577 | -0.0000 |
| gliclass-base-v3.0 | 0.6728 | 0.6764 | -0.0036 |

# UltraGliClass

**A ~150M-parameter zero-shot classifier and long-context decision model.** UltraGliClass puts the
[GLiClass](https://github.com/Knowledgator/GLiClass) uni-encoder head on top of
[`dmis-lab/UltraBERT-150M`](https://huggingface.co/dmis-lab/UltraBERT-150M) and trains it in three stages, ending at
**8,192-token inputs and up to 256 labels in one forward pass**. You give it a text and a list of label strings
(plus an optional instruction) and it scores every label at once.

Author: **Srimon Danguria**.

**Goal of v0.1:** beat `knowledgator/gliclass-modern-base-v3.0` (same encoder size) with the official GLiClass recipe
on an encoder that has a much longer native context (UltraBERT, 32K positions), and show the gain on long inputs.

This repository holds **everything needed to audit and reproduce v0.1**: the exact training code, the stage configs,
the data-build recipes, the evaluation harness, and the result files every number below comes from.

> **Weights:** [`Srimon12/ultragliclass-v0.1`](https://huggingface.co/Srimon12/ultragliclass-v0.1) on the Hugging Face Hub.
> The Hub model card is [`MODEL_CARD.md`](MODEL_CARD.md). Read its license notice (CC-BY-SA caveat) before use.

## Results (v0.1, full test splits, same harness and GPU for every model)

Source: [`proofs/hf_readme_eval/TABLES.md`](proofs/hf_readme_eval/TABLES.md) and `summary.json` (Kaggle run of 2 Nov 2026 IST,
status `complete`, 1.22 GPU-h). Peers are pinned by Hub commit in [`models.json`](proofs/hf_readme_eval/models.json).

| | **UltraGliClass v0.1** | gliclass-modern-base-v3.0 | gliclass-base-v3.0 |
|---|---|---|---|
| 14-set GLiClass zero-shot suite, avg macro-F1 | **0.688** | 0.558 | 0.673 |
| 12 sets zero-shot for every model ¹ | 0.663 | 0.586 | **0.668** |
| Banking77, all 77 labels in one pass (acc) | **0.584** | 0.318 | 0.454 |
| CLINC-150, all 150 labels in one pass (acc) | **0.636** | 0.473 | 0.608 |
| Long-Jev decisions at 8,192 tokens (acc, n=431) ² | **0.664** | 0.434 | 0.415 |

¹ Drops `massive` and `financial_phrasebank`, which are close to our training data. On these 12 sets v0.1 **ties**
gliclass-base-v3.0 (Δ −0.005, 95% CI [−0.011, +0.000]).
² Held-out Jev typed-decision rows with states ≥ 6,000 characters. **Not zero-shot for v0.1** (other Jev rows are in
training); zero-shot for the peers. Δ vs modern-base +0.230 [+0.167, +0.300], McNemar p = 7.1e-13.

Honest caveats, all detailed in the card: base-v3 is better on sst5, 20 Newsgroups, spam, cap_sotu and snips; CLINC-150
(0.636) is **below the pre-registered Stage 3 bar of 0.650** and was accepted for v0.1; typed-decision results are partly
in-domain. Per-set tables, CIs and significance tests: [`MODEL_CARD.md`](MODEL_CARD.md), index in [`docs/PROOFS.md`](docs/PROOFS.md).

## How v0.1 was built (map of this repo)

Each stage starts from the previous stage's final weights. Full detail, gates and caveats: [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md).

| Stage | What it adds | Train rows / steps | Train config → data recipe | Notebook as pushed | Proofs | Result (≤500-row quick eval) |
|---|---|---|---|---|---|---|
| S0 | smoke + resume test (not in the weights) | 10k / 150 | [`stage0_kaggle`](configs/stage0_kaggle.yaml) → [`data_stage0_kaggle`](configs/data_stage0_kaggle.yaml) | [s0a](notebooks/as_pushed/ultragliclass-s0a.ipynb), [s0b](notebooks/as_pushed/ultragliclass-s0b.ipynb) | [`s0/`](proofs/stage_summaries/s0/) | resume works on 2-GPU DDP |
| S1 | GLiClass on UltraBERT, 1,024 tokens, ≤ 50 labels | 410,422 / 6,413 | [`stage1_pilot`](configs/stage1_pilot.yaml) → [`data_stage1`](configs/data_stage1.yaml) | [s1](notebooks/as_pushed/ultragliclass-s1.ipynb) | [`s1/`](proofs/stage_summaries/s1/) | suite 0.586, CLINC 0.398 |
| S2 pilots / ablations | choose arm (A/B/C) and data mix (E1/E1z/E2) | 1,500 steps each | [`stage2_pilot_{A,B,C}`](configs/), [`stage2_abl_*`](configs/) | [s2pilot](notebooks/as_pushed/ultragliclass-s2pilot.ipynb), [s2ablation](notebooks/as_pushed/ultragliclass-s2ablation.ipynb) | [`S2_PILOT_INVESTIGATION.md`](docs/history/S2_PILOT_INVESTIGATION.md) | arm C + E2 mix chosen |
| S2 | 20–150 labels with hard negatives, label-order consistency loss | 1,731,505 / 27,000 (2 sessions) | [`stage2_full_C`](configs/stage2_full_C.yaml) → [`data_stage2_full`](configs/data_stage2_full.yaml) | [s2full](notebooks/as_pushed/ultragliclass-s2full.ipynb) | [`s2/`](proofs/stage_summaries/s2/) | suite 0.683, CLINC 0.650 |
| **S3 = v0.1** | 8,192 tokens, ≤ 256 labels; long Jev (no share-alike rows) + S2 replay | 61,184 / 956 | [`stage3_longctx`](configs/stage3_longctx.yaml) → [`data_stage3`](configs/data_stage3.yaml) | [s3smoke](notebooks/as_pushed/ultragliclass-s3smoke.ipynb), [s3full](notebooks/as_pushed/ultragliclass-s3full.ipynb) | [`s3/`](proofs/stage_summaries/s3/) | suite 0.689, CLINC 0.640, long-Jev 0.552 → 0.664 |
| Eval | full test splits vs two peers | — | [`scripts/hf_readme_eval.py`](scripts/hf_readme_eval.py) | [evalreadme](notebooks/as_pushed/ultragliclass-evalreadme.ipynb) | [`hf_readme_eval/`](proofs/hf_readme_eval/) | table above |

Lineage compute: 5.71 + 18.20 + 7.27 = **31.18 Kaggle GPU-h** on 2× T4, fp16 (41.69 h including pilots, smoke tests and the benchmark).
Software: `torch 2.11.0+cu128`, `transformers 5.20.2`, `gliclass 0.1.21`.
Which code version ran each stage: [`docs/PROVENANCE.md`](docs/PROVENANCE.md). Every claim → file: [`docs/PROOFS.md`](docs/PROOFS.md).

**Stage 3 did not pass all of its own pre-registered bars.** CLINC-150 came in at 0.640 against a bar of 0.650; the plan said
that keeps Stage 2 as the release. We released Stage 3 anyway for its long-context gain. Typed-decision accuracy also fell
(0.616 → 0.574) and we have not found out why.

## What was not done

- No same-recipe ModernBERT control, so the gain over gliclass-modern-base-v3.0 cannot be split between encoder and data.
- Nothing trained or tested beyond 8,192 tokens.
- One seed per stage; no variance estimate.
- No license-clean retrain (see the CC-BY-SA caveat below).

List with details: [`docs/TRAINING_RECIPE.md` §7](docs/TRAINING_RECIPE.md#7-what-was-not-done).

## Quick start (inference)

```bash
git clone https://github.com/srimon12/ultragliclass && cd ultragliclass
uv sync                      # or: pip install -e .   (install a CPU or CUDA torch first if you need a specific build)
```

```python
from ultragliclass import load_pipeline, predict

pipe = load_pipeline("Srimon12/ultragliclass-v0.1")   # or a local path
text = "The invoice total does not match the purchase order and the vendor is not on the approved list."
print(predict(pipe, text, ["approve", "reject", "escalate to finance"]))
```

Or from the shell: `ultragliclass-predict --model path/to/final_model --text "..." --labels approve reject "escalate to finance"`.

`load_pipeline` sets what v0.1 was trained with: `add_prefix_space=False`, `max_length=8192`, up to 256 labels.
(The Hub release sets `max_num_classes: 256` and tokenizer `max_length` / `model_max_length` 8192; the raw Stage 3 training output still says 50 / 1024.)

## Reproduce the training (S1 → S3)

The full recipe — data mixes, row counts, hyperparameters, steps, seeds, GPU-hours, data md5s — is in
**[`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md)**. In short:

```bash
uv sync --extra train        # or: pip install -e ".[train]"
python scripts/prepare_data.py --recipe configs/data_stage1.yaml            # -> data/stage1/
python scripts/train.py --config configs/stage1_pilot.yaml                  # -> runs/stage1_ultrabert150m/final_model
python scripts/prepare_data.py --recipe configs/data_stage2_full.yaml       # needs calib/stage2_pilot_val.jsonl (see data/README.md)
python scripts/train.py --config configs/stage2_full_C.yaml
python scripts/prepare_data.py --recipe configs/data_stage3.yaml            # needs data/stage2_full + calib/jev_long_test.jsonl
python scripts/train.py --config configs/stage3_longctx.yaml                # -> runs/stage3_full/final_model  (= v0.1)
```

Run every command from the repo root. Multi-GPU: `torchrun --nproc_per_node 2 scripts/train.py --config ...`.
Training resumes from the newest checkpoint automatically; override any value with `--set a.b=value`.
v0.1 itself was trained on Kaggle (2× T4, fp16); the exact notebooks that were pushed are in [`notebooks/as_pushed/`](notebooks/as_pushed/).

## Evaluate

```bash
python scripts/eval.py --model runs/stage3_full/final_model --suites gliclass intent typed --out results/quick.json
python scripts/eval_long_jev.py --model runs/stage3_full/final_model --slice calib/jev_long_test.jsonl --max_length 8192
python scripts/hf_readme_eval.py --help      # the full-split README benchmark (prep / worker / aggregate)
```

## Repository layout

```
src/ultragliclass/   public inference API + CLI (thin wrapper around `gliclass`)
src/ugc/             training internals: converters, many-label expansion, strict trainer, metrics (byte-identical to the v0.1 run)
scripts/             prepare_data.py, train.py, eval.py, long-Jev + README benchmark, recipe generators, sanity checks
scripts/analysis/    cost estimates, contamination/overlap checks, result comparisons
scripts/kaggle/      builders for the Kaggle notebooks + CPU dry-run harnesses
configs/             every stage and data recipe used (see configs/README.md for the v0.1 lineage)
calib/               eval/calibration slice manifests (the jsonl files are rebuilt, not committed)
data/                build instructions only; no dataset rows are committed
notebooks/           Kaggle notebook templates; as_pushed/ = the exact notebook + kernel metadata each job ran (no outputs)
proofs/              result files behind every claim + code snapshots of earlier stages
docs/                TRAINING_RECIPE (start here), PROOFS, PROVENANCE, DATASETS, PLAN, REFERENCES, LAUNCH_CHECKLIST, specs/ (pre-run plans), history/ (dev notes)
tests/               CPU tests for many-label expansion, length grouping, train/val dedupe
```

## License and data caveats

- **Code:** Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Training code builds on Knowledgator GLiClass (Apache-2.0).
- **Weights:** Apache-2.0, with the CC-BY-SA caveat below stated on the model card.
- **CC-BY-SA caveat.** Stages 1–2 used Jev rows filtered to `license_use == "commercial"`; that bucket **still contains
  CC-BY-SA text**. Stage 3 removed share-alike/copyleft rows from its new data, but its replay rows and its starting
  weights come from Stage 2, so **v0.1 is not free of the CC-BY-SA caveat**. Whether weights inherit share-alike duties
  is legally unsettled; get a legal check before commercial use.
- **Not used:** Bitext and DBpedia (share-alike licences) were excluded from all training data. Jev rows marked
  `unspecified` or `non-commercial` were never trained on.
- **No dataset rows are committed here.** Eval slices and training data are rebuilt from the Hub with the scripts
  ([`data/README.md`](data/README.md)). Several eval sets have non-commercial or unclear licences and are used for evaluation only.

Full dataset table with licences and transforms: [`docs/DATASETS.md`](docs/DATASETS.md).

## Citation

See [`MODEL_CARD.md`](MODEL_CARD.md#citation).

# UltraGliClass

**A ~150M-parameter zero-shot classifier and long-context decision model.** UltraGliClass puts the
[GLiClass](https://github.com/Knowledgator/GLiClass) uni-encoder head on top of
[`dmis-lab/UltraBERT-150M`](https://huggingface.co/dmis-lab/UltraBERT-150M) and trains it in three stages, ending at
**8,192-token inputs and up to 256 labels in one forward pass**. You give it a text and a list of label strings
(plus an optional instruction) and it scores every label at once.

This repository holds **everything needed to audit and reproduce v0.1**: the exact training code, the stage configs,
the data-build recipes, the evaluation harness, and the result files every number below comes from.

> **Status:** v0.1, pre-release. Weights are **not** published yet (no Hugging Face upload, no GitHub release).
> Open decisions are marked **[REVIEW]** in [`MODEL_CARD.md`](MODEL_CARD.md) and [`docs/LAUNCH_CHECKLIST.md`](docs/LAUNCH_CHECKLIST.md).

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

## Quick start (inference)

```bash
git clone https://github.com/srimon12/ultragliclass && cd ultragliclass
uv sync                      # or: pip install -e .   (install a CPU or CUDA torch first if you need a specific build)
```

```python
from ultragliclass import load_pipeline, predict

pipe = load_pipeline("path/to/final_model")   # [REVIEW] Hub id once published
text = "The invoice total does not match the purchase order and the vendor is not on the approved list."
print(predict(pipe, text, ["approve", "reject", "escalate to finance"]))
```

Or from the shell: `ultragliclass-predict --model path/to/final_model --text "..." --labels approve reject "escalate to finance"`.

`load_pipeline` sets what v0.1 was trained with: `add_prefix_space=False`, `max_length=8192`, up to 256 labels.
(The saved config still says `max_num_classes: 50` / tokenizer `max_length: 1024`; see [REVIEW] in the card.)

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
v0.1 itself was trained on Kaggle (2× T4, fp16) through the notebooks in [`notebooks/`](notebooks/).

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
notebooks/           the exact Kaggle notebooks that ran each stage (no outputs)
proofs/              result files behind every claim + code snapshots of earlier stages
docs/                TRAINING_RECIPE, DATASETS, PLAN, REFERENCES, PROOFS, PROVENANCE, LAUNCH_CHECKLIST, specs/, history/
tests/               CPU tests for many-label expansion, length grouping, train/val dedupe
```

## License and data caveats

- **Code:** Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Training code builds on Knowledgator GLiClass (Apache-2.0).
- **Weights (when released):** proposed Apache-2.0, **[REVIEW]** — pending the CC-BY-SA question below.
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

See [`MODEL_CARD.md`](MODEL_CARD.md#citation) (BibTeX entries still need author lists completed — [UNVERIFIED]).

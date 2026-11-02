# Launch checklist (v0.1 public release)

Status as of 2 Nov 2026 (IST). The repo is **private**; weights are **not** published.

## Decisions for Srimon ([REVIEW])

- [ ] **Release licence** for the weights (Apache-2.0 proposed) — needs a legal view on the CC-BY-SA Jev rows used in
      S1/S2 (S3 excluded `-sa/by-sa/gpl/odbl`). Options: release as-is with notice, or retrain S1/S2 without share-alike rows.
- [ ] Platform-terms question for Reddit (go_emotions) and Twitter (fin-news) text — `docs/PLAN.md` D3.
- [ ] Author / organisation name (README, NOTICE, MODEL_CARD, `pyproject.toml`).
- [ ] Distribution: Hugging Face repo id (card placeholder `<hf-user>/ultragliclass-v0.1`) and/or GitHub Release asset.
- [ ] Ship config with `max_num_classes: 256` and tokenizer `max_length: 8192`? (changes the config.json sha256).
- [ ] Base model licence wording: brief said MIT, Hub card says Apache-2.0 (card notes this).
- [ ] Accepting the CLINC-150 −0.010 miss and the typed-decisions regression (0.616 → 0.574 quick eval) in public copy.

## Must do before public

- [ ] Upload weights (615 MB fp32 safetensors + tokenizer + config) to the Hub/Release; verify sha256 against `docs/TRAINING_RECIPE.md` §3.
- [ ] Complete citation author lists ([UNVERIFIED] in the card).
- [ ] Remove absolute build-box paths (`/workspace/…`) from code comments, docs and history notes (46 lines in ~27 files as of 2 Nov 2026; `docs/PROOFS.md` lists box paths on purpose — replace with archive links).
- [ ] Archive the off-repo proofs (raw logits, logs) — e.g. a Kaggle dataset or Release asset — and link them from `docs/PROOFS.md`.
- [ ] Make the Kaggle notebooks/kernels public or state they are private.
- [ ] Add CI (ruff + `pytest tests/` + an inference smoke on a tiny model).
- [ ] Fix stale wording: `build_long_jev_slice.py` docstring ("TEST slice"), the HWU64 "eval_overlap: none" comment.

## Nice to have / open science

- [ ] Same-recipe ModernBERT control (`configs/stage1_control_modernbert.yaml`, never run) — needed to attribute gains to UltraBERT vs data.
- [ ] Verify the S1 data rebuild with current code (md5 `a9de9501…`) and the `calib/stage2_pilot_val.jsonl` rebuild (md5 `8e5e644e…`).
- [ ] Investigate the typed-decisions drop (invoice_processing −0.128) before v0.2.
- [ ] fp32 → fp16/bf16 weights variant (≈ 300 MB); ONNX.

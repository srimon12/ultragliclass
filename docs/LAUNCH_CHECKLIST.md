# Release checklist (v0.1)

Status as of 2 Nov 2026 (IST): **released.**

- Weights: [`Srimon12/ultragliclass-v0.1`](https://huggingface.co/Srimon12/ultragliclass-v0.1) on the Hugging Face Hub (public).
- Code, recipe and proofs: this repo, [`srimon12/ultragliclass`](https://github.com/srimon12/ultragliclass) (public).

## Decisions taken

- [x] Author: **Srimon Danguria** (README, NOTICE, MODEL_CARD, `pyproject.toml`).
- [x] Release licence: Apache-2.0 for code and weights, **with the CC-BY-SA notice** on the model card and in the README.
      No license-clean retrain for v0.1.
- [x] Hub repo id: `Srimon12/ultragliclass-v0.1`.
- [x] Release config ships `max_num_classes: 256` and tokenizer `max_length` / `model_max_length` 8192. Weights and
      `tokenizer.json` are byte-identical to the training output; old and new config hashes are on the model card.
- [x] Stage 3 released although it missed the pre-registered CLINC-150 bar (0.640 vs 0.650) and typed decisions fell
      (0.616 → 0.574). Stated in the README, the recipe and the card.
- [x] Base model licence: Apache-2.0 per the UltraBERT-150M Hub card (checked 1 Nov 2026).

## Still open

- [ ] Platform-terms question for Reddit (go_emotions) and Twitter (fin-news) text — `docs/PLAN.md` D3.
- [ ] Archive the off-repo proofs (README-eval raw logits, full Kaggle logs) as a Release asset or Kaggle dataset and link them from `docs/PROOFS.md`.
- [ ] The Kaggle kernels and the `ultragliclass-code` dataset are private. Make them public or keep relying on `notebooks/as_pushed/`.
- [ ] Hard-coded build-machine paths remain in some helper scripts and two tests (`scripts/analysis/`, `scripts/kaggle/`,
      `tests/test_length_group.py`, `tests/test_stage2.py`) and in a comment in `configs/stage3_longctx.yaml`. They were left
      unedited so those files stay byte-identical to the versions that ran; pass the path as an argument instead.
- [ ] Add CI (ruff + `pytest tests/` + an inference smoke on a tiny model).
- [ ] Fix stale wording: `build_long_jev_slice.py` docstring ("TEST slice"), the HWU64 "eval_overlap: none" comment.
- [ ] Verify the S1 data rebuild with current code (md5 `a9de950386ab7c99bdd6f1646254f30b`) and the
      `calib/stage2_pilot_val.jsonl` rebuild (md5 `8e5e644ea992bd05503652d507d5591d`).

## For v0.2

- [ ] License-clean retrain: drop every share-alike / copyleft row from S1 and S2, not only S3.
- [ ] Same-recipe ModernBERT control (`configs/stage1_control_modernbert.yaml`, never run) — needed to attribute gains to UltraBERT vs data.
- [ ] Investigate the typed-decisions drop (invoice_processing 0.518 → 0.390) before the next long-context stage.
- [ ] Measure 16K and 32K inputs (accuracy, memory, latency).
- [ ] Length-grouped batching on GPU (built, CPU-tested, never run).
- [ ] fp16/bf16 weights variant (≈ 300 MB); ONNX.

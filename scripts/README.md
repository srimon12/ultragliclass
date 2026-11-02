# scripts/

Run everything from the repo root after `pip install -e ".[train]"` (the scripts import the installed `ugc` package).
None of these files were edited when the repo was restructured (see `docs/PROVENANCE.md`).

| Script | Purpose |
|---|---|
| `prepare_data.py` | Build `data/<stage>/{train,val}.jsonl` + `manifest.json` from a `configs/data_*.yaml` recipe (Hub download, conversion, eval-id guard, dedupe). |
| `train.py` | Train from a `configs/stage*.yaml` (single GPU, CPU, or `torchrun` DDP); resumable; writes `STATUS.json`. |
| `eval.py` | Quick/full eval suites (gliclass 14-set, intent, typed decisions), temperature calibration, latency. |
| `eval_long_jev.py` | Long-Jev@8192 accuracy and paired comparison of two models. |
| `hf_readme_eval.py` | The full-split README benchmark (v0.1 vs peers) that produced `proofs/hf_readme_eval/`. |
| `build_long_jev_slice.py` | Build `calib/jev_long_test.jsonl` (eval only). |
| `make_calib_val.py` | Build only the val split of a recipe (for `calib/stage2_pilot_val.jsonl`). |
| `make_stage2_full_recipe.py`, `make_ablation_recipes.py`, `make_*_smoke_recipe.py` | Generate the derived data recipes in `configs/`. |
| `check_init.py`, `check_unused_params.py`, `dist_check.py` | Model-init / DDP sanity checks. |
| `smoke_cpu.sh` | End-to-end CPU smoke test (data → train → resume → DDP → eval). |
| `analysis/` | One-off analysis used for decisions (pilot arm comparison, overlap checks, cost estimates). |
| `kaggle/` | Builders for the notebooks in `notebooks/` and the code bundle; they assume the original flat layout. |

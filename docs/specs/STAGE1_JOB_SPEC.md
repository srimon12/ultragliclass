# Stage-1 job spec (for the Kaggle Handler; nothing has been uploaded or run on Kaggle yet)

> **Update 22 Oct (v2):** use the v2 notebook (see CHANGELOG.md). It adds a NCCL pre-flight, a 10-min collective timeout, a stall watchdog (`STALL_MINUTES=15`), 1-GPU auto-fallback, and logs in `/kaggle/working/logs/`. Return those logs plus STATUS.json/HEARTBEAT.json after every session. S0b rerun: S0B_RERUN.md.

**Prerequisites that need Srimon's OK:** (a) uploading this folder as a *private* Kaggle Dataset;
(b) about 10-15 GPU-hours of the weekly quota (stage 0 + stage 1 + optional control).

Code bundle: everything in `ultragliclass/` **except** `runs/ data/ results/ smoke_logs/`, uploaded as a private dataset,
e.g. `ultragliclass-code`.
Notebook: `kaggle_stage1.ipynb`. Accelerator **GPU T4 x2**. Internet **on**. Persistence off. Run it with "Save & Run All".

| Session | Notebook knobs (cell 1) | Expected result | Est. GPU time |
|---|---|---|---|
| S0a | `STAGE_CONFIG="configs/stage0_kaggle.yaml"`, `DATA_RECIPE="configs/data_stage0_kaggle.yaml"`, `DATA_DIR="data/stage0k"`, `PREP_ARGS="--streaming"`, `EXTRA_SET=["train.time_budget_hours=0.2"]`, `QUICK_EVAL=False` | STATUS `needs_resume` partway through 150 steps | 0.4 h |
| S0b | same as S0a, `EXTRA_SET=[]`, `PREV_OUTPUT=<S0a output>` | STATUS `complete` at step 150, resumed from S0a's checkpoint | 0.3 h |
| S1 | `STAGE_CONFIG="configs/stage1_pilot.yaml"`, `DATA_RECIPE="configs/data_stage1.yaml"`, `DATA_DIR="data/stage1"`, `PREP_ARGS=""`, `EXTRA_SET=[]`, `QUICK_EVAL=True` | about 6.5k steps; `complete` in 1 session if ≤ about 2.7 s/step, otherwise `needs_resume` → S1b with `PREV_OUTPUT=<S1 output>` | 4-7 h (EST) |
| S1-ctl (optional, recommended) | `STAGE_CONFIG="configs/stage1_control_modernbert.yaml"`, same data knobs as S1, `PREV_OUTPUT=<S1 output>` (reuses data/stage1) | ModernBERT-base control, same data/steps | 3-6 h (EST) |

Equivalent commands inside the notebook (for S1):
```bash
pip install -q -c /tmp/constraints.txt -r requirements.txt        # constraints = Kaggle's own torch==... pin
python prepare_data.py --recipe configs/data_stage1.yaml           # ~415k train / 4.3k val rows, deterministic (seed 42)
torchrun --nproc_per_node 2 train.py --config configs/stage1_pilot.yaml --resume auto --set train.time_budget_hours=<12 - elapsed - 0.75>
python eval.py --model runs/stage1_ultrabert150m/final_model --suites gliclass intent typed \
  --max_per_dataset 500 --permutations 3 --calib_file data/stage1/val.jsonl --out results/stage1_ultrabert150m_quick.json
```

Stage-1 config (resolved): UltraBERT-150M, uni-encoder, mlp scorer, first pooling, focal α 0.75 γ −1, dropout 0.3;
max_length 1024, ≤50 labels; batch 8 × accum 4 × 2 GPUs = 64; encoder LR 1e-5 / heads 3e-5; AdamW (0.9, 0.98, 1e-6);
wd 0.1 / 0.01; linear schedule, 5% warmup; fp16; gradient checkpointing; save + eval every 500 steps, keep 2; 1 epoch.

**Report back from each session:**
- `runs/<stage>/STATUS.json`
- the last 20 log lines (they include `sec_per_step`, `eta_hours`, `peak_mem_gb`, `loss`, `grad_norm`)
- `data/<stage>/manifest.json` (S1 only)
- `results/*_quick.json` (when complete)
- any NaN stop or OOM

**Stop conditions:** a NaN stop (STATUS `nan_stop`); OOM (rerun with `EXTRA_SET=["train.per_device_train_batch_size=4","train.gradient_accumulation_steps=8"]`); or projected time > 2 sessions.
**GPU-hours total (EST):** S0a+S0b about 0.7 h, plus S1 4-7 h, plus the optional S1-ctl 3-6 h, so about 8-14 h of the 30 h/week quota.
**Output size:** at most about 6 GB (two 1.8 GB checkpoints, the older one deleted at session end, 0.6 GB final model, about 0.5 GB data), under the 20 GB limit.
If the Handler picks **TPU** instead, do not run this spec: the TPU launch path is not written or tested yet (see PLAN.md §6).

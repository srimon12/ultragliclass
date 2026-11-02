#!/usr/bin/env bash
# CPU smoke test, end to end (what was run on 22 Oct 2026; ~15 min on 8 CPU cores).
# Usage: bash scripts/smoke_cpu.sh   (from the project root, after `pip install -e ".[train]"` with CPU torch)
set -euo pipefail
mkdir -p smoke_logs
O="train.max_steps=6 train.save_steps=3 train.eval_steps=3 train.logging_steps=1"

echo "[1] data: stream a tiny stage-0 mix";  python scripts/prepare_data.py --recipe configs/data_stage0.yaml --streaming
echo "[2] GLiClass init keeps UltraBERT weights"; python scripts/check_init.py
echo "[3] no unused parameters (DDP)";        python scripts/check_unused_params.py
echo "[4] train 6 steps";                     rm -rf runs/stage0 && python scripts/train.py --config configs/stage0_smoke.yaml --set $O
echo "[5] resume from step 3";                rm -rf runs/stage0_resume && cp -r runs/stage0 runs/stage0_resume && \
  rm -rf runs/stage0_resume/checkpoint-6 runs/stage0_resume/final_model && \
  python scripts/train.py --config configs/stage0_smoke.yaml --set $O train.output_dir=runs/stage0_resume
echo "[6] time budget stop + resume";         rm -rf runs/budget && B="train.max_steps=20 train.save_steps=10 train.output_dir=runs/budget" && \
  python scripts/train.py --config configs/stage0_smoke.yaml --set $B train.time_budget_hours=0.005 && \
  python scripts/train.py --config configs/stage0_smoke.yaml --set $B
echo "[7] DDP, 2 processes (gloo)";           rm -rf runs/ddp && OMP_NUM_THREADS=4 torchrun --nproc_per_node 2 scripts/train.py \
  --config configs/stage0_smoke.yaml --set train.max_steps=4 train.save_steps=2 train.eval_steps=2 train.output_dir=runs/ddp
echo "[8] eval, tiny subsets";                python scripts/eval.py --model runs/stage0/final_model --suites gliclass intent typed \
  --max_per_dataset 24 --typed_max_cases 6 --calib_file data/stage0/val.jsonl --calib_n 32 --out results/smoke_stage0.json
echo "SMOKE OK"

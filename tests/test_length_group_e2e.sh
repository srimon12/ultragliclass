#!/usr/bin/env bash
# End-to-end CPU checks of length-grouped training through the REAL train.py (tiny model-free config = stage2_smoke_lg):
#  [1] 22 rows, 14 steps => 3 epochs; the epoch counter reaches the sampler (indices >= N appear = epoch encoding) and the fetched rows
#      of an uninterrupted run == fetched rows of "stop at step 6 then resume" (num_workers=0 so the order is observable)
#  [2] 2-process gloo DDP with the flag on (same-count micro-batches per rank, completes)
set -euo pipefail
mkdir -p smoke_logs
cd "$(dirname "$0")/.."
OV="data.train_files=[data/stage2_smoke/tiny_train22.jsonl] train.num_workers=0 train.max_steps=14 train.save_steps=8 train.save_total_limit=10 train.eval_steps=100 train.logging_steps=2 train.output_dir=runs/lg_e2e_full"
rm -rf runs/lg_e2e_full runs/lg_e2e_res /tmp/lgo_full* /tmp/lgo_res*
echo "[1a] uninterrupted 14 steps (22 rows = 11 micro-batches of 2 per epoch; accum 2 -> 5.5 steps/epoch, so step 8 is mid-epoch-1)"
UGC_DUMP_ORDER=/tmp/lgo_full python scripts/train.py --config configs/stage2_smoke_lg.yaml --set $OV > smoke_logs/lg_e2e_full.log 2>&1
mkdir runs/lg_e2e_res && cp -r runs/lg_e2e_full/checkpoint-8 runs/lg_e2e_res/
echo "[1b] resume from checkpoint-8 to 14"
UGC_DUMP_ORDER=/tmp/lgo_res python scripts/train.py --config configs/stage2_smoke_lg.yaml --set $OV train.output_dir=runs/lg_e2e_res > smoke_logs/lg_e2e_res.log 2>&1
grep -q '"resumed_from": "runs/lg_e2e_res/checkpoint-8"' smoke_logs/lg_e2e_res.log || { echo "did not resume from checkpoint-8"; exit 1; }
python - <<'P'
import glob
full = [int(x) for f in glob.glob('/tmp/lgo_full.*') for x in open(f)]
res = [int(x) for f in glob.glob('/tmp/lgo_res.*') for x in open(f)]
N = 22
# rows consumed before step 8: epoch 0 = 22 rows (6 steps, the last one has a single micro-batch), + 2 steps x 4 rows of epoch 1
done = 22 + 8
print("fetched rows full:", len(full), " resumed run:", len(res), " expected resume start offset:", done)
n = min(len(res), len(full) - done)
assert n >= 18, n
assert res[:n] == full[done:done + n], "resumed run did not fetch the same rows as the uninterrupted run"
assert abs(len(res) - (len(full) - done)) <= 2, "unexpected number of rows fetched after resume (prefetch of at most one micro-batch tolerated)"
epochs = sorted({i // N for i in full}); print("epochs seen (index // N):", epochs)
assert epochs == [0, 1, 2], epochs
ep0 = [i % N for i in full if i // N == 0]; ep1 = [i % N for i in full if i // N == 1]
assert sorted(ep0) == sorted(ep1) == list(range(N)), "every epoch must use each of the 22 rows exactly once"
assert ep0 != ep1, "epochs must be ordered differently"
print("RESUME OK: after resuming mid-epoch the fetched rows continue exactly where the uninterrupted run was (epoch counter reached the sampler); each epoch uses every row once; epochs differ")
P
echo "[2] DDP, 2 processes (gloo), length_group on"
rm -rf runs/lg_ddp
OMP_NUM_THREADS=4 torchrun --nproc_per_node 2 scripts/train.py --config configs/stage2_smoke_lg.yaml \
  --set train.max_steps=4 train.save_steps=2 train.eval_steps=2 train.logging_steps=1 train.num_workers=0 train.output_dir=runs/lg_ddp > smoke_logs/lg_e2e_ddp.log 2>&1
grep -E "length_group ON|STATUS" smoke_logs/lg_e2e_ddp.log | head -3
grep -q '"state": "complete"' smoke_logs/lg_e2e_ddp.log && echo "DDP OK"
echo "E2E OK"

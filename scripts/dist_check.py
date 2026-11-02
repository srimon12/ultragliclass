"""Pre-flight for multi-GPU: init the process group, all_reduce and barrier, with a short timeout.
Run with: torchrun --nproc_per_node 2 scripts/dist_check.py   (the notebook runs it under a 180 s kill timer)."""
import datetime
import os
import time

import torch
import torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
tag = lambda: f"[dist_check {time.strftime('%H:%M:%S')} rank{rank}/{world}]"  # noqa: E731
cuda = torch.cuda.is_available()
backend = "nccl" if cuda else "gloo"
kw = {}
if cuda:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    kw["device_id"] = torch.device("cuda", int(os.environ["LOCAL_RANK"]))  # eager NCCL init: hangs show up here
print(f"{tag()} init_process_group({backend}) ...", flush=True)
t0 = time.time()
dist.init_process_group(backend, timeout=datetime.timedelta(seconds=90), **kw)
print(f"{tag()} init done in {time.time() - t0:.1f}s", flush=True)
x = torch.ones(1024, device="cuda" if cuda else "cpu") * (rank + 1)
dist.all_reduce(x)
if cuda:
    torch.cuda.synchronize()
print(f"{tag()} all_reduce ok: {x[0].item()} (expect {world * (world + 1) / 2})", flush=True)
dist.barrier(**({"device_ids": [torch.cuda.current_device()]} if cuda else {}))
print(f"{tag()} barrier ok; DIST_CHECK_OK", flush=True)
dist.destroy_process_group()

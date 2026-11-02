"""Builds kaggle_stage1.ipynb (v2, 22 Oct 2026: S0b-hang hardening) and kaggle_stage2_pilot.ipynb (v3, 23 Oct 2026).
Run: python scripts/build_notebook.py   (from the project root; cell sources of the pilot notebook: scripts/pilot_cells.py)"""
import os
import sys

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# UltraGliClass: one resumable training session

GLiClass uni-encoder on `dmis-lab/UltraBERT-150M`. **One run of this notebook = one Kaggle session.**
It restores the previous session's output, trains until the time budget, saves, and stops. Re-run it with the
previous version's output attached until `STATUS.json` says `complete`.

**Inputs:** the code dataset (`ultragliclass-code`), plus from session 2 on the output of the previous version.
**Settings:** GPU T4 x2, Internet on.
Everything is logged live (unbuffered) and also to `/kaggle/working/logs/`. A watchdog kills a run that prints
nothing for `STALL_MINUTES`, so the session always ends and saves its output and logs. It then falls back to 1 GPU.
"""),
code("""# ---- knobs ----
STAGE_CONFIG  = "configs/stage1_pilot.yaml"   # stage 0: configs/stage0_kaggle.yaml; control: configs/stage1_control_modernbert.yaml
DATA_RECIPE   = "configs/data_stage1.yaml"    # stage 0: configs/data_stage0_kaggle.yaml
DATA_DIR      = "data/stage1"                 # stage 0: data/stage0k
PREP_ARGS     = ""                            # stage 0: "--streaming"
CODE_INPUT    = "auto"     # auto-detect under /kaggle/input (e.g. /kaggle/input/datasets/<user>/ultragliclass-code)
PREV_OUTPUT   = None       # None = fresh start; "auto" = find the attached previous output; or an explicit path
USE_SINGLE_GPU = False     # True = never launch 2 processes (python -u train.py on GPU 0, grad-accum doubled)
AUTO_FALLBACK_SINGLE_GPU = True   # if the 2-GPU run stalls or crashes, resume on 1 GPU in the same session
DIST_CHECK    = True       # 20-s NCCL pre-flight (init + all_reduce + barrier) before the 2-GPU launch
NCCL_P2P_DISABLE = False   # set True to start with NCCL P2P off (auto-retried if the pre-flight hangs)
STALL_MINUTES = 15         # watchdog: no output for this long => kill the run (step logs come every ~30 s)
SESSION_HOURS = 12.0
RESERVE_HOURS = 0.75
QUICK_EVAL    = True
EXTRA_SET     = []         # dotted overrides, e.g. ["train.time_budget_hours=0.07"]
import os, time, glob, json, sys
T0 = time.time()
LOG_DIR = "/kaggle/working/logs"; os.makedirs(LOG_DIR, exist_ok=True)
NB_LOG = open(os.path.join(LOG_DIR, f"notebook_{time.strftime('%Y%m%d_%H%M%S')}.log"), "a", buffering=1)
def say(*a):
    line = f"[nb {time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.1f}m] " + " ".join(str(x) for x in a)
    print(line, flush=True); NB_LOG.write(line + "\\n")

def find_code_input():
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.py", recursive=True)]
    # the code dataset has no runs/ dir; previous-session outputs do (they also contain a copy of the code)
    hits = [h for h in hits if not os.path.isdir(os.path.join(h, "runs"))] or hits
    hits.sort(key=lambda p: (0 if "ultragliclass-code" in p else 1, len(p)))
    say("code candidates:", hits)
    return hits[0] if hits else None

def find_prev_output():
    hits = sorted({p.split("/runs/")[0] for p in glob.glob("/kaggle/input/**/runs/*/STATUS.json", recursive=True)}, key=len)
    say("previous-output candidates:", hits)
    return hits[0] if hits else None

if CODE_INPUT == "auto":
    CODE_INPUT = find_code_input()
if PREV_OUTPUT == "auto":
    PREV_OUTPUT = find_prev_output()
say("CODE_INPUT =", CODE_INPUT, "| PREV_OUTPUT =", PREV_OUTPUT, "| USE_SINGLE_GPU =", USE_SINGLE_GPU)
assert CODE_INPUT and os.path.exists(os.path.join(CODE_INPUT, "train.py")), "code dataset not found under /kaggle/input"
"""),
code("""import shutil, subprocess
WORK = "/kaggle/working/ultragliclass"
if not os.path.exists(os.path.join(WORK, "train.py")):
    shutil.copytree(CODE_INPUT, WORK, dirs_exist_ok=True, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns("runs", "data", "smoke_logs", "results", "__pycache__"))
subprocess.run(f"chmod -R u+w {WORK}", shell=True)  # copytree copies read-only dir modes from /kaggle/input
os.chdir(WORK)
say("work dir", WORK)
torch_pin = subprocess.run("pip freeze | grep -E '^torch=='", shell=True, capture_output=True, text=True).stdout.strip()
open("/tmp/constraints.txt", "w").write(torch_pin + "\\n")
say("torch pin:", torch_pin)
subprocess.run("pip install -q -c /tmp/constraints.txt -r requirements.txt", shell=True, check=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
import torch, transformers
NGPU = torch.cuda.device_count()
say("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", NGPU, "transformers", transformers.__version__)"""),
code("""# Restore the previous output. Runs HERE, in the single notebook process, BEFORE torchrun: no rank ever touches
# /kaggle/input. Plain file copies (no read-only mode bits from the input mount), then chmod u+w, then verify sizes.
def restore(src_root):
    for sub in ("runs", "data", "results"):
        src = os.path.join(src_root, sub)
        if not os.path.isdir(src):
            say("restore: no", sub, "in", src_root); continue
        n = b = 0; t = time.time()
        for dirpath, _, files in os.walk(src):
            rel = os.path.relpath(dirpath, src_root)
            os.makedirs(rel, exist_ok=True)
            for f in files:
                s, d = os.path.join(dirpath, f), os.path.join(rel, f)
                size = os.path.getsize(s)
                if not (os.path.exists(d) and os.path.getsize(d) == size):
                    shutil.copyfile(s, d)
                assert os.path.getsize(d) == size, f"size mismatch {d}"
                n += 1; b += size
                if size > 50e6: say(f"restore: copied {d} ({size / 2**20:.0f} MB)")
        say(f"restore: {sub}: {n} files, {b / 2**30:.2f} GB in {time.time() - t:.0f}s")
    subprocess.run("chmod -R u+w runs data results 2>/dev/null; true", shell=True)

if PREV_OUTPUT:
    say("restore from", PREV_OUTPUT)
    restore(PREV_OUTPUT)
for st in glob.glob("runs/*/STATUS.json"):
    s = json.load(open(st)); say("found", st, s.get("state"), "step", s.get("global_step"), "last", s.get("last_checkpoint"))
for c in sorted(glob.glob("runs/*/checkpoint-*")):
    say("checkpoint", c, sorted(os.listdir(c)))"""),
code("""if not os.path.exists(os.path.join(DATA_DIR, "train.jsonl")):
    say("preparing data:", DATA_RECIPE, PREP_ARGS)
    subprocess.run(f"python -u prepare_data.py --recipe {DATA_RECIPE} {PREP_ARGS}", shell=True, check=True)
else:
    say("data already present:", DATA_DIR)
print(open(os.path.join(DATA_DIR, "manifest.json")).read()[:1200])"""),
code("""# Launch with live, unbuffered streaming to this notebook AND to /kaggle/working/logs/, a no-output watchdog,
# a NCCL pre-flight, and a single-GPU fallback.
import signal, threading, queue
sys.path.insert(0, WORK)
from ugc.config import load_config
CFG = load_config(STAGE_CONFIG, EXTRA_SET)
OUT_DIR = CFG["train"]["output_dir"]
BASE_ACCUM = int(CFG["train"].get("gradient_accumulation_steps", 1))
for _kv in EXTRA_SET:  # an EXTRA_SET override wins over the yaml value
    if _kv.startswith("train.gradient_accumulation_steps="): BASE_ACCUM = int(_kv.split("=", 1)[1])

def run_streamed(cmd, env_extra, log_name, stall_min):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
           "TORCH_NCCL_DUMP_ON_TIMEOUT": "1", "NCCL_DEBUG": "WARN", **env_extra}
    path = os.path.join(LOG_DIR, f"{log_name}_{time.strftime('%H%M%S')}.log")
    say("RUN", cmd, "| env", env_extra, "| log", path)
    p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, start_new_session=True)
    q = queue.Queue()
    threading.Thread(target=lambda: ([q.put(l) for l in p.stdout], q.put(None)), daemon=True).start()
    last, stalled = time.time(), False
    with open(path, "a", buffering=1) as f:
        while True:
            try:
                line = q.get(timeout=10)
            except queue.Empty:
                if time.time() - last > stall_min * 60:
                    say(f"WATCHDOG: no output for {stall_min} min -> killing process group"); stalled = True
                    os.killpg(p.pid, signal.SIGTERM); time.sleep(30)
                    if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
                    break
                continue
            if line is None: break
            last = time.time(); print(line, end="", flush=True); f.write(line)
    rc = p.wait()
    say("EXIT", rc, "stalled" if stalled else "")
    return rc, stalled

def budget():
    return max(0.25, SESSION_HOURS - (time.time() - T0) / 3600 - RESERVE_HOURS)

def train_cmd(single):
    sets = " ".join([f"train.time_budget_hours={budget():.2f}"] + EXTRA_SET +
                    ([f"train.gradient_accumulation_steps={BASE_ACCUM * max(NGPU, 1)}"] if single and NGPU > 1 else []))
    launcher = "python -u" if single else f"torchrun --standalone --nproc_per_node {NGPU}"
    return f"{launcher} train.py --config {STAGE_CONFIG} --resume auto --set {sets}"

single = USE_SINGLE_GPU or NGPU < 2
nccl_env = {"NCCL_P2P_DISABLE": "1"} if NCCL_P2P_DISABLE else {}
if not single and DIST_CHECK:
    rc, st = run_streamed(f"torchrun --standalone --nproc_per_node {NGPU} scripts/dist_check.py", nccl_env, "dist_check", 3)
    if rc != 0 and not nccl_env:
        say("pre-flight failed: retrying with NCCL_P2P_DISABLE=1"); nccl_env = {"NCCL_P2P_DISABLE": "1"}
        rc, st = run_streamed(f"torchrun --standalone --nproc_per_node {NGPU} scripts/dist_check.py", nccl_env, "dist_check_p2poff", 3)
    if rc != 0:
        say("pre-flight failed twice: using 1 GPU"); single = True

rc, stalled = run_streamed(train_cmd(single), ({"CUDA_VISIBLE_DEVICES": "0"} if single else nccl_env),
                           "train_1gpu" if single else "train_ddp", STALL_MINUTES)
status_now = json.load(open(os.path.join(OUT_DIR, "STATUS.json"))) if os.path.exists(os.path.join(OUT_DIR, "STATUS.json")) else {}
if (rc != 0 or stalled) and not single and AUTO_FALLBACK_SINGLE_GPU and status_now.get("state") != "complete":
    say("2-GPU run failed/stalled -> resuming on 1 GPU from the newest checkpoint")
    rc, stalled = run_streamed(train_cmd(True), {"CUDA_VISIBLE_DEVICES": "0"}, "train_1gpu_fallback", STALL_MINUTES)"""),
code("""hb = os.path.join(OUT_DIR, "HEARTBEAT.json")
if os.path.exists(hb): say("last heartbeat:", open(hb).read()[:400])
status = json.load(open(os.path.join(OUT_DIR, "STATUS.json")))
print(json.dumps(status, indent=2))
if status["state"] == "needs_resume":
    say("NOT DONE: save this version, then rerun with PREV_OUTPUT = 'auto' and this version's output attached.")
elif status["state"] == "running":
    say("Run died without writing a final status (crash/stall): see /kaggle/working/logs/.")
elif status["state"] == "nan_stop":
    say("STOPPED ON NaN/inf loss: lower LR or use fp32 for the first steps.")"""),
code("""if QUICK_EVAL and status["state"] == "complete":
    run_streamed(f"python -u eval.py --model {status['final_model']} --suites gliclass intent typed "
                 f"--max_per_dataset 500 --permutations 3 --calib_file {DATA_DIR}/val.jsonl "
                 f"--out results/{os.path.basename(OUT_DIR)}_quick.json", {}, "quick_eval", 30)"""),
code("""ckpts = sorted(glob.glob(os.path.join(OUT_DIR, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
for c in ckpts[:-1]:
    shutil.rmtree(c)
subprocess.run("du -sh runs data results /kaggle/working/logs 2>/dev/null", shell=True)
say("session hours used:", round((time.time() - T0) / 3600, 2)); NB_LOG.close()"""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}}
nb["cells"] = cells
nb["metadata"] = meta
nbf.write(nb, "kaggle_stage1.ipynb")
print("wrote kaggle_stage1.ipynb")

# ---- v3 (23 Oct 2026): stage-2 pilot notebook: INIT_MODEL from the S1 kernel output, arms A/B/C in sequence, quick eval per arm ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_cells as pc  # noqa: E402
pnb = nbf.v4.new_notebook()
pnb["cells"] = [md(pc.MD_INTRO), code(pc.KNOBS), code(pc.SETUP), code(pc.INIT), cells[3], code(pc.DATA),
                code(pc.ARMS_RUN), code(pc.REPORT), code(pc.CLEAN)]
pnb["metadata"] = meta
nbf.write(pnb, "kaggle_stage2_pilot.ipynb")
print("wrote kaggle_stage2_pilot.ipynb")

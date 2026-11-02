"""Cell sources of kaggle_stage2_pilot.ipynb (imported by scripts/build_notebook.py). Raw strings: what you read is what runs."""

MD_INTRO = r"""# UltraGliClass stage 2 PILOTS (A / B / C) in one notebook

Starts from the **stage-1 model** (attached as an input: the output of kernel `ultragliclass-s1`) and trains up to three arms
for 1,500 steps each on the SAME data (`data/stage2_pilot`, built once) with the SAME seed and schedule:

| arm | what it adds to "S1 continued" |
|---|---|
| **A** | nothing (control) |
| **B** | many-label expansion (20-150 labels, hard negatives) |
| **C** | B + permutation-consistency term (paired label orders, symmetric KL) |

After each arm it runs the quick eval (14 GLiClass sets + intent + typed decisions, 500 per set, 3 label orders, global-T
calibration on `calib/stage1_val.jsonl`) and writes `results/stage2_pilot_<arm>_quick.json`.
**Inputs:** the code dataset (`ultragliclass-code`, v3 or later) and the S1 kernel output. To resume an unfinished pilot session,
also attach THAT version's output and set `PREV_OUTPUT = "auto"`. **Settings:** GPU T4 x2, Internet on.
All v2 safety features are kept: live unbuffered logs in `/kaggle/working/logs/`, no-output watchdog, NCCL pre-flight, 10-min
collective timeout, single-GPU fallback, restore of previous output BEFORE torchrun (no rank touches `/kaggle/input`).
Stops (and reports) on: NaN loss, OOM, an arm that needs more than `ARM_MAX_HOURS`, or `TOTAL_MAX_HOURS` of session time.
"""

KNOBS = r"""# ---- knobs ----
ARMS          = ["A", "B", "C"]   # arms to run, in this order; e.g. ["B"] to run one arm only
INIT_MODEL    = "auto"     # S1 final_model dir, OR the S1 kernel-output root, OR "auto" = search /kaggle/input for runs/stage1_ultrabert150m/final_model
DATA_RECIPE   = "configs/data_stage2_pilot.yaml"
DATA_DIR      = "data/stage2_pilot"
PREP_ARGS     = ""                              # smoke tests: "--streaming"
CALIB_FILE    = "calib/stage1_val.jsonl"       # = data/stage1/val.jsonl used by the S1 and ModernBERT evals (same global-T fit)
CODE_INPUT    = "auto"     # auto-detect the code dataset under /kaggle/input
PREV_OUTPUT   = None       # None = fresh; "auto" = previous version's output of THIS pilot (runs/stage2_pilot_*), never the S1 output
USE_SINGLE_GPU = False     # True = never launch 2 processes (python -u train.py on GPU 0, grad-accum doubled)
AUTO_FALLBACK_SINGLE_GPU = True   # 2-GPU run stalls/crashes (not OOM) => resume on 1 GPU in the same session
DIST_CHECK    = True       # 20-s NCCL pre-flight before the first 2-GPU launch
NCCL_P2P_DISABLE = False   # True = start with NCCL P2P off (auto-retried if the pre-flight hangs)
STALL_MINUTES = 15         # watchdog: no output for this long => kill the run
SESSION_HOURS = 12.0
RESERVE_HOURS = 0.5
ARM_MAX_HOURS = 2.0        # STOP: train budget per arm (train.py saves and exits; the notebook then stops and reports)
TOTAL_MAX_HOURS = 6.0      # STOP: no new arm starts if the session clock would pass this (GPU-hours = session hours)
EVAL_RESERVE_HOURS = 0.2   # kept free for the quick eval after an arm
QUICK_EVAL    = True
EVAL_MAX_PER_DATASET = 500
EVAL_PERMS    = 3
EVAL_CALIB_N  = 2000             # rows of CALIB_FILE used to fit the global temperature (S1: 2000)
EVAL_TYPED_MAX_CASES = None      # None = all 400 cases / 2,000 decisions (as in S1)
DROP_CKPT_AFTER_ARM = True       # delete checkpoint-* of a COMPLETED arm (final_model is kept): saves about 1.8 GB per arm
EXTRA_SET     = []         # dotted overrides for every arm, e.g. ["train.max_steps=20"]
ARM_SET       = {}         # per-arm overrides, e.g. {"C": ["data.consistency.pair_prob=0.1"]}
import os, time, glob, json, sys, shutil, subprocess
T0 = time.time()
LOG_DIR = "/kaggle/working/logs"; os.makedirs(LOG_DIR, exist_ok=True)
NB_LOG = open(os.path.join(LOG_DIR, f"notebook_{time.strftime('%Y%m%d_%H%M%S')}.log"), "a", buffering=1)
def say(*a):
    line = f"[nb {time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.1f}m] " + " ".join(str(x) for x in a)
    print(line, flush=True); NB_LOG.write(line + "\n")

def find_code_input():
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.py", recursive=True)]
    hits = [h for h in hits if os.path.exists(os.path.join(h, "configs", "stage2_pilot_A.yaml"))]   # needs the v3 code
    hits = [h for h in hits if not os.path.isdir(os.path.join(h, "runs"))] or hits
    hits.sort(key=lambda p: (0 if "ultragliclass-code" in p else 1, len(p)))
    say("code candidates:", hits)
    return hits[0] if hits else None

def find_prev_output():
    hits = sorted({p.split("/runs/")[0] for p in glob.glob("/kaggle/input/**/runs/stage2_pilot_*/STATUS.json", recursive=True)}, key=len)
    say("previous pilot-output candidates:", hits)
    return hits[0] if hits else None

def find_init_model():
    cands = []
    if INIT_MODEL != "auto":
        cands = [INIT_MODEL] if os.path.exists(os.path.join(INIT_MODEL, "config.json")) else \
            [os.path.dirname(p) for p in glob.glob(os.path.join(INIT_MODEL, "**", "final_model", "config.json"), recursive=True)]
    else:
        cands = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/runs/stage1_ultrabert150m/final_model/config.json", recursive=True)]
    cands.sort(key=len)
    say("S1 model candidates:", cands)
    return cands[0] if cands else None

if CODE_INPUT == "auto":
    CODE_INPUT = find_code_input()
if PREV_OUTPUT == "auto":
    PREV_OUTPUT = find_prev_output()
INIT_SRC = find_init_model()
say("CODE_INPUT =", CODE_INPUT, "| INIT_SRC =", INIT_SRC, "| PREV_OUTPUT =", PREV_OUTPUT, "| ARMS =", ARMS, "| USE_SINGLE_GPU =", USE_SINGLE_GPU)
assert CODE_INPUT and os.path.exists(os.path.join(CODE_INPUT, "configs", "stage2_pilot_A.yaml")), "v3 code dataset (with configs/stage2_pilot_*.yaml) not found under /kaggle/input"
assert INIT_SRC, "S1 model not found: attach the output of kernel ultragliclass-s1 (runs/stage1_ultrabert150m/final_model) or set INIT_MODEL"
"""

SETUP = r"""import shutil, subprocess
WORK = "/kaggle/working/ultragliclass"
if not os.path.exists(os.path.join(WORK, "train.py")):
    shutil.copytree(CODE_INPUT, WORK, dirs_exist_ok=True, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns("runs", "data", "smoke_logs", "results", "__pycache__"))
subprocess.run(f"chmod -R u+w {WORK}", shell=True)  # copytree copies read-only dir modes from /kaggle/input
os.chdir(WORK)
say("work dir", WORK)
torch_pin = subprocess.run("pip freeze | grep -E '^torch=='", shell=True, capture_output=True, text=True).stdout.strip()
open("/tmp/constraints.txt", "w").write(torch_pin + "\n")
say("torch pin:", torch_pin)
subprocess.run("pip install -q -c /tmp/constraints.txt -r requirements.txt", shell=True, check=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
import torch, transformers
NGPU = torch.cuda.device_count()
say("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", NGPU, "transformers", transformers.__version__)"""

INIT = r"""# The S1 model: validate it, then COPY it out of /kaggle/input (plain file copies, in this single process, before torchrun) so
# no rank ever reads the read-only input mount. INIT_LOCAL is what every arm gets as model.init_from.
INIT_LOCAL = "/tmp/ugc_init_model"
need = ["config.json", "tokenizer.json", "tokenizer_config.json"]
for f in need:
    assert os.path.exists(os.path.join(INIT_SRC, f)), f"S1 model dir {INIT_SRC} lacks {f}"
weights = [f for f in os.listdir(INIT_SRC) if f.endswith((".safetensors", ".bin"))]
assert weights, f"no weight file in {INIT_SRC}: {os.listdir(INIT_SRC)}"
size = sum(os.path.getsize(os.path.join(INIT_SRC, f)) for f in os.listdir(INIT_SRC))
assert size > 200e6, f"S1 model dir is only {size / 1e6:.0f} MB: incomplete"
cfg_s1 = json.load(open(os.path.join(INIT_SRC, "config.json")))
say("S1 model:", INIT_SRC, f"{size / 2**20:.0f} MB", weights, "| architectures", cfg_s1.get("architectures"), "| max_num_classes", cfg_s1.get("max_num_classes"))
shutil.rmtree(INIT_LOCAL, ignore_errors=True); os.makedirs(INIT_LOCAL)
for f in os.listdir(INIT_SRC):
    s = os.path.join(INIT_SRC, f)
    if os.path.isfile(s):
        shutil.copyfile(s, os.path.join(INIT_LOCAL, f))
        assert os.path.getsize(os.path.join(INIT_LOCAL, f)) == os.path.getsize(s), f"copy size mismatch {f}"
subprocess.run(f"chmod -R u+w {INIT_LOCAL}", shell=True)
say("S1 model copied to", INIT_LOCAL, sorted(os.listdir(INIT_LOCAL)))"""

DATA = r"""if not os.path.exists(os.path.join(DATA_DIR, "train.jsonl")):
    say("preparing data:", DATA_RECIPE, PREP_ARGS)
    subprocess.run(f"python -u prepare_data.py --recipe {DATA_RECIPE} {PREP_ARGS}", shell=True, check=True)
else:
    say("data already present (restored or built earlier):", DATA_DIR)
man = json.load(open(os.path.join(DATA_DIR, "manifest.json")))
os.makedirs("results", exist_ok=True)
shutil.copyfile(os.path.join(DATA_DIR, "manifest.json"), "results/stage2_pilot_data_manifest.json")
# sanity: every arm needs max_steps x 64 rows (one epoch) for one epoch; no eval-set / excluded source may have contributed rows
sys.path.insert(0, WORK)
from ugc.config import load_config as _lc
_t = _lc("configs/stage2_pilot_A.yaml", [f"model.init_from={INIT_LOCAL}"] + EXTRA_SET)["train"]
need_rows = int(_t["max_steps"]) * int(_t["per_device_train_batch_size"]) * int(_t.get("gradient_accumulation_steps", 1)) * \
    (NGPU if NGPU >= 2 and not USE_SINGLE_GPU else 1)      # = 1500 x 64 = 96,000 on the real run
used = {k: v for k, v in man["sources"].items() if v.get("train_rows", 0) > 0}
bad = [k for k in used if any(x in k.lower() for x in ("dbpedia", "bitext", "banking77", "clinc", "massive", "snips", "phrasebank", "sst5"))]
say("train rows", man["train_rows"], "val rows", man["val_rows"], "pools", man.get("taxonomy", {}).get("pool_sizes"), "labels/row", man["labels_per_row"])
say("sources used:", {k: v["train_rows"] for k, v in used.items()})
assert not bad, f"excluded / eval-set source in the data: {bad}"
assert man["train_rows"] >= need_rows, f"only {man['train_rows']} train rows, need {need_rows}"
print(json.dumps({k: v.get("license") for k, v in used.items()}, indent=1))"""

ARMS_RUN = r"""# Launch with live, unbuffered streaming to this notebook AND to /kaggle/working/logs/, a no-output watchdog, a NCCL pre-flight,
# a single-GPU fallback, then the quick eval, one arm after the other.
import signal, threading, queue, re
sys.path.insert(0, WORK)
from ugc.config import load_config
LAST_OOM = False

def run_streamed(cmd, env_extra, log_name, stall_min):
    global LAST_OOM
    LAST_OOM = False
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
            if re.search(r"OutOfMemoryError|CUDA out of memory", line): LAST_OOM = True
            last = time.time(); print(line, end="", flush=True); f.write(line)
    rc = p.wait()
    say("EXIT", rc, "stalled" if stalled else "", "OOM" if LAST_OOM else "")
    return rc, stalled

def elapsed_h():
    return (time.time() - T0) / 3600

def arm_budget_h():
    return min(ARM_MAX_HOURS, TOTAL_MAX_HOURS - elapsed_h() - EVAL_RESERVE_HOURS, SESSION_HOURS - elapsed_h() - RESERVE_HOURS)

def arm_sets(arm):
    return [f"model.init_from={INIT_LOCAL}"] + EXTRA_SET + ARM_SET.get(arm, [])

def train_cmd(arm, single):
    sets = [f"train.time_budget_hours={max(0.1, arm_budget_h()):.2f}"] + arm_sets(arm)
    accum = int(load_config(f"configs/stage2_pilot_{arm}.yaml", arm_sets(arm))["train"].get("gradient_accumulation_steps", 1))
    if single and NGPU > 1:
        sets.append(f"train.gradient_accumulation_steps={accum * NGPU}")
    launcher = "python -u" if single else f"torchrun --standalone --nproc_per_node {NGPU}"
    return f"{launcher} train.py --config configs/stage2_pilot_{arm}.yaml --resume auto --set {' '.join(sets)}"

single = USE_SINGLE_GPU or NGPU < 2
nccl_env = {"NCCL_P2P_DISABLE": "1"} if NCCL_P2P_DISABLE else {}
if not single and DIST_CHECK:
    rc, st = run_streamed(f"torchrun --standalone --nproc_per_node {NGPU} scripts/dist_check.py", nccl_env, "dist_check", 3)
    if rc != 0 and not nccl_env:
        say("pre-flight failed: retrying with NCCL_P2P_DISABLE=1"); nccl_env = {"NCCL_P2P_DISABLE": "1"}
        rc, st = run_streamed(f"torchrun --standalone --nproc_per_node {NGPU} scripts/dist_check.py", nccl_env, "dist_check_p2poff", 3)
    if rc != 0:
        say("pre-flight failed twice: using 1 GPU"); single = True

def read_status(out_dir):
    p = os.path.join(out_dir, "STATUS.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def run_arm(arm):
    global single
    cfg = load_config(f"configs/stage2_pilot_{arm}.yaml", arm_sets(arm))
    out_dir = cfg["train"]["output_dir"]
    res_path = f"results/stage2_pilot_{arm}_quick.json"
    rec = {"arm": arm, "out_dir": out_dir, "state": None, "t0_hours": round(elapsed_h(), 3)}
    st = read_status(out_dir)
    if st.get("state") == "complete" and os.path.isdir(st.get("final_model", "")):
        say(f"ARM {arm}: already complete (restored from PREV_OUTPUT): skipping training")
    else:
        if arm_budget_h() < 0.4:
            rec["state"] = "not_started_budget"; say(f"ARM {arm}: NOT STARTED, only {arm_budget_h():.2f} h of budget left"); return rec
        say(f"===== ARM {arm}: {cfg['train']['max_steps']} steps, budget {arm_budget_h():.2f} h, many_label={cfg['data']['many_label']['enabled']}, "
            f"consistency={cfg['data']['consistency']['enabled']} =====")
        rc, stalled = run_streamed(train_cmd(arm, single), ({"CUDA_VISIBLE_DEVICES": "0"} if single else nccl_env),
                                   f"train_{arm}_{'1gpu' if single else 'ddp'}", STALL_MINUTES)
        st = read_status(out_dir)
        if (rc != 0 or stalled) and not LAST_OOM and not single and AUTO_FALLBACK_SINGLE_GPU and st.get("state") != "complete":
            say(f"ARM {arm}: 2-GPU run failed/stalled -> resuming on 1 GPU from the newest checkpoint")
            single = True
            rc, stalled = run_streamed(train_cmd(arm, True), {"CUDA_VISIBLE_DEVICES": "0"}, f"train_{arm}_1gpu_fallback", STALL_MINUTES)
            st = read_status(out_dir)
        rec["oom"] = bool(LAST_OOM)
    hb = os.path.join(out_dir, "HEARTBEAT.json")
    if os.path.exists(hb): say("last heartbeat:", open(hb).read()[:300])
    rec["state"] = st.get("state", "no_status")
    rec["status"] = st
    print(json.dumps(st, indent=2))
    if rec["state"] != "complete":
        say(f"ARM {arm}: state={rec['state']} -> STOP (needs_resume = used the {ARM_MAX_HOURS} h arm budget; nan_stop = NaN loss; running = crash/OOM/stall)")
        return rec
    if QUICK_EVAL and not os.path.exists(res_path):
        cmd = (f"python -u eval.py --model {st['final_model']} --suites gliclass intent typed --max_per_dataset {EVAL_MAX_PER_DATASET} "
               f"--permutations {EVAL_PERMS} --calib_file {CALIB_FILE} --calib_n {EVAL_CALIB_N} --out {res_path}" +
               (f" --typed_max_cases {EVAL_TYPED_MAX_CASES}" if EVAL_TYPED_MAX_CASES else ""))
        rec["eval_rc"], _ = run_streamed(cmd, {}, f"quick_eval_{arm}", 30)
    if os.path.exists(res_path):
        r = json.load(open(res_path)); d = r["datasets"]
        rec["eval"] = {"avg_macro_f1": r.get("gliclass_avg_macro_f1"), "global_T": r.get("global_T"),
                       "typed_acc": (r.get("typed_decisions") or {}).get("T1", {}).get("accuracy"),
                       "flip_many_label": {k: d[k]["perm_flip_rate"] for k in ("banking77", "clinc150", "massive", "SetFit/20_newsgroups") if k in d}}
    if DROP_CKPT_AFTER_ARM:
        for c in glob.glob(os.path.join(out_dir, "checkpoint-*")):
            shutil.rmtree(c, ignore_errors=True)
    rec["wall_hours"] = round(elapsed_h() - rec["t0_hours"], 3)
    return rec

SUMMARY = []
def write_summary():
    json.dump({"session_hours": round(elapsed_h(), 3), "arms": SUMMARY, "init_model_src": INIT_SRC}, open("results/stage2_pilot_summary.json", "w"), indent=2)

for arm in ARMS:
    if elapsed_h() > TOTAL_MAX_HOURS - 0.4:
        say(f"STOP: session clock {elapsed_h():.2f} h is past the {TOTAL_MAX_HOURS} h total cap; arm {arm} not started"); break
    rec = run_arm(arm); SUMMARY.append(rec); write_summary()
    say(f"ARM {arm} done: state={rec['state']} wall={rec.get('wall_hours')} h eval={rec.get('eval')}")
    if rec["state"] != "complete":
        break"""

REPORT = r"""print(json.dumps({"session_hours": round(elapsed_h(), 2), "arms": [{k: v for k, v in r.items() if k != "status"} for r in SUMMARY]}, indent=2))
for r in SUMMARY:
    s = r.get("status", {})
    say(f"ARM {r['arm']}: {r['state']} step {s.get('global_step')}/{s.get('max_steps')} train_loss {s.get('train_loss')} "
        f"final_eval_loss {(s.get('final_eval') or {}).get('eval_loss')} peak_mem {(s.get('final_eval') or {}).get('peak_mem_gb')} avg_F1 {(r.get('eval') or {}).get('avg_macro_f1')}")
if any(r["state"] != "complete" for r in SUMMARY) or len(SUMMARY) < len(ARMS):
    say("NOT ALL ARMS COMPLETE: save this version; to continue attach its output, set PREV_OUTPUT='auto' and ARMS to the remaining arms (+ the unfinished one).")"""

CLEAN = r"""for out in sorted(glob.glob("runs/stage2_pilot_*")):          # unfinished arms keep their newest checkpoint only
    ck = sorted(glob.glob(os.path.join(out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    for c in ck[:-1]:
        shutil.rmtree(c)
subprocess.run("du -sh runs data results /kaggle/working/logs 2>/dev/null", shell=True)
say("session hours used:", round(elapsed_h(), 2)); NB_LOG.close()"""

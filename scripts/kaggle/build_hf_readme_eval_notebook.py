#!/usr/bin/env python
"""Builds kaggle_eval_readme.ipynb: the lean HF-README benchmark (scripts/hf_readme_eval.py) on Kaggle T4 x2.
Inputs: code dataset ultragliclass-code v9 + output of kernel ultragliclass-s3full (runs/stage3_full/final_model).
Peers download from the HF Hub (internet ON). See notes/HF_README_EVAL_JOB.md.   Run: python scripts/build_hf_readme_eval_notebook.py"""
import ast, hashlib
import nbformat as nbf

LJ_MD5 = hashlib.md5(open("calib/jev_long_test.jsonl", "rb").read()).hexdigest()
INTRO = """# UltraGliClass v0.1: HF-README benchmark (full test splits, 3 models, T4 x2)
14 GLiClass-v3 card sets (full splits, 1024, macro-F1) + Banking77 / CLINC-150 accuracy + long-Jev @ 8192, for
UltraGliClass v0.1 (S3 final), knowledgator/gliclass-modern-base-v3.0 and knowledgator/gliclass-base-v3.0.
Resumable per (model, dataset, label order); hard stop by `HARD_STOP_HOURS`. Outputs: `results/hf_readme/` (TABLES.md, summary.json,
per-cell logits npz, long-Jev JSONs). No texts in the outputs. Nothing is published. Spec: notes/HF_README_EVAL_JOB.md."""

KNOBS = f'''# ---- knobs ----
HARD_STOP_HOURS = 5.5        # session cap (workers stop claiming 10 min before; a running cell aborts 1 min before); GPU-h = session hours
PERMUTATIONS    = 3          # label orders: 0 = canonical (HEADLINE, runs first for every model), 1..2 = seeded shuffles (robustness)
MAX_LENGTH      = 1024       # suite + intent (= the official GLiClass pipeline default)
LONGJEV_MAXLEN  = 8192
LONGJEV_MD5     = "{LJ_MD5}"
PEER_ADD_PREFIX_SPACE = True # Knowledgator's test_gliclass.py setting for their models; ours stays False (as trained)
SMOKE_ROWS      = None       # e.g. 50 = quick pipeline test (NOT publishable); None = full test splits
BOOTSTRAP       = 1000
MODELS = {{   # name -> spec; "path": "auto" = v0.1 final_model found under /kaggle/input
    "ultragliclass-v0.1":        {{"path": "auto", "add_prefix_space": False, "batch": 16, "longjev_batch": 2}},
    "gliclass-modern-base-v3.0": {{"path": "knowledgator/gliclass-modern-base-v3.0", "add_prefix_space": PEER_ADD_PREFIX_SPACE, "batch": 16, "longjev_batch": 2}},
    "gliclass-base-v3.0":        {{"path": "knowledgator/gliclass-base-v3.0", "add_prefix_space": PEER_ADD_PREFIX_SPACE, "batch": 16, "longjev_batch": 1}},
}}
AFFINITY = ["ultragliclass-v0.1,gliclass-modern-base-v3.0,gliclass-base-v3.0",   # GPU0
            "gliclass-base-v3.0,gliclass-modern-base-v3.0,ultragliclass-v0.1"]   # GPU1 (DeBERTa = slowest, starts at once)
import os, time, glob, json, sys, shutil, subprocess, hashlib
T0 = time.time(); HARD_STOP_UNIX = T0 + HARD_STOP_HOURS * 3600
LOG_DIR = "/kaggle/working/logs"; os.makedirs(LOG_DIR, exist_ok=True)
NB_LOG = open(os.path.join(LOG_DIR, f"notebook_{{time.strftime('%Y%m%d_%H%M%S')}}.log"), "a", buffering=1)
def say(*a):
    line = f"[nb {{time.strftime('%H:%M:%S')}} +{{(time.time() - T0) / 60:.1f}}m] " + " ".join(str(x) for x in a)
    print(line, flush=True); NB_LOG.write(line + "\\n")
'''

SETUP = r'''# Inputs: code v9 (has scripts/hf_readme_eval.py) + v0.1 final_model from the ultragliclass-s3full output
code = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/scripts/hf_readme_eval.py", recursive=True)]
code = sorted([c for c in code if os.path.exists(os.path.join(c, "eval.py"))], key=lambda p: (0 if "ultragliclass-code" in p else 1, len(p)))
say("code candidates:", code)
assert code, "code dataset ultragliclass-code v9 (with scripts/hf_readme_eval.py) not attached"
v01 = sorted([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/runs/stage3_full/final_model/config.json", recursive=True)], key=len)
say("v0.1 model candidates:", v01)
assert v01, "v0.1 model not found: attach the output of kernel srimondanguria/ultragliclass-s3full (runs/stage3_full/final_model)"
assert sum(os.path.getsize(os.path.join(v01[0], f)) for f in os.listdir(v01[0])) > 200e6, "v0.1 final_model looks incomplete"
if MODELS["ultragliclass-v0.1"]["path"] == "auto":
    MODELS["ultragliclass-v0.1"]["path"] = v01[0]
WORK = "/kaggle/working/ultragliclass"
if not os.path.exists(os.path.join(WORK, "scripts", "hf_readme_eval.py")):
    shutil.copytree(code[0], WORK, dirs_exist_ok=True, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns("runs", "data", "smoke_logs", "results", "__pycache__"))
subprocess.run(f"chmod -R u+w {WORK}", shell=True)
os.chdir(WORK); say("work dir", WORK)
md5 = hashlib.md5(open("calib/jev_long_test.jsonl", "rb").read()).hexdigest()
assert md5 == LONGJEV_MD5, f"long-Jev slice md5 {md5} != {LONGJEV_MD5}"
torch_pin = subprocess.run("pip freeze | grep -E '^torch=='", shell=True, capture_output=True, text=True).stdout.strip()
open("/tmp/constraints.txt", "w").write(torch_pin + "\n")
subprocess.run("pip install -q -c /tmp/constraints.txt -r requirements.txt", shell=True, check=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
import torch; NGPU = torch.cuda.device_count(); say("torch", torch.__version__, "gpus", NGPU)
assert NGPU >= 1, "no GPU: set the accelerator to GPU T4 x2"
OUT = "results/hf_readme"; os.makedirs(OUT, exist_ok=True)
json.dump(MODELS, open("/tmp/hf_models.json", "w"), indent=1)
'''

PREP = r'''# Prep: parse every eval set once (texts -> /tmp only), pre-download peers (records the HF commit sha), write the job list
if not os.path.exists(os.path.join(OUT, "jobs.json")):
    t = time.time()
    cmd = (f"python -u scripts/hf_readme_eval.py prep --out {OUT} --models_json /tmp/hf_models.json --permutations {PERMUTATIONS} "
           f"--max_length {MAX_LENGTH} --longjev_max_length {LONGJEV_MAXLEN}" + (f" --max_rows {SMOKE_ROWS}" if SMOKE_ROWS else ""))
    say(cmd); subprocess.run(cmd + f" 2>&1 | tee {LOG_DIR}/prep.log", shell=True, check=True)
    say(f"prep {(time.time() - t) / 60:.1f} min")
say(open(os.path.join(OUT, "evalsets_manifest.json")).read()[:3000])
'''

RUN = r'''# Two workers (one per GPU) pull resumable jobs: tier 0 calib, 1 headline perm 0, 2 long-Jev, 3 perms 1..K-1
procs = []
for w in range(min(NGPU, 2)):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w), TOKENIZERS_PARALLELISM="false")
    lf = open(f"{LOG_DIR}/worker{w}.log", "a")
    cmd = (f"python -u scripts/hf_readme_eval.py worker --out {OUT} --worker {w} --affinity {AFFINITY[w]} "
           f"--hard_stop_unix {HARD_STOP_UNIX} --reserve_min 10 --max_length {MAX_LENGTH} --longjev_max_length {LONGJEV_MAXLEN}")
    procs.append(subprocess.Popen(cmd, shell=True, env=env, stdout=lf, stderr=subprocess.STDOUT)); say("started", cmd)
jobs = json.load(open(os.path.join(OUT, "jobs.json")))
def outp(j):
    if j["kind"] == "cls": return os.path.join(OUT, "raw", j["model"], f"{j['dataset'].replace('/', '__')}__p{j['perm']}.npz")
    if j["kind"] == "calib": return os.path.join(OUT, "raw", j["model"], "calib.json")
    return os.path.join(OUT, "longjev", f"{j['model']}.json")
last = 0
while any(p.poll() is None for p in procs):
    time.sleep(30)
    if time.time() > HARD_STOP_UNIX + 120:
        say("HARD STOP: killing workers"); [p.kill() for p in procs]; break
    if time.time() - last > 300:
        last = time.time(); done = [j for j in jobs if os.path.exists(outp(j))]
        by = {t: f"{sum(1 for j in done if j['tier'] == t)}/{sum(1 for j in jobs if j['tier'] == t)}" for t in range(4)}
        say(f"progress {len(done)}/{len(jobs)} by tier {by}", subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
            shell=True, capture_output=True, text=True).stdout.replace("\n", " | "))
        json.dump({"time": time.time(), "done": len(done), "total": len(jobs), "by_tier": by}, open(os.path.join(OUT, "HEARTBEAT.json"), "w"))
say("workers exited:", [p.poll() for p in procs])
for w in range(len(procs)):
    subprocess.run(f"grep -E 'error|oom|stopped' {OUT}/worker{w}.jsonl | tail -20", shell=True)
'''

AGG = r'''# Aggregate -> summary.json + TABLES.md (README-ready markdown). Re-runnable on the box from the output (gold ids are saved, no texts).
subprocess.run(f"python -u scripts/hf_readme_eval.py aggregate --out {OUT} --bootstrap {BOOTSTRAP} 2>&1 | tail -5", shell=True)
print(open(os.path.join(OUT, "TABLES.md")).read())
S = json.load(open(os.path.join(OUT, "summary.json")))
st = {"state": "complete" if not S["missing"] else "partial", "missing": S["missing"], "hours": round((time.time() - T0) / 3600, 3),
      "smoke_rows": SMOKE_ROWS}
json.dump(st, open(os.path.join(OUT, "STATUS.json"), "w"), indent=1); say("STATUS", st)
shutil.rmtree("/tmp/ugc_evalsets", ignore_errors=True)
'''

cells = [KNOBS, SETUP, PREP, RUN, AGG]
for c in cells:
    ast.parse(c)
nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(INTRO)] + [nbf.v4.new_code_cell(c) for c in cells]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "kaggle_eval_readme.ipynb")
print("wrote kaggle_eval_readme.ipynb; long-Jev md5", LJ_MD5)

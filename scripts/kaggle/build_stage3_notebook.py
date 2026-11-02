#!/usr/bin/env python
"""Builds kaggle_stage3.ipynb from the (Kaggle-proven) kaggle_stage2_full.ipynb cells with checked substitutions.
Stage 3 = long-context (8,192) + many-option (<= 256) finetune of the stage-2 FULL arm C model (configs/stage3_longctx.yaml).
MODE knob:
  "smoke" : (current Kaggle account, ~1.5 GPU-h, cap 2 h) = the FIRST CHUNK of the real run: the REAL data (configs/data_stage3.yaml,
            same recipe as the full run), S2 baseline on the long-Jev slice at 8192, then train.py on the REAL config (956-step
            schedule, runs/stage3_full) with a 1.0 h train cap -> a few hundred steps, checkpoint + exit needs_resume (NaN stop,
            OOM retry, stall watchdog as in S2). Then a long-Jev eval of the newest checkpoint (early signal, not the bar).
  "full"  : (Kaggle account #2, deferred) same data/config; PREV_OUTPUT="auto" restores the smoke output (data md5-checked,
            checkpoint resumed) and trains to completion, then final quick eval (= S2 protocol) + long-Jev eval + comparison.
OOM: ONE resume with data.max_length=4096 (fallback; changes the experiment -> reported). A 2nd OOM stops.
Run: python scripts/build_stage3_notebook.py"""
import ast, os, sys
import nbformat as nbf


def sub(s, old, new, count=None):
    n = s.count(old)
    assert n >= 1 and (count is None or n == count), f"substitution target found {n}x: {old[:80]!r}"
    return s.replace(old, new)


src = nbf.read("kaggle_stage2_full.ipynb", as_version=4)["cells"]
knobs, setup, init, restore, _data, run, report, clean = [c.source for c in src[1:9]]

INTRO = r"""# UltraGliClass stage 3: long context (8,192 tokens) + many options (<= 256), from the stage-2 FULL arm C model

**Inputs:** code dataset `ultragliclass-code` **v8** (has `configs/stage3_longctx.yaml`) + the output of kernel `ultragliclass-s2full`
**version 4** (has `runs/stage2_full_C/final_model` AND `data/stage2_full/train.jsonl` + `label_pools.json`, used for replay).
**MODE = "smoke"** (current account, ~1.5 GPU-h, hard cap 1.8 h + upload): REAL data + REAL config, S2 baseline on the long-Jev slice, ~1 h of
training (a few hundred of the 956 steps; s/step, peak memory, NaN), checkpoint + `needs_resume`, long-Jev eval of that checkpoint.
**MODE = "full"** is DEFERRED to Kaggle account #2: attach this smoke output, `PREV_OUTPUT="auto"` resumes it. See STAGE3_JOB_SPEC.md.
**Partly in-domain** (inherited from stage 2 through replay): typed-decisions customer_service + agent_trace_observability; massive and financial_phrasebank are NOT zero-shot.
"""

# ---------------- knobs ----------------
k = knobs
k = sub(k, 'ARMS          = ["full_C"]   # ONE arm: configs/stage2_full_C.yaml (do not change)',
        'MODE          = "smoke"      # "smoke" (current account, ~1.5 GPU-h) or "full" (account #2, resumes the smoke output); STAGE3_JOB_SPEC.md\n'
        'ARMS          = ["s3"]       # ONE arm: configs/stage3_longctx.yaml (do not change)\n'
        'S3_CONFIG     = "configs/stage3_longctx.yaml"\n'
        'SMOKE_TRAIN_HOURS = 1.0      # smoke: train cap (train.py checkpoints + exits needs_resume); ~250 steps at ~14 s/step (EST)\n'
        'SMOKE_SESSION_HOURS = 1.8    # smoke: whole-session cap (setup + data + baseline + train + checkpoint long-Jev eval); + upload <= 2 h\n'
        'SMOKE_QUICK_EVAL = False     # smoke: also run the 0.12-h suite quick eval on the checkpoint (adds ~0.12 h)')
k = sub(k, 'INIT_MODEL    = "auto"     # S1 final_model dir, OR the S1 kernel-output root, OR "auto" = search /kaggle/input for runs/stage1_ultrabert150m/final_model',
        'INIT_MODEL    = "auto"     # S2 final_model dir, OR "auto" = search /kaggle/input for runs/stage2_full_C/final_model (kernel ultragliclass-s2full v4)')
k = sub(k, 'DATA_RECIPE   = "configs/data_stage2_full.yaml"\nDATA_DIR      = "data/stage2_full"\n',
        'DATA_RECIPE   = "configs/data_stage3.yaml"\nDATA_DIR      = "data/stage3"\n'
        'LONGJEV_SLICE = "calib/jev_long_test.jsonl"   # 431 rows, built on the box 25 Oct (scripts/build_long_jev_slice.py); md5 checked below\n'
        'LONGJEV_MD5   = "__LONGJEV_MD5__"\n'
        'LONGJEV_MAXLEN = 8192\n')
k = sub(k, 'OOM_RETRY_SMALLER_BATCH = True  # on OOM: ONE resume with per-device batch 2 x accum 16 (same 64 rows/step); a 2nd OOM stops the session',
        'OOM_RETRY_SMALLER_BATCH = True  # on OOM: ONE resume with data.max_length=4096 (batch is already 1); a 2nd OOM stops the session')
k = sub(k, "SEMANTIC_DATA_APPROVED = True   # Srimon approved the LocalLLaMA/typed-decisions TRAIN split (2 workflows, labelled in-domain), 24 Oct 2026",
        "SEMANTIC_DATA_APPROVED = True   # replay carries S2's LocalLLaMA/typed-decisions TRAIN rows (approved 24 Oct 2026, labelled in-domain)")
k = sub(k, "PREV_OUTPUT   = \"auto\"     # \"auto\" = previous version's output of THIS full run (runs/stage2_full_C) if attached, else fresh; None = force fresh",
        "PREV_OUTPUT   = \"auto\"     # \"auto\" = previous version's output of THIS kernel (runs/stage3_*/STATUS.json or results/stage3_*), else fresh")
k = sub(k, '"configs", "stage2_full_C.yaml"', '"configs", "stage3_longctx.yaml"', 2)
k = sub(k, "# needs the v6 code (full-run configs)", "# needs the v8 code (stage-3 configs)")
k = sub(k, "v6 code dataset (with configs/stage2_full_C.yaml)", "v8 code dataset (with configs/stage3_longctx.yaml)")
k = sub(k, 'glob.glob("/kaggle/input/**/runs/stage2_full_C/STATUS.json", recursive=True)',
        'glob.glob("/kaggle/input/**/runs/stage3_*/STATUS.json", recursive=True) + '
        'glob.glob("/kaggle/input/**/results/stage3_baseline_s2_longjev_8192.json", recursive=True)')
k = sub(k, 'p.split("/runs/")[0] for p in', '(p.split("/runs/")[0] if "/runs/" in p else p.split("/results/")[0]) for p in')
k = sub(k, 'say("previous full-run output candidates:", hits)', 'say("previous stage-3 output candidates:", hits)')
k = sub(k, '"/kaggle/input/**/runs/stage1_ultrabert150m/final_model/config.json"', '"/kaggle/input/**/runs/stage2_full_C/final_model/config.json"')
k = sub(k, 'say("S1 model candidates:", cands)', 'say("S2 model candidates:", cands)')
k = sub(k, 'assert ARMS == ["full_C"], ARMS', 'assert ARMS == ["s3"] and MODE in ("smoke", "full"), (ARMS, MODE)')
k = sub(k, '"S1 model not found: attach the output of kernel ultragliclass-s1 (runs/stage1_ultrabert150m/final_model) or set INIT_MODEL"',
        '"S2 model not found: attach the output of kernel ultragliclass-s2full v4 (runs/stage2_full_C/final_model) or set INIT_MODEL"')
k += r'''
def find_s2_data():
    c = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/data/stage2_full/train.jsonl", recursive=True)}, key=len)
    say("S2 data candidates:", c)
    return c[0] if c else None
S2_DATA = find_s2_data()
assert S2_DATA and os.path.exists(os.path.join(S2_DATA, "label_pools.json")), \
    "S2 data (data/stage2_full/train.jsonl + label_pools.json) not found: attach the output of kernel ultragliclass-s2full v4"
if MODE == "smoke":   # same data + config + schedule as the full run; only the time caps differ (resume-safe first chunk)
    ARM_MAX_HOURS = SMOKE_TRAIN_HOURS; TOTAL_MAX_HOURS = SMOKE_SESSION_HOURS; QUICK_EVAL = SMOKE_QUICK_EVAL
say("MODE", MODE, "| S2_DATA", S2_DATA, "| EXTRA_SET", EXTRA_SET)
'''
init = init.replace("S1 model", "S2 model").replace("# The S1 model", "# The S2 model").replace("cfg_s1", "cfg_s2")

# ---------------- data ----------------
DATA = r'''# Data. Stage-2 mix is linked read-only for replay; the long-Jev eval slice ships in the code dataset (md5-checked).
import hashlib
sys.path.insert(0, WORK)
from ugc.config import load_config as _lc
os.makedirs("results", exist_ok=True); os.makedirs("data", exist_ok=True)
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
assert SEMANTIC_DATA_APPROVED
assert os.path.exists("calib/stage2_pilot_val.jsonl") and os.path.exists(LONGJEV_SLICE), "calib files missing from the code dataset"
assert md5(LONGJEV_SLICE) == LONGJEV_MD5, f"{LONGJEV_SLICE} md5 {md5(LONGJEV_SLICE)} != {LONGJEV_MD5}"
if not os.path.exists("data/stage2_full"):
    os.symlink(S2_DATA, "data/stage2_full")           # read-only is fine: prepare_data only reads (replay + pools copy)
say("data/stage2_full ->", os.path.realpath("data/stage2_full"), sorted(os.listdir("data/stage2_full")))
recipe, ddir = DATA_RECIPE, DATA_DIR   # smoke and full share the REAL recipe
MD5_REC = f"results/{os.path.basename(ddir)}_data_md5.json"
if not os.path.exists(os.path.join(ddir, "train.jsonl")):
    say("preparing data:", recipe); t = time.time()
    subprocess.run(f"python -u prepare_data.py --recipe {recipe}", shell=True, check=True)
    say(f"data prep took {(time.time() - t) / 60:.1f} min")
DATA_MD5 = md5(os.path.join(ddir, "train.jsonl"))
if os.path.exists(MD5_REC):
    assert json.load(open(MD5_REC))["train_md5"] == DATA_MD5, "STOP: rebuilt stage-3 data differs from session 1 (resume order would differ)"
else:
    json.dump({"train_md5": DATA_MD5, "recipe": recipe}, open(MD5_REC, "w"))
man = json.load(open(os.path.join(ddir, "manifest.json")))
shutil.copyfile(os.path.join(ddir, "manifest.json"), f"results/{os.path.basename(ddir)}_data_manifest.json")
used = {k: v for k, v in man["sources"].items() if v.get("train_rows", 0) > 0}
bad = [k for k in used if any(x in k.lower() for x in ("dbpedia", "bitext", "banking77", "clinc", "massive", "snips", "phrasebank", "sst5", "procedural-typed-decisions"))]
slice_txt = {json.loads(l)["text"] for l in open(LONGJEV_SLICE)}
val_txt = {json.loads(l)["text"] for l in open(os.path.join(ddir, "val.jsonl"))}
n_rows = n_long = n_slice = n_val = n_sa = 0
with open(os.path.join(ddir, "train.jsonl")) as f:
    for line in f:
        ex = json.loads(line); n_rows += 1
        n_long += ex["source"].startswith("tasksource/tasksource-jev-typed-decisions")
        n_slice += ex["text"] in slice_txt; n_val += ex["text"] in val_txt
cfg_t = _lc(S3_CONFIG, [f"model.init_from={INIT_LOCAL}"] + EXTRA_SET)["train"]
eff = int(cfg_t["per_device_train_batch_size"]) * int(cfg_t["gradient_accumulation_steps"]) * (NGPU if NGPU >= 2 and not USE_SINGLE_GPU else 1)
STEPS_PLANNED = int(cfg_t.get("max_steps") or 0) or (n_rows // eff)
say(f"train rows {n_rows} (long Jev {n_long}, replay {n_rows - n_long}) | val {man['val_rows']} | pools {man.get('taxonomy')} | labels/row {man['labels_per_row']}")
say(f"leak: train rows whose text is in the long-Jev EVAL slice {n_slice}, in the stage-3 val {n_val} | steps planned {STEPS_PLANNED} x {eff}")
say("sources used:", {k: v["train_rows"] for k, v in used.items()})
json.dump({"mode": MODE, "train_rows": n_rows, "long_jev_rows": n_long, "replay_rows": n_rows - n_long, "train_in_eval_slice": n_slice,
           "train_in_val": n_val, "md5": DATA_MD5, "steps_planned": STEPS_PLANNED, "licenses": {k: v.get("license") for k, v in used.items()}},
          open(f"results/{os.path.basename(ddir)}_data_check.json", "w"), indent=1)
assert not bad, f"excluded / eval-set / dedicated procedural source in the data: {bad}"
assert n_slice == 0 and n_val == 0, "STOP: eval-slice or val text found in train"
if True:
    assert 20_000 <= n_long <= 26_000 and 39_000 <= n_rows - n_long <= 40_000, (n_long, n_rows - n_long)
'''

# ---------------- run ----------------
r = run
r = sub(r, 'configs/stage2_full_C.yaml', '" + S3_CONFIG + "' if False else 'configs/stage2_full_C.yaml', None)
r = r.replace('f"configs/stage2_full_C.yaml"', 'S3_CONFIG').replace('"configs/stage2_full_C.yaml"', 'S3_CONFIG')
r = sub(r, 'f"{launcher} train.py --config configs/stage2_full_C.yaml --resume auto', 'f"{launcher} train.py --config {S3_CONFIG} --resume auto')
r = sub(r, '''    if SMALL_BATCH:   # OOM retry: half the micro-batch, double the accumulation (same rows per optimizer step)
        _c = load_config(S3_CONFIG, arm_sets(arm))["train"]
        sets += [f"train.per_device_train_batch_size={int(_c['per_device_train_batch_size']) // 2}", f"train.gradient_accumulation_steps={int(_c['gradient_accumulation_steps']) * 2}"]
    accum = int(load_config(S3_CONFIG, arm_sets(arm))["train"].get("gradient_accumulation_steps", 1)) * (2 if SMALL_BATCH else 1)''',
          '''    if SMALL_BATCH:   # OOM retry: batch is already 1 -> shorten the context (fallback; reported as a deviation)
        sets += ["data.max_length=4096"]
    accum = int(load_config(S3_CONFIG, arm_sets(arm))["train"].get("gradient_accumulation_steps", 1))''')
head, _tail = r.split("def run_arm(arm):", 1)
RUN = r'''def longjev_eval(model_dir, res_path, tag):
    cmd = f"python -u scripts/eval_long_jev.py --model {model_dir} --slice {LONGJEV_SLICE} --max_length {LONGJEV_MAXLEN} --batch_size 1 --dtype fp16 --out {res_path}"
    rc, _ = run_streamed(cmd, {"CUDA_VISIBLE_DEVICES": "0"}, f"longjev_{tag}", 20)
    if not os.path.exists(res_path):
        return {"rc": rc}
    x = json.load(open(res_path))
    return {"rc": rc, "acc": x["accuracy"], "ci95": x["accuracy_ci95_cluster"], "n": x["n"], "truncated_frac": x["truncated_frac"], "seconds": x["seconds"]}

BASE_LJ = "results/stage3_baseline_s2_longjev_8192.json"
def baseline():
    if os.path.exists(BASE_LJ):
        say("baseline long-Jev eval of S2 already present:", BASE_LJ); return {"note": "present"}
    say("BASELINE: stage-2 final_model on the long-Jev slice at", LONGJEV_MAXLEN)
    return longjev_eval(INIT_LOCAL, BASE_LJ, "baseline_s2")

def smoke_report(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(LOG_DIR, "train_s3_*.log"))):
        for line in open(p, errors="ignore"):
            if "sec_per_step" in line or "peak_reserved_gb" in line: rows.append(line.strip()[:300])
    say("speed/memory metric lines (last 12):"); [print(x) for x in rows[-12:]]
    st = read_status(out_dir); hb = os.path.join(out_dir, "HEARTBEAT.json")
    return {"status": st.get("state"), "step": st.get("global_step"), "heartbeat": json.load(open(hb)) if os.path.exists(hb) else None,
            "metric_lines": rows[-12:], "oom": bool(LAST_OOM), "small_ctx_retry": SMALL_BATCH}

def run_arm(arm):
    global single, SMALL_BATCH
    cfg = load_config(S3_CONFIG, arm_sets(arm))
    out_dir = cfg["train"]["output_dir"]
    st0 = read_status(out_dir)
    rec = {"arm": arm, "mode": MODE, "out_dir": out_dir, "state": None, "t0_hours": round(elapsed_h(), 3), "start_step": st0.get("global_step", 0)}
    rec["baseline_longjev"] = baseline()
    st = st0
    if st.get("state") == "complete" and os.path.isdir(st.get("final_model", "")):
        say("STAGE 3 already complete (restored): skipping training")
    else:
        if arm_budget_h() < 0.3:
            rec["state"] = "not_started_budget"; say(f"NOT STARTED, only {arm_budget_h():.2f} h of budget left"); return rec
        say(f"===== STAGE 3 {MODE}: step {rec['start_step']} -> {STEPS_PLANNED}, session train budget {arm_budget_h():.2f} h =====")
        rc, stalled = run_streamed(train_cmd(arm, single), ({"CUDA_VISIBLE_DEVICES": "0"} if single else nccl_env),
                                   f"train_{arm}_{'1gpu' if single else 'ddp'}", STALL_MINUTES)
        st = read_status(out_dir)
        if LAST_OOM and OOM_RETRY_SMALLER_BATCH and not SMALL_BATCH and st.get("state") != "complete" and arm_budget_h() >= 0.3:
            say("OOM at 8192 -> ONE resume from the newest checkpoint with data.max_length=4096 (FALLBACK, report it)")
            SMALL_BATCH = True; rec["oom_retry_maxlen4096"] = True
            rc, stalled = run_streamed(train_cmd(arm, single), ({"CUDA_VISIBLE_DEVICES": "0"} if single else nccl_env),
                                       f"train_{arm}_ctx4096", STALL_MINUTES)
            st = read_status(out_dir)
        if (rc != 0 or stalled) and not LAST_OOM and not single and AUTO_FALLBACK_SINGLE_GPU and st.get("state") not in ("complete", "nan_stop") \
                and arm_budget_h() >= 0.4 and MODE == "full":
            say("2-GPU run failed/stalled -> resuming on 1 GPU from the newest checkpoint (2x slower: report before relying on it)")
            single = True
            rc, stalled = run_streamed(train_cmd(arm, True), {"CUDA_VISIBLE_DEVICES": "0"}, f"train_{arm}_1gpu_fallback", STALL_MINUTES)
            st = read_status(out_dir)
        rec["oom"] = bool(LAST_OOM)
    rec["state"] = st.get("state", "no_status"); rec["status"] = st; rec["end_step"] = st.get("global_step")
    print(json.dumps(st, indent=2))
    rec["speed_mem"] = smoke_report(out_dir)   # sec_per_step / peak_reserved_gb lines, heartbeat, OOM / 4096-fallback flags
    if rec["state"] == "complete":
        res_path = "results/stage3_final_quick.json"
        if QUICK_EVAL and not os.path.exists(res_path):
            rec["eval"] = quick_eval(st["final_model"], res_path, "final")
        lj = "results/stage3_final_longjev_8192.json"
        if not os.path.exists(lj):
            rec["longjev"] = longjev_eval(st["final_model"], lj, "final")
        if os.path.exists(lj) and os.path.exists(BASE_LJ):
            subprocess.run(f"python scripts/eval_long_jev.py --compare {BASE_LJ} {lj} > results/stage3_longjev_compare.json", shell=True)
            rec["longjev_compare"] = json.load(open("results/stage3_longjev_compare.json"))
        if DROP_CKPT_AFTER_ARM and os.path.exists(res_path) and os.path.exists(lj):
            for c in glob.glob(os.path.join(out_dir, "checkpoint-*")):
                shutil.rmtree(c, ignore_errors=True)
    else:
        say(f"STAGE 3: state={rec['state']} at step {rec['end_step']} (needs_resume -> attach this output, run the next session; nan_stop/no_status -> report)")
        ck = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        if MID_EVAL and ck and rec["state"] == "needs_resume" and SESSION_HOURS - elapsed_h() > 0.3:
            step = int(ck[-1].rsplit("-", 1)[1])
            rec["longjev_mid"] = longjev_eval(ck[-1], f"results/stage3_mid_step{step}_longjev_8192.json", f"mid{step}")
            if os.path.exists(BASE_LJ) and os.path.exists(f"results/stage3_mid_step{step}_longjev_8192.json"):
                _o = f"results/stage3_mid_step{step}_longjev_compare.json"
                subprocess.run(f"python scripts/eval_long_jev.py --compare {BASE_LJ} results/stage3_mid_step{step}_longjev_8192.json > {_o}", shell=True)
                rec["longjev_mid_compare"] = json.load(open(_o))
            if QUICK_EVAL and SESSION_HOURS - elapsed_h() > 0.3:
                rec["eval_mid"] = quick_eval(ck[-1], f"results/stage3_mid_step{step}_quick.json", f"mid{step}")
    rec["wall_hours"] = round(elapsed_h() - rec["t0_hours"], 3)
    return rec

SESSION_IDX = 1 + len(glob.glob(f"results/stage3_{MODE}_summary_s*.json"))
SUMMARY = []
def write_summary():
    json.dump({"mode": MODE, "session": SESSION_IDX, "session_hours": round(elapsed_h(), 3), "arms": SUMMARY, "init_model_src": INIT_SRC,
               "prev_output": PREV_OUTPUT, "s2_data": S2_DATA, "data_md5": DATA_MD5, "partly_in_domain": True},
              open(f"results/stage3_{MODE}_summary_s{SESSION_IDX}.json", "w"), indent=2)

say(f"STAGE 3 {MODE} SESSION {SESSION_IDX}")
rec = run_arm("s3"); SUMMARY.append(rec); write_summary()
say(f"SESSION done: mode={MODE} state={rec['state']} steps {rec.get('start_step')} -> {rec.get('end_step')} wall={rec.get('wall_hours')} h")'''
run3 = head + RUN

rep = r'''print(json.dumps({"mode": MODE, "session": SESSION_IDX, "session_hours": round(elapsed_h(), 2),
                  "arms": [{k: v for k, v in r.items() if k != "status"} for r in SUMMARY]}, indent=2, default=str))
for r in SUMMARY:
    s = r.get("status", {})
    say(f"STAGE 3 {MODE}: {r['state']} step {s.get('global_step')}/{s.get('max_steps')} train_loss {s.get('train_loss')} "
        f"eval_loss {(s.get('final_eval') or {}).get('eval_loss')} | baseline {r.get('baseline_longjev')} | longjev {r.get('longjev')} | compare {r.get('longjev_compare')}")'''
cl = sub(clean, 'glob.glob("runs/stage2_full_*")', 'glob.glob("runs/stage3_*")')
cl = 'if os.path.islink("data/stage2_full"):\n    os.unlink("data/stage2_full")   # never save the 1.8 GB S2 data link into this output\n' + cl

cells = [k, setup, init, restore, DATA, run3, rep, cl]
md5 = __import__("hashlib").md5(open("calib/jev_long_test.jsonl", "rb").read()).hexdigest()
cells[0] = cells[0].replace("__LONGJEV_MD5__", md5)
for c in cells:
    ast.parse(c)
nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(INTRO)] + [nbf.v4.new_code_cell(c) for c in cells]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "kaggle_stage3.ipynb")
print("wrote kaggle_stage3.ipynb; long-Jev slice md5", md5)

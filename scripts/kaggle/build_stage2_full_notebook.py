#!/usr/bin/env python
"""Builds kaggle_stage2_full.ipynb: the FULL stage-2 run (arm C, configs/stage2_full_C.yaml) as ONE session-aware notebook.
Reuses the pilot notebook cells (scripts/pilot_cells.py: setup, S1 init copy, previous-output restore, run_streamed watchdog, NCCL pre-flight,
1-GPU fallback) with checked substitutions, plus:
  * PREV_OUTPUT="auto" by default: session 1 finds nothing (fresh start), session k>1 restores runs/ data/ results/ of the attached previous
    version of THIS kernel's output (runs/stage2_full_C/STATUS.json) and train.py resumes from the newest checkpoint.
  * data: built once (session 1), restored afterwards; if it has to be rebuilt, its md5 must equal the one recorded in session 1 (else STOP:
    a different row order would break the resume data order). Asserts: rows, sources, licences, procedural share, 3,000 typed-train rows.
  * mid-run quick eval of the newest checkpoint at every session end (needs_resume), final quick eval of final_model at completion.
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; OOM -> ONE retry in the same session with per-device batch 2 x accum 16 (same 64 rows/step).
Run: python scripts/build_stage2_full_notebook.py"""
import ast, os, sys
import nbformat as nbf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_cells as pc

def sub(s, old, new, count=None):
    n = s.count(old)
    assert n >= 1 and (count is None or n == count), f"substitution target found {n}x: {old[:70]!r}"
    return s.replace(old, new)

INTRO = r"""# UltraGliClass stage 2 FULL run (arm C), session-aware

**What:** pilot arm C recipe (many-label expansion 20-150 labels + permutation-consistency) on the FULL stage-2 mix (`configs/data_stage2_full.yaml`, ~1.74M rows),
27,000 steps x 64 rows, init = the stage-1 model, seed 42. Fast settings measured on T4 x2: per-device batch 4 x accum 8, gradient checkpointing OFF, fp16,
length grouping OFF, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, DataLoader workers 0 (host-RAM safety).
**Partly in-domain:** 3,000 `LocalLLaMA/typed-decisions` TRAIN rows (customer_service + agent_trace_observability). massive and financial_phrasebank are NOT zero-shot for this model.

**Sessions (<= 11 h each, see STAGE2_FULL_JOB_SPEC.md):** session 1 = fresh (attach code dataset v6 + S1 output). Session 2+ = attach ALSO the previous version's
output of THIS kernel; `PREV_OUTPUT = "auto"` finds it and training resumes from the newest checkpoint (saved every 500 steps and at the session time cap).
At each session end: quick eval of the newest checkpoint -> `results/stage2_full_C_mid_step<N>.json`. At completion: final quick eval -> `results/stage2_full_C_final_quick.json`.
Stops (and reports) on NaN, OOM (after one smaller-micro-batch retry), stall (15 min no output -> kill, then one 1-GPU resume), or the session cap.
"""

knobs = pc.KNOBS
knobs = sub(knobs, 'ARMS          = ["A", "B", "C"]   # arms to run, in this order; e.g. ["B"] to run one arm only',
            'ARMS          = ["full_C"]   # ONE arm: configs/stage2_full_C.yaml (do not change)')
knobs = sub(knobs, 'DATA_RECIPE   = "configs/data_stage2_pilot.yaml"\nDATA_DIR      = "data/stage2_full"\n' if False else
            'DATA_RECIPE   = "configs/data_stage2_pilot.yaml"\nDATA_DIR      = "data/stage2_pilot"\n',
            'DATA_RECIPE   = "configs/data_stage2_full.yaml"\nDATA_DIR      = "data/stage2_full"\n'
            'SEMANTIC_DATA_APPROVED = True   # Srimon approved the LocalLLaMA/typed-decisions TRAIN split (2 workflows, labelled in-domain), 24 Oct 2026\n'
            'OOM_RETRY_SMALLER_BATCH = True  # on OOM: ONE resume with per-device batch 2 x accum 16 (same 64 rows/step); a 2nd OOM stops the session\n')
knobs = sub(knobs, 'PREV_OUTPUT   = None       # None = fresh; "auto" = previous version\'s output of THIS pilot (runs/stage2_pilot_*), never the S1 output',
            'PREV_OUTPUT   = "auto"     # "auto" = previous version\'s output of THIS full run (runs/stage2_full_C) if attached, else fresh; None = force fresh')
knobs = sub(knobs, 'ARM_MAX_HOURS = 2.0        # STOP: train budget per arm (train.py saves and exits; the notebook then stops and reports)',
            'ARM_MAX_HOURS = 10.6       # train budget per SESSION (train.py saves a checkpoint and exits needs_resume; next session resumes)')
knobs = sub(knobs, 'TOTAL_MAX_HOURS = 6.0      # STOP: no new arm starts if the session clock would pass this (GPU-hours = session hours)',
            'TOTAL_MAX_HOURS = 10.8     # STOP: session clock cap incl. setup, data, train, eval; + ~0.2 h output upload => <= 11 h (GPU-hours = session hours)')
knobs = sub(knobs, 'DROP_CKPT_AFTER_ARM = True       # delete checkpoint-* of a COMPLETED arm (final_model is kept): saves about 1.8 GB per arm',
            'DROP_CKPT_AFTER_ARM = True       # delete checkpoint-* only once the run is COMPLETE and its final eval is written (final_model is kept)\n'
            'MID_EVAL      = True             # quick eval of the newest checkpoint at the end of an unfinished session')
knobs = sub(knobs, '"configs", "stage2_pilot_A.yaml"', '"configs", "stage2_full_C.yaml"', 2)
knobs = sub(knobs, "runs/stage2_pilot_*/STATUS.json", "runs/stage2_full_C/STATUS.json")
knobs = sub(knobs, "configs/stage2_pilot_*.yaml", "configs/stage2_full_C.yaml")
knobs = sub(knobs, "# needs the v3 code", "# needs the v6 code (full-run configs)")
knobs = sub(knobs, "v3 code dataset", "v6 code dataset")
knobs = sub(knobs, 'say("previous pilot-output candidates:", hits)', 'say("previous full-run output candidates:", hits)')
knobs = sub(knobs, 'say("CODE_INPUT =", CODE_INPUT,', 'assert ARMS == ["full_C"], ARMS\nsay("CODE_INPUT =", CODE_INPUT,')

DATA = r"""# Data: built once in session 1 (deterministic: recipe + seed 43), restored from the previous output afterwards.
import hashlib
sys.path.insert(0, WORK)
from ugc.config import load_config as _lc
assert os.path.exists("calib/stage2_pilot_val.jsonl"), "calib/stage2_pilot_val.jsonl missing from the code dataset"
assert SEMANTIC_DATA_APPROVED, "the full recipe contains LocalLLaMA/typed-decisions TRAIN rows (owner approval D10)"
os.makedirs("results", exist_ok=True)
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
MD5_REC = "results/stage2_full_data_md5.json"
rebuilt = False
if not os.path.exists(os.path.join(DATA_DIR, "train.jsonl")):
    say("preparing data:", DATA_RECIPE, PREP_ARGS); t = time.time()
    subprocess.run(f"python -u prepare_data.py --recipe {DATA_RECIPE} {PREP_ARGS}", shell=True, check=True)
    rebuilt = True; say(f"data prep took {(time.time() - t) / 60:.1f} min")
else:
    say("data already present (restored from the previous output):", DATA_DIR)
DATA_MD5 = md5(os.path.join(DATA_DIR, "train.jsonl"))
if os.path.exists(MD5_REC):
    rec_md5 = json.load(open(MD5_REC))["train_md5"]
    assert rec_md5 == DATA_MD5, (f"STOP: train.jsonl md5 {DATA_MD5} != session-1 md5 {rec_md5}; the resume data order would differ. "
                                 "Attach the previous output that contains data/stage2_full (do not rebuild).")
    say("train.jsonl md5 matches session 1:", DATA_MD5, "(rebuilt)" if rebuilt else "")
else:
    json.dump({"train_md5": DATA_MD5, "recipe": DATA_RECIPE, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(MD5_REC, "w"))
man = json.load(open(os.path.join(DATA_DIR, "manifest.json")))
shutil.copyfile(os.path.join(DATA_DIR, "manifest.json"), "results/stage2_full_data_manifest.json")
_t = _lc("configs/stage2_full_C.yaml", [f"model.init_from={INIT_LOCAL}"] + EXTRA_SET)["train"]
need_rows = int(_t["max_steps"]) * int(_t["per_device_train_batch_size"]) * int(_t.get("gradient_accumulation_steps", 1)) * \
    (NGPU if NGPU >= 2 and not USE_SINGLE_GPU else 1)      # = 27,000 x 64 = 1,728,000 on the real run
used = {k: v for k, v in man["sources"].items() if v.get("train_rows", 0) > 0}
bad = [k for k in used if any(x in k.lower() for x in ("dbpedia", "bitext", "banking77", "clinc", "massive", "snips", "phrasebank", "sst5"))]
sem = {k: v["train_rows"] for k, v in used.items() if "localllama/typed-decisions" in k.lower()}
proc_dedicated = sum(v["train_rows"] for k, v in used.items() if k.startswith("tasksource/procedural-typed-decisions"))
n_proc = n_tot = 0
with open(os.path.join(DATA_DIR, "train.jsonl")) as f:
    for line in f:
        n_tot += 1; n_proc += ("procedural-typed-decisions" in json.loads(line)["source"])
PROC_SHARE = round(n_proc / max(n_tot, 1), 4)
val_txt = {json.loads(l)["text"] for l in open("calib/stage2_pilot_val.jsonl")}
n_val_in_train = 0
with open(os.path.join(DATA_DIR, "train.jsonl")) as f:
    for line in f:
        n_val_in_train += json.loads(line)["text"] in val_txt
say("train rows", man["train_rows"], "need", need_rows, "| val (own) rows", man["val_rows"], "| pools", man.get("taxonomy", {}).get("pool_sizes"),
    "| labels/row", man["labels_per_row"])
say(f"procedural-style rows {n_proc} ({100 * PROC_SHARE:.2f} %), dedicated procedural rows {proc_dedicated} | typed-decisions TRAIN rows {sem} | "
    f"train rows whose text equals a val text: {n_val_in_train}")
say("sources used:", {k: v["train_rows"] for k, v in used.items()})
print(json.dumps({k: v.get("license") for k, v in used.items()}, indent=1))
json.dump({"train_rows": man["train_rows"], "need_rows": need_rows, "procedural_share": PROC_SHARE, "procedural_rows": n_proc,
           "semantic_rows": sem, "train_rows_text_in_val": n_val_in_train, "md5": DATA_MD5,
           "licenses": {k: v.get("license") for k, v in used.items()}}, open("results/stage2_full_data_check.json", "w"), indent=1)
assert not bad, f"excluded / eval-set source in the data: {bad}"
assert man["train_rows"] >= need_rows, f"only {man['train_rows']} train rows, need {need_rows}"
assert proc_dedicated == 0, f"dedicated procedural rows present: {proc_dedicated}"
assert sorted(k.split(":")[1] for k in sem) == ["agent_trace_observability", "customer_service"] and sum(sem.values()) == 3000, sem
assert 0.035 <= PROC_SHARE <= 0.065 or man["train_rows"] < 100_000, f"procedural share {PROC_SHARE} not at the E1 level (~0.049)"
assert n_val_in_train <= 0.01 * len(val_txt), f"{n_val_in_train} train rows share a text with the val set"
"""

run = pc.ARMS_RUN
run = sub(run, '"NCCL_DEBUG": "WARN", **env_extra}', '"NCCL_DEBUG": "WARN", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **env_extra}')
run = sub(run, 'configs/stage2_pilot_{arm}.yaml', 'configs/stage2_full_C.yaml', 3)
run = sub(run, 'def train_cmd(arm, single):\n    sets = [f"train.time_budget_hours={max(0.1, arm_budget_h()):.2f}"] + arm_sets(arm)\n',
          'SMALL_BATCH = False\ndef train_cmd(arm, single):\n    sets = [f"train.time_budget_hours={max(0.1, arm_budget_h()):.2f}"] + arm_sets(arm)\n'
          '    if SMALL_BATCH:   # OOM retry: half the micro-batch, double the accumulation (same rows per optimizer step)\n'
          '        _c = load_config("configs/stage2_full_C.yaml", arm_sets(arm))["train"]\n'
          '        sets += [f"train.per_device_train_batch_size={int(_c[\'per_device_train_batch_size\']) // 2}", '
          'f"train.gradient_accumulation_steps={int(_c[\'gradient_accumulation_steps\']) * 2}"]\n')
run = sub(run, '        accum = int(load_config(f"configs/stage2_full_C.yaml", arm_sets(arm))["train"].get("gradient_accumulation_steps", 1))' if False else
          '    accum = int(load_config(f"configs/stage2_full_C.yaml", arm_sets(arm))["train"].get("gradient_accumulation_steps", 1))\n',
          '    accum = int(load_config(f"configs/stage2_full_C.yaml", arm_sets(arm))["train"].get("gradient_accumulation_steps", 1)) * (2 if SMALL_BATCH else 1)\n')
run = sub(run, '    if single and NGPU > 1:\n        sets.append(f"train.gradient_accumulation_steps={accum * NGPU}")\n',
          '    if single and NGPU > 1:\n        sets = [s for s in sets if not s.startswith("train.gradient_accumulation_steps=")] + [f"train.gradient_accumulation_steps={accum * NGPU}"]\n')
head, _tail = run.split("def run_arm(arm):", 1)
RUN_FULL = r'''def quick_eval(model_dir, res_path, tag):
    cmd = (f"python -u eval.py --model {model_dir} --suites gliclass intent typed --max_per_dataset {EVAL_MAX_PER_DATASET} "
           f"--permutations {EVAL_PERMS} --calib_file {CALIB_FILE} --calib_n {EVAL_CALIB_N} --max_length 1024 --out {res_path}" +
           (f" --typed_max_cases {EVAL_TYPED_MAX_CASES}" if EVAL_TYPED_MAX_CASES else ""))
    rc, _ = run_streamed(cmd, {}, f"quick_eval_{tag}", 30)
    if not os.path.exists(res_path):
        return {"eval_rc": rc}
    r = json.load(open(res_path)); d = r["datasets"]; t1 = (r.get("typed_decisions") or {}).get("T1", {})
    return {"eval_rc": rc, "avg_macro_f1": r.get("gliclass_avg_macro_f1"), "global_T": r.get("global_T"), "typed_acc": t1.get("accuracy"),
            "typed_by_type": t1.get("accuracy_by_type"), "typed_by_workflow": t1.get("accuracy_by_workflow"),
            "flip_many_label": {k: d[k]["perm_flip_rate"] for k in ("banking77", "clinc150", "massive", "SetFit/20_newsgroups") if k in d},
            "NOT_zero_shot": ["massive (HWU64 shares 56/59 intents)", "financial_phrasebank (fin-news sentiment, same 3 labels)",
                              "typed-decisions customer_service + agent_trace_observability (trained on their TRAIN split)"]}

def run_arm(arm):
    global single, SMALL_BATCH
    cfg = load_config("configs/stage2_full_C.yaml", arm_sets(arm))
    out_dir = cfg["train"]["output_dir"]
    st0 = read_status(out_dir)
    rec = {"arm": arm, "out_dir": out_dir, "state": None, "t0_hours": round(elapsed_h(), 3), "start_step": st0.get("global_step", 0)}
    st = st0
    if st.get("state") == "complete" and os.path.isdir(st.get("final_model", "")):
        say("FULL RUN already complete (restored): skipping training")
    else:
        if arm_budget_h() < 0.4:
            rec["state"] = "not_started_budget"; say(f"NOT STARTED, only {arm_budget_h():.2f} h of budget left"); return rec
        say(f"===== FULL arm C: step {rec['start_step']} -> {cfg['train']['max_steps']}, session train budget {arm_budget_h():.2f} h =====")
        rc, stalled = run_streamed(train_cmd(arm, single), ({"CUDA_VISIBLE_DEVICES": "0"} if single else nccl_env),
                                   f"train_{arm}_{'1gpu' if single else 'ddp'}", STALL_MINUTES)
        st = read_status(out_dir)
        if LAST_OOM and OOM_RETRY_SMALLER_BATCH and not SMALL_BATCH and st.get("state") != "complete" and arm_budget_h() >= 0.4:
            say("OOM -> ONE resume from the newest checkpoint with per-device batch 2 x accum 16 (same 64 rows/step)")
            SMALL_BATCH = True; rec["oom_retry_small_batch"] = True
            rc, stalled = run_streamed(train_cmd(arm, single), ({"CUDA_VISIBLE_DEVICES": "0"} if single else nccl_env),
                                       f"train_{arm}_smallbatch", STALL_MINUTES)
            st = read_status(out_dir)
        if (rc != 0 or stalled) and not LAST_OOM and not single and AUTO_FALLBACK_SINGLE_GPU and st.get("state") not in ("complete", "nan_stop") \
                and arm_budget_h() >= 0.4:
            say("2-GPU run failed/stalled -> resuming on 1 GPU from the newest checkpoint")
            single = True
            rc, stalled = run_streamed(train_cmd(arm, True), {"CUDA_VISIBLE_DEVICES": "0"}, f"train_{arm}_1gpu_fallback", STALL_MINUTES)
            st = read_status(out_dir)
        rec["oom"] = bool(LAST_OOM)
    hb = os.path.join(out_dir, "HEARTBEAT.json")
    if os.path.exists(hb): say("last heartbeat:", open(hb).read()[:300])
    rec["state"] = st.get("state", "no_status"); rec["status"] = st; rec["end_step"] = st.get("global_step")
    print(json.dumps(st, indent=2))
    if rec["state"] == "complete":
        res_path = "results/stage2_full_C_final_quick.json"
        if QUICK_EVAL and not os.path.exists(res_path):
            rec["eval"] = quick_eval(st["final_model"], res_path, "final")
        elif os.path.exists(res_path):
            rec["eval"] = {"note": "final eval already present", "path": res_path}
        if DROP_CKPT_AFTER_ARM and os.path.exists(res_path):
            for c in glob.glob(os.path.join(out_dir, "checkpoint-*")):
                shutil.rmtree(c, ignore_errors=True)
    else:
        say(f"FULL RUN: state={rec['state']} at step {rec['end_step']} (needs_resume = session cap reached -> attach this output and run the next session; "
            f"nan_stop = NaN loss; running/no_status = crash/OOM/stall -> report)")
        ck = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        if MID_EVAL and QUICK_EVAL and ck and rec["state"] == "needs_resume" and SESSION_HOURS - elapsed_h() > 0.3:
            step = int(ck[-1].rsplit("-", 1)[1])
            rec["eval"] = quick_eval(ck[-1], f"results/stage2_full_C_mid_step{step}.json", f"mid_step{step}")
            rec["eval"]["checkpoint"] = ck[-1]
    rec["wall_hours"] = round(elapsed_h() - rec["t0_hours"], 3)
    return rec

SESSION_IDX = 1 + len(glob.glob("results/stage2_full_summary_s*.json"))
SUMMARY = []
def write_summary():
    json.dump({"session": SESSION_IDX, "session_hours": round(elapsed_h(), 3), "arms": SUMMARY, "init_model_src": INIT_SRC, "prev_output": PREV_OUTPUT,
               "data_md5": DATA_MD5, "procedural_share": PROC_SHARE, "partly_in_domain": True},
              open(f"results/stage2_full_summary_s{SESSION_IDX}.json", "w"), indent=2)

say(f"SESSION {SESSION_IDX} of the full run")
rec = run_arm("full_C"); SUMMARY.append(rec); write_summary()
say(f"SESSION {SESSION_IDX} done: state={rec['state']} steps {rec.get('start_step')} -> {rec.get('end_step')} wall={rec.get('wall_hours')} h eval={rec.get('eval')}")'''
run = head + RUN_FULL

report = r"""print(json.dumps({"session": SESSION_IDX, "session_hours": round(elapsed_h(), 2), "arms": [{k: v for k, v in r.items() if k != "status"} for r in SUMMARY]}, indent=2))
for r in SUMMARY:
    s = r.get("status", {})
    say(f"FULL C: {r['state']} step {s.get('global_step')}/{s.get('max_steps')} train_loss {s.get('train_loss')} "
        f"final_eval_loss {(s.get('final_eval') or {}).get('eval_loss')} peak_mem {(s.get('final_eval') or {}).get('peak_mem_gb')} avg_F1 {(r.get('eval') or {}).get('avg_macro_f1')}")
if any(r["state"] != "complete" for r in SUMMARY):
    say("NOT COMPLETE: save this version; next session: attach code v6 + S1 output + THIS version's output, keep PREV_OUTPUT='auto', run all.")"""
clean = sub(pc.CLEAN, 'glob.glob("runs/stage2_pilot_*")', 'glob.glob("runs/stage2_full_*")')

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
shared_cell = nbf.read("kaggle_stage2_pilot.ipynb", as_version=4)["cells"][4]
assert shared_cell.cell_type == "code" and "def restore" in shared_cell.source
cells = [knobs, pc.SETUP, pc.INIT, shared_cell.source, DATA, run, report, clean]
for c in cells:
    ast.parse(c)
nb["cells"] = [md(INTRO)] + [code(c) for c in cells]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "kaggle_stage2_full.ipynb")
print("wrote kaggle_stage2_full.ipynb")

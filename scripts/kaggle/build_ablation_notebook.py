#!/usr/bin/env python
"""Builds kaggle_stage2_ablation.ipynb from the pilot notebook cells (scripts/pilot_cells.py) with checked text substitutions: arms E1/E1z/E2/E3/C0
(configs/stage2_abl_<arm>.yaml), per-arm data prep (each arm has its own recipe/data dir), per-arm data assertions, a switch that keeps the
LocalLLaMA/typed-decisions TRAIN rows (E2, E3) off until the owner approves them. Run: python scripts/build_ablation_notebook.py"""
import os, sys
import nbformat as nbf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_cells as pc

def sub(s, old, new, count=None):
    n = s.count(old)
    assert n >= 1 and (count is None or n == count), f"substitution target found {n}x: {old[:70]!r}"
    return s.replace(old, new)

INTRO = r"""# UltraGliClass stage 2 typed-decisions ABLATION (E1 / E1z / E2 / E3 / C0), one notebook

Question: did the procedural typed-decision rows added in the stage-2 pilot cost the typed-decisions `choice` accuracy (S1 0.485 -> 0.41-0.42)?
Every arm = **pilot arm C** (many-label expansion + permutation consistency, seed 42, 1,500 steps x 64 rows, init = the stage-1 model), only the data mix differs:

| arm | data change vs pilot C (all arms: 96,035 train rows, same val file `calib/stage2_pilot_val.jsonl`) | procedural-style share |
|---|---|---|
| **E1** | dedicated procedural dataset (11,750 rows) removed, filled with gliclass-v2.0 | 4.9 % (the part inside Jev; about S1's own share) |
| **E1z** | E1 and the procedural-style rows inside Jev removed too (Jev drawn from its non-procedural rest) | 0 % |
| **E2** | E1 + 3,000 rows of `LocalLLaMA/typed-decisions` **train** (customer_service + agent_trace_observability; semantic prose options) | 4.9 % |
| **E3** | pilot C + the same 3,000 semantic rows (replacing 3,000 gliclass rows) | 17.2 % |
| **C0** | pilot C unchanged (optional reproducibility / noise rerun) | 17.2 % |

E2 and E3 stay **blocked** (`SEMANTIC_DATA_APPROVED = False`) until the owner approves training on the LocalLLaMA/typed-decisions train split.
After each arm: quick eval (14 GLiClass sets + intent + typed decisions with per-type and per-workflow accuracy) -> `results/stage2_abl_<arm>_quick.json`.
**Inputs:** code dataset v4 (`ultragliclass-code`, with the ablation configs) + the S1 kernel output (as for the pilot). **Settings:** GPU T4 x2, Internet on.
Same safety features as the pilot notebook (watchdog, NCCL pre-flight, 1-GPU fallback, NaN/OOM/arm-time stops, no GPU spend beyond `TOTAL_MAX_HOURS`).
"""

knobs = pc.KNOBS
knobs = sub(knobs, 'ARMS          = ["A", "B", "C"]   # arms to run, in this order; e.g. ["B"] to run one arm only',
            'ARMS          = ["E1"]   # in this order: plan P1 = ["E1"]; P2 = ["E1", "E2"] (E2 needs SEMANTIC_DATA_APPROVED); see STAGE2_ABLATION_JOB_SPEC.md')
knobs = sub(knobs, 'DATA_RECIPE   = "configs/data_stage2_pilot.yaml"\nDATA_DIR      = "data/stage2_pilot"\n',
            'ARM_DATA      = {a: (f"configs/data_stage2_abl_{a}.yaml", f"data/stage2_abl_{a}") for a in ("E1", "E1z", "E2", "E3")}\n'
            'ARM_DATA["C0"] = ("configs/data_stage2_pilot.yaml", "data/stage2_pilot")\n'
            'SEM_ARMS      = ("E2", "E3")   # contain LocalLLaMA/typed-decisions TRAIN rows\n'
            'SEMANTIC_DATA_APPROVED = False   # owner decision needed before E2/E3: the benchmark train split has the same 4 workflows and option strings as the eval set\n')
knobs = sub(knobs, 'PREV_OUTPUT   = None       # None = fresh; "auto" = previous version\'s output of THIS pilot (runs/stage2_pilot_*), never the S1 output',
            'PREV_OUTPUT   = None       # None = fresh; "auto" = previous version\'s output of THIS ablation (runs/stage2_abl_*), never the S1 output or the pilot output')
knobs = sub(knobs, '"configs", "stage2_pilot_A.yaml"', '"configs", "stage2_abl_E1.yaml"', 2)
knobs = sub(knobs, "runs/stage2_pilot_*/STATUS.json", "runs/stage2_abl_*/STATUS.json")
knobs = sub(knobs, "configs/stage2_pilot_*.yaml", "configs/stage2_abl_*.yaml")
knobs = sub(knobs, "# needs the v3 code", "# needs the v4 code (ablation configs)")
knobs = sub(knobs, "v3 code dataset", "v4 code dataset")
knobs = sub(knobs, 'say("CODE_INPUT =", CODE_INPUT,', 'assert all(a in ARM_DATA for a in ARMS), f"unknown arm in {ARMS}"\n'
            'assert SEMANTIC_DATA_APPROVED or not any(a in SEM_ARMS for a in ARMS), "E2/E3 train on LocalLLaMA/typed-decisions TRAIN rows: set SEMANTIC_DATA_APPROVED=True only with the owner\'s approval"\n'
            'say("CODE_INPUT =", CODE_INPUT,')

DATA = r"""# Per-arm data: each arm has its own recipe and data dir; prepared lazily, right before the arm trains.
sys.path.insert(0, WORK)
from ugc.config import load_config as _lc
assert os.path.exists("calib/stage2_pilot_val.jsonl"), "calib/stage2_pilot_val.jsonl missing from the code dataset (v4)"
os.makedirs("results", exist_ok=True)
PROC_SHARE = {}
def prep_arm_data(arm):
    recipe, ddir = ARM_DATA[arm]
    if arm in SEM_ARMS:
        assert SEMANTIC_DATA_APPROVED, "E2/E3 are blocked until SEMANTIC_DATA_APPROVED = True"
    if not os.path.exists(os.path.join(ddir, "train.jsonl")):
        say("preparing data for arm", arm, ":", recipe, PREP_ARGS)
        subprocess.run(f"python -u prepare_data.py --recipe {recipe} {PREP_ARGS}", shell=True, check=True)
    else:
        say("data already present for arm", arm, ":", ddir)
    man = json.load(open(os.path.join(ddir, "manifest.json")))
    shutil.copyfile(os.path.join(ddir, "manifest.json"), f"results/stage2_abl_{arm}_data_manifest.json")
    _t = _lc(f"configs/stage2_abl_{arm}.yaml", [f"model.init_from={INIT_LOCAL}"] + EXTRA_SET)["train"]
    need_rows = int(_t["max_steps"]) * int(_t["per_device_train_batch_size"]) * int(_t.get("gradient_accumulation_steps", 1)) * \
        (NGPU if NGPU >= 2 and not USE_SINGLE_GPU else 1)      # 1500 x 64 = 96,000 on the real run
    used = {k: v for k, v in man["sources"].items() if v.get("train_rows", 0) > 0}
    bad = [k for k in used if any(x in k.lower() for x in ("dbpedia", "bitext", "banking77", "clinc", "massive", "snips", "phrasebank", "sst5"))]
    sem = {k: v["train_rows"] for k, v in used.items() if "localllama/typed-decisions" in k.lower()}
    proc_dedicated = sum(v["train_rows"] for k, v in used.items() if k.startswith("tasksource/procedural-typed-decisions"))
    n_proc = n_tot = 0                                         # procedural-style rows incl. the ones inside Jev (source tag)
    with open(os.path.join(ddir, "train.jsonl")) as f:
        for line in f:
            n_tot += 1; n_proc += ("procedural-typed-decisions" in json.loads(line)["source"])
    PROC_SHARE[arm] = round(n_proc / max(n_tot, 1), 4)
    say(f"[{arm}] train rows {man['train_rows']} val {man['val_rows']} | procedural-style rows {n_proc} ({100 * PROC_SHARE[arm]:.1f} %) | semantic rows {sem} | sources", {k: v["train_rows"] for k, v in used.items()})
    assert not bad, f"excluded / eval-set source in the data: {bad}"
    assert man["train_rows"] >= need_rows, f"only {man['train_rows']} train rows, need {need_rows}"
    assert bool(sem) == (arm in SEM_ARMS), f"[{arm}] semantic (LocalLLaMA typed-decisions train) rows present={bool(sem)} but expected {arm in SEM_ARMS}"
    if arm in SEM_ARMS:
        assert sum(sem.values()) == 3000, sem
    exp = {"E1": (0, 0.03, 0.07), "E1z": (0, 0.0, 0.0), "E2": (0, 0.03, 0.07), "E3": (11750, 0.15, 0.20), "C0": (11750, 0.15, 0.20)}[arm]
    assert (proc_dedicated == exp[0] and exp[1] <= PROC_SHARE[arm] <= exp[2]), \
        f"[{arm}] procedural share {PROC_SHARE[arm]} / dedicated {proc_dedicated} not as designed {exp}"
    return man"""

run = pc.ARMS_RUN
run = sub(run, 'configs/stage2_pilot_{arm}.yaml', 'configs/stage2_abl_{arm}.yaml', 3)
run = sub(run, 'res_path = f"results/stage2_pilot_{arm}_quick.json"', 'res_path = f"results/stage2_abl_{arm}_quick.json"')
run = sub(run, '        if arm_budget_h() < 0.4:', '        prep_arm_data(arm)\n        if arm_budget_h() < 0.4:')
run = sub(run, 'open("results/stage2_pilot_summary.json", "w")', 'open("results/stage2_abl_summary.json", "w")')
run = sub(run, '"typed_acc": (r.get("typed_decisions") or {}).get("T1", {}).get("accuracy"),',
          '"typed_acc": (r.get("typed_decisions") or {}).get("T1", {}).get("accuracy"),\n'
          '                       "typed_by_type": (r.get("typed_decisions") or {}).get("T1", {}).get("accuracy_by_type"),\n'
          '                       "typed_by_workflow": (r.get("typed_decisions") or {}).get("T1", {}).get("accuracy_by_workflow"),\n'
          '                       "procedural_share": PROC_SHARE.get(arm),')
run = sub(run, '"arms": SUMMARY, "init_model_src": INIT_SRC}', '"arms": SUMMARY, "init_model_src": INIT_SRC, "procedural_share": PROC_SHARE}')
clean = sub(pc.CLEAN, 'glob.glob("runs/stage2_pilot_*")', 'glob.glob("runs/stage2_abl_*")')
report = pc.REPORT

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
# cell 4 of kaggle_stage2_pilot.ipynb is the stage-1 notebook's shared cell (reused unchanged by build_notebook.py)
shared_cell = nbf.read("kaggle_stage2_pilot.ipynb", as_version=4)["cells"][4]
assert shared_cell.cell_type == "code"
nb["cells"] = [md(INTRO), code(knobs), code(pc.SETUP), code(pc.INIT), shared_cell, code(DATA), code(run), code(report), code(clean)]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "kaggle_stage2_ablation.ipynb")
print("wrote kaggle_stage2_ablation.ipynb")

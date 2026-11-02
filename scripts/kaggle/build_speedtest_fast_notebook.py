#!/usr/bin/env python
"""Derive kaggle_speedtest_fast.ipynb (C nogroup vs C grouped, ckpt off, b4 x accum 8) from the already-used kaggle_speedtest.ipynb
(the T0-T3 speed test that ran on 23 Oct 2026). PREPARES a file only; nothing is uploaded or run. Usage: python scripts/build_speedtest_fast_notebook.py [src.ipynb]"""
import json, sys, ast
src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/kaggle/ultragliclass/speedtest/ultragliclass/kaggle_speedtest.ipynb"
nb = json.load(open(src))

def sub(s, old, new, count=1):
    assert s.count(old) >= 1, f"pattern not found: {old[:60]!r}"
    return s.replace(old, new) if count == 0 else s.replace(old, new, count)

for c in nb["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "markdown" and s.startswith("# UltraGliClass SPEED TEST"):
        s = """# UltraGliClass FAST SPEED TEST (length-grouped batching; throughput / memory / loss-sanity probe, NOT a quality run)

50 optimizer steps per config, from the **stage-1 model** on the pilot data (`data/stage2_pilot`, same prep as the pilot), arm-C settings (many-label + paired views + consistency),
per-device batch 4 x accum 8 x 2 GPUs = 64 rows/step, gradient checkpointing OFF, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, no eval, no intermediate save.
Order: **fast_C_nogroup** (random batches, the pilot sampler) -> **fast_C_grouped** (`data.length_group`: seeded mega-batch length grouping).
Per config: sec/step (all log intervals + mean of steps 21-50), peak allocated/reserved memory on both ranks, loss / grad_norm / cons_loss per interval. Stops on OOM or any failure.
Inputs: code dataset `ultragliclass-code` (NEW version with `ugc/length_group.py` and `configs/speedtest_fast_C_*.yaml`) + output of kernel `ultragliclass-s1`. GPU T4 x2, Internet on.
Needs Srimon's approval before it is run (see STAGE2_FAST_SPEEDTEST_SPEC.md)."""
        c["source"] = s.splitlines(keepends=True)
        continue
    if c["cell_type"] != "code":
        continue
    if "CONFIGS_ORDER =" in s:
        s = sub(s, 'CONFIGS_ORDER = ["T0_baseline", "T2_nockpt_b4", "T1_nockpt_b8", "T3_ckpt_b16"]', 'CONFIGS_ORDER = ["fast_C_nogroup", "fast_C_grouped"]')
    s = s.replace("TOTAL_MAX_HOURS = 0.9 ", "TOTAL_MAX_HOURS = 0.45").replace("CFG_MAX_HOURS = 0.25", "CFG_MAX_HOURS = 0.12")
    s = s.replace("speedtest_T0_baseline.yaml", "speedtest_fast_C_grouped.yaml")
    s = s.replace("assert man[\"train_rows\"] >= 40 * 64", "assert man[\"train_rows\"] >= 50 * 64")
    s = s.replace('"NCCL_DEBUG": "WARN", **env_extra}', '"NCCL_DEBUG": "WARN", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **env_extra}')
    if "def parse_log" in s:
        s = sub(s, '    return {"per_rank_torch": ranks,',
                '''    def series(key): return [float(x) for x in re.findall(r"'" + key + r"': '?([-\\w.+]+)", txt)]
    steady = sps[4:]    # logging_steps=5 -> intervals ending at steps 25..50 (drops the first 20 steps: warm-up / allocator / cache effects)
    extra = {"loss_by_interval": series("loss"), "grad_norm_by_interval": series("grad_norm"), "cons_loss_by_interval": series("cons_loss"),
             "sec_per_step_steps21_50_mean": round(sum(steady) / len(steady), 3) if steady else None,
             "length_group_log": re.findall(r"length_group ON[^\\n]*|label-slot normaliser[^\\n]*", txt)[:2]}
    return {**extra, "per_rank_torch": ranks,''')
    c["source"] = s.splitlines(keepends=True)
    ast.parse(s.replace("%", "#%") if False else s)   # syntax check
out = "kaggle_speedtest_fast.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)

# docs/specs/

Plans and Kaggle job specs, written **before** each run. They are kept as written: a "Status: prepared only" line means
"as of writing", and estimates in them are estimates. What actually ran, and the results, are in
[`../TRAINING_RECIPE.md`](../TRAINING_RECIPE.md) and [`../../proofs/`](../../proofs/). Path placeholders
(`<artifacts>`, `<dev-tree>`, `<build-box>`) are explained in [`../PROVENANCE.md`](../PROVENANCE.md).

| File | Stage |
|---|---|
| `SMOKE_TEST.md`, `S0B_RERUN.md` | Stage 0 CPU smoke tests and the S0b resume rerun |
| `STAGE1_JOB_SPEC.md` | Stage 1 |
| `STAGE2_PLAN.md` | Stage 2 design and its go/no-go bars |
| `STAGE2_PILOT_JOB_SPEC.md`, `STAGE2_ABLATION_JOB_SPEC.md`, `STAGE2_FAST_SPEEDTEST_SPEC.md` | Stage 2 pilots, data ablations, speed test |
| `STAGE2_FULL_JOB_SPEC.md` | Stage 2 full run (arm C) |
| `STAGE3_PLAN.md`, `STAGE3_JOB_SPEC.md` | Stage 3 design, bars and job |

Note: `STAGE2_FAST_SPEEDTEST_SPEC.md` (length-grouped batching test) was prepared but never run; the speed test that ran is the T0–T3 one (`proofs/stage_summaries/s2_speedtest/`).

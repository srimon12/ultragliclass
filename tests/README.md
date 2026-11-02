# tests/

Script-style CPU checks (run with `python tests/<name>.py`, after `pip install -e ".[train,dev]"`). They need small
local data that is not committed:

| Test | Needs |
|---|---|
| `test_stage2.py` | `data/stage2_smoke/` (`python scripts/prepare_data.py --recipe configs/data_stage2_smoke.yaml --streaming`) |
| `test_length_group.py` | `data/stage2_pilot_smoke/` (`configs/data_stage2_pilot_smoke.yaml`) |
| `test_train_val_dedupe.py` | `calib/stage2_pilot_val.jsonl` (see `data/README.md`) |
| `test_length_group_e2e.sh` | `data/stage2_smoke/tiny_train22.jsonl` (22 rows of the stage-2 smoke data) |

All three Python tests printed `ALL PASS` on 2 Nov 2026 (IST) against the restructured layout. They are not
`pytest`-collectable yet (they read data at import time) — converting them is a launch TODO.

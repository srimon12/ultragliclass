# calib/

Eval and calibration slices. Only manifests are committed; the `.jsonl` files contain dataset text and are rebuilt
(see `data/README.md`). Expected md5s of the files the v0.1 runs used:

| file | rows | md5 | used for |
|---|---|---|---|
| `stage1_val.jsonl` | 4,300 | `cacbb98152c871dc57d602a35c0c5cd8` | global temperature fit (2,000 rows) in every eval |
| `stage2_pilot_val.jsonl` | 1,850 | `8e5e644ea992bd05503652d507d5591d` | S2/S3 val loss; S2 train/val dedupe |
| `jev_long_test.jsonl` | 431 | `cc664b35683a6703d4a8be15983fd5cb` | long-Jev@8192 eval; S3 train dedupe guard |

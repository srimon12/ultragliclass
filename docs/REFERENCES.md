# References (all opened or fetched on 22 Oct 2026; R-ids are cited in PLAN.md and configs/*.yaml)

## GLiClass (official recipe)
- **R1** Knowledgator GLiClass repo, `train.py` (training recipe, special tokens, optimizer groups, augmentation flags), main @ 6a00f80 (16 Oct 2026): https://github.com/Knowledgator/GLiClass/blob/main/train.py (local clone: ../gliclass_repo)
- **R2** `gliclass/data_processing.py` (GLiClassDataset, DataCollatorWithPadding, AugmentationConfig, data format): https://github.com/Knowledgator/GLiClass/blob/main/gliclass/data_processing.py
- R2a `gliclass/training.py` (Trainer with others_lr, EWC, batch_rebalance sampler, the exception-swallowing training_step): https://github.com/Knowledgator/GLiClass/blob/main/gliclass/training.py
- R2b `gliclass/model.py` (uni-encoder, scorers, focal loss, logit_scale) and `gliclass/pipeline.py` (inference format): https://github.com/Knowledgator/GLiClass/tree/main/gliclass
- R2c `train_rl.py` (PPO-style RL stage) and `calibrate.py` (calibrator): https://github.com/Knowledgator/GLiClass
- R2d Official fine-tuning notebook: https://github.com/Knowledgator/GLiClass/blob/main/notebooks/finetuning.ipynb
- **R5** gliclass 0.1.21 on PyPI: https://pypi.org/project/gliclass/0.1.21/
- **R6** gliclass-v4-edge card + config (uni-encoder, mlp scorer, first pooling, focal 0.75/−1, dropout 0.3, 3-stage training, typed decisions 49.3): https://huggingface.co/knowledgator/gliclass-v4-edge
- **R7** gliclass-modern-base-v3.0 card (zero-shot table, average 0.5577): https://huggingface.co/knowledgator/gliclass-modern-base-v3.0
- **R8** gliclass-v4-2b card (Jev / System One → GLiClass mapping): https://huggingface.co/knowledgator/gliclass-v4-2b
- **R9** `test_gliclass.py` (official zero-shot eval: datasets, single-label, macro-F1): https://github.com/Knowledgator/GLiClass/blob/main/test_gliclass.py
- R11 GLiClass paper: Stepanov et al., "GLiClass: Generalist Lightweight Model for Sequence Classification Tasks", arXiv 2508.07662: https://arxiv.org/abs/2508.07662
- R12 Other GLiClass cards used for settings (v3 LoRA, focal α 0.7; large/edge variants): https://huggingface.co/knowledgator/gliclass-modern-large-v3.0 , https://huggingface.co/knowledgator/gliclass-edge-v3.0 , https://huggingface.co/knowledgator/gliclass-instruct-large-v1.0
- R13 SCX Router paper cited by the v4 cards (decoder-KV GLiClass): arXiv 2609.02292: https://arxiv.org/abs/2609.02292

## UltraBERT (backbone)
- **R3** dmis-lab/UltraBERT-150M card (fine-tuning recipe: AdamW 0.9/0.98/1e-6, wd 0.1, linear 5% warmup, LR sweep, "up to 10× lower LR"; FA2/SDPA; bf16): https://huggingface.co/dmis-lab/UltraBERT-150M
- R3a UltraBERT-380M (later upgrade path): https://huggingface.co/dmis-lab/UltraBERT-380M
- R3b UltraBERT-Embed-150M: https://huggingface.co/dmis-lab/UltraBERT-Embed-150M
- **R4** dmis-lab/ultrabert-embed-training (official fine-tuning code: bf16, lr 5e-6, warmup 0.05, linear, max_grad_norm 1.0, β2 0.98, FA2): https://github.com/dmis-lab/ultrabert-embed-training
- **R10** Transformers UltraBERT docs (native since 5.20): https://huggingface.co/docs/transformers/model_doc/ultrabert
- R14 UltraBERT paper: "UltraBERT: Beyond the Limit of BERT Pre-Training", arXiv 2610.15028: https://arxiv.org/abs/2610.15028
- R15 dmis-lab blog post (14 Oct 2026): https://huggingface.co/blog/dmis-lab/ultrabert
- R16 Control backbone ModernBERT-base: https://huggingface.co/answerdotai/ModernBERT-base

## Data
- R17 https://huggingface.co/datasets/knowledgator/gliclass-v2.0
- R18 https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset
- R19 https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions
- **R20** LocalLLaMA/typed-decisions (benchmark, leaderboard, scorer `bench/score_with_jobs.py`; local copy in notes/typed_decisions_bench/): https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- R21 https://huggingface.co/datasets/tasksource/procedural-typed-decisions
- **R22** Reasoning Core: "Designing Broad Procedural Data for Completion-Supervised Reasoning Training", arXiv 2608.05148: https://arxiv.org/abs/2608.05148
- R23 Reserve sources: https://huggingface.co/datasets/knowledgator/gliclass-v2.0-RAC , https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1 , https://huggingface.co/datasets/tasksource/zero-shot-label-nli , https://huggingface.co/datasets/BioMike/formal-logic-reasoning-gliclass-2k
- R24 Eval sets: https://huggingface.co/datasets/mteb/banking77 , https://huggingface.co/datasets/PolyAI/banking77 , https://huggingface.co/datasets/clinc/clinc_oos , https://huggingface.co/datasets/mteb/amazon_massive_intent , https://huggingface.co/datasets/AmazonScience/massive , https://huggingface.co/datasets/benayas/snips , https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree , https://huggingface.co/datasets/takala/financial_phrasebank , https://huggingface.co/datasets/SetFit/CR , https://huggingface.co/datasets/SetFit/sst2 , https://huggingface.co/datasets/SetFit/sst5 , https://huggingface.co/datasets/SetFit/20_newsgroups , https://huggingface.co/datasets/SetFit/enron_spam , https://huggingface.co/datasets/stanfordnlp/imdb , https://huggingface.co/datasets/fancyzhx/ag_news , https://huggingface.co/datasets/dair-ai/emotion , https://huggingface.co/datasets/MoritzLaurer/cap_sotu , https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes

## Controls and context
- **R25** Laya (pip package and repo): https://pypi.org/project/laya/0.4.13/ , https://huggingface.co/convaiinnovations/laya , https://github.com/NandhaKishorM/laya
- **R26** laya-typed-decisions: https://huggingface.co/convaiinnovations/laya-typed-decisions
- R27 "Benchmarking System One decision models against trained classifiers and language models for automated decision gates", arXiv 2610.00346 (many-option collapse of encoder decision models): https://arxiv.org/abs/2610.00346
- R28 Guo et al. 2017, "On Calibration of Modern Neural Networks" (ECE, temperature scaling): https://arxiv.org/abs/1706.04599
- R29 Lin et al. 2017, "Focal Loss for Dense Object Detection": https://arxiv.org/abs/1708.02002

## Infrastructure
- **R30** Kaggle notebook limits (12 h GPU / 9 h TPU sessions, 20 GB /kaggle/working, T4×2 and P100 specs, 4 CPU cores, 29 GB RAM): https://www.kaggle.com/docs/notebooks ; GPU quota of 30 h/week: https://www.kaggle.com/docs/efficient-gpu-usage ; TPU 20 h/week: https://www.kaggle.com/docs/tpu
- R31 HF Trainer checkpointing / `resume_from_checkpoint`: https://huggingface.co/docs/transformers/main_classes/trainer
- R32 torchrun / DDP: https://pytorch.org/docs/stable/elastic/run.html , https://pytorch.org/docs/stable/notes/ddp.html
- R33 PyTorch/XLA (TPU path, static shapes): https://docs.pytorch.org/xla/
- R34 Hub API used for the license checks: https://huggingface.co/api/datasets/<id> ; dataset-viewer statistics: https://datasets-server.huggingface.co

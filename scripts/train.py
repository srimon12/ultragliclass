#!/usr/bin/env python
"""UltraGliClass staged trainer: GLiClass (uni-encoder) on dmis-lab/UltraBERT, resumable on Kaggle.

Built on the official `gliclass` package (GLiClassModel, GLiClassDataset, DataCollatorWithPadding,
gliclass.training.Trainer/TrainingArguments). Adds what Kaggle's 12 h sessions need:
  * fixed max_steps per stage (the LR schedule spans the whole stage, across sessions),
  * a wall-clock budget: save a checkpoint and exit cleanly before the session dies,
  * automatic resume from the newest checkpoint in output_dir (optimizer, scheduler, scaler, RNG, data order),
  * STATUS.json telling the notebook whether the stage is complete or needs another session,
  * fp16 (T4/P100) or bf16 (A100/TPU) switch, gradient accumulation, DDP via torchrun.

    python train.py --config configs/stage1_pilot.yaml                       # 1 GPU / CPU
    torchrun --nproc_per_node 2 train.py --config configs/stage1_pilot.yaml  # Kaggle 2x T4
    python train.py --config configs/stage0_smoke.yaml --set train.max_steps=20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# fail loudly instead of hanging: NCCL watchdog aborts the job when a collective exceeds ddp_timeout
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
import contextlib
import io

import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset  # noqa: E402
from gliclass.training import TrainingArguments  # noqa: E402

from ugc.config import load_config  # noqa: E402
from ugc.model_init import build_model, no_decay_norm_params  # noqa: E402
from ugc.many_label import LabelPools, ManyLabelDataset, PairedCollator  # noqa: E402
from ugc.length_group import (LengthGroupedSampler, LengthModel, calibrate_label_norm,  # noqa: E402
                              estimate_lengths, parse_length_group)
from ugc.trainer import StrictTrainer  # noqa: E402


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    """Rank-0-only progress line."""
    if is_main():
        print(f"[ugc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rlog(msg: str) -> None:
    """Rank-tagged line printed by EVERY rank (resume / barrier tracing only)."""
    print(f"[ugc {time.strftime('%H:%M:%S')} rank{os.environ.get('RANK', '0')}/{os.environ.get('WORLD_SIZE', '1')}] {msg}",
          flush=True)


def quiet_replica():
    """Silence stdout of non-zero ranks while building model/data (gliclass prints per process)."""
    return contextlib.nullcontext() if is_main() else contextlib.redirect_stdout(io.StringIO())


def init_dist(timeout_s: int) -> None:
    """Create the process group OURSELVES, early, with an explicit timeout, so every collective
    (incl. our barriers and HF's own) raises after timeout_s instead of hanging. HF reuses an
    existing group. NCCL gets an eager device_id so the communicator is built (or fails) here."""
    import datetime
    import torch.distributed as dist
    if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not dist.is_available():
        return
    if dist.is_initialized():
        rlog("process group was already initialized by someone else: timeout may NOT be applied")
        return
    t0 = time.time()
    kw = {"timeout": datetime.timedelta(seconds=timeout_s)}
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        kw.update(backend="nccl", device_id=torch.device("cuda", local))
    else:
        kw.update(backend="gloo")
    rlog(f"init_process_group({kw['backend']}, timeout={timeout_s}s) ...")
    dist.init_process_group(**kw)
    rlog(f"init_process_group done in {time.time() - t0:.1f}s")


def dist_barrier(tag: str) -> None:
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return
    rlog(f"barrier '{tag}': waiting")
    t0 = time.time()
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
    rlog(f"barrier '{tag}': passed in {time.time() - t0:.1f}s")


def describe_checkpoint(path: str | None) -> None:
    if not path:
        rlog("no checkpoint to resume from (fresh start)")
        return
    files = sorted(os.listdir(path))
    sizes = {f: os.path.getsize(os.path.join(path, f)) for f in files}
    rlog(f"resume checkpoint {path}: {len(files)} files, {sum(sizes.values()) / 2**30:.2f} GB, "
         f"rng files {[f for f in files if f.startswith('rng_state')]}, scaler={'scaler.pt' in files}")


def load_jsonl(paths) -> list[dict]:
    rows = []
    for path in [paths] if isinstance(paths, str) else paths:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


class TimeBudgetCallback(TrainerCallback):
    """Save + stop once the wall-clock budget is used, so a Kaggle session never dies mid-step.
    The budget is checked on every optimizer step; the check is broadcast so all DDP ranks agree."""

    def __init__(self, budget_seconds: float | None, start: float):
        self.budget = budget_seconds
        self.start = start
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.budget:
            return control
        over = torch.tensor([1.0 if time.time() - self.start > self.budget else 0.0])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.cuda.is_available():
                over = over.cuda()
            torch.distributed.all_reduce(over, op=torch.distributed.ReduceOp.MAX)
        if over.item() > 0 and not self.triggered:
            self.triggered = True
            log(f"time budget reached at step {state.global_step}: saving checkpoint and stopping")
            control.should_save = True
            control.should_training_stop = True
        return control


class ProgressCallback(TrainerCallback):
    """Seconds per optimizer step, ETA for the stage, and a NaN/inf guard (fp16 on T4 is untested
    for UltraBERT, whose pretraining used bf16)."""

    def __init__(self, nan_patience: int = 3):
        self.t_last = None
        self.s_last = None
        self.bad = 0
        self.nan_patience = nan_patience
        self.nan_stop = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        now = time.time()
        if self.t_last is not None and state.global_step > self.s_last:
            sps = (now - self.t_last) / (state.global_step - self.s_last)
            remaining = max(state.max_steps - state.global_step, 0) * sps
            logs["sec_per_step"] = round(sps, 3)
            logs["eta_hours"] = round(remaining / 3600, 2)
        if torch.cuda.is_available():
            logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            logs["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            rlog(f"mem step={state.global_step} peak_alloc_gb={logs['peak_mem_gb']} peak_reserved_gb={logs['peak_reserved_gb']}")  # every rank
        self.t_last, self.s_last = now, state.global_step
        loss = logs.get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self.bad += 1
            if self.bad >= self.nan_patience:
                log(f"loss is {loss} for {self.bad} logs in a row: stopping (switch LR / precision)")
                self.nan_stop = True
                control.should_training_stop = True
        else:
            self.bad = 0
        return control


class ResumeTraceCallback(TrainerCallback):
    """Rank-tagged lines at train begin and around the first optimizer step of this session."""

    def __init__(self):
        self.first_begin = True
        self.first_end = True
        self.t = None

    def on_train_begin(self, args, state, control, **kwargs):
        rlog(f"train_begin: global_step={state.global_step} max_steps={state.max_steps} "
             f"(checkpoint state loaded; skipping already-seen batches next)")
        self.t = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.first_begin:
            self.first_begin = False
            rlog(f"first step of this session begins at global_step={state.global_step} "
                 f"({time.time() - self.t:.1f}s after train_begin)")

    def on_step_end(self, args, state, control, **kwargs):
        if self.first_end:
            self.first_end = False
            rlog(f"first step of this session done: global_step={state.global_step} "
                 f"({time.time() - self.t:.1f}s after train_begin)")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:  # heartbeat for the notebook watchdog / post-mortem
            hb = os.path.join(args.output_dir, "HEARTBEAT.json")
            with open(hb, "w") as f:
                json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "global_step": state.global_step,
                           "logs": logs or {}}, f)


class StaticShapeCollator:
    """TPU/XLA portability: pad every batch to max_length tokens and a fixed label count so XLA compiles
    once. Wraps the official collator; label slots beyond a row's labels are padded with 0 and masked by
    GLiClass's class-token mask exactly as with dynamic padding."""

    def __init__(self, max_length: int, max_labels: int):
        self.base = DataCollatorWithPadding()
        self.max_length = max_length
        self.max_labels = max_labels

    def __call__(self, batch):
        out = self.base(batch)
        for key in ("input_ids", "attention_mask"):
            t = out[key]
            out[key] = torch.nn.functional.pad(t, (0, self.max_length - t.shape[1]), value=0)
        lab = out["labels"]
        if lab.dim() == 2:
            out["labels"] = torch.nn.functional.pad(lab, (0, self.max_labels - lab.shape[1]), value=0)
        out["max_num_classes"] = self.max_labels
        return out


class IndexLogger(torch.utils.data.Dataset):
    """Test hook (env UGC_DUMP_ORDER=<file prefix>): appends every fetched row index to <prefix>.<pid>; used to prove that
    arms A/B/C see the same base data order (scripts/check_pilot_arms.py). Not used in real runs."""

    def __init__(self, ds, prefix):
        self.ds, self.prefix = ds, prefix

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        with open(f"{self.prefix}.{os.getpid()}", "a") as f:
            f.write(f"{int(i)}\n")
        return self.ds[i]


def write_status(path: str, status: dict) -> None:
    if is_main():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(status, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.max_steps=10")
    ap.add_argument("--resume", default="auto", help="'auto' (newest checkpoint in output_dir), 'none', or a path")
    args = ap.parse_args()
    t_start = time.time()
    cfg = load_config(args.config, args.set)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    out_dir = tcfg["output_dir"]
    status_path = os.path.join(out_dir, "STATUS.json")

    seed = int(tcfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------- precision ----------------
    precision = tcfg.get("precision", "fp16")
    cuda = torch.cuda.is_available()
    if precision == "fp16" and not cuda:
        log("fp16 needs CUDA; falling back to fp32 on this machine")
        precision = "fp32"
    if precision == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 requested but this GPU has no bf16 (T4/P100): use precision: fp16")

    # ---------------- model ----------------
    rlog(f"process up: pid={os.getpid()} local_rank={os.environ.get('LOCAL_RANK', '0')} "
         f"cuda={torch.cuda.is_available()} visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    init_dist(int(tcfg.get("ddp_timeout", 600)))
    with quiet_replica():
        model, tokenizer = build_model(mcfg)
    no_decay = no_decay_norm_params(model)
    model.config.problem_type = mcfg.get("problem_type", "multi_label_classification")
    enc = model.model.encoder_model
    log(f"backbone {type(enc).__name__}, attn={getattr(enc.config, '_attn_implementation', '?')}, "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, "
        f"norm params without decay={len(no_decay)}, transformers {transformers.__version__}")

    # ---------------- data ----------------
    train_rows = load_jsonl(dcfg["train_files"])
    val_rows = load_jsonl(dcfg["val_files"])[: dcfg.get("max_eval_samples", 2000)] if dcfg.get("val_files") else []
    aug = dcfg.get("augmentation", {}) or {}
    augment_config = AugmentationConfig(
        enabled=aug.get("enabled", False),
        random_label_removal_prob=aug.get("random_label_removal_prob", 0.05),
        random_label_addition_prob=aug.get("random_label_addition_prob", 0.05),
        random_text_addition_prob=aug.get("random_text_addition_prob", 0.05),
        random_add_description_prob=aug.get("random_add_description_prob", 0.05),
        random_add_synonyms_prob=aug.get("random_add_synonyms_prob", 0.05),
        random_add_examples_prob=aug.get("random_add_examples_prob", 0.1),
        max_num_examples=aug.get("max_num_examples", 5),
    )
    common = dict(max_length=dcfg["max_length"], problem_type=model.config.problem_type,
                  architecture_type=model.config.architecture_type, prompt_first=model.config.prompt_first,
                  labels_tokenizer=None, shuffle_labels=mcfg.get("shuffle_labels", True),
                  max_labels=int(mcfg.get("max_num_classes", 50)))
    ml_cfg = dcfg.get("many_label") or {}
    cons_cfg = dcfg.get("consistency") or {}
    use_ml = bool(ml_cfg.get("enabled"))
    pair_prob = float(cons_cfg.get("pair_prob", 0.0)) if cons_cfg.get("enabled") else 0.0
    if (use_ml or pair_prob > 0) and dcfg.get("static_shapes"):
        raise SystemExit("data.many_label / data.consistency are not supported with static_shapes (TPU) yet")
    with quiet_replica():
        if use_ml or pair_prob > 0:
            pools = None
            if use_ml:
                pools = LabelPools.from_file(ml_cfg["pools_file"], top_m=int(ml_cfg.get("top_m", 48)))
                for pid in pools.pools:  # build neighbour tables once, before DataLoader workers fork
                    pools.neighbours(pid)
                log("label pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.pools.items()))
            train_ds = ManyLabelDataset(train_rows, tokenizer, augment_config, {}, pools=pools,
                                        many_label=ml_cfg.get("params"), pair_prob=pair_prob, seed=seed, **common)
        else:
            train_ds = GLiClassDataset(train_rows, tokenizer, augment_config, {}, **common)
    # validation: no augmentation and a fixed label order, so eval_loss is comparable across sessions
    with quiet_replica():
        val_ds = (GLiClassDataset(val_rows, tokenizer, AugmentationConfig(enabled=False), {},
                                  **{**common, "shuffle_labels": False}) if val_rows else None)
    # ---------------- optional length-grouped batching (data.length_group; default OFF = random batches as before) ----------
    lg = parse_length_group(dcfg)
    train_sampler, label_norm = None, None
    if lg:
        if dcfg.get("static_shapes"):
            raise SystemExit("data.length_group is not compatible with static_shapes")
        world_ = int(os.environ.get("WORLD_SIZE", "1"))
        mb = int(tcfg["per_device_train_batch_size"])
        cache_dir = lg["cache_dir"] or os.path.join(out_dir, "length_cache")
        if is_main():                      # one rank tokenises (~1 min) and writes the cache; the others wait, then load it
            estimate_lengths(train_rows, tokenizer, dcfg["max_length"], cache_dir, log)
        dist_barrier("length-cache")
        base_len = estimate_lengths(train_rows, tokenizer, dcfg["max_length"], cache_dir, log if is_main() else (lambda *_: None))
        planning = isinstance(train_ds, ManyLabelDataset)
        if planning:
            train_ds.enable_planning()     # fetches seeded from (seed, epoch, row): the sampler can predict expanded lengths
        lm = LengthModel(base_len, train_rows if (planning and use_ml) else None, tokenizer, pools if use_ml else None,
                         ml_cfg.get("params"), dcfg["max_length"], seed)
        train_sampler = LengthGroupedSampler(lm.lengths, mb, world_, lg["mega_batch_mult"], seed=seed,
                                             encode_epoch=planning, n_total=len(train_rows))
        if lg["loss_norm"]:
            label_norm = calibrate_label_norm(train_ds, mb, seed=seed, log=log)
        log(f"length_group ON: micro-batch {mb} x world {world_}, mega-batch {train_sampler.chunk} rows, "
            f"{len(train_sampler)} rows/epoch = {len(train_sampler) // (mb * world_)} micro-batches per rank, "
            f"expansion planning={planning}, loss_norm={label_norm}")
    if os.environ.get("UGC_DUMP_ORDER"):
        train_ds = IndexLogger(train_ds, os.environ["UGC_DUMP_ORDER"])
    if dcfg.get("static_shapes"):
        collator = StaticShapeCollator(dcfg["max_length"], int(mcfg.get("max_num_classes", 50)))
    else:
        collator = DataCollatorWithPadding()
    if pair_prob > 0:
        collator = PairedCollator(collator)
    log(f"train rows {len(train_rows)}, val rows {len(val_rows)}, max_length {dcfg['max_length']}")

    # ---------------- schedule ----------------
    world = int(os.environ.get("WORLD_SIZE", "1"))
    eff_batch = tcfg["per_device_train_batch_size"] * tcfg.get("gradient_accumulation_steps", 1) * world
    max_steps = tcfg.get("max_steps")
    if not max_steps:  # derive from epochs so every session shares one fixed schedule
        max_steps = math.ceil(len(train_rows) * float(tcfg.get("num_epochs", 1)) / eff_batch)
    log(f"world={world}, effective batch={eff_batch}, max_steps={max_steps} "
        f"(= {max_steps * eff_batch / max(len(train_rows), 1):.2f} epochs)")

    targs = TrainingArguments(
        output_dir=out_dir,
        max_steps=int(max_steps),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(tcfg["encoder_lr"]),
        weight_decay=float(tcfg.get("encoder_weight_decay", 0.1)),
        others_lr=float(tcfg["others_lr"]),
        others_weight_decay=float(tcfg.get("others_weight_decay", 0.01)),
        adam_beta1=float(tcfg.get("adam_beta1", 0.9)),
        adam_beta2=float(tcfg.get("adam_beta2", 0.98)),
        adam_epsilon=float(tcfg.get("adam_epsilon", 1e-6)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "linear"),
        warmup_steps=float(tcfg.get("warmup_ratio", 0.05)),  # transformers v5: float < 1 is a ratio
        optim=tcfg.get("optim", "adamw_torch"),
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        use_cpu=not cuda,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        save_strategy="steps",
        save_steps=int(tcfg.get("save_steps", 500)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        save_only_model=False,  # optimizer/scheduler/scaler/RNG are needed to resume
        save_optimizer_state=True,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=int(tcfg.get("eval_steps", tcfg.get("save_steps", 500))),
        prediction_loss_only=True,  # real metrics come from eval.py
        logging_steps=int(tcfg.get("logging_steps", 50)),
        logging_first_step=True,
        dataloader_num_workers=int(tcfg.get("num_workers", 2)),
        dataloader_drop_last=bool(tcfg.get("drop_last", True)),
        ddp_find_unused_parameters=bool(tcfg.get("ddp_find_unused_parameters", False)),
        seed=seed,
        data_seed=seed,
        report_to="none",
        ddp_timeout=int(tcfg.get("ddp_timeout", 600)),  # 10 min: a stuck collective raises instead of hanging
        log_level_replica="error",
        disable_tqdm=bool(tcfg.get("disable_tqdm", True)),
    )

    budget_h = tcfg.get("time_budget_hours")
    budget_cb = TimeBudgetCallback(float(budget_h) * 3600 if budget_h else None, t_start)
    progress_cb = ProgressCallback(int(tcfg.get("nan_patience", 3)))
    trainer = StrictTrainer(max_oom_skips=int(tcfg.get("max_oom_skips", 0)), train_sampler=train_sampler, label_norm=label_norm, model=model, args=targs,
                            train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
                            processing_class=tokenizer, callbacks=[budget_cb, progress_cb, ResumeTraceCallback()])
    if pair_prob > 0:
        trainer.consistency_weight = float(cons_cfg.get("weight", 0.1))
        trainer.consistency_kind = cons_cfg.get("kind", "skl")
        log(f"consistency: pair_prob={pair_prob} weight={trainer.consistency_weight} kind={trainer.consistency_kind}")
    rlog(f"trainer built: device={targs.device} world_size={targs.world_size} ddp_timeout={targs.ddp_timeout}s")

    resume = None
    if args.resume == "auto":
        resume = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    elif args.resume not in ("none", ""):
        resume = args.resume
    log(f"resume_from_checkpoint={resume}")
    describe_checkpoint(resume)
    if is_main():  # fail fast (before any collective) if the restored output dir is not writable
        os.makedirs(out_dir, exist_ok=True)
        probe = os.path.join(out_dir, ".write_probe")
        try:
            open(probe, "w").close()
            os.remove(probe)
        except OSError as e:
            raise SystemExit(f"output_dir {out_dir} is not writable ({e}); copy runs/ out of /kaggle/input "
                             f"with plain file copies and chmod -R u+w") from e
    write_status(status_path, {"state": "running", "resume_from": resume, "max_steps": int(max_steps),
                               "config": os.path.abspath(args.config), "overrides": args.set})
    if os.environ.get("UGC_DEBUG_HANG_RANK") is not None and os.environ.get("UGC_DEBUG_HANG_RANK") == os.environ.get("RANK", "0"):  # test hook: simulate a stuck rank
        rlog("UGC_DEBUG_HANG_RANK set: this rank now sleeps to simulate a hang")
        time.sleep(10**6)
    dist_barrier("before-train")

    result = trainer.train(resume_from_checkpoint=resume)
    rlog(f"train() returned at global_step={trainer.state.global_step}")
    step = trainer.state.global_step
    complete = step >= int(max_steps) and not progress_cb.nan_stop
    status = {"state": "complete" if complete else ("nan_stop" if progress_cb.nan_stop else "needs_resume"),
              "global_step": step, "max_steps": int(max_steps), "resumed_from": resume,
              "train_loss": result.training_loss, "session_seconds": round(time.time() - t_start, 1),
              "last_checkpoint": get_last_checkpoint(out_dir), "precision": precision, "world_size": world}
    if complete:
        final_dir = os.path.join(out_dir, "final_model")
        if is_main():
            trainer.model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
        status["final_model"] = final_dir
        if val_ds is not None:
            status["final_eval"] = trainer.evaluate()
    write_status(status_path, status)
    log(f"STATUS {json.dumps(status)}")
    dist_barrier("after-status")
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

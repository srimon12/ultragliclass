"""gliclass.training.Trainer with a strict training_step.

The official Trainer.training_step wraps forward/backward in `except Exception: print(...); return 0`.
Our CPU DDP smoke test showed why that is dangerous: a DDP error was printed and every later step
silently trained on nothing (loss 0, grad_norm 0). Here errors raise, except CUDA OOM, which may be
skipped up to `max_oom_skips` times in single-process runs only (in DDP a skipped rank would deadlock)."""
from __future__ import annotations

import torch
from gliclass.training import Trainer


class StrictTrainer(Trainer):
    def __init__(self, *args, max_oom_skips: int = 0, train_sampler=None, label_norm: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_oom_skips = max_oom_skips
        self.oom_skips = 0
        # length-grouped batching (ugc.length_group; both None = the original random batching, unchanged behaviour)
        self.train_sampler = train_sampler
        self.label_norm = label_norm

    # ---- length-grouped batching ------------------------------------------------------------------------------
    def _get_train_sampler(self, train_dataset=None):
        if self.train_sampler is None:
            return super()._get_train_sampler(train_dataset)
        if self.args.world_size != self.train_sampler.world or self.args.train_batch_size != self.train_sampler.mb or self.args.n_gpu > 1:
            raise RuntimeError(f"length-grouped sampler was built for world={self.train_sampler.world}, micro-batch={self.train_sampler.mb}; "
                               f"trainer has world={self.args.world_size}, batch={self.args.train_batch_size}, n_gpu={self.args.n_gpu}")
        return self.train_sampler

    def _run_epoch(self, *args, **kwargs):
        # the trainer's own epoch number selects the sampler's epoch (exact, also for the first epoch after a resume)
        if self.train_sampler is not None and "epoch" in kwargs:
            self.train_sampler.set_epoch(int(kwargs["epoch"]))
        return super()._run_epoch(*args, **kwargs)

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        inputs.pop("labels_text", None)
        inputs.pop("input_texts", None)
        inputs = self._prepare_inputs(inputs)
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            del inputs
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps
        except torch.OutOfMemoryError:
            if self.args.world_size > 1 or self.oom_skips >= self.max_oom_skips:
                raise
            self.oom_skips += 1
            print(f"[ugc] CUDA OOM, skipping batch ({self.oom_skips}/{self.max_oom_skips})", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.args.device)

    # ---- stage 2: permutation-consistency regulariser (paired views are stacked into one forward) ----
    consistency_weight: float = 0.0
    consistency_kind: str = "skl"

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pair_idx = inputs.pop("pair_idx", None)
        pair_perm = inputs.pop("pair_perm", None)
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.label_norm and model.training and labels is not None and getattr(outputs, "class_mask", None) is not None:
            # The model's loss is (sum over valid label slots) / (valid slots in THIS micro-batch). With random batches every
            # micro-batch has about the same number of slots, so every label slot has the same weight. Length-grouped
            # micro-batches differ a lot (many-label rows are long), which would up-weight the slots of few-label rows.
            # Dividing by a FIXED expected slot count instead restores equal weight per label slot (same expectation as random batches).
            c = outputs.class_mask.shape[-1]
            valid = (labels[:, :c] != -100) & outputs.class_mask.bool() if labels.dim() > 1 else None
            if valid is not None:
                loss = loss * valid.sum().clamp_min(1).to(loss.dtype) / self.label_norm
        if pair_idx is not None and pair_idx.numel() and self.consistency_weight > 0:
            from ugc.many_label import consistency_loss
            cons = consistency_loss(outputs.logits, outputs.class_mask, pair_idx, pair_perm, self.consistency_kind)
            self._cons_log = getattr(self, "_cons_log", []) + [float(cons.detach())]
            self._pairs_log = getattr(self, "_pairs_log", 0) + int(pair_idx.shape[0])
            loss = loss + self.consistency_weight * cons
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        cons = getattr(self, "_cons_log", None)
        if cons and "loss" in logs:
            logs["cons_loss"] = round(sum(cons) / len(cons), 5)
            logs["pairs_since_last_log"] = self._pairs_log
            self._cons_log, self._pairs_log = [], 0
        return super().log(logs, *args, **kwargs)

    def _load_scaler(self, checkpoint):
        # A checkpoint written under fp16 carries scaler.pt; resuming it with bf16/fp32 (e.g. the CPU repro or
        # a precision switch) has no GradScaler, and transformers then crashes with AttributeError. Skip instead.
        if checkpoint and getattr(self.accelerator, "scaler", None) is None:
            print(f"[ugc rank{self.args.process_index}] no GradScaler in this run: skipping scaler.pt", flush=True)
            return
        return super()._load_scaler(checkpoint)

    # ---- rank-tagged tracing of every resume step (S0b hung with an empty log) ----
    def _trace(self, what, fn, *a, **k):
        import os
        import time
        tag = f"[ugc {time.strftime('%H:%M:%S')} rank{self.args.process_index}/{self.args.world_size}]"
        print(f"{tag} resume: {what} ...", flush=True)
        t0 = time.time()
        out = fn(*a, **k)
        print(f"{tag} resume: {what} done in {time.time() - t0:.1f}s (device {self.args.device}, pid {os.getpid()})",
              flush=True)
        return out

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        return self._trace(f"load model weights from {resume_from_checkpoint}",
                           super()._load_from_checkpoint, resume_from_checkpoint, model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if not checkpoint:
            return super()._load_optimizer_and_scheduler(checkpoint)
        return self._trace("load optimizer + scheduler", super()._load_optimizer_and_scheduler, checkpoint)

    def _load_rng_state(self, checkpoint):
        if not checkpoint:
            return super()._load_rng_state(checkpoint)
        return self._trace("restore RNG state", super()._load_rng_state, checkpoint)

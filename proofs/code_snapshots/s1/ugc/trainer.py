"""gliclass.training.Trainer with a strict training_step.

The official Trainer.training_step wraps forward/backward in `except Exception: print(...); return 0`.
Our CPU DDP smoke test showed why that is dangerous: a DDP error was printed and every later step
silently trained on nothing (loss 0, grad_norm 0). Here errors raise, except CUDA OOM, which may be
skipped up to `max_oom_skips` times in single-process runs only (in DDP a skipped rank would deadlock)."""
from __future__ import annotations

import torch
from gliclass.training import Trainer


class StrictTrainer(Trainer):
    def __init__(self, *args, max_oom_skips: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_oom_skips = max_oom_skips
        self.oom_skips = 0

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

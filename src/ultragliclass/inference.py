"""Load an UltraGliClass checkpoint and score labels.

The released v0.1 config still carries training-time defaults (`max_num_classes: 50`, tokenizer `max_length: 1024`).
v0.1 was trained up to 8,192 tokens and up to 256 labels, so this wrapper passes those limits explicitly.
"""
from __future__ import annotations

from typing import Iterable

TRAINED_MAX_LENGTH = 8192   # Stage 3 context (configs/stage3_longctx.yaml)
TRAINED_MAX_LABELS = 256    # Stage 3 model.max_num_classes


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_pipeline(model: str, *, device: str | None = None, max_length: int = TRAINED_MAX_LENGTH,
                  classification_type: str = "single-label", progress_bar: bool = False):
    """Return a `gliclass.ZeroShotClassificationPipeline` for a local path or Hub id.

    UltraGliClass was trained with `add_prefix_space=False` (the Knowledgator models use True).
    """
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    mdl = GLiClassModel.from_pretrained(model)
    tok = AutoTokenizer.from_pretrained(model, add_prefix_space=False)
    return ZeroShotClassificationPipeline(
        mdl, tok, max_classes=TRAINED_MAX_LABELS, max_length=max_length,
        classification_type=classification_type, device=_device(device), progress_bar=progress_bar,
    )


def predict(pipe, texts: str | list[str], labels: Iterable[str], *, threshold: float = 0.5,
            prompt: str | None = None, batch_size: int = 8):
    """Score `labels` for each text. Returns gliclass's list of [{"label", "score"}, ...] per text."""
    return pipe(texts, list(labels), threshold=threshold, prompt=prompt, batch_size=batch_size)

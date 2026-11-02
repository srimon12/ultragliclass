"""UltraGliClass: GLiClass uni-encoder on dmis-lab/UltraBERT-150M.

Public inference API (thin wrapper around the `gliclass` package). Training internals live in the
separate `ugc` package, kept byte-identical to the code that trained v0.1 (see docs/PROVENANCE.md).
"""
from ultragliclass.inference import TRAINED_MAX_LABELS, TRAINED_MAX_LENGTH, load_pipeline, predict

__all__ = ["load_pipeline", "predict", "TRAINED_MAX_LENGTH", "TRAINED_MAX_LABELS"]
__version__ = "0.1.0"

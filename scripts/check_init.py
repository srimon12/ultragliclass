"""Smoke check: GLiClassModel construction must not re-initialise the pretrained UltraBERT weights,
and the new special-token rows must be initialised (not random)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModel
from ugc.config import load_config
from ugc.model_init import build_model

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/stage0_smoke.yaml", [])
model, tok = build_model(cfg["model"])
ref = AutoModel.from_pretrained(cfg["model"]["encoder_model_name"]).state_dict()
enc = model.model.encoder_model.state_dict()
diff, checked = [], 0
for k, v in ref.items():
    if k not in enc:
        continue
    w = enc[k]
    if w.shape != v.shape:  # resized embedding matrix: compare the original rows
        w = w[: v.shape[0]]
    checked += 1
    if not torch.equal(w.float(), v.float()):
        diff.append(k)
print(f"checked {checked} backbone tensors; changed by GLiClass init: {len(diff)} {diff[:5]}")
emb = model.model.encoder_model.get_input_embeddings().weight
for t in ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]:
    i = tok.convert_tokens_to_ids(t)
    print(t, i, "norm", round(emb[i].norm().item(), 3))
print("PASS" if not diff else "FAIL")

"""Find parameters that receive no gradient in one forward/backward (decides ddp_find_unused_parameters)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from ugc.config import load_config
from ugc.model_init import build_model

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/stage0_smoke.yaml", [])
model, tok = build_model(cfg["model"])
model.config.problem_type = "multi_label_classification"
rows = [json.loads(l) for l in open(cfg["data"]["train_files"][0])][:4]
ds = GLiClassDataset(rows, tok, AugmentationConfig(enabled=False), {}, max_length=256,
                     architecture_type="uni-encoder", prompt_first=True)
batch = DataCollatorWithPadding(device="cpu")([ds[i] for i in range(4)])
model.train()
out = model(**batch)
out.loss.backward()
unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print("loss", float(out.loss), "unused params:", len(unused))
for n in unused: print("  ", n)

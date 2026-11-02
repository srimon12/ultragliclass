"""`ultragliclass-predict --model PATH --text "..." --labels a b c [--prompt ...] [--multi-label]`"""
from __future__ import annotations

import argparse
import json
import sys

from ultragliclass.inference import TRAINED_MAX_LENGTH, load_pipeline, predict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ultragliclass-predict", description=__doc__)
    ap.add_argument("--model", required=True, help="local final_model dir or Hub id")
    ap.add_argument("--text", help="input text; reads stdin when omitted")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--prompt", default=None, help="optional instruction, placed before the text")
    ap.add_argument("--multi-label", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.0, help="min score to print (default: print all)")
    ap.add_argument("--max-length", type=int, default=TRAINED_MAX_LENGTH)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    text = a.text if a.text is not None else sys.stdin.read()
    pipe = load_pipeline(a.model, device=a.device, max_length=a.max_length,
                         classification_type="multi-label" if a.multi_label else "single-label")
    out = predict(pipe, text, a.labels, threshold=a.threshold, prompt=a.prompt)[0]
    print(json.dumps(sorted(out, key=lambda r: -r["score"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

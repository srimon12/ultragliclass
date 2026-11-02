"""Convert source rows into the official GLiClass training format.

GLiClass format (Knowledgator/GLiClass README, "How to Train"):
    {"text": str, "all_labels": [str, ...], "true_labels": [str, ...] | {label: prob}}
plus the optional "prompt" key that gliclass.data_processing.GLiClassDataset places after <<SEP>>.

System One rows (tasksource-jev-typed-decisions) follow the same mapping knowledgator documents for
serving a Jev-style API with GLiClass (gliclass-v4 card, "Serving a Jev-style decision API"):
the state is the text, the question is the prompt, every option is a label; `noul` uses the labels
"yes" / "no". Soft targets are kept as {label: probability} dicts, which GLiClass supports natively.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

NOUL_LABELS = ("yes", "no")


def text_hash(text: str, labels: Iterable[str]) -> str:
    h = hashlib.sha1()
    h.update(text.strip().lower().encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update("|".join(sorted(labels)).encode("utf-8", "ignore"))
    return h.hexdigest()


def _clean_labels(labels: Iterable[Any]) -> list[str]:
    return [str(lbl).strip() for lbl in labels]


def from_gliclass(row: dict, source: str, max_options: int, max_chars: int) -> dict | None:
    text = row.get("text")
    all_labels = _clean_labels(row.get("all_labels") or [])
    true_labels = row.get("true_labels") or []
    if not text or not all_labels:
        return None
    if len(all_labels) > max_options or len(set(all_labels)) != len(all_labels) or any(not l for l in all_labels):
        return None
    if isinstance(true_labels, dict):
        true = {str(k).strip(): float(v) for k, v in true_labels.items() if str(k).strip() in all_labels}
    else:
        true = [str(t).strip() for t in true_labels if str(t).strip() in all_labels]
    out = {"text": str(text)[:max_chars], "all_labels": all_labels, "true_labels": true,
           "source": source, "kind": "gliclass"}
    if row.get("prompt"):
        out["prompt"] = str(row["prompt"])
    return out


def from_jev(row: dict, source: str, max_options: int, max_chars: int,
             max_state_chars: int | None = None) -> dict | None:
    """tasksource-jev-typed-decisions row -> GLiClass row (soft targets)."""
    state, question, kind = row.get("state"), row.get("question") or "", row.get("kind")
    target = list(row.get("target") or [])
    if not state or kind not in {"choice", "score", "noul"}:
        return None
    if max_state_chars is not None and len(state) > max_state_chars:
        return None
    if kind == "noul":
        if len(target) != 1:
            return None
        p = min(max(float(target[0]), 0.0), 1.0)
        labels = list(NOUL_LABELS)
        true = {"yes": p, "no": 1.0 - p}
    else:
        labels = _clean_labels(row.get("options") or [])
        if len(labels) < 2 or len(labels) != len(target) or len(labels) > max_options:
            return None
        if len(set(labels)) != len(labels) or any(not l for l in labels):
            return None
        true = {lbl: float(p) for lbl, p in zip(labels, target) if float(p) > 0}
        if not true:
            return None
    return {
        "text": str(state)[:max_chars],
        "prompt": str(question),
        "all_labels": labels,
        "true_labels": true,
        "source": f"{source}:{row.get('source', '')}",
        "kind": f"jev-{kind}",
        "license_use": row.get("license_use"),
    }


def render_state(state: Any) -> str:
    """Render a JSON state as `key: value` lines (string lists as `- item` lines), as the GLiClass v4
    cards describe for Jev-style serving. Plain strings pass through."""
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
        except (ValueError, TypeError):
            return state
        if not isinstance(parsed, (dict, list)):
            return state
        state = parsed
    lines: list[str] = []

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(val, (dict, list)) and val and not (
                    isinstance(val, list) and all(isinstance(v, str) for v in val)
                ):
                    walk(val, name)
                elif isinstance(val, list):
                    lines.append(f"{name}:")
                    lines.extend(f"- {v}" for v in val)
                else:
                    lines.append(f"{name}: {val}")
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                walk(val, f"{prefix}[{i}]") if isinstance(val, (dict, list)) else lines.append(f"- {val}")

    walk(state)
    return "\n".join(lines)


def system_one_options(question: dict) -> tuple[list[str], list[str]]:
    """Return (option_keys, label_texts) for a System One question, using the GLiClass-v4 mapping:
    choice -> option descriptions (key when missing); noul -> true/false descriptions (yes/no when
    missing), keys ("true", "false"); score -> level descriptions in order, keys "0".."n-1"."""
    qtype = question.get("type")
    crit = question.get("criteria")
    if qtype == "choice":
        if isinstance(crit, dict):
            keys = [str(k) for k in crit]
            texts = [str(v) if v else str(k) for k, v in crit.items()]
        else:
            keys = [str(c) for c in crit or []]
            texts = list(keys)
        # GLiClass labels must be unique strings; keep the key when two descriptions collide
        if len(set(texts)) != len(texts):
            texts = [f"{k}: {t}" if t != k else k for k, t in zip(keys, texts)]
        return keys, texts
    if qtype == "noul":
        crit = crit if isinstance(crit, dict) else {}
        return ["true", "false"], [str(crit.get("true") or "yes"), str(crit.get("false") or "no")]
    if qtype == "score":
        levels = crit if isinstance(crit, list) else list((crit or {}).values())
        return [str(i) for i in range(len(levels))], [str(l) for l in levels]
    raise ValueError(f"unknown question type {qtype!r}")


def from_system_one(state: Any, questions: Any, answers: Any, source: str, max_options: int,
                    max_chars: int) -> list[dict]:
    """A (state, {name: question}, {name: answer}) request -> one GLiClass row per question."""
    questions = json.loads(questions) if isinstance(questions, str) else questions
    answers = json.loads(answers) if isinstance(answers, str) else (answers or {})
    text = render_state(state)[:max_chars]
    rows = []
    for name, q in questions.items():
        try:
            keys, labels = system_one_options(q)
        except ValueError:
            continue
        if len(labels) < 2 or len(labels) > max_options or len(set(labels)) != len(labels):
            continue
        ans = answers.get(name) or {}
        probs = ans.get("probabilities")
        if probs is None and q.get("type") == "noul" and "noul" in ans:
            probs = {"true": ans["noul"], "false": 1 - ans["noul"]}
        if not probs:
            continue
        true = {lbl: float(probs.get(k, 0.0)) for k, lbl in zip(keys, labels) if float(probs.get(k, 0.0)) > 0}
        if not true:
            continue
        rows.append({"text": text, "prompt": str(q.get("instructions") or ""), "all_labels": labels,
                     "true_labels": true, "source": source, "kind": f"s1-{q.get('type')}"})
    return rows

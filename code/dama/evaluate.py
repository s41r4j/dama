"""Model-level metrics on labelled examples (no downstream QA or cost claims)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .contracts import OPERATIONS


def mean(xs):
    return float(np.mean(xs)) if xs else None


def f1(p, r):
    return 2 * p * r / (p + r) if p is not None and r is not None and p + r else 0.0


def span_f1(pred, gold):
    if pred is None:
        return 0.0
    overlap = max(0, min(pred.end, gold.end) - max(pred.start, gold.start))
    return f1(overlap / (pred.end - pred.start), overlap / (gold.end - gold.start))


def calibration_error(confidence, correct, bins=10):
    if not confidence:
        return None
    confidence, correct = np.array(confidence), np.array(correct, dtype=float)
    which = np.minimum((confidence * bins).astype(int), bins - 1)
    return float(sum(abs(confidence[which == b].mean() - correct[which == b].mean()) * (which == b).mean() for b in range(bins) if (which == b).any()))


def evaluate(predictor, examples, output=None, warmup=3):
    for row in examples[:warmup]:
        predictor.predict(row.input)
    m = {k: [] for k in ["op", "type", "target", "span_em", "span_f1", "precision", "recall", "fallback", "conf", "raw_ok", "latency", "tokens"]}
    confusion = {g: {p: 0 for p in OPERATIONS} for g in OPERATIONS}
    writes_made = writes_right = useful = missed = 0
    rows = []
    for row in examples:
        d, timing = predictor.predict(row.input)
        y = row.label
        m["latency"].append(timing["seconds"]); m["tokens"].append(timing["input_tokens"]); m["fallback"].append(d.needs_fallback == y.needs_fallback)
        if y.operation is not None:
            confusion[y.operation][d.operation] += 1
            m["op"].append(d.operation == y.operation)
            if d.operation_scores:
                raw = max(d.operation_scores, key=d.operation_scores.get)
                m["conf"].append(d.operation_scores[raw]); m["raw_ok"].append(raw == y.operation)
            if y.memory_type is not None:
                m["type"].append(d.memory_type == y.memory_type)
            if y.target_id is not None:
                m["target"].append(d.target_id == y.target_id)
            if y.evidence_span is not None:
                m["span_em"].append(d.evidence_span == y.evidence_span); m["span_f1"].append(span_f1(d.evidence_span, y.evidence_span))
            if d.operation != "NOOP":
                writes_made += 1; writes_right += d.operation == y.operation and d.target_id == y.target_id
            if y.operation != "NOOP":
                useful += 1; missed += d.operation == "NOOP"
        else:
            gold, pred = set(y.relevant_ids), set(d.selected_ids)
            m["precision"].append(len(gold & pred) / len(pred) if pred else float(not gold))
            m["recall"].append(len(gold & pred) / len(gold) if gold else float(not pred))
        rows.append({"id": row.id, "family": row.family, "decision": d.model_dump(), "label": y.model_dump()})

    per_class = {}
    for op in OPERATIONS:
        tp = confusion[op][op]; predicted = sum(confusion[g][op] for g in OPERATIONS); actual = sum(confusion[op].values())
        per_class[op] = f1(tp / predicted if predicted else 0.0, tp / actual if actual else 0.0) if actual else None
    precision, recall = mean(m["precision"]), mean(m["recall"])
    summary = {
        "examples": len(examples),
        "operation_accuracy": mean(m["op"]),
        "operation_macro_f1": mean([v for v in per_class.values() if v is not None]),
        "operation_f1": per_class,
        "type_accuracy": mean(m["type"]),
        "target_accuracy": mean(m["target"]),
        "span_exact_match": mean(m["span_em"]),
        "span_f1": mean(m["span_f1"]),
        "retrieval_precision": precision, "retrieval_recall": recall, "retrieval_f1": f1(precision, recall) if precision is not None else None,
        "fallback_accuracy": mean(m["fallback"]),
        "write_precision": writes_right / writes_made if writes_made else None,
        "useful_writes_missed": missed / useful if useful else None,
        "operation_ece": calibration_error(m["conf"], m["raw_ok"]),
        "latency_p50_ms": float(np.percentile(m["latency"], 50) * 1000), "latency_p95_ms": float(np.percentile(m["latency"], 95) * 1000),
        "mean_input_tokens": mean(m["tokens"]),
        "confusion": confusion,
    }
    if output:
        root = Path(output); root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (root / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return summary


def score(summary):
    """Single dev-selection score: mean of the headline metrics that exist."""
    keys = ["operation_macro_f1", "target_accuracy", "span_f1", "retrieval_f1", "fallback_accuracy"]
    return mean([summary[k] for k in keys if summary[k] is not None])

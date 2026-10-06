"""Decode head outputs into one typed Decision; anything unsafe or unsure abstains."""
from __future__ import annotations
import time
import torch
from .batching import Collator, InputOverflow, to_device
from .contracts import Decision, ModelInput, OPERATIONS, TYPES, Span


def abstain(x, reason):
    return Decision(task=x.task, operation="NOOP" if x.task == "write" else None, needs_fallback=True, fallback_score=1.0, reason=reason)


def retrievable(candidate, mode):
    if candidate.status == "archived":
        return False
    return mode == "all" or candidate.status == ("current" if mode == "current" else "superseded")


def best_span(start, end, valid):
    """Joint argmax over start <= end within the observed text (independent argmaxes can cross)."""
    allowed = torch.triu(torch.ones(len(start), len(start), dtype=torch.bool)) & valid[:, None] & valid[None, :]
    if not allowed.any():
        return None
    joint = (start[:, None] + end[None, :]).masked_fill(~allowed, -float("inf"))
    return divmod(int(joint.argmax()), len(start))


def decode(x: ModelInput, out, offsets, source_mask, config):
    if any(not torch.isfinite(v).all() for v in out.values()):
        return abstain(x, "invalid_scores")
    op_scores = torch.softmax(out["operation_logits"].float(), -1).tolist()
    fallback = float(torch.softmax(out["fallback_logits"].float(), -1)[1])
    relevance = torch.sigmoid(out["relevance_logits"].float()).tolist()
    base = {"task": x.task, "operation_scores": dict(zip(OPERATIONS, op_scores)),
            "relevance_scores": {c.id: relevance[i] for i, c in enumerate(x.candidates)},
            "fallback_score": fallback, "needs_fallback": fallback >= config.fallback_threshold, "reason": "model_decision"}

    if x.task == "retrieve":
        if not x.candidates:
            return abstain(x, "no_candidates")
        chosen = [c.id for i, c in enumerate(x.candidates) if retrievable(c, x.temporal_mode) and relevance[i] >= config.relevance_threshold]
        return Decision(**base, selected_ids=sorted(chosen, key=lambda i: -base["relevance_scores"][i]))

    if x.source_role != "user":
        return abstain(x, "unsupported_source")
    index = max(range(len(OPERATIONS)), key=op_scores.__getitem__)
    operation = OPERATIONS[index]
    if op_scores[index] < config.decision_threshold or base["needs_fallback"]:
        return Decision(**{**base, "operation": "NOOP", "needs_fallback": True, "reason": "low_confidence"})
    result = {"operation": operation}
    if operation in {"UPDATE", "ARCHIVE"}:
        t = int(out["target_logits"].argmax())
        if t >= len(x.candidates) or x.candidates[t].status != "current":
            return abstain(x, "invalid_target")
        result["target_id"] = x.candidates[t].id
    if operation in {"ADD", "UPDATE"}:
        found = best_span(out["start_logits"].float(), out["end_logits"].float(), source_mask)
        if found is None:
            return abstain(x, "invalid_span")
        a, b = int(offsets[found[0], 0]), int(offsets[found[1], 1])
        result.update(evidence_span=Span(start=a, end=b), evidence_text=x.text[a:b], memory_type=TYPES[int(out["type_logits"].argmax())])
    return Decision(**base, **result)


class Predictor:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model, self.device = model.to(device).eval(), device
        self.collator = Collator(tokenizer, model.config)

    @torch.inference_mode()
    def predict(self, x: ModelInput):
        """Return (decision, timing) for one input; overflow abstains instead of truncating."""
        started = time.perf_counter()
        try:
            batch = self.collator([x])
        except InputOverflow:
            return abstain(x, "input_overflow"), {"seconds": time.perf_counter() - started, "input_tokens": 0}
        outputs = self.model(**to_device(batch, self.device))
        decision = decode(x, {k: v[0].cpu() for k, v in outputs.items()}, batch["offsets"][0], batch["model_inputs"]["source_mask"][0], self.model.config)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return decision, {"seconds": time.perf_counter() - started, "input_tokens": int(batch["model_inputs"]["attention_mask"].sum())}

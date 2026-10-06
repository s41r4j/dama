"""Turn typed inputs into encoder tensors plus structured metadata ids.

Each example becomes a row of items: [current text, recent context, candidate 1..N].
Only real texts are tokenized (no padded empty strings); metadata such as task, role,
status and memory type travels as ids for learned embeddings instead of as prose.
"""
from __future__ import annotations
import torch
from .contracts import Example, ModelInput, OPERATIONS, ROLES, STATUSES, TASKS, TEMPORAL_MODES, TYPES

EVENT, CONTEXT = 0, 1  # item positions; candidates start at 2


class InputOverflow(ValueError):
    pass


def context_text(x: ModelInput):
    return "\n".join(f"{t.role}: {t.text}" for t in x.recent)


class Collator:
    def __init__(self, tokenizer, config):
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer with offset mappings is required")
        self.tokenizer, self.config = tokenizer, config

    def __call__(self, rows: list[Example] | list[ModelInput]):
        if not rows:
            raise ValueError("empty batch")
        examples = rows if isinstance(rows[0], Example) else None
        inputs = [r.input for r in rows] if examples else list(rows)
        batch, count = len(inputs), max(len(x.candidates) for x in inputs)
        if count > self.config.max_candidates:
            raise InputOverflow(f"{count} candidates exceeds {self.config.max_candidates}")

        texts, item_index = [], torch.full((batch, count + 2), -1, dtype=torch.long)
        for b, x in enumerate(inputs):
            for i, text in enumerate([x.text, context_text(x)] + [c.text for c in x.candidates]):
                if text:
                    item_index[b, i] = len(texts)
                    texts.append(text)
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_offsets_mapping=True, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(1)
        if int(lengths.max()) > self.config.max_length:
            raise InputOverflow(f"{int(lengths.max())} tokens exceeds {self.config.max_length}; inputs are never truncated")

        events = item_index[:, EVENT]
        offsets = encoded.pop("offset_mapping")[events]
        source_mask = offsets[..., 1] > offsets[..., 0]  # real text tokens; specials and padding map to (0, 0)
        candidate_mask = torch.tensor([[i < len(x.candidates) for i in range(count)] for x in inputs], dtype=torch.bool).reshape(batch, count)
        def ids(values): return torch.tensor(values, dtype=torch.long).reshape(batch, -1)
        def per_candidate(field, vocabulary): return ids([[vocabulary.index(getattr(c, field)) for c in x.candidates] + [0] * (count - len(x.candidates)) for x in inputs])
        meta = {"task": ids([TASKS.index(x.task) for x in inputs])[:, 0],
                "role": ids([ROLES.index(x.source_role) for x in inputs])[:, 0],
                "temporal_mode": ids([TEMPORAL_MODES.index(x.temporal_mode) for x in inputs])[:, 0],
                "status": per_candidate("status", STATUSES),
                "candidate_type": per_candidate("memory_type", TYPES)}
        model_inputs = {**encoded, "item_index": item_index, "candidate_mask": candidate_mask, "source_mask": source_mask, **meta}
        labels = self.labels(examples, offsets, source_mask, count) if examples else None
        return {"model_inputs": model_inputs, "labels": labels, "inputs": inputs, "offsets": offsets}

    @staticmethod
    def labels(examples, offsets, source_mask, count):
        batch = len(examples)
        y = {k: torch.full((batch,), -100, dtype=torch.long) for k in ["operation", "memory_type", "target", "start", "end", "fallback"]}
        y["relevance"] = torch.full((batch, count), -100.0)
        for b, row in enumerate(examples):
            label, candidates = row.label, row.input.candidates
            y["fallback"][b] = int(label.needs_fallback)
            if label.operation is not None:
                y["operation"][b] = OPERATIONS.index(label.operation)
                # Index `count` is the null target used by ADD and NOOP.
                y["target"][b] = next((i for i, c in enumerate(candidates) if c.id == label.target_id), count)
            if label.memory_type is not None:
                y["memory_type"][b] = TYPES.index(label.memory_type)
            if label.evidence_span is not None:
                span, tokens = label.evidence_span, offsets[b].tolist()
                starts = [i for i, (s, e) in enumerate(tokens) if source_mask[b, i] and s <= span.start < e]
                ends = [i for i, (s, e) in enumerate(tokens) if source_mask[b, i] and s < span.end <= e]
                if not starts or not ends:
                    raise ValueError(f"{row.id}: evidence span does not align with tokens")
                y["start"][b], y["end"][b] = starts[0], ends[-1]
            if label.relevant_ids is not None:
                for i, c in enumerate(candidates):
                    y["relevance"][b, i] = float(c.id in label.relevant_ids)
        return y


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch["model_inputs"].items()}

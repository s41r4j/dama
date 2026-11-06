"""Decision model: one cross-encoder, four heads, one forward pass.

Input is always a text pair (a, b): (request, stored turn) for rel, (turn, previous turn) for sal and
(request, "") for need/route. Output is one logits tensor [batch, 3 + len(ROUTES)]:
    column 0 rel, 1 sal, 2 need, 3.. route (one per ROUTES entry). Every column is an independent sigmoid.

The rel head starts from the MS MARCO cross-encoder's own relevance classifier, so retrieval quality is
good from step 0 and fine-tuning only adapts it to conversational memory.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from ..config import ROUTES

HEADS = ["rel", "sal", "need"] + [f"route:{r}" for r in ROUTES]


class DecisionModel(nn.Module):
    def __init__(self, encoder, warm_classifier: nn.Module | None = None, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rel = nn.Linear(hidden, 1)
        self.sal = nn.Linear(hidden, 1)
        self.need = nn.Linear(hidden, 1)
        self.route = nn.Linear(hidden, len(ROUTES))
        self.warm_started = False
        if isinstance(warm_classifier, nn.Linear) and warm_classifier.out_features == 1 and warm_classifier.in_features == hidden:
            self.rel.load_state_dict(warm_classifier.state_dict())
            self.warm_started = True
        self.uses_pooler = getattr(encoder, "pooler", None) is not None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = getattr(out, "pooler_output", None)
        h = pooled if (self.uses_pooler and pooled is not None) else out.last_hidden_state[:, 0]
        h = self.dropout(h)
        return torch.cat([self.rel(h), self.sal(h), self.need(h), self.route(h)], dim=-1)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def encoder_parameters(self):
        return list(self.encoder.parameters())


def save_decision(model: DecisionModel, tokenizer, directory, meta: dict | None = None):
    from safetensors.torch import save_file
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(root / "encoder")
    tokenizer.save_pretrained(root / "tokenizer")
    heads = {n: p.detach().cpu().contiguous() for n, p in model.state_dict().items() if not n.startswith("encoder.")}
    save_file(heads, str(root / "heads.safetensors"))
    (root / "meta.json").write_text(json.dumps({"kind": "decision", "heads": HEADS, "routes": ROUTES, **(meta or {})}, indent=1) + "\n")


def load_decision(directory, device="cpu"):
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    root = Path(directory)
    encoder = AutoModel.from_pretrained(root / "encoder")
    model = DecisionModel(encoder)
    missing, unexpected = model.load_state_dict(load_file(str(root / "heads.safetensors")), strict=False)
    if unexpected or [m for m in missing if not m.startswith("encoder.")]:
        raise ValueError(f"head weights do not match: missing={missing} unexpected={unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    return model.to(device).eval(), tokenizer, json.loads((root / "meta.json").read_text())

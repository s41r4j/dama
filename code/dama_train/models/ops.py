"""Op model: an NLI cross-encoder re-headed to ADD / UPDATE / DELETE / NOOP.

Input pair: (existing memory, new statement). The new 4-way head is initialised from the NLI head:
    NOOP <- entailment     (the new statement is already covered)
    UPDATE <- contradiction (same slot, different value)
    ADD <- neutral          (new information)
    DELETE <- contradiction (a retraction contradicts the stored memory; fine-tuning separates it from UPDATE)
so the model already "understands" contradiction before seeing a single memory example.
"""
from __future__ import annotations
import torch
from torch import nn
from ..config import OPS

NLI_SOURCE = {"NOOP": "entail", "UPDATE": "contradict", "ADD": "neutral", "DELETE": "contradict"}


def rehead_for_ops(model):
    """Replace a 3-way NLI classifier with a 4-way op classifier initialised from it."""
    clf = getattr(model, "classifier", None)
    if not isinstance(clf, nn.Linear):
        raise ValueError("expected a sequence-classification model with a linear `classifier`")
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    new = nn.Linear(clf.in_features, len(OPS))
    warm = len(id2label) == clf.out_features
    with torch.no_grad():
        for j, op in enumerate(OPS):
            src = next((i for i, name in id2label.items() if NLI_SOURCE[op] in name), None) if warm else None
            if src is None:
                warm = False
                break
            new.weight[j] = clf.weight[src]
            new.bias[j] = clf.bias[src]
        if not warm:
            nn.init.normal_(new.weight, std=0.02)
            nn.init.zeros_(new.bias)
        else:  # tiny noise so DELETE and UPDATE can separate
            new.weight[OPS.index("DELETE")] += 0.01 * torch.randn_like(new.weight[OPS.index("DELETE")])
    model.classifier = new
    model.num_labels = len(OPS)
    model.config.num_labels = len(OPS)
    model.config.id2label = dict(enumerate(OPS))
    model.config.label2id = {op: i for i, op in enumerate(OPS)}
    model.config.problem_type = "single_label_classification"
    return model, warm


class OpLogits(nn.Module):
    """Plain-tensor wrapper (used for training and ONNX export)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs).logits

    def head_parameters(self):
        return list(self.model.classifier.parameters())

    def encoder_parameters(self):
        head = {id(p) for p in self.head_parameters()}
        return [p for p in self.model.parameters() if id(p) not in head]

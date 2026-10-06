"""Shared pretrained encoder + order-free candidate contextualizer + parallel decision heads.

Flow for one example (all items encoded in one batched encoder call):

  current text ─┐
  recent turns ─┼─ encoder ─ pool ─ project + metadata embeddings ─ contextualizer ─┐
  candidates ───┘   │                                                               │
                    └─ token states of current text ──────────── span head ◄───────┤
                                                                                    ├─ operation / type / fallback (event)
                                       cosine(event, candidate) on raw embeddings ─►├─ relevance / target (event × candidate)
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ModelConfig, OPERATIONS, ROLES, STATUSES, TASKS, TEMPORAL_MODES, TYPES

NEG = -1e4


class DAMADecisionModel(nn.Module):
    def __init__(self, config: ModelConfig, encoder=None):
        super().__init__()
        self.config = config
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(config.encoder, revision=config.encoder_revision)
        if config.encoder_mode == "lora":
            from peft import LoraConfig, TaskType, get_peft_model
            encoder = get_peft_model(encoder, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_rank, lora_alpha=config.lora_alpha,
                                                         lora_dropout=config.dropout, target_modules=["query", "value"], bias="none"))
        else:
            encoder.requires_grad_(False)
        self.encoder = encoder
        width, dim = encoder.config.hidden_size, config.hidden_dim

        self.projection = nn.Sequential(nn.Linear(width, dim), nn.LayerNorm(dim), nn.GELU())
        # Structured state as learned embeddings: kind (event/context/candidate), event metadata, candidate metadata.
        self.kind = nn.Embedding(3, dim)
        self.task = nn.Embedding(len(TASKS), dim)
        self.role = nn.Embedding(len(ROLES), dim)
        self.temporal_mode = nn.Embedding(len(TEMPORAL_MODES), dim)
        self.status = nn.Embedding(len(STATUSES), dim)
        self.candidate_type = nn.Embedding(len(TYPES), dim)
        self.context = None
        if config.context_layers:
            layer = nn.TransformerEncoderLayer(dim, config.attention_heads, dim * 2, dropout=config.dropout, batch_first=True, activation="gelu", norm_first=True)
            self.context = nn.TransformerEncoder(layer, config.context_layers, enable_nested_tensor=False)

        def mlp(i, o): return nn.Sequential(nn.Linear(i, dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim, o))
        self.operation = mlp(dim, len(OPERATIONS))
        self.memory_type = mlp(dim, len(TYPES))
        self.fallback = mlp(dim, 2)
        pair = dim * 4 + 1  # event, candidate, difference, product, raw cosine similarity
        self.relevance = mlp(pair, 1)
        self.target = mlp(pair, 1)
        self.null_target = nn.Linear(dim, 1)
        self.span = nn.Sequential(nn.Linear(width + dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def train(self, mode=True):
        super().train(mode)
        if self.config.encoder_mode == "frozen":
            self.encoder.eval()  # frozen features stay deterministic; heads still use dropout
        return self

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            args["token_type_ids"] = token_type_ids
        with torch.set_grad_enabled(self.config.encoder_mode == "lora" and torch.is_grad_enabled()):
            hidden = self.encoder(**args).last_hidden_state
        if self.config.pooling == "cls":
            return hidden, hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, item_index, candidate_mask, source_mask,
                task, role, temporal_mode, status, candidate_type, token_type_ids=None):
        hidden, pooled = self.encode(input_ids, attention_mask, token_type_ids)
        batch, items = item_index.shape
        present = item_index >= 0
        raw = pooled.new_zeros(batch, items, pooled.shape[-1])
        raw[present] = pooled[item_index[present]]

        kinds = torch.tensor([0, 1] + [2] * (items - 2), device=raw.device)
        event_meta = (self.task(task) + self.role(role) + self.temporal_mode(temporal_mode)).unsqueeze(1)
        candidate_meta = self.status(status) + self.candidate_type(candidate_type)
        meta = torch.cat([event_meta, torch.zeros_like(event_meta), candidate_meta], dim=1)
        x = self.projection(raw) + self.kind(kinds) + meta
        if self.context is not None:
            x = self.context(x, src_key_padding_mask=~present)
        x = x.masked_fill(~present.unsqueeze(-1), 0)

        event, candidates = x[:, 0], x[:, 2:]
        e = event.unsqueeze(1).expand_as(candidates)
        cosine = F.cosine_similarity(raw[:, :1].float(), raw[:, 2:].float(), dim=-1).unsqueeze(-1).to(x.dtype)
        pair = torch.cat([e, candidates, e - candidates, e * candidates, cosine], dim=-1)
        relevance = self.relevance(pair).squeeze(-1).masked_fill(~candidate_mask, NEG)
        target = self.target(pair).squeeze(-1).masked_fill(~candidate_mask, NEG)
        target = torch.cat([target, self.null_target(event)], dim=1)

        tokens = hidden[item_index[:, 0]]
        span = self.span(torch.cat([tokens, event.unsqueeze(1).expand(-1, tokens.shape[1], -1)], dim=-1))
        span = span.masked_fill(~source_mask.unsqueeze(-1), NEG)
        return {"operation_logits": self.operation(event), "type_logits": self.memory_type(event), "fallback_logits": self.fallback(event),
                "relevance_logits": relevance, "target_logits": target, "start_logits": span[..., 0], "end_logits": span[..., 1]}

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder": sum(p.numel() for p in self.encoder.parameters())}


def masked_ce(logits, labels):
    active = labels >= 0
    return F.cross_entropy(logits[active].float(), labels[active]) if active.any() else logits.new_zeros((), dtype=torch.float32)


def decision_loss(outputs, labels, candidate_mask):
    """Multitask loss; each head only learns from rows where its label exists."""
    losses = {"operation": masked_ce(outputs["operation_logits"], labels["operation"]),
              "memory_type": masked_ce(outputs["type_logits"], labels["memory_type"]),
              "target": masked_ce(outputs["target_logits"], labels["target"]),
              "fallback": masked_ce(outputs["fallback_logits"], labels["fallback"]),
              "span_start": masked_ce(outputs["start_logits"], labels["start"]),
              "span_end": masked_ce(outputs["end_logits"], labels["end"])}
    active = (labels["relevance"] >= 0) & candidate_mask
    logits = outputs["relevance_logits"]
    losses["relevance"] = F.binary_cross_entropy_with_logits(logits[active].float(), labels["relevance"][active]) if active.any() else logits.new_zeros((), dtype=torch.float32)
    return sum(losses.values()), {k: float(v.detach()) for k, v in losses.items()}

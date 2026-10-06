"""Typed inputs, labels, decisions and configs for the DAMA decision model."""
from __future__ import annotations
import math
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OPERATIONS = ["ADD", "NOOP", "UPDATE", "ARCHIVE"]
TYPES = ["key_fact", "preference", "plan", "routine", "emotional"]
TASKS = ["write", "retrieve"]
ROLES = ["user", "assistant", "tool"]
STATUSES = ["current", "superseded", "archived"]
TEMPORAL_MODES = ["current", "historical", "all"]

Operation = Literal["ADD", "NOOP", "UPDATE", "ARCHIVE"]
MemoryType = Literal["key_fact", "preference", "plan", "routine", "emotional"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Turn(Strict):
    role: Literal["user", "assistant", "tool"]
    text: str = Field(min_length=1)


class Candidate(Strict):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal["current", "superseded", "archived"] = "current"
    memory_type: MemoryType = "key_fact"


class ModelInput(Strict):
    task: Literal["write", "retrieve"]
    text: str = Field(min_length=1)
    source_role: Literal["user", "assistant", "tool"] = "user"
    recent: list[Turn] = Field(default_factory=list, max_length=8)
    candidates: list[Candidate] = Field(default_factory=list)
    temporal_mode: Literal["current", "historical", "all"] = "current"

    @model_validator(mode="after")
    def unique_ids(self):
        if len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate candidate IDs")
        return self


class Span(Strict):
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("empty or reversed span")
        return self


class Label(Strict):
    operation: Operation | None = None
    memory_type: MemoryType | None = None
    target_id: str | None = None
    evidence_span: Span | None = None
    relevant_ids: list[str] | None = None
    needs_fallback: bool = False


class Example(Strict):
    id: str
    family: str
    group: str  # split unit: examples sharing a group never cross train/dev/test
    input: ModelInput
    label: Label

    @model_validator(mode="after")
    def consistent(self):
        x, y = self.input, self.label
        current = {c.id for c in x.candidates if c.status == "current"}
        if x.task == "write":
            if y.operation is None or y.relevant_ids is not None:
                raise ValueError("write labels need an operation and no relevant_ids")
            if y.operation in {"ADD", "UPDATE"}:
                if y.memory_type is None or y.evidence_span is None:
                    raise ValueError("ADD/UPDATE need memory_type and evidence_span")
                if y.evidence_span.end > len(x.text):
                    raise ValueError("evidence span outside text")
            elif y.memory_type is not None or y.evidence_span is not None:
                raise ValueError("NOOP/ARCHIVE carry no content")
            if y.operation in {"UPDATE", "ARCHIVE"}:
                if y.target_id not in current:
                    raise ValueError("UPDATE/ARCHIVE need a current target")
            elif y.target_id is not None:
                raise ValueError("ADD/NOOP cannot target a candidate")
            if y.operation != "NOOP" and x.source_role != "user":
                raise ValueError("only user evidence may change memory")
        else:
            if any(v is not None for v in [y.operation, y.memory_type, y.target_id, y.evidence_span]):
                raise ValueError("retrieval labels cannot carry write targets")
            ids = {c.id for c in x.candidates}
            if y.relevant_ids is None or not set(y.relevant_ids) <= ids or len(set(y.relevant_ids)) != len(y.relevant_ids):
                raise ValueError("relevant_ids must be unique candidate IDs")
        return self


Reason = Literal["model_decision", "low_confidence", "invalid_target", "unsupported_source",
                 "input_overflow", "invalid_span", "invalid_scores", "no_candidates"]


class Decision(Strict):
    task: Literal["write", "retrieve"]
    operation: Operation | None = None
    memory_type: MemoryType | None = None
    target_id: str | None = None
    evidence_span: Span | None = None
    evidence_text: str | None = None
    selected_ids: list[str] = Field(default_factory=list)
    operation_scores: dict[str, float] = Field(default_factory=dict)
    relevance_scores: dict[str, float] = Field(default_factory=dict)
    fallback_score: float = Field(ge=0, le=1)
    needs_fallback: bool
    reason: Reason

    @field_validator("operation_scores", "relevance_scores")
    @classmethod
    def probabilities(cls, scores):
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError("scores must be finite probabilities")
        return scores


class ModelConfig(Strict):
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    pooling: Literal["mean", "cls"] = "mean"
    encoder_mode: Literal["frozen", "lora"] = "frozen"
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    hidden_dim: int = Field(default=192, ge=16)
    context_layers: int = Field(default=2, ge=0, le=4)
    attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    max_length: int = Field(default=256, ge=16, le=512)
    max_candidates: int = Field(default=32, ge=1, le=128)
    decision_threshold: float = Field(default=0.6, ge=0, le=1)
    relevance_threshold: float = Field(default=0.5, ge=0, le=1)
    fallback_threshold: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def compatible(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.encoder_revision):
            raise ValueError("pin the encoder to an exact 40-character revision")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        return self


class TrainConfig(Strict):
    model: ModelConfig = Field(default_factory=ModelConfig)
    epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    encoder_learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.06, ge=0, lt=1)
    grad_clip: float = Field(default=1.0, gt=0)
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    seed: int = 42

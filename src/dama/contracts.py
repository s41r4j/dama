from __future__ import annotations
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OPERATIONS = ["ADD","NOOP","UPDATE","ARCHIVE"]
TYPES = ["key_fact","preference","plan","routine","emotional"]


def timestamp(value):
    dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    if dt.tzinfo is None: raise ValueError("timezone required")
    return dt.astimezone(timezone.utc).isoformat()


class Strict(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True,frozen=True)
    schema_version: Literal[1]=1


class Scope(Strict):
    user: str=Field(min_length=1,max_length=128)
    project: str=Field(min_length=1,max_length=128)


class ContextTurn(Strict):
    id: str=Field(min_length=1)
    role: Literal["user","assistant","tool"]
    text: str=Field(min_length=1,max_length=16000)
    at: str
    _at=field_validator("at")(timestamp)


class Candidate(Strict):
    id: str=Field(min_length=1,max_length=128)
    scope: Scope
    text: str=Field(min_length=1,max_length=16000)
    status: Literal["current","superseded","archived"]="current"
    valid_from: str
    valid_to: str | None=None
    version: int=Field(default=1,ge=1)
    memory_type: Literal["key_fact","preference","plan","routine","emotional"]="key_fact"
    _from=field_validator("valid_from")(timestamp)
    @field_validator("valid_to")
    @classmethod
    def to(cls,v): return timestamp(v) if v else v
    @model_validator(mode="after")
    def interval(self):
        if self.valid_to is not None and self.valid_to<self.valid_from: raise ValueError("invalid interval")
        return self


class ModelInput(Strict):
    task: Literal["write","retrieve"]
    scope: Scope
    at: str
    text: str=Field(min_length=1,max_length=16000)
    source_id: str | None=None
    source_role: Literal["user","assistant","tool"]="user"
    recent: list[ContextTurn]=Field(default_factory=list,max_length=16)
    candidates: list[Candidate]=Field(default_factory=list,max_length=128)
    temporal_mode: Literal["current","historical","all"]="current"
    budget_tokens: int=Field(default=1024,ge=0)
    _at=field_validator("at")(timestamp)
    @model_validator(mode="after")
    def scope_time(self):
        if self.task=="write" and not self.source_id: raise ValueError("write requires source ID")
        if len({c.id for c in self.candidates})!=len(self.candidates): raise ValueError("duplicate candidate IDs")
        for c in self.candidates:
            if c.scope!=self.scope: raise ValueError("candidate scope differs from request")
            if c.valid_from>self.at: raise ValueError("future candidate")
            if c.valid_to and c.valid_to>self.at: raise ValueError("future validity metadata")
        if any(t.at>self.at for t in self.recent): raise ValueError("future recent turn")
        return self


class Span(Strict):
    start: int=Field(ge=0)
    end: int=Field(ge=1)
    @model_validator(mode="after")
    def order(self):
        if self.end<=self.start: raise ValueError("empty or reversed span")
        return self


class Label(Strict):
    operation: Literal["ADD","NOOP","UPDATE","ARCHIVE"] | None=None
    memory_type: Literal["key_fact","preference","plan","routine","emotional"] | None=None
    target_id: str | None=None
    evidence_span: Span | None=None
    relevant_ids: list[str] | None=None
    needs_fallback: bool=False


class Example(Strict):
    id: str
    family: str
    conversation: str
    input: ModelInput
    label: Label
    provenance: dict[str,str]
    private: bool=False
    @model_validator(mode="after")
    def targets(self):
        x,y=self.input,self.label; candidates={c.id:c for c in x.candidates}
        if self.private: raise ValueError("private data excluded by default")
        for key in ["source","license","label_method","review_status"]:
            if not self.provenance.get(key): raise ValueError("missing provenance: "+key)
        if x.task=="write":
            if y.operation is None or y.relevant_ids is not None: raise ValueError("write labels need operation, not retrieval labels")
            if y.operation in {"ADD","UPDATE"}:
                if y.memory_type is None or y.evidence_span is None: raise ValueError("supported content span and type required")
                if x.source_role!="user": raise ValueError("V1 mutation requires user evidence")
                if y.evidence_span.end>len(x.text): raise ValueError("span out of source bounds")
            if y.operation in {"UPDATE","ARCHIVE"}:
                if y.target_id not in candidates or candidates[y.target_id].status!="current": raise ValueError("current target required")
            elif y.target_id is not None: raise ValueError("ADD/NOOP cannot target candidate")
            if y.operation=="NOOP" and (y.evidence_span or y.memory_type): raise ValueError("NOOP cannot have content")
            if y.operation=="ARCHIVE" and x.text!=f"Archive memory {y.target_id}.": raise ValueError("archive needs matching explicit request")
        else:
            if y.operation is not None or y.target_id is not None or y.evidence_span is not None or y.memory_type is not None: raise ValueError("retrieval must not have write targets")
            if y.relevant_ids is None or not set(y.relevant_ids).issubset(candidates): raise ValueError("retrieval target outside candidates")
            if len(set(y.relevant_ids))!=len(y.relevant_ids): raise ValueError("duplicate retrieval labels")
        return self


class Decision(Strict):
    task: Literal["write","retrieve"]
    operation: Literal["ADD","NOOP","UPDATE","ARCHIVE"] | None=None
    memory_type: Literal["key_fact","preference","plan","routine","emotional"] | None=None
    target_id: str | None=None
    expected_version: int | None=Field(default=None,ge=1)
    evidence_span: Span | None=None
    evidence_text: str | None=None
    source_id: str | None=None
    selected_ids: list[str]=Field(default_factory=list)
    operation_scores: dict[str,float]=Field(default_factory=dict)
    relevance_scores: dict[str,float]=Field(default_factory=dict)
    fallback_score: float=Field(ge=0,le=1)
    needs_fallback: bool
    reason: Literal["model_decision","low_score","invalid_target","unsupported_source","input_overflow","invalid_span","invalid_scores","explicit_archive_required","no_candidates"]
    scores_calibrated: bool=False
    model_version: str

    @field_validator("operation_scores","relevance_scores")
    @classmethod
    def finite_scores(cls,scores):
        import math
        if any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
            raise ValueError("scores must be finite values in [0,1]")
        return scores

    @model_validator(mode="after")
    def decision_contract(self):
        if self.task=="retrieve":
            if any(v is not None for v in [self.operation,self.memory_type,self.target_id,self.evidence_span,self.evidence_text,self.expected_version,self.source_id]):
                raise ValueError("retrieval cannot mutate content")
        else:
            if self.operation is None or self.selected_ids: raise ValueError("write needs one operation and no retrieval IDs")
            if self.operation in {"UPDATE","ARCHIVE"} and (self.target_id is None or self.expected_version is None):
                raise ValueError("target and version required")
            if self.operation in {"ADD","UPDATE"} and any(v is None for v in [self.memory_type,self.evidence_span,self.evidence_text,self.source_id]):
                raise ValueError("supported evidence required")
        return self


class ModelConfig(Strict):
    encoder: str="sentence-transformers/all-MiniLM-L6-v2"
    encoder_revision: str="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    pooling: Literal["mean","cls"]="mean"
    hidden_dim: int=Field(default=192,ge=16)
    context_layers: int=Field(default=2,ge=0,le=4)
    attention_heads: int=Field(default=4,ge=1)
    max_length: int=Field(default=256,ge=16,le=512)
    max_candidates: int=Field(default=32,ge=1,le=128)
    dropout: float=Field(default=0.1,ge=0,lt=1)
    encoder_mode: Literal["frozen","lora"]="frozen"
    lora_rank: int=Field(default=8,ge=1)
    lora_alpha: int=Field(default=16,ge=1)
    decision_threshold: float=Field(default=0.6,ge=0,le=1)
    relevance_threshold: float=Field(default=0.5,ge=0,le=1)
    fallback_threshold: float=Field(default=0.5,ge=0,le=1)
    @model_validator(mode="after")
    def compatible(self):
        import re
        if not re.fullmatch(r"[0-9a-f]{40}",self.encoder_revision): raise ValueError("exact encoder revision required")
        if self.hidden_dim%self.attention_heads: raise ValueError("hidden size must be divisible by attention heads")
        return self


class TrainConfig(Strict):
    model: ModelConfig=Field(default_factory=ModelConfig)
    epochs: int=Field(default=5,ge=1)
    microbatch: int=Field(default=4,ge=1)
    gradient_accumulation: int=Field(default=4,ge=1)
    learning_rate: float=Field(default=0.001,gt=0)
    encoder_learning_rate: float=Field(default=0.00005,gt=0)
    weight_decay: float=Field(default=0.01,ge=0)
    seed: int=42
    save_steps: int=Field(default=50,ge=1)
    max_steps: int | None=Field(default=None,ge=1)
    precision: Literal["auto","bf16","fp16","fp32"]="auto"
    grad_clip: float=Field(default=1.0,gt=0)

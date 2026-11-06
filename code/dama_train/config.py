"""All settings in one place. Defaults are sized for a free T4 (16 GB) and finish each stage in well under an hour."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROUTES = ["episodic", "facts", "timeline", "expand"]   # which stores a request should search
OPS = ["ADD", "UPDATE", "DELETE", "NOOP"]                 # write operation for (existing memory, new statement)
SPLITS = ["train", "val", "calib", "test"]               # calib is only for calibration / conformal, never for training
CHECKPOINT = "L1.2"                                       # label written into every run's output (see docs/making-of-dama.md)


@dataclass
class DataConfig:
    locomo_url: str = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    lme_repo: str = "xiaowu0162/longmemeval-cleaned"
    lme_oracle: str = "longmemeval_oracle.json"
    lme_s: str = "longmemeval_s_cleaned.json"     # 277 MB; full haystacks give realistic hard negatives
    use_lme_s: bool = True
    mnli_repo: str = "nyu-mll/multi_nli"
    dolly_repo: str = "databricks/databricks-dolly-15k"
    negatives_per_positive: int = 15                # relevance: hard (BM25) + random negatives per evidence turn
    hard_negative_fraction: float = 0.7
    pool_size: int = 64                             # candidate pool per question for recall@k and conformal
    salience_negatives_per_positive: int = 4
    max_turn_chars: int = 1000
    mnli_samples: int = 40000
    synthetic_ops: dict = field(default_factory=lambda: {"train": 24000, "val": 3000, "calib": 3000, "test": 3000})
    dolly_samples: int = 6000
    # need-head negatives that are questions too (L1.1): SQuAD = general-knowledge questions, Quora = first-person
    # "How do I ...?" questions that need no memory. Without them need learned "benchmark question vs instruction".
    squad_repo: str = "rajpurkar/squad"
    quora_repo: str = "sentence-transformers/quora-duplicates"
    question_samples: int = 3000                   # per source
    seed: int = 13


@dataclass
class TrainConfig:
    backbone: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    max_length: int = 256
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-5          # encoder
    head_lr: float = 1e-3     # new heads
    epochs: int = 3
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    eval_every: int = 400     # steps
    save_every: int = 400     # steps
    max_minutes: float = 100  # stop cleanly (and keep a resumable checkpoint) before a free session ends
    patience: int = 4         # evals without improvement before early stop
    seed: int = 42
    loss_weights: dict = field(default_factory=lambda: {"rel": 1.0, "sal": 1.0, "need": 0.5, "route": 0.5})
    tiny: bool = False        # offline smoke test: random tiny model, no downloads
    precision: str = "auto"   # "auto" = bf16 on A100/L4, fp16 on T4; "fp32" if you ever see non-finite losses


@dataclass
class RetrieverConfig:
    """First-stage dense retriever: bge-small fine-tuned on the train split. L1.1: in-batch + BM25 hard negatives.
    L1.2: negatives mined with the base dense model, and keys that include the start of the previous turn."""
    backbone: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 256        # stored turns
    query_max_length: int = 64
    batch_size: int = 64         # questions per batch; every other question's turns are negatives
    lr: float = 2e-5
    epochs: int = 6
    temperature: float = 0.05
    hard_negatives: int = 2        # per (question, gold turn), sampled from the base model's top-30 non-gold turns
    mine_depth: int = 30
    context_chars: int = 200       # previous-turn characters in each memory key (0 = the turn alone)
    warmup_ratio: float = 0.1
    seed: int = 42
    tiny: bool = False


def decision_defaults() -> TrainConfig:
    return TrainConfig()


def ops_defaults() -> TrainConfig:
    return TrainConfig(backbone="cross-encoder/nli-deberta-v3-xsmall", max_length=160, batch_size=64, lr=2e-5,
                       head_lr=2e-5, epochs=2, loss_weights={})


def save_config(cfg, path):
    Path(path).write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def load_config(cls, path):
    raw = json.loads(Path(path).read_text())
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})

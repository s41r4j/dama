"""Common in-memory form for LoCoMo and LongMemEval, so builders never care which benchmark a dialogue came from."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Turn:
    tid: str            # globally unique turn id
    speaker: str        # speaker name or role (user/assistant)
    text: str
    date: str | None    # session date as given by the benchmark
    session: str
    role: str = "user"  # "user" for the person whose memory we keep, "assistant" otherwise


@dataclass
class Dialogue:
    did: str            # split unit: all rows from one dialogue stay in one split
    source: str         # "locomo" | "lme"
    turns: list[Turn]


@dataclass
class Question:
    qid: str
    did: str
    text: str
    date: str | None
    qtype: str
    evidence: list[str] = field(default_factory=list)   # turn ids
    answer: str = ""
    abstention: bool = False


def render_turn(t: Turn, max_chars: int) -> str:
    """How a stored memory appears to the models (and later to the Primary Agent)."""
    text = t.text if len(t.text) <= max_chars else t.text[:max_chars] + " ..."
    date = f"[{t.date}] " if t.date else ""
    return f"{date}{t.speaker}: {text}"


def render_query(q: Question) -> str:
    return f"[{q.date}] {q.text}" if q.date else q.text

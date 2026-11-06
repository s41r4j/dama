"""DMBM (DAMA Memory Benchmark) item schema and ability taxonomy. Spec: docs/research/DMBM_spec.md.

Every source is converted to the same item: one question over one history of dated sessions. Systems see `sessions`,
`question` and `q_date` only; `gold`, `rubric` and `evidence` are for scoring and analysis.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field

ABILITIES = {
    "A1": "single-fact recall",
    "A2": "multi-hop",
    "A3": "cross-session aggregation",
    "A4": "temporal reasoning",
    "A5": "event ordering",
    "A6": "knowledge update / contradiction",
    "A7": "preference",
    "A8": "abstention",
    "G": "summarisation / instruction following (generative track)",
    "X": "reported, not in the headline (e.g. LoCoMo open-domain)",
}
HEADLINE = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")

# source label -> ability. LME `_abs` questions go to A8 whatever their type.
LOCOMO = {1: "A2", 2: "A4", 3: "X", 4: "A1", 5: "A8"}          # LoCoMo categories: multi-hop, temporal, open-domain, single-hop, adversarial
LME = {"single-session-user": "A1", "single-session-assistant": "A1", "multi-session": "A3", "temporal-reasoning": "A4",
       "knowledge-update": "A6", "single-session-preference": "A7"}
BEAM = {"information_extraction": "A1", "multi_hop_reasoning": "A2", "temporal_reasoning": "A4", "event_ordering": "A5",
        "knowledge_update": "A6", "contradiction_resolution": "A6", "preference_following": "A7", "abstention": "A8",
        "summarization": "G", "instruction_following": "G"}      # names from the BEAM README; checked when the adapter is built
MAB = {"FactConsolidation_sh": "A6", "FactConsolidation_mh": "A6"}


@dataclass
class Turn:
    speaker: str
    text: str
    turn_id: str


@dataclass
class Session:
    session_id: str
    date: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Item:
    item_id: str          # "<source>:<native id>"
    source: str           # locomo | lme | beam100k | beam500k | mab_fc
    split: str            # dev | test | seen-test
    ability: str          # key of ABILITIES
    native_type: str      # the source's own label
    history_id: str       # items over the same history share it (ingest once)
    question: str
    q_date: str | None
    gold: str
    rubric: list[str] = field(default_factory=list)     # BEAM nuggets; empty elsewhere
    evidence: list[str] = field(default_factory=list)   # turn ids

    def to_json(self):
        return asdict(self)


def lme_ability(qtype: str, qid: str) -> str:
    return "A8" if "_abs" in qid else LME[qtype]

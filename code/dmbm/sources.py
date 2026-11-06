"""Source adapters: the DAMA loaders' structures (experiments/v2_2026-10-28/v2lib.py: load_locomo, load_lme) -> DMBM
items and histories. BEAM and MemoryAgentBench adapters are added only after DAMA is frozen (spec §5), so those sets
stay unseen until then.
"""
from __future__ import annotations
from collections import defaultdict

from .schema import LOCOMO, Item, Session, Turn, lme_ability


def _sessions(turns, date_key, sid_key="session"):
    by = defaultdict(list)
    for t in turns:
        by[t[sid_key]].append(t)
    out = []
    for sid, ts in by.items():                   # loaders keep sessions in time order
        out.append(Session(str(sid), str(ts[0].get(date_key) or ts[0].get("date") or ""),
                           [Turn(t["speaker"], t["text"], t["dia"]) for t in ts]))
    return out


def locomo(convs, questions):
    """convs: [{"ci", "turns"}]; questions: LoCoMo questions (qid, ci, cat, question, gold, evidence, split)."""
    hist = {f"locomo:{c['ci']}": _sessions(c["turns"], "date_raw") for c in convs}
    items = [Item(f"locomo:{q['qid']}", "locomo", q.get("split", "dev"), LOCOMO[q["cat"]], str(q["cat"]), f"locomo:{q['ci']}",
                  q["question"], None, str(q["gold"]), evidence=list(q.get("evidence") or [])) for q in questions]
    return items, {h: hist[h] for h in {it.history_id for it in items}}


def lme(qs, split_name=None):
    """qs: LongMemEval items (qid, type, question, gold/answer, date, turns, evidence, split)."""
    items, hist = [], {}
    for q in qs:
        hid = f"lme:{q['qid']}"
        hist[hid] = _sessions(q["turns"], "date")
        items.append(Item(f"lme:{q['qid']}", "lme", split_name or q.get("split", "dev"), lme_ability(q["type"], q["qid"]), q["type"],
                          hid, q["question"], q.get("date"), str(q.get("gold", q.get("answer", ""))), evidence=list(q.get("evidence") or [])))
    return items, hist

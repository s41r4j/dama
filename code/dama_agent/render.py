"""Packet rendering R1 and reader prompt P2 for DAMA v3 (candidate A of docs/research/DAMA_v3_research_and_architecture.md,
§5.9.4-5.10). No model, no I/O.

Why (v2b, 2026-10-30): with every evidence turn in the packet, the reader (Qwen3.5-9B, notes-first prompt P1) still
answered only 61% of LongMemEval dev multi-session questions. The misses were counting ("3 of 4 festivals"), arithmetic
across two sessions, date anchoring and picking a value that had since changed. R1 turns date arithmetic into a lookup:
each line says which session it is from, the session's weekday and date, and how many days before the question it was
said, and each resolved time note carries the same offset. P2 asks for an evidence table, then one CALC line that
postcheck.py recomputes without a second model call.

Turns are v2lib's dicts: {"i", "dia", "speaker", "text", "date", "date_raw", "session"}. Lines are numbered #1..#n in
chronological order; `ids[k - 1]` is the turn id of line #k.
"""
from __future__ import annotations
import re
from datetime import date

from .timeparse import find, parse_date

WEEKDAYS = "Mon Tue Wed Thu Fri Sat Sun".split()
_DAY = re.compile(r"\d{1,2} [A-Z][a-z]+ \d{4}")        # timeparse's exact-day format; "about ...", months, spans get no offset
NOT_FOUND = "Not mentioned in the conversation."       # P1's abstention line, so judges and abstention scoring are unchanged
MAX_WORDS = {"locomo": 15, "lme": 30}


def session_order(turns) -> dict:
    """session id -> 1-based position in the whole history. Call it on the history, not the packet, so S-numbers do not
    depend on which turns were packed."""
    out = {}
    for t in sorted(turns, key=lambda t: t["i"]):
        out.setdefault(t.get("session"), len(out) + 1)
    return out


def days_before(d: date | None, q: date | None) -> str:
    if d is None or q is None:
        return ""
    n = (q - d).days
    if n == 0:
        return "same day as question"
    return f"{n} day{'s' * (n != 1)} before question" if n > 0 else f"{-n} day{'s' * (n != -1)} after question"


def annotate(text: str, ref, q: date | None) -> str:
    """timeparse.annotate, plus the offset to the question date for exact days: `yesterday [= 16 November 2023, 164 days
    before question]`."""
    for s, e, r in reversed(find(text, ref)):
        off = days_before(parse_date(r), q) if _DAY.fullmatch(r) else ""
        text = f"{text[:e]} [= {r}{', ' + off if off else ''}]{text[e:]}"
    return text


def render_line(k: int, t: dict, q: date | None, session_idx=None, time_notes=True) -> str:
    d = parse_date(t.get("date_raw") or t.get("date"))
    head = [f"S{session_idx}"] if session_idx is not None else []
    head.append(f"{WEEKDAYS[d.weekday()]} {d.isoformat()}" if d else str(t.get("date") or "date unknown"))
    if off := days_before(d, q):
        head.append(off)
    head.append(t["speaker"])
    text = annotate(t["text"], t.get("date_raw") or t.get("date"), q) if time_notes else t["text"]
    return f"#{k} [{' | '.join(head)}] {text}"


def render_packet(turns, q_date=None, sessions: dict | None = None, time_notes=True) -> tuple[list[str], list[str]]:
    """(lines, ids): the packet's turns oldest first. q_date None (LoCoMo has none) drops the offsets, nothing else."""
    q = parse_date(q_date)
    chosen = sorted(turns, key=lambda t: t["i"])
    lines = [render_line(k, t, q, (sessions or {}).get(t.get("session")), time_notes) for k, t in enumerate(chosen, 1)]
    return lines, [t["dia"] for t in chosen]


# ----------------------------------------------------------------------------------------------- prompt P2
_SUBJECT = {"lme": "a user's past conversations with an AI assistant",
            "locomo": "a long conversation between two people"}
_HEADER_HELP = ("MEMORY lines are copied from the conversation history, oldest first. Each line starts with a header: the "
                "line number, the session number, the session's weekday and date, how many days before the question date "
                "that session was, and the speaker. Text in [= ...] was added by the memory system: it gives the date a "
                "relative time expression refers to.")
_DIGEST_HELP = (" [LATEST as of ...] and [SUPERSEDED on ...] mark values that changed over time. The DIGEST lists records "
                "derived from the history; each one cites the MEMORY lines it came from. Raw MEMORY lines are the ground "
                "truth: if a DIGEST record disagrees with a raw line, trust the raw line. A DIGEST list may be incomplete; "
                "add any matching items you find in MEMORY.")
_PREF = {"lme": " If the question asks for a suggestion or recommendation, base it on the user's own preferences and "
                "details found in MEMORY.", "locomo": ""}
_DATES = {"locomo": " For a question about when something happened, answer with an absolute date (for example 7 May 2023).",
          "lme": ""}

P2_STEPS = """Answer in this order.
1. EVIDENCE: a table with one row per MEMORY{or_digest} line that bears on the question, including older values that later changed:
| # | date | session | fact (at most 12 words) |
2. RESOLVE, using exactly one of these:
- If the question asks how many, how much or a total: write each distinct item on its own line starting with "ITEM:", merging items that refer to the same thing, then write "CALC: COUNT" or "CALC: SUM(<number>, <number>, ...)".
- If the question asks for a current, latest or most recent value: write "CALC: LATEST(#<id>)" naming the evidence row with the latest date.
- If the question needs a duration, a date difference or an age: write "CALC: DATEDIFF(<YYYY-MM-DD>, <YYYY-MM-DD>, <days|weeks|months|years>)" or "CALC: <arithmetic on numbers>".
- Otherwise write "CALC: NONE".
3. If MEMORY does not contain what is needed, the answer is: {not_found}
4. End with exactly one line that starts with "ANSWER:" followed by the answer. Keep it short (at most {words} words), unless the question asks for a list, a recommendation or an explanation; then answer fully.{dates}"""


def system_p2(bench: str, digest: bool = False) -> str:
    return (f"You answer questions about {_SUBJECT[bench]} using only the MEMORY below. Do not use outside knowledge. "
            + _HEADER_HELP + (_DIGEST_HELP if digest else "") + _PREF[bench])


def messages_p2(bench: str, question: str, q_date, lines: list[str], digest: list[str] | None = None,
                evidence_status: str | None = None) -> list[dict]:
    """digest None: candidate A (no DIGEST block at all, so the reader is not told about records it never sees)."""
    user = [f"QUESTION DATE: {q_date}" if q_date else "", f"QUESTION: {question}", "", "MEMORY (oldest first):", *lines, ""]
    if digest is not None:
        user += ["DIGEST:", *(digest or ["(none)"]), ""]
    if evidence_status:
        user += [f"EVIDENCE_STATUS: {evidence_status}", ""]
    user.append(P2_STEPS.format(or_digest=" or DIGEST" if digest is not None else "", not_found=NOT_FOUND,
                                words=MAX_WORDS[bench], dates=_DATES[bench]))
    return [{"role": "system", "content": system_p2(bench, digest is not None)},
            {"role": "user", "content": "\n".join(x for i, x in enumerate(user) if x or i > 0)}]


# ----------------------------------------------------------------------------------------------- thinking router
COUNT_CUE = re.compile(r"\b(how many|how much|total|in total|altogether|combined|number of|count)\b", re.I)
LATEST_CUE = re.compile(r"\b(current(ly)?|now|still|latest|most recent|these days|at the moment)\b", re.I)
ARITH_CUE = re.compile(r"\b(older|younger|difference|how many (more|fewer)|age|years? old)\b", re.I)


def route_thinking(question: str) -> bool:
    """Thinking only for questions that need aggregation, arithmetic or the latest value (§5.10). Pre-registered as an
    arm (E4), not a default."""
    return bool(COUNT_CUE.search(question) or LATEST_CUE.search(question) or ARITH_CUE.search(question))

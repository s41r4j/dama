"""Event ledger (DAMA v3-L): dated, typed facts written once per session; counts, sums, date differences, order and
latest values computed in Python at question time, never by the reader.

Why (v3a, 2026-10-31; docs/research/DAMA_v3_new_arch_research_2026-10-31.md): retrieval already puts the evidence in
the packet for 92-95% of questions, but the 9B reader miscounts across sessions, does date arithmetic badly and loses
when asked to build structure itself (evidence-table prompt -6 points). So the structure is built at write time by an
extractor model, dates are resolved by rules, and the reader only gets finished numbers in a short LEDGER block next to
the unchanged raw-turn packet. Questions the router does not match get no block, so their packets stay byte-identical.

Write:  session turns -> extractor (any chat model, prompt EXTRACT_SYSTEM) -> parse_rows() -> resolve_when() -> rows
Read:   question -> route() -> match() over the history's rows -> operator -> render() -> "LEDGER: ..." (<= ~150 tokens)
No model and no I/O here; the caller supplies the extractor outputs and a similarity function.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .timeparse import MONTHS, NUM, WEEKDAYS, parse_date

# ---------------------------------------------------------------------------------------------------- write side
EXTRACT_SYSTEM = """You extract a memory ledger from one chat session.
List the concrete facts about the people in the session that someone might ask about later: things they did, bought, attended, visited, made, started, finished, received or plan to do; their current states (job, home, pet, relationship, habits, devices, amounts) and their preferences (likes, dislikes, needs).
Output one JSON object per line and nothing else. Keys:
 "t": the turn number in [#n] that states the fact (an integer)
 "who": the person ("user" or the person's name)
 "kind": "event" (something done or planned), "state" (a value that can change later) or "pref" (a like, dislike or need)
 "what": a short verb phrase, e.g. "attended", "bought", "works as", "likes"
 "item": the specific thing, e.g. "AFI Fest", "Trek road bike", "sushi"
 "cat": one generic category noun used for counting, e.g. "film festival", "bike", "restaurant", "museum visit"
 "qty": a number if the fact states one (amount, count, duration, price), otherwise null
 "unit": the unit of qty (e.g. "USD", "hours", "pounds", "days", "times"), otherwise ""
 "when": the time words exactly as written (e.g. "last Saturday", "on May 3", "two weeks ago"), or "" if none
Rules: one line per item (two festivals = two lines). At most 25 lines. Skip small talk, questions, greetings and the assistant's own suggestions. If nothing is worth keeping, output nothing."""

USER_TEMPLATE = "SESSION DATE: {date}\n{turns}\n\nLEDGER LINES:"
MAX_SESSION_CHARS = 16000          # longer sessions are cut into parts of about this size


def session_prompts(session_date, turns, speaker_filter=None, max_chars=MAX_SESSION_CHARS):
    """turns: [(t, speaker, text)] of one session. Returns [(messages, [t...])], one per part."""
    parts, cur, size = [], [], 0
    for t, spk, text in turns:
        if speaker_filter and spk not in speaker_filter:
            continue
        line = f"[#{t}] {spk}: {' '.join(str(text).split())}"
        if cur and size + len(line) > max_chars:
            parts.append(cur); cur, size = [], 0
        cur.append((t, line)); size += len(line)
    if cur:
        parts.append(cur)
    out = []
    for p in parts:
        user = USER_TEMPLATE.format(date=session_date, turns="\n".join(l for _, l in p))
        out.append(([{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": user}], [t for t, _ in p]))
    return out


_KINDS = {"event", "state", "pref"}


def parse_rows(output: str, valid_turns=None) -> list[dict]:
    """Extractor output -> rows. Tolerates code fences, prose lines, single quotes and a JSON list instead of lines."""
    text = re.sub(r"<think>.*?</think>", "", output or "", flags=re.S)
    objs = []
    s = text.strip()
    if s.startswith("[") or s.startswith("```"):
        body = re.sub(r"^```\w*|```$", "", s, flags=re.M).strip()
        try:
            v = json.loads(body)
            if isinstance(v, list):
                objs = [o for o in v if isinstance(o, dict)]
        except Exception:
            pass
    if not objs:
        for line in text.splitlines():
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            raw = m.group(0)
            for cand in (raw, re.sub(r",\s*}", "}", raw), raw.replace("'", '"')):
                try:
                    o = json.loads(cand)
                    if isinstance(o, dict):
                        objs.append(o)
                    break
                except Exception:
                    continue
    rows = []
    for o in objs:
        try:
            t = int(str(o.get("t", "")).strip().lstrip("#"))
        except ValueError:
            continue
        if valid_turns is not None and t not in valid_turns:
            continue
        item = _clean(o.get("item")); what = _clean(o.get("what"))
        if not item and not what:
            continue
        kind = str(o.get("kind", "event")).strip().lower()
        qty = o.get("qty")
        try:
            qty = None if qty in (None, "", "null") else float(str(qty).replace(",", "").replace("$", ""))
        except ValueError:
            qty = None
        rows.append({"t": t, "who": _clean(o.get("who")) or "user", "kind": kind if kind in _KINDS else "event",
                     "what": what, "item": item, "cat": _clean(o.get("cat")).lower(), "qty": qty,
                     "unit": _clean(o.get("unit")).lower(), "when": _clean(o.get("when"))})
    return rows


def _clean(x) -> str:
    return " ".join(str(x).split()).strip(" .") if x not in (None, "null") else ""


# ---------------------------------------------------------------------------------------------------- dates
_WORDNUM = dict(NUM, **{"a few": 3, "few": 3, "several": 3, "some": 3, "a couple of": 2})
_N = r"(\d+|" + "|".join(sorted((re.escape(k) for k in _WORDNUM), key=len, reverse=True)) + r")"
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def _mon(s):
    return [m[:3].lower() for m in MONTHS].index(s[:3].lower()) + 1


def _month_end(y, m):
    return (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))


def _past_year(ref, m, d=1):
    """The most recent (month, day) on or before ref; a later month this year is taken as last year."""
    y = ref.year if (m, d) <= (ref.month, ref.day) else ref.year - 1
    return y


@dataclass
class When:
    start: date
    end: date
    gran: str          # day | week | month | year | approx | mention


def resolve_when(text: str, ref) -> When | None:
    """The extractor's verbatim time words, said on date ref -> a date range. Unknown or empty -> the mention date."""
    ref = parse_date(ref)
    if ref is None:
        return None
    s = (text or "").lower().strip()
    if not s:
        return When(ref, ref, "mention")
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3))); return When(d, d, "day")
        except ValueError:
            pass
    m = re.search(_MON + r"\.?\s+(\d{1,2})(?!\d)(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?", s) or None
    if m:
        mo, dd = _mon(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else _past_year(ref, mo, dd)
        try:
            d = date(y, mo, dd); return When(d, d, "day")
        except ValueError:
            pass
    m = re.search(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"(?:,?\s+(\d{4}))?", s)
    if m:
        mo, dd = _mon(m.group(2)), int(m.group(1))
        y = int(m.group(3)) if m.group(3) else _past_year(ref, mo, dd)
        try:
            d = date(y, mo, dd); return When(d, d, "day")
        except ValueError:
            pass
    m = re.search(r"\b" + _MON + r"\b(?:\s+(\d{4}))?", s)
    if m and not re.search(r"\bmay (?:be|have|not)\b", s):
        mo = _mon(m.group(1)); y = int(m.group(2)) if m.group(2) else _past_year(ref, mo)
        return When(date(y, mo, 1), _month_end(y, mo), "month")
    m = re.search(r"\b(?:in\s+)?(19\d\d|20\d\d)\b", s)
    if m:
        y = int(m.group(1)); return When(date(y, 1, 1), date(y, 12, 31), "year")
    if "day before yesterday" in s:
        d = ref - timedelta(days=2); return When(d, d, "day")
    if re.search(r"\byesterday\b|\blast night\b", s):
        d = ref - timedelta(days=1); return When(d, d, "day")
    if re.search(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b|\bjust now\b", s):
        return When(ref, ref, "day")
    if re.search(r"\btomorrow\b", s):
        d = ref + timedelta(days=1); return When(d, d, "day")
    m = re.search(_N + r"\s+(day|week|month|year)s?\s+ago", s)
    if m:
        n, unit = _WORDNUM.get(m.group(1), None) if not m.group(1).isdigit() else int(m.group(1)), m.group(2)
        if n is None:
            return When(ref, ref, "mention")
        vague = m.group(1) in ("a few", "few", "several", "some")
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
        d = ref - timedelta(days=days)
        if vague:
            return When(d - timedelta(days=days // 2), d + timedelta(days=days // 2), "approx")
        return When(d, d, "day" if unit == "day" else ("week" if unit == "week" else ("month" if unit == "month" else "year")))
    m = re.search(r"\b(?:last|this past|on|this)\s+(" + "|".join(WEEKDAYS) + r")\b", s)
    if m:
        wd = WEEKDAYS.index(m.group(1))
        back = (ref.weekday() - wd) % 7 or 7
        d = ref - timedelta(days=back); return When(d, d, "day")
    m = re.search(r"\b(last|this|past)\s+weekend\b", s)
    if m:
        sat = ref + timedelta(days=5 - ref.weekday()) - timedelta(days=0 if m.group(1) == "this" else 7)
        return When(sat, sat + timedelta(days=1), "week")
    m = re.search(r"\b(last|this|next)\s+(week|month|year)\b", s)
    if m and not re.search(r"\b(?:the|over the|for the|in the|past)\s+(?:last|this|next)\s+(?:week|month|year)", s):
        which, unit = m.groups()
        k = {"last": -1, "this": 0, "next": 1}[which]
        if unit == "week":
            mon = ref - timedelta(days=ref.weekday()) + timedelta(days=7 * k)
            return When(mon, mon + timedelta(days=6), "week")
        if unit == "month":
            y, mo = ref.year + (ref.month - 1 + k) // 12, (ref.month - 1 + k) % 12 + 1
            return When(date(y, mo, 1), _month_end(y, mo), "month")
        y = ref.year + k
        return When(date(y, 1, 1), date(y, 12, 31), "year")
    if re.search(r"\brecently\b|\bthe other day\b|\blately\b|\bjust\b", s):
        return When(ref - timedelta(days=14), ref, "approx")
    return When(ref, ref, "mention")


def finish_rows(rows, session_id, session_date, dia_of, history_key):
    """Adds ids, mention date and resolved event dates. dia_of(t) maps the extractor's turn number to the turn id."""
    ref = parse_date(session_date)
    out = []
    for r in rows:
        w = resolve_when(r["when"], ref) if ref else None
        out.append({**r, "hid": history_key, "sid": session_id, "dia": dia_of(r["t"]),
                    "mention": ref.isoformat() if ref else None,
                    "start": w.start.isoformat() if w else None, "end": w.end.isoformat() if w else None,
                    "gran": w.gran if w else "unknown"})
    return out


def row_text(r) -> str:
    """What a row is matched on."""
    cat = f" ({r['cat']})" if r.get("cat") else ""
    return f"{r.get('who', '')} {r.get('what', '')} {r.get('item', '')}{cat}".strip()


# ---------------------------------------------------------------------------------------------------- read side
_DATEDIFF = re.compile(r"\bhow long ago\b|\bhow many (?:days|weeks|months|years)\b(?!.*\b(?:spend|spent|stay(?:ed)?|take|took|last(?:ed)?)\b)|"
                       r"\bhow long (?:had|have|has|did|was|were|is|since|after|before|between)\b|\bhow (?:old|long) was i when\b|"
                       r"\bwhat (?:is|was) the (?:time|gap|difference)\b.*\b(?:between|since)\b", re.I)
_SUM = re.compile(r"\b(?:in total|total|altogether|combined|sum)\b|\bhow much (?:money|time)\b|\bhow much (?:did|have|had) (?:i|we|you) (?:spend|spent|pay|paid|raise|raised|earn|earned|save|saved)\b|"
                  r"\bhow many (?:hours|minutes|days|weeks|months|years|miles|kilometers|km|pounds|dollars)\b.*\b(?:spend|spent|stay(?:ed)?|take|took|last(?:ed)?|run|ran|walk(?:ed)?|lose|lost)\b", re.I)
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.I)
_ORDER = re.compile(r"\bwhich (?:\w+ ){0,6}(?:first|earlier|earliest|last|later|latest)\b|\border of\b|\bin what order\b|"
                    r"\bhappened first\b|\bcame first\b|\bbefore or after\b", re.I)
_LATEST = re.compile(r"\bcurrent(?:ly)?\b|\bnow\b|\bstill\b|\bthese days\b|\bat the moment\b|\bmost recent(?:ly)?\b|\blatest\b|"
                     r"\bany ?more\b|\bupdated?\b|\bdo i (?:have|own|use)\b|\bwhere do i\b|\bwhat is my\b", re.I)
_PREF = re.compile(r"\brecommend|\bsuggest|\btips?\b|\badvice\b|\bideas?\b|\bwhat should i\b|\bany (?:good|other)\b|"
                   r"\bcan you (?:help|give|find|think)\b|\bwould i (?:like|enjoy)\b|\bwhat (?:kind|type) of\b.*\b(?:should|could|would)\b", re.I)
ROUTES = ("DATEDIFF", "SUM", "COUNT", "ORDER", "LATEST", "PREF")


def route(question: str, prefs=True) -> str | None:
    """The question's operator, by fixed rules (first match wins), or None (no LEDGER block)."""
    q = question or ""
    if _DATEDIFF.search(q):
        return "DATEDIFF"
    if _SUM.search(q):
        return "SUM"
    if _COUNT.search(q):
        return "COUNT"
    if _ORDER.search(q):
        return "ORDER"
    if _LATEST.search(q):
        return "LATEST"
    if prefs and _PREF.search(q):
        return "PREF"
    return None


_ART = re.compile(r"^(?:the|a|an|my|our|his|her|their)\s+")


def norm_item(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    s = _ART.sub("", " ".join(s.split()))
    return re.sub(r"(?<=\w{3})s$", "", s)


def match(rows, scores, route_name, top=12, rel=0.08, floor=0.0):
    """rows of one history, scores = similarity of each row to the question. Keeps rows within `rel` of the best score and
    above `floor`, best first, at most `top`; PREF keeps pref rows only, LATEST prefers state rows."""
    pairs = [(s, r) for s, r in zip(scores, rows) if s >= floor]
    if route_name == "PREF":
        pairs = [(s, r) for s, r in pairs if r["kind"] == "pref"]
    if not pairs:
        return []
    best = max(s for s, _ in pairs)
    keep = sorted(((s, r) for s, r in pairs if s >= best - rel), key=lambda x: -x[0])[:top]
    return [r for _, r in keep]


@dataclass
class Block:
    route: str
    text: str
    rows: list = field(default_factory=list)
    value: float | None = None


def _d(s):
    return date.fromisoformat(s) if s else None


def _when_str(r):
    if r.get("gran") in ("day", "mention"):
        return r["start"]
    if r.get("gran") == "month":
        return r["start"][:7]
    if r.get("gran") == "year":
        return r["start"][:4]
    return f"~{r['start']}" if r.get("start") else "date unknown"


def _ago(d, q):
    n = (q - d).days
    if n < 0:
        return f"{-n} days after the question date"
    return f"{n} days ({n / 7:.1f} weeks, {n / 30.4375:.1f} months) before the question date"


def _line(r, q_date=None, ago=False):
    tail = f"; {_ago(_d(r['start']), q_date)}" if ago and q_date and r.get("start") else ""
    qty = f" [{r['qty']:g} {r['unit']}]".rstrip() if r.get("qty") is not None else ""
    return f"- {r['who']} {r['what']} {r['item']}{qty} ({_when_str(r)}{tail}; turn {r['dia']})"


HEADER = "LEDGER (facts extracted from the whole history and computed by code; may be incomplete or include near-misses, so check them against the memory lines):"


def operate(route_name, rows, question, q_date=None) -> Block | None:
    """Matched rows -> one finished computation. q_date: the question's date (LongMemEval) or None (LoCoMo)."""
    if not rows:
        return None
    q = parse_date(q_date) if q_date else None
    by_time = sorted(rows, key=lambda r: (r.get("start") or "", r.get("mention") or ""))
    if route_name == "COUNT":
        seen, uniq = set(), []
        for r in by_time:
            k = norm_item(r["item"]) or norm_item(r["what"])
            if k not in seen:
                seen.add(k); uniq.append(r)
        body = [f"{len(uniq)} distinct matching items, oldest first:"] + [_line(r) for r in uniq]
        return Block("COUNT", "\n".join(body), uniq, float(len(uniq)))
    if route_name == "SUM":
        q_rows = [r for r in by_time if r.get("qty") is not None]
        if not q_rows:
            return operate("COUNT", rows, question, q_date)
        units = {}
        for r in q_rows:
            units.setdefault(r["unit"] or "", []).append(r)
        unit, rs = max(units.items(), key=lambda kv: len(kv[1]))
        tot = sum(r["qty"] for r in rs)
        body = [f"total {tot:g} {unit}".rstrip() + f" over {len(rs)} matching items:"] + [_line(r) for r in rs]
        return Block("SUM", "\n".join(body), rs, tot)
    if route_name == "DATEDIFF":
        pick = by_time[:6] if len(by_time) <= 6 else sorted(rows[:6], key=lambda r: r.get("start") or "")
        body = ["matching dated items, oldest first:"] + [_line(r, q, ago=q is not None) for r in pick]
        if len(pick) >= 2:
            a, b = _d(pick[0]["start"]), _d(pick[-1]["start"])
            if a and b:
                n = (b - a).days
                body.append(f"first to last: {n} days ({n / 7:.1f} weeks, {n / 30.4375:.1f} months)")
        return Block("DATEDIFF", "\n".join(body), pick, None)
    if route_name == "ORDER":
        body = ["matching items in date order:"] + [_line(r) for r in by_time[:8]]
        return Block("ORDER", "\n".join(body), by_time[:8], None)
    if route_name == "LATEST":
        st = [r for r in by_time if r["kind"] == "state"] or by_time
        body = ["matching values over time, oldest first (the last one is the latest):"] + [_line(r) for r in st[-6:]]
        return Block("LATEST", "\n".join(body), st[-6:], None)
    if route_name == "PREF":
        body = ["the user's own preferences and details that may matter:"] + [_line(r) for r in by_time[-8:]]
        return Block("PREF", "\n".join(body), by_time[-8:], None)
    return None


def render(block: Block | None) -> str:
    return "" if block is None else f"{HEADER}\n{block.text}"

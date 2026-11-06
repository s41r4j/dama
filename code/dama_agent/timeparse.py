"""Write-time resolution of relative time expressions (Tier 0: deterministic, no model).

Why (paper round 2, 2026-10-28): with every evidence turn in the packet, the reader still answered LoCoMo temporal
questions with the speaker's own words ("Last Friday.", "Next month.") instead of a date: 26.5% temporal accuracy for
DAMA-H, 27.6% for the full history. The date a turn was said is known when it is stored, so the memory resolves
"yesterday" once, at write time, and the packet carries both: `yesterday [= 16 November 2023]`.

Only expressions with one clear reading are resolved. Ambiguous ones are left alone: a bare weekday ("on Friday"),
"next Friday" and "next weekend" (this coming one, or the one after?), "in two weeks" (a date, or a duration as in
"lost 5 kg in two weeks"?), "this summer" said outside summer, "recently", "the other day", "the last month" (the
past 30 days, not the previous calendar month) and "last week of school" (the final week, not the previous one). Vague counts keep their vagueness:
"a few days ago" -> "a few days before 17 November 2023".
"""
from __future__ import annotations
import re
from datetime import date, timedelta

MONTHS = "January February March April May June July August September October November December".split()
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
       "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "couple of": 2}
VAGUE = ("a few", "few", "several", "some")
SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}

_N = r"(\d+|a couple of|a couple|couple of|a few|few|several|some|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT = r"(day|week|month|year)s?"
_WD = "|".join(WEEKDAYS)
# order matters: longer phrases first, so "the day before yesterday" wins over "yesterday"
_PATTERNS = [
    ("day_before_yesterday", r"the day before yesterday"),
    ("day_after_tomorrow", r"the day after tomorrow"),
    ("yesterday", r"yesterday|last night"),
    ("today", r"today|tonight|this morning|this afternoon|this evening"),
    ("tomorrow", r"tomorrow"),
    ("ago", _N + r"\s+" + _UNIT + r"\s+ago"),
    ("last_wd", r"(?:last|this past)\s+(" + _WD + r")"),
    ("weekend", r"(last|this)\s+weekend"),
    ("rel", r"(last|this|next)\s+(week|month|year)(?!\s+of\b)"),
    ("season", r"(last|this|next)\s+(spring|summer|fall|autumn|winter)"),
]
_RX = re.compile("|".join(f"(?P<{name}>\\b(?:{p})\\b)" for name, p in _PATTERNS), re.IGNORECASE)
_PRECEDED_BY_THE = re.compile(r"\b(the|over the|for the|in the|during the|past)\s*$", re.IGNORECASE)


def parse_date(raw) -> date | None:
    """A session or turn date: LoCoMo '1:56 pm on 8 May, 2023', LongMemEval '2023/05/20 (Sat) 02:21', ISO '2023-05-20'."""
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    s = str(raw)
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        return _safe(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s)
    if m and m.group(2).capitalize() in MONTHS:
        return _safe(int(m.group(3)), MONTHS.index(m.group(2).capitalize()) + 1, int(m.group(1)))
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1).capitalize() in MONTHS:
        return _safe(int(m.group(3)), MONTHS.index(m.group(1).capitalize()) + 1, int(m.group(2)))
    return None


def _safe(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def fmt_day(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def fmt_month(y: int, m: int) -> str:
    y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    return f"{MONTHS[m - 1]} {y}"


def _count(word: str):
    w = word.lower()
    if w.isdigit():
        return int(w)
    return NUM.get(w)


def _shift(ref: date, n: int, unit: str, sign: int, vague: str | None) -> str:
    """ref moved by n units (sign -1 = past). Days give a date; weeks an approximate date; months a month; years a year."""
    where = "before" if sign < 0 else "after"
    if vague:
        return f"{vague} {unit}s {where} {fmt_day(ref)}"
    if unit == "day":
        return fmt_day(ref + timedelta(days=sign * n))
    if unit == "week":
        return f"about {fmt_day(ref + timedelta(days=sign * 7 * n))}"
    if unit == "month":
        return fmt_month(ref.year, ref.month + sign * n)
    return str(ref.year + sign * n)


def _weekend(ref: date, which: str) -> str:
    wd = ref.weekday()                                   # Monday 0 ... Sunday 6
    this_sat = ref + timedelta(days=5 - wd)              # Saturday of ref's week (may be before ref on Sunday)
    sat = this_sat - timedelta(days=7) if which == "last" else this_sat
    sun = sat + timedelta(days=1)
    span = f"{sat.day}-{sun.day} {MONTHS[sun.month - 1]} {sun.year}" if sat.month == sun.month else f"{fmt_day(sat)} - {fmt_day(sun)}"
    return f"the weekend of {span}"


def _season(ref: date, which: str, name: str) -> str:
    start, end = SEASONS[name]
    label = "autumn" if name == "fall" else name
    def year_of(y):                                      # winter spans two years: name it by the December
        return f"{label} {y}" if name != "winter" else f"winter {y}/{y + 1}"
    # the season's latest start on or before ref
    y = ref.year if (ref.month, ref.day) >= (start, 1) else ref.year - 1
    in_it = (ref.month >= start or ref.month <= end) if start > end else (start <= ref.month <= end)
    if which == "this":
        return year_of(y) if in_it else None             # outside the season "this summer" may be past or coming
    if which == "last":
        return year_of(y - 1 if in_it else y)
    return year_of(y + 1)


def resolve(kind: str, m: re.Match, ref: date) -> str | None:
    g = m.group(0).lower()
    if kind == "day_before_yesterday":
        return fmt_day(ref - timedelta(days=2))
    if kind == "day_after_tomorrow":
        return fmt_day(ref + timedelta(days=2))
    if kind == "yesterday":
        return fmt_day(ref - timedelta(days=1))
    if kind == "today":
        return fmt_day(ref)
    if kind == "tomorrow":
        return fmt_day(ref + timedelta(days=1))
    if kind == "ago":
        mm = re.match(_N + r"\s+" + _UNIT, g)
        if not mm:
            return None
        word, unit = mm.group(1), mm.group(2)
        vague = word if word in VAGUE else None
        n = None if vague else _count(word)
        if n is None and not vague:
            return None
        return _shift(ref, n or 0, unit, -1, vague)
    if kind == "last_wd":
        wd = WEEKDAYS.index(re.search(_WD, g).group(0))
        back = (ref.weekday() - wd) % 7 or 7
        return fmt_day(ref - timedelta(days=back))
    if kind == "weekend":
        return _weekend(ref, g.split()[0])
    if kind == "rel":
        which, unit = g.split()[0], g.split()[-1]
        if unit == "week":
            return {"last": f"the week before {fmt_day(ref)}", "this": f"the week of {fmt_day(ref)}",
                    "next": f"the week after {fmt_day(ref)}"}[which]
        if unit == "month":
            return fmt_month(ref.year, ref.month + {"last": -1, "this": 0, "next": 1}[which])
        return str(ref.year + {"last": -1, "this": 0, "next": 1}[which])
    if kind == "season":
        which, name = g.split()
        return _season(ref, which, name)
    return None


def find(text: str, ref) -> list[tuple[int, int, str]]:
    """(start, end, resolved) for every resolvable expression in text, said on date ref."""
    ref = parse_date(ref)
    if ref is None or not text:
        return []
    out = []
    for m in _RX.finditer(text):
        kind = m.lastgroup
        if kind in ("rel", "season") and _PRECEDED_BY_THE.search(text[:m.start()]):
            continue                                     # "the last month", "over the past year": a span, not a calendar unit
        if text[m.end():m.end() + 2] == " [":
            continue                                     # already annotated
        r = resolve(kind, m, ref)
        if r:
            out.append((m.start(), m.end(), r))
    return out


def annotate(text: str, ref) -> str:
    """text with `expression [= resolved]` after each resolvable relative time expression; unchanged if ref is unknown."""
    hits = find(text, ref)
    for s, e, r in reversed(hits):
        text = f"{text[:e]} [= {r}]{text[e:]}"
    return text


TIME_HINT = re.compile(r"\b(?:\d{4}|yesterday|today|tonight|tomorrow|ago|last|next|weekend|week|month|year|"
                       r"january|february|march|april|may|june|july|august|september|october|november|december|"
                       r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|evening|night)\b", re.IGNORECASE)
TEMPORAL_QUERY = re.compile(r"^\s*when\b|\bwhat (?:year|month|date|day|time)\b|\bhow long\b|\bhow many (?:days|weeks|months|years)\b|"
                            r"\b(?:before|after|ago|since|until|first|last|recently|latest|earlier|later)\b", re.IGNORECASE)

"""DAMA v4 read-path checks (docs/research/DAMA_v4_plan_2026-11-02.md). CPU only, no model, a few ms per question.

DMBM run 1 error analysis (DAMA A0, dev): 36 of 47 wrong LoCoMo answers had every gold evidence turn in the packet, so
the misses were the reader's, not retrieval's. Three error kinds are worth a packet-level fix:

1. Wrong-person premise (LoCoMo "adversarial"): the question names person S, but the matching memories are person O
   speaking about O's own life. A0 answered the swapped question or wrote "S did not; O did".
   -> attribution check: one CHECK line in the packet names who said the matching lines.
2. List questions ("what books has ...", "both", "how many"): the reader stopped at the first item, or the packet held
   only some of the items. -> coverage: matching turns from all sessions are swapped in for the packet's least
   related lines, so the packet never grows.
3. Over-abstention and inference questions: handled by the reader prompt (P3), not here.

Everything works on plain turn dicts {"dia", "speaker", "text"} plus the packet's line ids, so the same code runs on
stored packets (offline replay) and inside a Kaggle run.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field

_STOP = set("""a an the of to in on at for with and or but is are was were be been being do does did done has have had
having i you he she it we they his her its their my your our me him them this that these those what which who whom whose
when where why how would could should might may can will shall from by as about into over after before during since until
than then there here also any some all both each other more most such only own same so too very just not no nor if while
once again ever up out off get got go went make made take took like one ones thing things kind kinds type types way ways
lot lots really much many s t don didn doesn isn wasn aren weren hasn haven re ve ll d m let let's""".split())
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|we|we're|we've|our|ours|us)\b", re.I)
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

LIST_Q = [re.compile(p, re.I) for p in (
    r"\bboth\b", r"\bhow many\b", r"\bnames?\s+of\b", r"\bwhat (?:are|were) (?:the|some|all)\b",
    r"\b(?:what|which)\s+(?:\w+\s+){0,2}\w{3,}s\s+(?:has|have|did|does|do|is|are|was|were)\b",
    r"\bwhat\s+(?:has|have)\s+\w+\s+\w+(?:ed|en|done|read|made|seen|painted|bought)\b",
    r"\bwhat (?:kinds?|types?|sorts?) of\b", r"\bin what ways\b",
)]


def stem(w: str) -> str:
    """Crude, but the same on both sides: turtle/turtles -> turtl, hike/hiking -> hik, story/stories -> story."""
    w = w.lower().split("'")[0]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and re.search(r"(?:s|x|z|ch|sh)es$", w):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def is_list_question(q: str) -> bool:
    return any(p.search(q) for p in LIST_Q)


@dataclass
class History:
    """One conversation's turns with an idf table over their stems (built once per history)."""
    turns: list[dict]
    speakers: list[str] = field(default_factory=list)
    idf: dict = field(default_factory=dict)
    stems: list[set] = field(default_factory=list)

    def __post_init__(self):
        self.stems = [{stem(w) for w in words(t["text"])} for t in self.turns]
        df = Counter(s for ss in self.stems for s in ss)
        n = max(1, len(self.turns))
        self.idf = {s: math.log((n + 1) / (c + 0.5)) for s, c in df.items()}
        if not self.speakers:
            self.speakers = [s for s, _ in Counter(t["speaker"] for t in self.turns).most_common()]


def named_speakers(q: str, h: History) -> list[str]:
    """Speakers named in the question, in order of first mention (full name, possessive, or a 3+ letter nickname such
    as "Mel" are not used: questions name people in full)."""
    low = q.lower()
    hits = []
    for s in h.speakers:
        m = re.search(r"\b" + re.escape(s.lower()) + r"(?:'s)?\b", low)
        if m:
            hits.append((m.start(), s))
    return [s for _, s in sorted(hits)]


def topic_terms(q: str, h: History) -> set[str]:
    names = {s.lower() for s in h.speakers}
    return {stem(w) for w in words(q) if w not in _STOP and w.split("'")[0] not in names and len(w) > 2} - {""}


def scores(q: str, h: History) -> list[float]:
    """Share of the question's topic weight (idf) that each turn covers, 0..1."""
    T = [t for t in topic_terms(q, h) if t in h.idf]
    tot = sum(h.idf[t] for t in T)
    if not tot:
        return [0.0] * len(h.turns)
    return [sum(h.idf[t] for t in T if t in ss) / tot for ss in h.stems]


def mentions(text: str, person: str) -> bool:
    p = person.lower()
    low = (text or "").lower()
    return bool(re.search(r"\b" + re.escape(p) + r"\b", low) or re.search(r"\b" + re.escape(p[:3]) + r"\b", low))


_SENT = re.compile(r"(?<=[.!?])\s+")
_SECOND_PERSON = re.compile(r"\b(you|you're|you've|you'd|you'll|your|yours|yourself)\b", re.I)


def about_whom(q: str, h: History, i: int) -> str | None:
    """Who the best-matching sentence of turn i is about: "I/my" = its speaker, "you/your" = the other speaker of a
    two-person conversation, both = "both", neither = None. A turn that states something about its speaker and then
    asks it back ("I love sci-fi movies! What's your favorite genre?") is about both."""
    t = h.turns[i]
    T = [x for x in topic_terms(q, h) if x in h.idf]
    sents = [s for s in _SENT.split(t["text"]) if s.strip()] or [t["text"]]
    cov = lambda s: sum(h.idf[x] for x in T if x in {stem(w) for w in words(s)})
    sent = max(sents, key=cov)
    first, second = bool(_FIRST_PERSON.search(sent)), bool(_SECOND_PERSON.search(sent))
    other = next((s for s in h.speakers if s != t["speaker"]), None)
    if first and second:
        return "both"
    if second and sent.rstrip().endswith("?") and any(_FIRST_PERSON.search(s) and not s.rstrip().endswith("?")
                                                       for s in sents if s is not sent):
        return "both"
    return t["speaker"] if first else (other if second else None)


@dataclass
class CheckResult:
    ids: list[str]                 # packet line ids after coverage changes, in history order
    header: str = ""               # CHECK lines for the top of the packet ("" = none)
    attribution: str | None = None  # the other speaker, when the wrong-person check fired
    list_q: bool = False
    added: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


ATTR_MIN, ATTR_REL, ATTR_SUPPORT = 0.6, 0.8, 0.55         # a "matching" turn covers >= 50% of the topic weight and >= 80% of the best turn's
COVER_MIN, COVER_REL, COVER_MAX = 0.5, 0.6, 8


def check(q: str, h: History, packet_ids: list[str], ntok=None, budget: int | None = None,
          attribution: bool = True, coverage: bool = True) -> CheckResult:
    """packet_ids: the dia ids of the base packet (A0). ntok(text) counts tokens; budget caps the packet after coverage
    (default: the base packet's own size, so the packet never grows)."""
    sc = scores(q, h)
    pos = {t["dia"]: i for i, t in enumerate(h.turns)}
    ids = [d for d in packet_ids if d in pos]
    res = CheckResult(ids=list(ids), list_q=is_list_question(q))
    best = max(sc) if sc else 0.0
    named = named_speakers(q, h)

    # 1. wrong-person premise: exactly one speaker is named, and every strongly matching turn is about the OTHER
    #    speaker (LoCoMo's best match is often the other person asking: "your new hair color looks amazing!")
    if attribution and len(named) == 1 and len(h.speakers) == 2 and best >= ATTR_MIN:
        subj = named[0]
        other = next(s for s in h.speakers if s != subj)
        strong = [i for i, s in enumerate(sc) if s >= max(ATTR_MIN, ATTR_REL * best)]
        about = {i: about_whom(q, h, i) for i in strong}
        # any fair match about the asked-about speaker (their own statement) vetoes the check
        support = [i for i, s in enumerate(sc) if s >= ATTR_SUPPORT * best and i not in about
                   and about_whom(q, h, i) in (subj, "both")]
        if other in about.values() and subj not in about.values() and "both" not in about.values() and not support:
            res.attribution = other
            show = sorted(strong, key=lambda i: -sc[i])[:3]
            for i in show:                               # the reader must see the lines the check talks about
                if h.turns[i]["dia"] not in res.ids:
                    res.ids.append(h.turns[i]["dia"]); res.added.append(h.turns[i]["dia"])
            tags = ", ".join(f"[m:{h.turns[i]['dia']}]" for i in sorted(show))
            res.header = (f"CHECK: the memories that match this question ({tags}) are about {other}, not {subj}; "
                          f"no memory says this about {subj}.")

    # 2. list questions: matching turns of the asked-about speaker(s) from every session
    if coverage and res.list_q and res.attribution is None and best >= COVER_MIN:
        who = set(named) or set(h.speakers)
        cand = [i for i, s in sorted(enumerate(sc), key=lambda x: -x[1])
                if s >= max(COVER_MIN, COVER_REL * best)
                and (h.turns[i]["speaker"] in who or any(mentions(h.turns[i]["text"], p) for p in who))]
        for i in cand[:COVER_MAX]:
            if h.turns[i]["dia"] not in res.ids:
                res.ids.append(h.turns[i]["dia"]); res.added.append(h.turns[i]["dia"])

    # keep the packet within budget: drop the least related base lines (never one just added)
    if res.added and ntok is not None:
        cap = budget if budget is not None else sum(ntok(h.turns[pos[d]]["text"]) + 12 for d in ids)
        keep = set(res.added)
        size = lambda: sum(ntok(h.turns[pos[d]]["text"]) + 12 for d in res.ids)
        for d in sorted([d for d in res.ids if d not in keep], key=lambda d: (sc[pos[d]], pos[d])):
            if size() <= cap:
                break
            res.ids.remove(d); res.dropped.append(d)
    res.ids.sort(key=lambda d: pos[d])
    return res

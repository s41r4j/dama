"""Training rows for the op model: (existing memory, new statement) -> ADD / UPDATE / DELETE / NOOP.

Three sources, each tagged so metrics are reported per source:
* synthetic : slot templates with functional slots (one current value: city, job, database ...) and multi-valued
              slots (hobbies, pets ...). Templates and values are partitioned by split, so val/test use wording
              and entities never seen in training. Hard cases: implicit updates, hedges and hearsay (-> NOOP),
              new values of multi-valued slots (-> ADD, not UPDATE), retractions (-> DELETE).
              L1.1 adds implicit updates: a new value of a one-value slot with no "changed from" cue ("I relocated to
              X", "I live in X") -> UPDATE, against plain new values of many-value slots -> ADD. L1.0 had almost
              none and answered ADD for "lives in Berlin" vs "I moved to Pune".
* mnli      : MNLI pairs mapped entailment->NOOP, neutral->ADD, contradiction->UPDATE. Broad language coverage.
* lme_ku    : real LongMemEval knowledge-update evidence: (older evidence turn, newer evidence turn) -> UPDATE.
              Small, but it is the only real-dialogue update signal, so it is kept mostly for evaluation.
"""
from __future__ import annotations
import random
import re
from collections import Counter
from datetime import datetime
from ..config import OPS, SPLITS, DataConfig
from .schema import Dialogue, Question
from .splits import hashed_split, split_of_dialogue

ADD, UPDATE, DELETE, NOOP = (OPS.index(x) for x in OPS)

# slot: (functional?, subject, values, user statements, stored memory form)
SLOTS = {
    "city": (True, "where I live", ["Berlin", "Toronto", "Mumbai", "Austin", "Lisbon", "Seoul", "Nairobi", "Melbourne", "Denver", "Dublin", "Pune", "Oslo", "Lagos", "Kyoto"],
             ["I live in {v}", "I'm based in {v}", "home for me is {v} these days", "I've been living in {v}", "I stay in {v}"], "The user lives in {v}."),
    "job": (True, "my job", ["data scientist", "nurse", "teacher", "product manager", "backend engineer", "designer", "accountant", "lawyer", "pharmacist", "researcher", "chef", "electrician"],
            ["I work as a {v}", "I'm a {v} by profession", "my day job is {v}", "professionally I'm a {v}", "I've been working as a {v}"], "The user works as a {v}."),
    "employer": (True, "where I work", ["Infosys", "a small startup", "the city hospital", "Siemens", "a law firm", "the university", "Flipkart", "a bakery", "the bank", "Shopify", "a game studio"],
                 ["I work at {v}", "I'm employed at {v}", "my employer is {v}", "I joined {v}", "I'm with {v} now"], "The user works at {v}."),
    "database": (True, "our database", ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis", "Cassandra", "DynamoDB", "MariaDB", "CockroachDB", "Neo4j", "Firestore"],
                 ["the project uses {v} as its database", "we store all the data in {v}", "our database is {v}", "the backend runs on {v}", "we went with {v} for storage"], "The project database is {v}."),
    "framework": (True, "our frontend framework", ["React", "Vue", "Svelte", "Angular", "SolidJS", "Next.js", "Nuxt", "Remix", "Preact", "Qwik"],
                  ["the frontend is built with {v}", "we use {v} for the UI", "our web app runs on {v}", "we picked {v} for the frontend", "the client is written in {v}"], "The frontend uses {v}."),
    "hosting": (True, "our hosting", ["AWS", "Google Cloud", "Azure", "Fly.io", "Heroku", "DigitalOcean", "Vercel", "Netlify", "Render", "Hetzner"],
                ["we deploy to {v}", "the app is hosted on {v}", "production lives on {v}", "our infra runs on {v}", "we host everything on {v}"], "The app is hosted on {v}."),
    "launch": (True, "the launch date", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
               ["we plan to launch in {v}", "the release is scheduled for {v}", "we're aiming to ship in {v}", "go-live is planned for {v}", "launch is in {v}"], "The launch is planned for {v}."),
    "diet": (True, "my diet", ["vegetarian", "vegan", "pescatarian", "keto", "gluten-free", "on no special diet", "dairy-free", "halal"],
             ["I'm {v}", "I eat {v}", "my diet is {v}", "I've been {v} for a while", "I follow a {v} diet"], "The user's diet: {v}."),
    "phone": (True, "my phone", ["an iPhone 15", "a Pixel 8", "a Galaxy S24", "a OnePlus 12", "a Nothing Phone", "an iPhone 13", "a Moto G"],
              ["I use {v}", "my phone is {v}", "I'm on {v}", "I carry {v}", "I switched to {v}"], "The user's phone is {v}."),
    "meeting": (True, "my preferred meeting time", ["in the morning", "after lunch", "before noon", "late afternoon", "on Tuesdays", "on Fridays", "early in the week"],
                ["I prefer meetings {v}", "please book my calls {v}", "meetings work best for me {v}", "schedule syncs {v} for me", "I like calls {v}"], "The user prefers meetings {v}."),
    "hobby": (False, "my hobbies", ["hiking", "pottery", "chess", "painting", "rock climbing", "baking", "photography", "cycling", "gardening", "salsa dancing", "birdwatching", "knitting"],
              ["I love {v}", "I've been getting into {v}", "{v} is my favourite hobby", "I spend weekends on {v}", "I enjoy {v}"], "The user enjoys {v}."),
    "pet": (False, "my pets", ["a golden retriever", "two cats", "a parrot", "a rabbit", "a beagle", "a goldfish", "a tortoise", "a husky"],
            ["I have {v}", "I adopted {v}", "we have {v} at home", "I live with {v}", "I look after {v}"], "The user has {v}."),
    "allergy": (False, "my allergies", ["peanuts", "shellfish", "pollen", "penicillin", "dust", "lactose", "cats", "gluten"],
                ["I'm allergic to {v}", "I have an allergy to {v}", "{v} make me sick", "I react badly to {v}", "I can't have {v}, allergy"], "The user is allergic to {v}."),
    "spoken": (False, "languages I speak", ["Spanish", "Hindi", "French", "Japanese", "German", "Marathi", "Portuguese", "Mandarin", "Arabic"],
               ["I speak {v}", "I'm fluent in {v}", "I can talk in {v}", "I grew up speaking {v}", "I learned {v}"], "The user speaks {v}."),
}
CHANGES = ["{s_new}, I moved on from {old}", "I switched from {old} to {new}", "it's {new} now, not {old}", "correction: {new}, no longer {old}",
           "we replaced {old} with {new}", "update: {s_new}", "{s_new} now", "{old} is out, {new} is in", "things changed, {s_new}"]
# How people announce a new value of a one-value slot without naming the old one. Lists of 8+ are held out by split.
MOVES = {
    "city": ["I relocated to {v}", "we just settled in {v}", "I've packed up and I'm now in {v}", "my new place is in {v}",
             "I'm living in {v} since the spring", "finally made the move to {v}", "I'm a {v} resident now", "we bought a flat in {v}"],
    "job": ["I just started a new role as a {v}", "I got hired as a {v}", "I changed careers, I'm a {v}", "my new job title is {v}",
            "I'm working as a {v} since last week", "I took a position as a {v}", "I switched careers to become a {v}", "they made me a {v}"],
    "employer": ["I just joined {v}", "I accepted an offer from {v}", "my first day at {v} was Monday", "I'm now on the payroll at {v}",
                 "I left my old company for {v}", "I signed with {v}", "starting at {v} next week", "I got a job at {v}"],
}
GENERIC_MOVES = ["we've migrated to {v}", "it's {v} from now on", "we went over to {v} last sprint", "these days it's {v}",
                 "we settled on {v} instead", "recently changed to {v}", "starting this month: {v}", "lately, {s_new}",
                 "as of today {s_new}", "{s_new} since last month"]
TECH_SLOTS = {"database", "framework", "hosting", "phone", "employer"}
WHEN = ["", "", " last month", " recently", " a few weeks ago", " this year", " last week"]
ALSO = ["I also {verb} {v}", "{s_v} too", "besides that, {s_v}", "on top of that, {s_v}", "another one: {s_v}",
        "and {s_v} as well", "in addition, {s_v}", "add this: {s_v}", "one more thing, {s_v}"]
RETRACT = ["please forget {subject}", "you can delete what you know about {subject}", "that note about {subject} is wrong, remove it",
           "stop remembering {subject}", "erase {subject} from memory", "drop the memory about {subject}",
           "don't keep {subject} on file", "wipe whatever you saved about {subject}", "ignore what I told you about {subject}"]
RETRACT_VALUE = ["I don't have {v} anymore", "I'm no longer into {v}", "{v} is not a thing for me anymore", "scratch {v}, that's over",
                 "remove {v} from my list", "{v} is no longer part of my life", "forget about {v}, that ended", "I gave up {v}",
                 "{v} doesn't apply to me now"]
HEDGE = ["maybe {s_v}, not sure yet", "I might {soft}, still deciding", "possibly {s_v}, I need to check", "thinking about it: {s_v}? no idea yet",
         "if things work out, {s_v}", "not decided, but perhaps {s_v}", "there's a chance {s_v}", "I'm considering it: {s_v}", "we may {soft}, unclear"]
HEARSAY = ["my colleague thinks {s_v}", "someone said {s_v}, unconfirmed", "a friend guessed {s_v}", "I heard a rumour that {s_v}",
           "my manager claims {s_v}, no idea if true", "on Slack someone wrote {s_v}", "the intern believes {s_v}",
           "apparently {s_v}, according to gossip", "a contractor mentioned {s_v}"]
CHATTER = ["Thanks, that worked.", "Can you refactor this function?", "lol nice", "What's the capital of France?", "Good morning!", "Run the tests again.",
           "Write a haiku about rain.", "ok", "How do I center a div?", "Let's take a break.", "That's hilarious.", "Explain recursion simply.",
           "Translate 'hello' to Spanish.", "Cool, thanks!", "Make the button blue.", "What does HTTP 404 mean?"]


def held(items, split):
    """Partition a list across splits: train ~60%, others ~13% each; lists shorter than 8 are shared."""
    n = len(items)
    if n < 8:
        return list(items)
    k = max(1, n // 8)
    cuts = {"train": (0, n - 3 * k), "val": (n - 3 * k, n - 2 * k), "calib": (n - 2 * k, n - k), "test": (n - k, n)}[split]
    return list(items[cuts[0]:cuts[1]])


def cap(s):
    return s[:1].upper() + s[1:]


class OpGenerator:
    def __init__(self, split, rng):
        self.split, self.rng = split, rng

    def pick(self, items):
        return self.rng.choice(held(items, self.split))

    def value(self, slot, exclude=()):
        pool = [v for v in held(SLOTS[slot][2], self.split) if v not in exclude] or [v for v in SLOTS[slot][2] if v not in exclude]
        return self.rng.choice(pool)

    def said(self, slot, v):
        return self.pick(SLOTS[slot][3]).format(v=v)

    def stored(self, slot, v):
        """Existing memory: either the canonical fact or an earlier verbatim user turn."""
        return SLOTS[slot][4].format(v=v) if self.rng.random() < 0.5 else cap(self.said(slot, v)) + "."

    def row(self, kind):
        slot = self.rng.choice(list(SLOTS))
        functional, subject = SLOTS[slot][0], SLOTS[slot][1]
        v = self.value(slot)
        if kind == "update":
            slot = self.rng.choice([s for s in SLOTS if SLOTS[s][0]]); subject = SLOTS[slot][1]
            old, new = self.value(slot), None
            new = self.value(slot, exclude={old})
            t = self.pick(CHANGES)
            text = t.format(s_new=self.said(slot, new), old=old, new=new)
            return self.stored(slot, old), cap(text), UPDATE, f"update/{t}"
        if kind == "implicit_update":
            slot = self.rng.choice([s for s in SLOTS if SLOTS[s][0]]); subject = SLOTS[slot][1]
            old = self.value(slot); new = self.value(slot, exclude={old})
            r = self.rng.random()
            if r < 0.35:  # only the bare statement gets a time phrase; the other frames carry their own
                t = "plain"; text = self.said(slot, new) + self.rng.choice(WHEN)
            elif r < 0.7 and slot in MOVES:
                t = self.pick(MOVES[slot]); text = t.format(v=new)
            else:  # "{v}"-only frames read oddly for personal slots ("changed to in the morning"), so those get "{s_new}" frames
                frames = GENERIC_MOVES if slot in TECH_SLOTS else [g for g in GENERIC_MOVES if "{s_new}" in g]
                t = self.pick(frames); text = t.format(v=new, s_new=self.said(slot, new))
            return self.stored(slot, old), cap(text), UPDATE, f"update/implicit/{t}"
        if kind == "add_multi_plain":
            slot = self.rng.choice([s for s in SLOTS if not SLOTS[s][0]])
            first = self.value(slot); second = self.value(slot, exclude={first})
            return self.stored(slot, first), cap(self.said(slot, second)), ADD, f"add/multi_plain/{slot}"
        if kind == "add_unrelated":
            other = self.rng.choice([s for s in SLOTS if s != slot])
            return self.stored(other, self.value(other)), cap(self.said(slot, v)), ADD, f"add/unrelated/{slot}"
        if kind == "add_multi":
            slot = self.rng.choice([s for s in SLOTS if not SLOTS[s][0]])
            first, second = self.value(slot), None
            second = self.value(slot, exclude={first})
            t = self.pick(ALSO)
            verb = {"hobby": "enjoy", "pet": "have", "allergy": "react to", "spoken": "speak"}[slot]
            text = t.format(verb=verb, v=second, s_v=self.said(slot, second))
            return self.stored(slot, first), cap(text), ADD, f"add/multi/{t}"
        if kind == "duplicate":
            return self.stored(slot, v), cap(self.said(slot, v)), NOOP, f"noop/duplicate/{slot}"
        if kind == "hedge":
            slot = self.rng.choice([s for s in SLOTS if SLOTS[s][0]]); old = self.value(slot); new = self.value(slot, exclude={old})
            t = self.pick(HEDGE)
            text = t.format(s_v=self.said(slot, new), soft=f"go with {new}")
            return self.stored(slot, old), cap(text), NOOP, f"noop/hedge/{t}"
        if kind == "hearsay":
            slot = self.rng.choice([s for s in SLOTS if SLOTS[s][0]]); old = self.value(slot); new = self.value(slot, exclude={old})
            t = self.pick(HEARSAY)
            return self.stored(slot, old), cap(t.format(s_v=self.said(slot, new))), NOOP, f"noop/hearsay/{t}"
        if kind == "chatter":
            t = self.pick(CHATTER)
            return self.stored(slot, v), t, NOOP, f"noop/chatter/{t}"
        if kind == "delete_slot":
            t = self.pick(RETRACT)
            return self.stored(slot, v), cap(t.format(subject=subject)), DELETE, f"delete/slot/{t}"
        if kind == "delete_value":
            slot = self.rng.choice([s for s in SLOTS if not SLOTS[s][0]]); v = self.value(slot)
            t = self.pick(RETRACT_VALUE)
            return self.stored(slot, v), cap(t.format(v=v)), DELETE, f"delete/value/{t}"
        raise ValueError(kind)


KIND_WEIGHTS = {"update": 15, "implicit_update": 15, "add_unrelated": 15, "add_multi": 10, "add_multi_plain": 8, "duplicate": 12,
                "hedge": 6, "hearsay": 4, "chatter": 8, "delete_slot": 8, "delete_value": 7}


def parse_date(s):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s or "")
    return datetime(*map(int, m.groups())) if m else datetime.min


def build_op_rows(dialogues: list[Dialogue], questions: list[Question], mnli: list[dict], cfg: DataConfig):
    rows = {s: [] for s in SPLITS}
    kinds, weights = list(KIND_WEIGHTS), list(KIND_WEIGHTS.values())
    for split, n in cfg.synthetic_ops.items():
        rng = random.Random(f"{cfg.seed}-ops-{split}")
        g = OpGenerator(split, rng)
        for _ in range(n):
            a, b, y, group = g.row(rng.choices(kinds, weights)[0])
            rows[split].append({"task": "op", "a": a, "b": b, "y": y, "group": f"syn/{group}", "src": "synthetic"})

    nli_to_op = {0: NOOP, 1: ADD, 2: UPDATE}
    for r in mnli:
        split = hashed_split("mnli:" + r["premise"])
        rows[split].append({"task": "op", "a": r["premise"], "b": r["hypothesis"], "y": nli_to_op[int(r["label"])], "group": "mnli", "src": "mnli"})

    locomo_order = {d.did: i for i, d in enumerate(x for x in dialogues if x.source == "locomo")}
    turns = {t.tid: t for d in dialogues for t in d.turns}
    for q in questions:
        if q.qtype != "knowledge-update" or q.abstention or len(q.evidence) < 2:
            continue
        ev = sorted((turns[e] for e in q.evidence if e in turns), key=lambda t: parse_date(t.date))
        if len(ev) >= 2 and ev[0].session != ev[-1].session:
            split = split_of_dialogue(q.did, locomo_order)
            rows[split].append({"task": "op", "a": ev[0].text[:800], "b": ev[-1].text[:800], "y": UPDATE, "group": q.did, "src": "lme_ku"})

    for s in rows:
        random.Random(f"{cfg.seed}-shuffle-{s}").shuffle(rows[s])
    leak = template_leaks(rows)
    stats = {s: dict(Counter(f"{r['src']}:{OPS[r['y']]}" for r in rs)) for s, rs in rows.items()}
    stats["synthetic_template_groups_shared_across_splits"] = len(leak)
    return rows, stats


def template_leaks(rows):
    """Synthetic phrasing frames seen in more than one split. Slot statements and slot names are shared on
    purpose (there are only 14 slots); frames and slot values are held out, so this should report frames only."""
    seen = {}
    for s, rs in rows.items():
        for r in rs:
            if r["src"] == "synthetic" and not r["group"].split("/")[2] in ("duplicate", "unrelated", "multi_plain", "plain"):
                seen.setdefault(r["group"], set()).add(s)
    return [g for g, ss in seen.items() if len(ss) > 1]

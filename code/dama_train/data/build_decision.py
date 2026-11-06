"""Training rows for the decision model, built from real benchmark annotations.

rel   : (request, stored turn) -> 1 if the turn is annotated evidence for the request.
        Negatives are mostly BM25 hard negatives from the same conversation, the case a reranker must get right.
sal   : (turn, previous turn) -> 1 if the turn is evidence for any question about this conversation.
        This is a weak label: a turn nobody asked about may still be worth remembering. Fine for a first model;
        replace with teacher labels later if needed.
need  : request -> 1 for benchmark memory questions and casual memory requests (NEED_TEMPLATES), 0 for general
        instructions (Dolly-15k) and general questions (SQuAD, first-person Quora "How do I ...?"). The question
        negatives stop need from learning "benchmark question vs instruction" (it did in L1.0).
route : request -> multi-hot over ROUTES, mapped from the benchmark question type (see ROUTE_OF).

Pools (val/calib/test only): for each question, the BM25 top-k turns of its conversation, with gold flags.
They measure ranking quality (recall@k) and feed the conformal threshold.
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict
from ..config import ROUTES, SPLITS, DataConfig
from .bm25 import BM25
from .schema import Dialogue, Question, render_query, render_turn
from .build_ops import SLOTS, held
from .splits import hashed_split, split_of_dialogue

NOT_DIALOGUE = {"dolly", "questions", "need_tpl"}   # need rows that do not come from a conversation

# Casual requests that need the user's memory. Held out by split like the op templates; probes.py uses other wording.
NEED_TEMPLATES = [
    "What did I tell you about {subject}?", "Do you remember {subject}?", "Remind me what I said about {subject}.",
    "Can you recall {subject} from our earlier chats?", "Based on what you know about me, what's {subject}?",
    "Earlier I mentioned {subject}, what was it again?", "What have I shared with you about {subject}?",
    "Last time we talked about {subject}, what did I say?", "Did I ever mention {subject}?",
    "You should know {subject} by now, what is it?", "Going by our past conversations, what about {subject}?",
    "Quick check: what do you have on file about {subject}?", "From memory, tell me {subject}.",
    "What did we decide about {subject}?", "Remember when I told you about {subject}? Remind me.",
    "Using what I've told you before, summarize {subject}.",
]
PERSONAL_REQUESTS = [
    "Plan a weekend that suits my hobbies.", "Pick a restaurant I would like.", "Draft a bio for me using what you know.",
    "Is the thing I was worried about last week sorted?", "Continue where we left off yesterday.",
    "Any update on the issue I raised before?", "Suggest a gift for my partner, you know what they like.",
    "Write my status update based on what I did this week.", "What should I pack, given where I'm going?",
    "Schedule it at my usual time.", "Use my usual settings for the report.", "How is my plan going so far?",
    "Recommend a book similar to the ones I enjoyed.", "Send it to the same address as last time.",
    "What was the conclusion of our previous discussion?", "Which option did I prefer?",
]
NEED_TEMPLATE_ROWS = {"train": 600, "val": 100, "calib": 100, "test": 100}   # fewer than the real benchmark questions

ROUTE_OF = {
    # LoCoMo
    "single-hop": ["episodic", "facts"], "open-domain": ["episodic", "facts"], "adversarial": ["episodic", "facts"],
    "multi-hop": ["episodic", "facts", "expand"], "temporal": ["episodic", "timeline"],
    # LongMemEval
    "single-session-user": ["episodic", "facts"], "single-session-assistant": ["episodic"],
    "single-session-preference": ["facts", "episodic"], "temporal-reasoning": ["episodic", "timeline"],
    "knowledge-update": ["facts", "timeline"], "multi-session": ["episodic", "facts", "expand"],
}


def multi_hot(names):
    return [1.0 if r in names else 0.0 for r in ROUTES]


def need_template_rows(split, n, rng):
    subjects = held([v[1] for v in SLOTS.values()], split)
    frames, requests = held(NEED_TEMPLATES, split), held(PERSONAL_REQUESTS, split)
    out = []
    for _ in range(n):
        text = rng.choice(frames).format(subject=rng.choice(subjects)) if rng.random() < 0.7 else rng.choice(requests)
        out.append({"task": "need", "a": text, "b": "", "y": 1.0, "group": "need_tpl", "src": "need_tpl"})
    return out


def build_decision_rows(dialogues: list[Dialogue], questions: list[Question], dolly: list[dict], cfg: DataConfig,
                        general_questions: list[dict] = ()):
    rng = random.Random(cfg.seed)
    locomo_order = {d.did: i for i, d in enumerate(x for x in dialogues if x.source == "locomo")}
    by_did = {d.did: d for d in dialogues}
    split_of = {d.did: split_of_dialogue(d.did, locomo_order) for d in dialogues}
    rows = {s: [] for s in SPLITS}
    pools = {s: [] for s in SPLITS if s != "train"}
    eval_dialogues = {s: [] for s in SPLITS}   # train too: the retriever trains on whole dialogues (neighbour keys, mined negatives)
    q_by_did = defaultdict(list)
    for q in questions:
        q_by_did[q.did].append(q)

    for did, qs in q_by_did.items():
        d = by_did[did]
        split = split_of[did]
        texts = [render_turn(t, cfg.max_turn_chars) for t in d.turns]
        index = {t.tid: i for i, t in enumerate(d.turns)}
        bm25 = BM25(texts)
        evidence_any = set()
        eval_dialogues[split].append({"did": did, "src": d.source, "texts": texts, "questions": [
            {"qid": q.qid, "query": render_query(q), "qtype": q.qtype, "gold": [index[e] for e in q.evidence if e in index]}
            for q in qs if q.evidence and not (q.abstention and d.source == "lme")]})
        for q in qs:
            query = render_query(q)
            gold = [index[e] for e in q.evidence if e in index]
            evidence_any.update(gold)
            # route / need rows exist for every question, including abstention ones. They use the bare question:
            # only LongMemEval questions carry a date, so a "[date] " prefix would let need learn "dated = needs memory"
            rows[split].append({"task": "need", "a": q.text, "b": "", "y": 1.0, "group": did, "src": d.source})
            if q.qtype in ROUTE_OF:
                rows[split].append({"task": "route", "a": q.text, "b": "", "y": multi_hot(ROUTE_OF[q.qtype]), "group": did, "src": d.source})
            if not gold or (q.abstention and d.source == "lme"):
                continue
            ranked = bm25.top(query, cfg.pool_size + len(gold))
            if split == "train":
                hard = [i for i in ranked if i not in gold]
                n_neg = cfg.negatives_per_positive * len(gold)
                n_hard = min(len(hard), int(n_neg * cfg.hard_negative_fraction))
                rest = [i for i in range(len(texts)) if i not in gold and i not in hard[:n_hard]]
                negatives = hard[:n_hard] + rng.sample(rest, min(len(rest), n_neg - n_hard))
                for i in gold:
                    rows[split].append({"task": "rel", "a": query, "b": texts[i], "y": 1.0, "group": did, "src": d.source})
                for i in negatives:
                    rows[split].append({"task": "rel", "a": query, "b": texts[i], "y": 0.0, "group": did, "src": d.source})
            else:
                pool = ranked[:cfg.pool_size]
                pools[split].append({"qid": q.qid, "did": did, "src": d.source, "qtype": q.qtype, "query": query,
                                     "n_gold": len(gold), "gold_in_pool": sum(i in pool for i in gold),
                                     "cands": [{"text": texts[i], "gold": int(i in gold), "rank": r} for r, i in enumerate(pool)]})
                # val/calib/test still get rel rows (gold + same number of negatives as train) for AUC
                negatives = [i for i in pool if i not in gold][:cfg.negatives_per_positive * len(gold)]
                for i in gold:
                    rows[split].append({"task": "rel", "a": query, "b": texts[i], "y": 1.0, "group": did, "src": d.source})
                for i in negatives:
                    rows[split].append({"task": "rel", "a": query, "b": texts[i], "y": 0.0, "group": did, "src": d.source})

        # salience: evidence turns vs. other turns of the same dialogue
        others = [i for i in range(len(texts)) if i not in evidence_any]
        sampled = rng.sample(others, min(len(others), cfg.salience_negatives_per_positive * max(1, len(evidence_any))))
        for i in sorted(evidence_any) + sampled:
            prev = texts[i - 1][:300] if i > 0 and d.turns[i - 1].session == d.turns[i].session else ""
            rows[split].append({"task": "sal", "a": texts[i], "b": prev, "y": float(i in evidence_any), "group": did, "src": d.source})

    for r in dolly:
        text = r["instruction"].strip()
        if text:
            s = hashed_split("dolly:" + text)
            rows[s].append({"task": "need", "a": text, "b": "", "y": 0.0, "group": "dolly", "src": "dolly"})
    for r in general_questions:
        text = r["text"].strip()
        if text:
            rows[hashed_split("q:" + text)].append({"task": "need", "a": text, "b": "", "y": 0.0, "group": "questions", "src": r["src"]})
    for s, n in NEED_TEMPLATE_ROWS.items():
        rows[s] += need_template_rows(s, n, random.Random(f"{cfg.seed}-need-{s}"))

    for s in rows:
        rng.shuffle(rows[s])
    stats = {s: dict(Counter(f"{r['task']}:{r['src']}:{int(r['y']) if not isinstance(r['y'], list) else 'multi'}" for r in rows[s])) for s in rows}
    stats["pools"] = {s: {"questions": len(p), "gold_in_pool_rate": round(sum(x["gold_in_pool"] for x in p) / max(1, sum(x["n_gold"] for x in p)), 3)}
                      for s, p in pools.items()}
    leak = check_groups(rows)
    if leak:
        raise ValueError(f"dialogue groups appear in more than one split: {leak[:5]}")
    return rows, pools, eval_dialogues, stats


def check_groups(rows) -> list[str]:
    seen = defaultdict(set)
    for s, rs in rows.items():
        for r in rs:
            if r["group"] not in NOT_DIALOGUE:
                seen[r["group"]].add(s)
    return sorted(g for g, ss in seen.items() if len(ss) > 1)

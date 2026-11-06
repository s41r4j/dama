"""Floors and retrieval baselines over raw turns (no LLM at write time). Budget = reader packet size in tokens."""
from __future__ import annotations
import re

import numpy as np

from .base import NOT_FOUND, System, lines, ntok

BUDGET = 3000


def _flat(sessions):
    return [(s.date, f"{t.speaker}: {t.text}") for s in sessions for t in s.turns]


class NoMemory(System):
    name, native = "no_memory", "floor"

    def ingest(self, hid, sessions):
        pass

    def context(self, hid, question, q_date):
        return NOT_FOUND


class Recent(System):
    """The most recent turns that fit the budget (full context truncated to the window when budget = reader window)."""
    native = "floor"

    def __init__(self, budget=BUDGET, **kw):
        super().__init__(**kw)
        self.budget, self.name, self.mem = budget, f"recent_{budget}", {}

    def ingest(self, hid, sessions):
        self.mem[hid] = _flat(sessions)

    def context(self, hid, question, q_date):
        out, used = [], 0
        for d, t in reversed(self.mem[hid]):
            n = _cost(d, t)
            if used + n > self.budget:
                break
            out.append((d, t)); used += n
        return lines(reversed(out))

    def close(self, hid):
        self.mem.pop(hid, None)


class BM25(System):
    name, native = "bm25", "retrieval"

    def __init__(self, budget=BUDGET, **kw):
        super().__init__(**kw)
        self.budget, self.mem = budget, {}

    def ingest(self, hid, sessions):
        from rank_bm25 import BM25Okapi
        turns = _flat(sessions)
        self.mem[hid] = (turns, BM25Okapi([re.findall(r"\w+", t.lower()) for _, t in turns]))

    def context(self, hid, question, q_date):
        turns, bm = self.mem[hid]
        order = np.argsort(-bm.get_scores(re.findall(r"\w+", question.lower())))
        return lines([turns[i] for i in sorted(_budgeted(order, turns, self.budget))])

    def close(self, hid):
        self.mem.pop(hid, None)


class Dense(System):
    name, native = "dense_bge_small", "retrieval"
    PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, budget=BUDGET, encoder=None, **kw):
        super().__init__(**kw)
        self.budget, self.mem, self._enc = budget, {}, encoder

    def enc(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        return self._enc

    def ingest(self, hid, sessions):
        turns = _flat(sessions)
        v = self.enc().encode([t for _, t in turns], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        self.mem[hid] = (turns, v)

    def context(self, hid, question, q_date):
        turns, v = self.mem[hid]
        q = self.enc().encode([self.PREFIX + question], normalize_embeddings=True, convert_to_numpy=True)[0]
        order = np.argsort(-(v @ q))
        return lines([turns[i] for i in sorted(_budgeted(order, turns, self.budget))])

    def close(self, hid):
        self.mem.pop(hid, None)


def _cost(d, t):
    """Tokens of one packet line, date prefix included (so every system's packet fits the same budget)."""
    return ntok(f"[m:000] ({d}) {t}")


def _budgeted(order, turns, budget):
    """Highest-scoring turns until the budget is used (a turn larger than the budget is skipped)."""
    out, used = [], 0
    for i in order:
        n = _cost(*turns[i])
        if used + n > budget:
            if used:
                break
            continue
        out.append(int(i)); used += n
    return out

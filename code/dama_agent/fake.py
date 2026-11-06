"""A deterministic stand-in for runtime.Tier1 (numpy only): lets the agent logic be tested without onnxruntime or
trained models. Embeddings are hashed bags of words; scores are word overlap. Not a model of real quality."""
from __future__ import annotations
import re
import zlib
import numpy as np

OPS = ["ADD", "UPDATE", "DELETE", "NOOP"]
_W = re.compile(r"[a-z0-9]+")
STOP = set("a an the i you to of and is are was in on at my me it that this for with do did what where when how".split())


def words(t):
    return {w for w in _W.findall(t.lower()) if w not in STOP}


class FakeTier1:
    def __init__(self, dim=256):
        self.dim = dim
        self.meta = {"retriever": {"context_chars": 0}}

    def memory_key(self, text, previous=""):
        return text

    def embed(self, texts, query=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in words(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def first_stage(self, request, memory_vecs, depth=64):
        return [int(i) for i in np.argsort(-(memory_vecs @ self.embed([request], query=True)[0]))[:depth]]

    def relevance(self, request, memories):
        q = words(request)
        return np.array([len(q & words(m)) / max(1, len(q)) for m in memories])

    def need(self, request):
        return 0.9 if words(request) & {"remember", "told", "said", "live", "name", "did", "my"} or "?" in request else 0.1

    def salience(self, turn, previous=""):
        body = turn.split(": ", 1)[-1]   # rendered as "[date] speaker: text"
        return 0.1 if len(words(body)) < 3 else 0.8

    def op(self, existing, new):
        n = new.lower()
        if any(x in n for x in ("forget", "delete", "remove")):
            p = [0.05, 0.05, 0.85, 0.05]
        elif any(x in n for x in ("moved to", "switched to", "now")):
            p = [0.1, 0.8, 0.05, 0.05]
        elif words(existing) == words(new):
            p = [0.05, 0.05, 0.05, 0.85]
        else:
            p = [0.8, 0.1, 0.05, 0.05]
        return dict(zip(OPS, p))

    def select(self, request, candidates, alpha=0.1, budget_tokens=1500, rerank_k=None):
        k = rerank_k or 8
        p = self.relevance(request, candidates[:k])
        admitted = [int(i) for i in np.argsort(-p) if p[i] > 0]
        chosen, used = [], 0.0
        for i in admitted:
            cost = len(candidates[i]) / 4
            if used + cost <= budget_tokens:
                chosen.append(i); used += cost
        return {"ids": chosen, "p": [float(p[i]) for i in chosen], "complete": len(chosen) == len(admitted), "tokens": int(used)}

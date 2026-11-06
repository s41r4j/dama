"""RAG and vector-memory baselines (no LLM at write time unless noted). Same packet budget as every other system.

- rag_chunk:      the textbook RAG pipeline. Each session is cut into ~256-token chunks (50-token overlap), embedded
                  (bge-small), top chunks by cosine until the budget.
- rag_hybrid:     BM25 + dense over turns, reciprocal-rank fusion (k = 60), the top 100 re-ranked by the
                  ms-marco-MiniLM cross-encoder (the pool-level reranker of v2's V2_acc).
- vector_memory:  "vector store memory" as in LangChain's VectorStoreRetrieverMemory: one document per exchange
                  (a turn and its reply), FAISS inner-product index, top-k exchanges until the budget.
- summary_memory: MemoryBank-style (Zhong et al., AAAI 2024) session summaries: the writer LLM summarises each session
                  once; summaries are retrieved by cosine and shown with their date, followed by the best raw turns.
"""
from __future__ import annotations
import re

import numpy as np

from .base import System, count_llm, lines
from .native import BUDGET, _budgeted, _cost, _flat

PREFIX = "Represent this sentence for searching relevant passages: "
_ENC = {}


def encoder():
    if "bge" not in _ENC:
        from sentence_transformers import SentenceTransformer
        _ENC["bge"] = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    return _ENC["bge"]


def embed(texts):
    return encoder().encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True) if texts else np.zeros((0, 384), "float32")


def qvec(q):
    return encoder().encode([PREFIX + q], normalize_embeddings=True, convert_to_numpy=True)[0]


class RagChunk(System):
    name, native = "rag_chunk", "retrieval"
    SIZE, OVERLAP = 256, 50

    def __init__(self, budget=BUDGET, **kw):
        super().__init__(**kw)
        self.budget, self.mem = budget, {}

    def ingest(self, hid, sessions):
        chunks = []
        for s in sessions:
            words = " ".join(f"{t.speaker}: {t.text}" for t in s.turns).split()
            step = max(1, int((self.SIZE - self.OVERLAP) / 1.3))
            size = int(self.SIZE / 1.3)
            for i in range(0, max(1, len(words)), step):
                piece = " ".join(words[i:i + size])
                if piece:
                    chunks.append((s.date, piece))
                if i + size >= len(words):
                    break
        self.mem[hid] = (chunks, embed([c for _, c in chunks]))

    def context(self, hid, question, q_date):
        chunks, v = self.mem[hid]
        order = np.argsort(-(v @ qvec(question))) if len(chunks) else []
        return lines([chunks[i] for i in sorted(_budgeted(order, chunks, self.budget))])

    def close(self, hid):
        self.mem.pop(hid, None)


class RagHybrid(System):
    name, native = "rag_hybrid_rerank", "retrieval"
    POOL, RRF_K = 100, 60

    def __init__(self, budget=BUDGET, **kw):
        super().__init__(**kw)
        self.budget, self.mem, self._ce = budget, {}, None

    def ce(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=256)
        return self._ce

    def ingest(self, hid, sessions):
        from rank_bm25 import BM25Okapi
        turns = _flat(sessions)
        self.mem[hid] = (turns, BM25Okapi([re.findall(r"\w+", t.lower()) for _, t in turns]), embed([t for _, t in turns]))

    def context(self, hid, question, q_date):
        turns, bm, v = self.mem[hid]
        r_bm = np.argsort(-bm.get_scores(re.findall(r"\w+", question.lower())))
        r_de = np.argsort(-(v @ qvec(question)))
        fused = {}
        for ranks in (r_bm, r_de):
            for pos, i in enumerate(ranks[:self.POOL * 2]):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.RRF_K + pos + 1)
        pool = sorted(fused, key=lambda i: -fused[i])[:self.POOL]
        sc = self.ce().predict([(question, turns[i][1]) for i in pool], batch_size=32) if pool else []
        order = [pool[j] for j in np.argsort(-np.asarray(sc))]
        return lines([turns[i] for i in sorted(_budgeted(order, turns, self.budget))])

    def close(self, hid):
        self.mem.pop(hid, None)


class VectorMemory(System):
    name, native = "vector_memory", "retrieval"

    def __init__(self, budget=BUDGET, **kw):
        super().__init__(**kw)
        self.budget, self.mem = budget, {}

    def ingest(self, hid, sessions):
        import faiss
        docs = []
        for s in sessions:
            ts = s.turns
            for i in range(0, len(ts), 2):
                pair = ts[i:i + 2]
                docs.append((s.date, " | ".join(f"{t.speaker}: {t.text}" for t in pair)))
        v = embed([d for _, d in docs]).astype("float32")
        idx = faiss.IndexFlatIP(v.shape[1] if len(v) else 384)
        if len(v):
            idx.add(v)
        self.mem[hid] = (docs, idx)

    def context(self, hid, question, q_date):
        docs, idx = self.mem[hid]
        if not docs:
            return lines([])
        _, I = idx.search(qvec(question)[None].astype("float32"), min(len(docs), 200))
        return lines([docs[i] for i in sorted(_budgeted([i for i in I[0] if i >= 0], docs, self.budget))])

    def close(self, hid):
        self.mem.pop(hid, None)


SUMMARY_PROMPT = """Summarize this conversation session for a long-term memory. Keep every concrete fact about the people: events with their dates, names, places, numbers, plans, preferences and changes. Write at most 120 words in plain sentences.
SESSION DATE: {date}
{text}
SUMMARY:"""


class SummaryMemory(System):
    """MemoryBank-style: one LLM summary per session (re-implemented; the forgetting curve is not used because every
    question is asked after the whole history). Packet: the top session summaries (half the budget), then the best raw
    turns by cosine for the rest."""
    name, needs_llm, native = "summary_memory", True, "re-implemented"
    MAX_CHARS = 12000

    def __init__(self, budget=BUDGET, **kw):
        super().__init__(**kw)
        count_llm()
        from openai import OpenAI
        self.client = OpenAI(base_url=self.llm_url, api_key="local")
        self.budget, self.mem = budget, {}

    def ingest(self, hid, sessions):
        sums = []
        for s in sessions:
            if self.out_of_time():
                break
            text = "\n".join(f"{t.speaker}: {t.text}" for t in s.turns)[:self.MAX_CHARS]
            r = self.client.chat.completions.create(model=self.llm_model, max_tokens=220,
                                                    messages=[{"role": "user", "content": SUMMARY_PROMPT.format(date=s.date, text=text)}])
            out = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
            if out:
                sums.append((s.date, f"[session summary] {out}"))
        turns = _flat(sessions)
        self.mem[hid] = (sums, embed([x for _, x in sums]), turns, embed([t for _, t in turns]))

    def context(self, hid, question, q_date):
        sums, vs, turns, vt = self.mem[hid]
        q = qvec(question)
        half = self.budget // 2
        s_idx = sorted(_budgeted(np.argsort(-(vs @ q)) if len(sums) else [], sums, half))
        used = sum(_cost(*sums[i]) for i in s_idx)
        t_idx = sorted(_budgeted(np.argsort(-(vt @ q)), turns, self.budget - used))
        return lines([sums[i] for i in s_idx] + [turns[i] for i in t_idx])

    def close(self, hid):
        self.mem.pop(hid, None)

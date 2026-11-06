"""Hybrid read path (Level 2.1): union pool -> prior -> cross-encoder on the top N -> admission -> packing.

Why (paper run v0, 2026-10-24): the v0 pipeline lost 6 points to a plain cross-encoder top-k (B3). Its ablations named
the causes: reciprocal-rank fusion cut the pool to K and dropped evidence that dense search alone kept (74.9% vs 81.2% of
questions fully covered at K = 128), MMR packing pushed out relevant near-duplicates, and the conformal threshold
admitted everything. This read path keeps every list's candidates (a union, not a fused cut), lets a small
cross-encoder order the most promising N, and packs by calibrated probability.

Steps (all numpy; the two models are plain callables, so ONNX on CPU, PyTorch on GPU or a fake all plug in):
  1. pool       dense top-k  U  BM25 top-k  U  entity-overlap top-k (each list's own ranking; nothing is cut by fusion)
  2. prior      a calibrated logistic head on 5 cheap features (dense cosine, BM25 min-max in pool, reciprocal-rank
                score, entity overlap, log length); without one, the reciprocal-rank score orders the pool
  3. rerank     the cross-encoder re-scores the prior's top N (calibrated probabilities); deadline-aware: N shrinks
                to what the time left allows, using a running per-pair cost estimate
  4. admit      p >= lam (a split-conformal threshold fitted on dev; -inf admits the whole pool)
  5. pack       highest p first within the token budget; the first `neighbours` hits also bring their previous turn
                of the same session (conversational context: the question a reply answers). Re-scored candidates
                go before the rest (`rescored_first`): the prior and the cross-encoder are not on one scale unless both
                were calibrated on the same pools, and the review of 2026-10-28 found DAMA-HT sorting Platt
                probabilities against max-scaled reciprocal-rank scores (packet evidence recall 80.5% -> 49.5%).
`read_many()` serves several requests in one go (pillar 4): one query-embedding batch and one cross-encoder batch.

Features: "v1" = the 5 above (DAMA-H). "v2" (paper v2) adds 8 cheap ones, all from what is stored at write time:
rank of the turn by dense and by BM25 inside the pool, the share of the request's terms the turn contains, whether the
turn's speaker is named in the request, whether the turn mentions a time, that flag times "the request asks about
time", the best dense score of the turn's previous/next turn in the same session, and its position in the history.

BM25 and entities match the v0 experiment exactly (dama_train's tokenizer, k1 1.5, b 0.75; capitalised words and
numbers minus a stop list), but scoring touches only the postings of the query's own terms.
"""
from __future__ import annotations
import math
import re
import time
from array import array
from collections import Counter
from dataclasses import dataclass, replace
import numpy as np
from dama_train.data.bm25 import tokenize
from .timeparse import TEMPORAL_QUERY, TIME_HINT

ENT_STOP = set("What When Where Who Whom Which Why How Did Does Do Is Was Were Are Has Have Had Would Could Should Can Will "
               "The A An In On At Of For To And Or But If Then This That These Those It Its I My Me We Our You Your He She "
               "His Her They Their Them".split())
_ENT = re.compile(r"\b[A-Z][a-zA-Z'-]+\b|\b\d+\b")
FEATURES = ["dense_cos", "bm25_minmax_in_pool", "rrf", "entity_overlap", "log_turn_chars"]
FEATURES_V2 = FEATURES + ["dense_rank_in_pool_log", "bm25_rank_in_pool_log", "query_term_coverage", "speaker_in_query",
                          "turn_has_time", "temporal_query_x_time", "context_dense", "position"]


def entities(text: str) -> set[str]:
    return {w for w in _ENT.findall(text) if w not in ENT_STOP}


class SparseIndex:
    """Appendable BM25 + entity postings. Scores cost O(postings of the query's terms), not O(store)."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.post: dict[str, tuple[array, array]] = {}
        self.ent: dict[str, array] = {}
        self.lengths = array("f")
        self.total = 0.0

    def __len__(self):
        return len(self.lengths)

    def add(self, texts):
        for t in texts:
            i = len(self.lengths)
            c = Counter(tokenize(t))
            for w, f in c.items():
                d, tf = self.post.setdefault(w, (array("i"), array("f")))
                d.append(i); tf.append(f)
            n = sum(c.values())
            self.lengths.append(n); self.total += n
            for e in entities(t):
                self.ent.setdefault(e, array("i")).append(i)

    def bm25(self, query: str) -> np.ndarray:
        n = len(self.lengths)
        out = np.zeros(n, dtype=np.float32)
        if not n:
            return out
        L = np.frombuffer(self.lengths, dtype=np.float32)
        avg = self.total / n or 1.0
        for w in set(tokenize(query)):
            p = self.post.get(w)
            if p is None:
                continue
            d, tf = np.frombuffer(p[0], dtype=np.int32), np.frombuffer(p[1], dtype=np.float32)
            idf = math.log(1 + (n - len(d) + 0.5) / (len(d) + 0.5))
            out[d] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L[d] / avg))
        return out

    def entity_overlap(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.lengths), dtype=np.float32)
        for e in entities(query):
            p = self.ent.get(e)
            if p is not None:
                out[np.frombuffer(p, dtype=np.int32)] += 1
        return out


def _top(scores: np.ndarray, k: int, positive: bool = False, tie=None) -> np.ndarray:
    """Indices of the k highest scores, best first (ties broken by `tie`, higher first)."""
    if k <= 0 or not len(scores):
        return np.zeros(0, dtype=np.int64)
    idx = np.flatnonzero(scores > 0) if positive else np.arange(len(scores))
    if len(idx) > k:
        part = idx[np.argpartition(-scores[idx], k - 1)[:k]]
        # keep everything tied with the k-th score, so the tie-break below decides among equals
        kth = scores[part].min()
        idx = idx[scores[idx] >= kth]
    order = np.lexsort((-tie[idx], -scores[idx])) if tie is not None else np.argsort(-scores[idx], kind="stable")
    return idx[order][:k]


@dataclass
class HybridConfig:
    dense_k: int = 128
    bm25_k: int = 32
    entity_k: int = 16
    rerank_n: int = 32               # prior's top N re-scored by the cross-encoder (0 = prior only)
    neighbours: int = 0              # first m packed hits bring their previous turn of the same session
    budget_tokens: int = 3000
    lam: float = -math.inf           # admission threshold on the final probability
    deadline_ms: float | None = None # None: always re-score N (accuracy runs); else shrink N to fit the time left
    rrf_k: int = 60
    features: str = "v1"             # "v1": the 5 DAMA-H features; "v2": FEATURES_V2 (the prior must be fitted on the same set)
    rescored_first: bool = True      # pack re-scored candidates before un-rescored ones (their scores are not comparable)


class HybridReader:
    """One memory space (a user, a team, a conversation). add() memories as they arrive; read() a packet per request.

    encode_query(list[str]) -> (n, d) L2-normalised vectors, matching the stored vectors.
    scorer(queries, docs) -> calibrated probabilities for the (query, doc) pairs (the cross-encoder); optional.
    prior: anything with predict_proba(features) (sklearn LogisticRegression on FEATURES); optional.
    """
    def __init__(self, encode_query, scorer=None, prior=None, config: HybridConfig | None = None):
        self.encode_query, self.scorer, self.prior = encode_query, scorer, prior
        self.cfg = config or HybridConfig()
        self.sparse = SparseIndex()
        self._vec = None                 # growable (capacity, d) buffer
        self.n = 0
        self.texts, self.pair_texts, self.ntok, self.prev, self.next = [], [], [], [], []
        self.speakers = []
        self.loglen = array("f")
        self.has_time = array("f")
        self.ms_per_pair: float | None = None

    # ------------------------------------------------------------------ write
    def add(self, texts, vecs, *, pair_texts=None, ntok=None, prev=None, speakers=None):
        """texts: what the packet shows; vecs: their dense vectors; pair_texts: what the cross-encoder reads (default
        texts); ntok: token cost of each line in a packet (default len/4); prev: index of the previous turn of the same
        session (-1 = none; default: no neighbours); speakers: who said each turn (v2 feature speaker_in_query)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        m = len(texts)
        if self._vec is None:
            self._vec = np.zeros((max(64, 2 * m), vecs.shape[1]), dtype=np.float32)
        while self.n + m > len(self._vec):
            grown = np.zeros((2 * len(self._vec), self._vec.shape[1]), dtype=np.float32)
            grown[:self.n] = self._vec[:self.n]; self._vec = grown
        self._vec[self.n:self.n + m] = vecs
        self.n += m
        self.texts += list(texts)
        self.pair_texts += list(pair_texts if pair_texts is not None else texts)
        self.ntok += list(ntok if ntok is not None else [max(1, len(t) // 4) for t in texts])
        start = len(self.prev)
        self.prev += list(prev if prev is not None else [-1] * m)
        self.next += [-1] * m
        for i in range(start, start + m):
            if self.prev[i] >= 0:
                self.next[self.prev[i]] = i
        src = pair_texts if pair_texts is not None else texts
        self.speakers += [set(tokenize(s or "")) for s in (speakers if speakers is not None else [""] * m)]
        self.loglen.extend(math.log(max(1, len(t))) for t in src)
        self.has_time.extend(1.0 if TIME_HINT.search(t) else 0.0 for t in src)
        self.sparse.add(src)

    @property
    def vectors(self) -> np.ndarray:
        return self._vec[:self.n] if self._vec is not None else np.zeros((0, 1), dtype=np.float32)

    # ------------------------------------------------------------------ read
    def pool(self, query: str, qvec: np.ndarray, cfg: HybridConfig | None = None):
        """Union pool and its features. Returns (indices, features (n, 5), timings ms)."""
        c = cfg or self.cfg
        T, t = {}, time.perf_counter()
        dense = self.vectors @ qvec
        T["dense"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()
        bm = self.sparse.bm25(query)
        T["bm25"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()
        ent = self.sparse.entity_overlap(query)
        T["entity"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()
        lists = [_top(dense, c.dense_k), _top(bm, c.bm25_k, positive=True), _top(ent, c.entity_k, positive=True, tie=dense)]
        rrf = {}
        for lst in lists:
            for r, i in enumerate(lst.tolist()):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (c.rrf_k + r + 1)
        idx = np.fromiter(rrf.keys(), dtype=np.int64, count=len(rrf))
        if not len(idx):
            return idx, np.zeros((0, len(FEATURES)), dtype=np.float32), T
        b = bm[idx]; lo, hi = float(b.min()), float(b.max())
        cols = [dense[idx], (b - lo) / (hi - lo) if hi > lo else np.zeros(len(idx)), np.fromiter(rrf.values(), dtype=np.float64),
                ent[idx], np.frombuffer(self.loglen, dtype=np.float32)[idx]]
        if c.features == "v2":
            cols += self._v2_features(query, idx, dense, b)
        feats = np.stack(cols, axis=1).astype(np.float32)
        T["pool"] = (time.perf_counter() - t) * 1000
        return idx, feats, T

    def _v2_features(self, query: str, idx: np.ndarray, dense: np.ndarray, bm_pool: np.ndarray) -> list[np.ndarray]:
        """FEATURES_V2 beyond the first 5, for the pool members idx (O(pool), never O(store))."""
        d = dense[idx]
        rank_d = np.empty(len(idx)); rank_d[np.argsort(-d, kind="stable")] = np.arange(len(idx))
        rank_b = np.empty(len(idx)); rank_b[np.argsort(-bm_pool, kind="stable")] = np.arange(len(idx))
        q = set(tokenize(query))
        # tokenised on demand for the pool only: a token set per stored turn would grow memory with the store
        cover = np.array([len(q & set(tokenize(self.pair_texts[i]))) / len(q) if q else 0.0 for i in idx.tolist()])
        spk = np.array([1.0 if self.speakers[i] and self.speakers[i] <= q else 0.0 for i in idx.tolist()])
        has_t = np.frombuffer(self.has_time, dtype=np.float32)[idx].astype(np.float64)
        temporal = 1.0 if TEMPORAL_QUERY.search(query) else 0.0
        ctx = np.array([max(dense[self.prev[i]] if self.prev[i] >= 0 else 0.0, dense[self.next[i]] if self.next[i] >= 0 else 0.0)
                        for i in idx.tolist()])
        pos = idx / max(1, self.n - 1)
        return [np.log1p(rank_d), np.log1p(rank_b), cover, spk, has_t, temporal * has_t, ctx, pos]

    def prior_p(self, feats: np.ndarray) -> np.ndarray:
        if not len(feats):
            return np.zeros(0)
        if self.prior is not None:
            return self.prior.predict_proba(feats)[:, 1]
        return feats[:, 2] / max(float(feats[:, 2].max()), 1e-9)   # reciprocal-rank score, scaled to [0, 1]

    def read(self, query: str, **overrides) -> dict:
        return self.read_many([query], **overrides)[0]

    def read_many(self, queries: list[str], *, qvecs=None, **overrides) -> list[dict]:
        """Packets for several requests at once. overrides: any HybridConfig field (e.g. budget_tokens=1000)."""
        c = replace(self.cfg, **overrides) if overrides else self.cfg
        t0 = time.perf_counter()
        if qvecs is None:
            qvecs = self.encode_query(list(queries)) if queries else np.zeros((0, 1))
        t_embed = (time.perf_counter() - t0) * 1000 / max(1, len(queries))
        states = []
        for q, v in zip(queries, qvecs):
            idx, feats, T = self.pool(q, np.asarray(v, dtype=np.float32), c)
            t = time.perf_counter()
            p = self.prior_p(feats).astype(np.float64)
            T["prior"] = (time.perf_counter() - t) * 1000
            states.append({"q": q, "idx": idx, "p": p, "T": {"embed": t_embed, **T}, "rescored": 0, "resc": np.zeros(len(idx), dtype=bool)})
        # one cross-encoder batch for every request's top N
        n = c.rerank_n if self.scorer is not None else 0
        if n and c.deadline_ms is not None:
            spent = (time.perf_counter() - t0) * 1000 / max(1, len(queries))
            left = c.deadline_ms - spent - 5          # ~5 ms for packing and rendering
            if self.ms_per_pair is not None:
                n = max(0, min(n, int(left / self.ms_per_pair)))
        if n:
            pairs_q, pairs_d, where = [], [], []
            for si, s in enumerate(states):
                top = np.argsort(-s["p"], kind="stable")[:n]
                for j in top.tolist():
                    pairs_q.append(s["q"]); pairs_d.append(self.pair_texts[int(s["idx"][j])]); where.append((si, j))
            if pairs_q:
                t = time.perf_counter()
                z = np.asarray(self.scorer(pairs_q, pairs_d), dtype=np.float64)
                ms = (time.perf_counter() - t) * 1000
                per = ms / len(pairs_q)
                self.ms_per_pair = per if self.ms_per_pair is None else 0.8 * self.ms_per_pair + 0.2 * per
                for (si, j), v in zip(where, z):
                    states[si]["p"][j] = v; states[si]["rescored"] += 1; states[si]["resc"][j] = True
                for s in states:
                    s["T"]["rerank"] = ms / len(states)
        return [self._pack(s, c) for s in states]

    def _pack(self, s: dict, c: HybridConfig) -> dict:
        t = time.perf_counter()
        idx, p = s["idx"], s["p"]
        adm = np.flatnonzero(p >= c.lam) if math.isfinite(c.lam) else np.arange(len(idx))
        if c.rescored_first:
            order = adm[np.lexsort((-p[adm], ~s["resc"][adm]))]      # re-scored first, each group by p
        else:
            order = adm[np.argsort(-p[adm], kind="stable")]
        chosen, used, seen, hits = [], 0, set(), 0
        admitted_tokens = sum(self.ntok[int(idx[j])] for j in order)
        for j in order.tolist():
            i = int(idx[j])
            if i in seen or used + self.ntok[i] > c.budget_tokens:
                continue
            chosen.append(i); seen.add(i); used += self.ntok[i]
            hits += 1
            if hits <= c.neighbours:
                pv = self.prev[i]
                if pv >= 0 and pv not in seen and used + self.ntok[pv] <= c.budget_tokens:
                    chosen.append(pv); seen.add(pv); used += self.ntok[pv]
        pmap = {int(i): float(v) for i, v in zip(idx.tolist(), p)}
        s["T"]["pack"] = (time.perf_counter() - t) * 1000
        return {"ids": sorted(chosen), "order": chosen, "p": [pmap.get(i) for i in chosen], "tokens": used,
                "pool": len(idx), "admitted": int(len(adm)), "rescored": s["rescored"], "budget_limited": admitted_tokens > c.budget_tokens,
                "stages_ms": {k: round(v, 3) for k, v in s["T"].items()}, "ms": round(sum(s["T"].values()), 3)}

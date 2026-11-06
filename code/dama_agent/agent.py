"""DAMA v0 Memory Agent (Level 2): the sidecar that keeps memory for a Primary LLM.

Read path (push, before the Primary LLM answers; deadline ~100 ms, fail-open):
  1. need gate        Tier-1 `need`: does this request need long-term memory at all?
  2. first stage      dense search over the scope's episodes (fine-tuned retriever embeddings, one mat-vec)
  3. pack             the first stage's top-k within the token budget. Optional (use_rerank=True): Tier-1 `rel`
                      re-scores the top-k with a split-conformal threshold. Off by default since L2 on L1.2: over the
                      fine-tuned first stage it changed coverage by 0.000 and cost ~220 ms on CPU, because at k = 8 the
                      conformal threshold admits every candidate (it only reorders them)
  4. freshen          a turn whose fact was superseded brings its newer fact along and is marked outdated
  5. render           time-ordered lines with [m:ID] citations, or NOT_FOUND
With the rerank on, step 3 is deadline-aware: the agent keeps a running estimate of its rerank cost per pair and uses
the largest calibrated k (8, then 4) that fits the time left; if none fits, the packet keeps the first-stage order
(`degraded`). Each packet reports its per-stage milliseconds.
Hybrid read path (AgentConfig.hybrid set; Level 2.1, see hybrid.py): steps 2-3 become a union pool (dense U BM25 U
entities) ordered by Tier-1 `rel` on the top N (deadline-aware) and packed by probability; one HybridReader per scope is
kept in memory and extended with each new episode.
Identical requests to an unchanged scope (same store version) are answered from a small cache: with many agents on one
memory (pillar 4), a question asked by several agents at once is computed once.

Write path (after the reply; `observe(..., defer=True)` + `commit()`, or immediately):
  1. append the turn verbatim as an episode (with its retrieval key: previous-turn context + turn)
  2. salience gate    Tier-1 `sal`
  3. compare          the new statement against the nearest current facts of the scope (dense, same speaker)
  4. op               Tier-1 op model per pair -> ADD / UPDATE (supersede) / DELETE (retract) / NOOP (duplicate);
                      an UPDATE that is likely but not confident is stored as a CONFLICT next to the old fact
  5. audit            every decision with its probabilities

Tier 2 (an LLM extracting compact facts) is not in v0: a fact's text is its turn's text.
"""
from __future__ import annotations
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
import numpy as np
from .hybrid import HybridConfig, HybridReader
from .store import Store


@dataclass
class AgentConfig:
    need_threshold: float = 0.5
    sal_threshold: float = 0.5
    alpha: float = 0.1
    rerank_k: int = 8               # packet size: first-stage memories per packet (and rel depth when use_rerank)
    budget_tokens: int = 1500
    deadline_ms: float = 100.0
    first_stage_depth: int = 64
    op_neighbours: int = 3          # current facts compared against each salient new turn
    op_min_similarity: float = 0.55 # below this cosine a stored fact is not a candidate for UPDATE/DELETE/NOOP
    op_confident: float = 0.6
    op_conflict: float = 0.4        # UPDATE in [op_conflict, op_confident) -> stored as a conflict
    max_turn_chars: int = 1000      # as in training (dama_train.data.schema.render_turn)
    fallback_ks: tuple = (4,)       # smaller calibrated rerank depths to try when k does not fit the deadline
    use_rerank: bool = False        # True: rel re-scores the top rerank_k (adds ~0 coverage, ~220 ms on CPU; see docstring)
    cache_size: int = 512           # packets kept for identical requests to an unchanged scope (0 = off)
    hybrid: HybridConfig | None = None   # set: hybrid read path (union pool + rel on the top N); None: dense first stage


def render(ts, speaker, text, max_chars=1000) -> str:
    """How a stored turn is shown to the models and to the Primary LLM; matches the Level-1 training format."""
    text = text if len(text) <= max_chars else text[:max_chars] + " ..."
    return f"{f'[{ts}] ' if ts else ''}{speaker}: {text}"


class MemoryAgent:
    def __init__(self, tier1, store: Store | None = None, config: AgentConfig | None = None):
        self.t1, self.store, self.cfg = tier1, store or Store(), config or AgentConfig()
        self.queue: list[dict] = []
        self.ms_per_pair: float | None = None   # running estimate of rerank cost, learned from real calls
        self._cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inflight: dict[tuple, threading.Event] = {}
        self._hybrid: dict[str, dict] = {}
        self._hybrid_lock = threading.Lock()

    # ------------------------------------------------------------------ write path
    def observe(self, scope, text, *, speaker="user", role="user", session=None, ts=None, source_id=None, defer=False) -> dict:
        """Record one turn. defer=True queues it for commit() (run after the Primary LLM has replied)."""
        item = dict(scope=scope, text=text, speaker=speaker, role=role, session=session, ts=ts, source_id=source_id)
        if defer:
            self.queue.append(item)
            return {"queued": len(self.queue)}
        return self._write(**item)

    def commit(self) -> list[dict]:
        out, self.queue = [self._write(**item) for item in self.queue], []
        return out

    def note(self, scope, text, *, speaker="user", ts=None) -> dict:
        """The Primary LLM asks to remember something explicitly: skips the salience gate."""
        return self._write(scope=scope, text=text, speaker=speaker, role="note", session=None, ts=ts, source_id=None, force_fact=True)

    def _write(self, scope, text, speaker, role, session, ts, source_id, force_fact=False) -> dict:
        c = self.cfg
        prev = self.store.last_episode(scope)
        value = render(ts, speaker, text, c.max_turn_chars)
        prev_value = render(prev["ts"], prev["speaker"], prev["text"], c.max_turn_chars) if prev else ""
        vec = self.t1.embed([self.t1.memory_key(value, prev_value)])[0]
        eid = self.store.add_episode(scope, text, vec, session=session, speaker=speaker, role=role, ts=ts, source_id=source_id)
        p_sal = 1.0 if force_fact else self.t1.salience(value, prev_value[:300])
        self.store.set_salience(eid, p_sal)
        if p_sal < c.sal_threshold:
            return {"episode": eid, "op": "SKIP", "salience": p_sal}

        statement = f"{speaker}: {text}"
        ids, mat = self.store.current_fact_matrix(scope)
        candidates = []
        if ids:
            sims = mat @ vec
            facts = self.store.facts([ids[i] for i in sims.argsort()[::-1][:c.op_neighbours * 3] if sims[i] >= c.op_min_similarity])
            candidates = [f for f in sorted(facts.values(), key=lambda f: -float(sims[ids.index(f["id"])]))
                          if f["speaker"] == speaker][:c.op_neighbours]
        scored = [(f, self.t1.op(f"{f['speaker']}: {f['text']}", statement)) for f in candidates]

        def best(op):
            return max(scored, key=lambda x: x[1][op], default=(None, {op: 0.0}))

        upd, dele, noop = best("UPDATE"), best("DELETE"), best("NOOP")
        detail = {"episode": eid, "salience": round(p_sal, 3), "compared": [f["id"] for f in candidates],
                  "p": [{k: round(v, 3) for k, v in p.items()} for _, p in scored]}
        if dele[0] is not None and dele[1]["DELETE"] >= c.op_confident:
            self.store.close_fact(dele[0]["id"], "deleted", valid_to=ts)
            self.store.audit(scope, "DELETE", fact=dele[0]["id"], **detail)
            return {"episode": eid, "op": "DELETE", "fact": dele[0]["id"], "salience": p_sal}
        if noop[0] is not None and noop[1]["NOOP"] >= c.op_confident and not force_fact:
            self.store.audit(scope, "NOOP", fact=noop[0]["id"], **detail)
            return {"episode": eid, "op": "NOOP", "fact": noop[0]["id"], "salience": p_sal}
        if upd[0] is not None and upd[1]["UPDATE"] >= c.op_confident:
            fid = self.store.add_fact(scope, eid, text, vec, speaker=speaker, valid_from=ts)
            self.store.close_fact(upd[0]["id"], "superseded", valid_to=ts, superseded_by=fid)
            self.store.audit(scope, "UPDATE", fact=fid, replaces=upd[0]["id"], **detail)
            return {"episode": eid, "op": "UPDATE", "fact": fid, "replaces": upd[0]["id"], "salience": p_sal}
        conflict = upd[0] is not None and upd[1]["UPDATE"] >= c.op_conflict
        fid = self.store.add_fact(scope, eid, text, vec, speaker=speaker, valid_from=ts, conflict=conflict)
        self.store.audit(scope, "CONFLICT" if conflict else "ADD", fact=fid, **({"with": upd[0]["id"]} if conflict else {}), **detail)
        return {"episode": eid, "op": "CONFLICT" if conflict else "ADD", "fact": fid, "salience": p_sal}

    # ------------------------------------------------------------------ read path
    def packet(self, scope, request, *, force=False, deadline_ms=None, budget_tokens=None, alpha=None, k=None) -> dict:
        """The push: a compact memory packet for `request`, built within the deadline."""
        t0 = time.perf_counter()
        if not self.cfg.cache_size:
            return self._packet(scope, request, force, deadline_ms, budget_tokens, alpha, k, t0)
        key = (scope, request, self.store.version(scope), force, budget_tokens, alpha, k)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return {**hit, "cached": True, "late": False, "ms": (time.perf_counter() - t0) * 1000}
            waiter = self._inflight.get(key)     # the same request is being computed right now by another agent
            if waiter is None:
                self._inflight[key] = threading.Event()
        if waiter is not None:
            waiter.wait()
            with self._cache_lock:
                hit = self._cache.get(key)
            if hit is not None:
                ms = (time.perf_counter() - t0) * 1000
                return {**hit, "cached": True, "ms": ms, "late": ms > (deadline_ms or self.cfg.deadline_ms)}
            return self._packet(scope, request, force, deadline_ms, budget_tokens, alpha, k, t0)
        try:
            out = self._packet(scope, request, force, deadline_ms, budget_tokens, alpha, k, t0)
            with self._cache_lock:
                self._cache[key] = out
                while len(self._cache) > self.cfg.cache_size:
                    self._cache.popitem(last=False)
            return out
        finally:
            with self._cache_lock:
                self._inflight.pop(key).set()

    def _packet(self, scope, request, force, deadline_ms, budget_tokens, alpha, k, t0) -> dict:
        c = self.cfg
        deadline = (deadline_ms or c.deadline_ms) / 1000
        out = {"need": None, "skipped": False, "degraded": False, "ids": [], "text": "NOT_FOUND", "tokens": 0, "complete": True,
               "stages_ms": {}, "cached": False}
        lap = [t0]

        def mark(stage):
            now = time.perf_counter(); out["stages_ms"][stage] = round((now - lap[0]) * 1000, 2); lap[0] = now

        if not force:
            out["need"] = self.t1.need(request)
            mark("need")
            if out["need"] < c.need_threshold:
                out.update(skipped=True, ms=(time.perf_counter() - t0) * 1000)
                return out
        ids, mat = self.store.episode_matrix(scope)
        if not ids:
            out["ms"] = (time.perf_counter() - t0) * 1000
            return out
        if c.hybrid is not None:
            return self._hybrid_packet(scope, request, ids, mat, out, deadline_ms, budget_tokens, k, t0)
        order = self.t1.first_stage(request, mat, depth=c.first_stage_depth)
        out["first_stage"] = [ids[i] for i in order]
        mark("first_stage")
        want = k or c.rerank_k               # memories the packet may hold (first-stage order)
        rk = 0                               # how many of them rel re-scores (0 = no rerank)
        if c.use_rerank:
            rk = want
            if deadline_ms is None or deadline_ms < 1e8:   # largest calibrated depth whose estimated cost fits the time left
                left_ms = (deadline - (time.perf_counter() - t0)) * 1000 - 5   # ~5 ms for rendering
                fits = [kk for kk in (want, *c.fallback_ks) if kk <= want and (self.ms_per_pair is None or kk * self.ms_per_pair <= left_ms)]
                rk = fits[0] if fits and left_ms > 0 else 0
        out["k"], out["reranked_k"] = want, rk
        rows = self.store.episodes(out["first_stage"][:want])
        cands = [rows[e] for e in out["first_stage"][:want]]
        texts = [render(r["ts"], r["speaker"], r["text"], c.max_turn_chars) for r in cands]
        if rk == 0:  # first-stage order within budget: by choice (default) or fail-soft when no rerank fits the deadline
            chosen, used = [], 0.0
            for i, t in enumerate(texts):
                if used + len(t) / 4 > (budget_tokens or c.budget_tokens):
                    break
                chosen.append(i); used += len(t) / 4
            sel = {"ids": chosen, "p": [None] * len(chosen), "complete": len(chosen) == len(texts), "tokens": int(used)}
            out["degraded"] = c.use_rerank
        else:
            t_rr = time.perf_counter()
            sel = self.t1.select(request, texts[:rk], alpha=alpha or c.alpha, budget_tokens=budget_tokens or c.budget_tokens, rerank_k=rk)
            per = (time.perf_counter() - t_rr) * 1000 / rk
            self.ms_per_pair = per if self.ms_per_pair is None else 0.8 * self.ms_per_pair + 0.2 * per
            mark("rerank")
        picked = [cands[i]["id"] for i in sel["ids"]]
        lines, extra, extra_tokens = self._render(picked)
        mark("render")
        out.update(ids=picked + extra, p=sel["p"], complete=sel["complete"], tokens=sel["tokens"] + extra_tokens,
                   text="\n".join(lines) if lines else "NOT_FOUND", ms=(time.perf_counter() - t0) * 1000)
        out["late"] = out["ms"] > (deadline_ms or c.deadline_ms)
        return out

    def _hybrid_reader(self, scope, ids, mat) -> tuple[HybridReader, list[int]]:
        """The scope's HybridReader, extended with episodes stored since it was last used."""
        with self._hybrid_lock:
            h = self._hybrid.get(scope)
            if h is None or h["n"] > len(ids):
                h = {"reader": HybridReader(lambda qs: self.t1.embed(qs, query=True), scorer=self._rel_scorer,
                                            config=self.cfg.hybrid), "n": 0, "ids": [], "last": {}}
                self._hybrid[scope] = h
            if h["n"] < len(ids):
                new = ids[h["n"]:]
                rows = self.store.episodes(new)
                texts, prev = [], []
                for j, e in enumerate(new):
                    r = rows[e]
                    texts.append(render(r["ts"], r["speaker"], r["text"], self.cfg.max_turn_chars))
                    prev.append(h["last"].get(r["session"], -1) if r["session"] is not None else -1)
                    if r["session"] is not None:
                        h["last"][r["session"]] = h["n"] + j
                h["reader"].add(texts, mat[h["n"]:len(ids)], prev=prev)
                h["ids"] += list(new); h["n"] = len(ids)
            return h["reader"], h["ids"]

    def _rel_scorer(self, queries, docs):
        """Tier-1 `rel` for (request, rendered turn) pairs, one call per distinct request."""
        out = np.zeros(len(queries))
        by = {}
        for i, q in enumerate(queries):
            by.setdefault(q, []).append(i)
        for q, rows in by.items():
            out[rows] = self.t1.relevance(q, [docs[i] for i in rows])
        return out

    def _hybrid_packet(self, scope, request, ids, mat, out, deadline_ms, budget_tokens, k, t0) -> dict:
        c = self.cfg
        reader, rids = self._hybrid_reader(scope, ids, mat)
        hc = replace(c.hybrid, budget_tokens=budget_tokens or c.hybrid.budget_tokens,
                     deadline_ms=None if (deadline_ms or 0) >= 1e8 else (deadline_ms or c.deadline_ms) - (time.perf_counter() - t0) * 1000)
        if k:
            hc = replace(hc, rerank_n=max(hc.rerank_n, k))
        res = reader.read(request, **{f: getattr(hc, f) for f in ("budget_tokens", "deadline_ms", "rerank_n")})
        picked = [rids[i] for i in res["order"]]
        lines, extra, extra_tokens = self._render(picked)
        out["stages_ms"].update(res["stages_ms"])
        out.update(ids=picked + extra, p=res["p"], complete=not res["budget_limited"], tokens=res["tokens"] + extra_tokens,
                   text="\n".join(lines) if lines else "NOT_FOUND", pool=res["pool"], reranked_k=res["rescored"],
                   ms=(time.perf_counter() - t0) * 1000)
        out["late"] = out["ms"] > (deadline_ms or c.deadline_ms)
        return out

    def recall(self, scope, query, *, k=16, budget_tokens=3000) -> dict:
        """The pull (a tool the Primary LLM calls for more): no need gate, more memories, larger budget, no deadline."""
        return self.packet(scope, query, force=True, deadline_ms=1e9, budget_tokens=budget_tokens, k=k)

    def timeline(self, scope, query, *, k=10) -> list[dict]:
        """All facts close to `query`, including superseded and deleted ones, oldest first: how something changed."""
        facts = self.store.all_facts(scope)
        if not facts:
            return []
        import numpy as np
        mat = np.stack([np.frombuffer(f["vec"], dtype=np.float32) for f in facts])
        q = self.t1.embed([query], query=True)[0]
        top = sorted((mat @ q).argsort()[::-1][:k], key=lambda i: facts[i]["id"])
        return [{"id": facts[i]["id"], "speaker": facts[i]["speaker"], "text": facts[i]["text"], "status": facts[i]["status"],
                 "valid_from": facts[i]["valid_from"], "valid_to": facts[i]["valid_to"], "superseded_by": facts[i]["superseded_by"]}
                for i in top]

    def _render(self, episode_ids) -> tuple[list[str], list[int], int]:
        """Time-ordered lines with [m:ID]. A turn whose fact was superseded is marked outdated and the newer turn is
        added (once), so the Primary LLM sees the current value next to the old one."""
        rows = self.store.episodes(episode_ids)
        facts = self.store.facts_of_episodes(episode_ids)
        extra, notes = [], {}
        newer = self.store.facts([f["superseded_by"] for f in facts.values() if f["status"] == "superseded" and f["superseded_by"]])
        for eid, f in facts.items():
            if f["status"] == "superseded" and f["superseded_by"] in newer:
                new_eid = newer[f["superseded_by"]]["episode_id"]
                notes[eid] = f" (outdated: see m:{new_eid})"
                if new_eid not in rows and new_eid not in extra:
                    extra.append(new_eid)
            elif f["status"] == "deleted":
                notes[eid] = " (retracted by the user)"
        rows.update(self.store.episodes(extra))
        ordered = sorted(rows.values(), key=lambda r: r["id"])
        lines = [f"[m:{r['id']}] {render(r['ts'], r['speaker'], r['text'], self.cfg.max_turn_chars)}{notes.get(r['id'], '')}" for r in ordered]
        extra_tokens = sum(len(render(rows[e]["ts"], rows[e]["speaker"], rows[e]["text"], self.cfg.max_turn_chars)) // 4 for e in extra)
        return lines, extra, extra_tokens

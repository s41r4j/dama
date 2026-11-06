"""Minimal CPU runtime for the exported Tier-1 models (needs only numpy, onnxruntime and tokenizers/transformers).

    from runtime import Tier1
    t1 = Tier1("dama_tier1")
    vecs = t1.embed([t1.memory_key(m, prev) for m, prev in zip(memories, [""] + memories[:-1])])  # once, on write
    order = t1.first_stage("Where do I live now?", vecs)   # dense first stage, best first
    t1.relevance("Where do I live now?", ["[2023/05/01] user: I moved to Pune.", "[2023/04/02] user: I like tea."])
    t1.need("What did I say about my landlord?")
    t1.route("When did I start the new job?")
    t1.op("The user lives in Berlin.", "I moved to Pune last month")

All scores are calibrated probabilities. The bundle's calibration was fitted on the first-stage source named in
meta["calibration"]["first_stage"] (e.g. "bm25+tuned" = BM25 fused with these embeddings by reciprocal rank).
`select()` re-scores the first-stage top-k candidates, applies the
conformal threshold fitted for that k, then a token budget.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class Tier1:
    def __init__(self, directory, threads: int = 2):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self._load_meta(directory)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.dec = ort.InferenceSession(str(self.root / self.meta["decision"]["file"]), opts, providers=["CPUExecutionProvider"])
        self.ops = ort.InferenceSession(str(self.root / self.meta["ops"]["file"]), opts, providers=["CPUExecutionProvider"])
        self.dtok = AutoTokenizer.from_pretrained(self.root / self.meta["decision"]["tokenizer"])
        self.otok = AutoTokenizer.from_pretrained(self.root / self.meta["ops"]["tokenizer"])
        r = self.meta.get("retriever")
        if r:
            self.ret = ort.InferenceSession(str(self.root / r["file"]), opts, providers=["CPUExecutionProvider"])
            self.rtok = AutoTokenizer.from_pretrained(self.root / r["tokenizer"])

    def _load_meta(self, directory):
        """Bundle metadata and calibration (shared with dama_agent.gpu.TorchTier1, which runs the PyTorch weights)."""
        self.root = Path(directory)
        self.meta = json.loads((self.root / "dama_tier1.json").read_text())
        cal = self.meta.get("calibration") or {}
        self.platt = cal.get("platt", {})
        self.route_platt = cal.get("route_platt", {})
        self.temperature = cal.get("op_temperature", 1.0)
        by_k = cal.get("conformal_by_k") or {}
        self.conformal = {int(k): {float(a): lam for a, lam in lams.items()} for k, lams in by_k.items()}
        self.rerank_k = cal.get("default_rerank_k")

    def memory_key(self, text: str, previous: str = "") -> str:
        """The text to embed for a stored turn: the start of the previous turn, a newline, the turn (as trained)."""
        n = self.meta["retriever"].get("context_chars", 0)
        return f"{previous[:n]}\n{text}" if n > 0 and previous else text

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        """L2-normalised embeddings from the fine-tuned first-stage retriever (store them with each memory)."""
        r = self.meta["retriever"]
        texts = [r["query_prefix"] + t for t in texts] if query else list(texts)
        enc = self.rtok(texts, truncation=True, max_length=r["query_max_length"] if query else r["max_length"], padding=True, return_tensors="np")
        return self.ret.run(None, {n: enc[n].astype(np.int64) for n in r["inputs"]})[0]

    def first_stage(self, request: str, memory_vecs: np.ndarray, depth: int = 64) -> list[int]:
        """Dense ranking of stored memories (by their embed() vectors), best first."""
        if len(memory_vecs) == 0:
            return []
        return [int(i) for i in np.argsort(-(memory_vecs @ self.embed([request], query=True)[0]))[:depth]]

    def _run(self, sess, tok, spec, a, b):
        enc = tok(a, b, truncation="longest_first", max_length=spec["max_length"], padding=True, return_tensors="np")
        return sess.run(None, {n: enc[n].astype(np.int64) for n in spec["inputs"]})[0]

    def _p(self, name, z):
        a, b = self.platt.get(name, (1.0, 0.0))
        return _sigmoid(a * z + b)

    def relevance(self, request: str, memories: list[str]) -> np.ndarray:
        if not memories:
            return np.zeros(0)
        z = self._run(self.dec, self.dtok, self.meta["decision"], [request] * len(memories), memories)
        return self._p("rel", z[:, 0])

    def salience(self, turn: str, previous: str = "") -> float:
        return float(self._p("sal", self._run(self.dec, self.dtok, self.meta["decision"], [turn], [previous])[0, 1]))

    def need(self, request: str) -> float:
        return float(self._p("need", self._run(self.dec, self.dtok, self.meta["decision"], [request], [""])[0, 2]))

    def route(self, request: str) -> dict:
        z = self._run(self.dec, self.dtok, self.meta["decision"], [request], [""])[0, 3:]
        out = {}
        for k, r in enumerate(self.meta["routes"]):
            a, b = self.route_platt.get(r, (1.0, 0.0))
            out[r] = float(_sigmoid(a * z[k] + b))
        return out

    def op(self, existing: str, new: str) -> dict:
        z = self._run(self.ops, self.otok, self.meta["ops"], [existing], [new])[0] / self.temperature
        p = np.exp(z - z.max()); p /= p.sum()
        return {label: float(v) for label, v in zip(self.meta["ops"]["labels"], p)}

    def select(self, request: str, candidates: list[str], alpha: float = 0.1, budget_tokens: int = 1500, rerank_k: int | None = None) -> dict:
        """Build a memory packet from `candidates`, which must be ordered best-first by the first-stage retriever
        (BM25 + embeddings). Only the top `rerank_k` are re-scored by the rel head (its cost grows with k).

        Returns {"ids": candidate indices, best first, "p": their probabilities, "complete": bool, "tokens": int}.
        `complete` is False when the token budget dropped a memory that passed the conformal threshold: the
        1 - alpha coverage guarantee only holds for complete packets.
        """
        k = rerank_k or self.rerank_k
        if k not in self.conformal or alpha not in self.conformal[k]:
            raise ValueError(f"no conformal threshold for k={k}, alpha={alpha}; calibrated: "
                             + ", ".join(f"k={kk} alpha={sorted(v)}" for kk, v in sorted(self.conformal.items())))
        lam = self.conformal[k][alpha]
        p = self.relevance(request, candidates[:k])
        admitted = [int(i) for i in np.argsort(-p) if p[i] >= lam]
        chosen, used = [], 0.0
        for i in sorted(admitted, key=lambda i: -p[i] / max(1, len(candidates[i]) / 4)):
            cost = len(candidates[i]) / 4
            if used + cost <= budget_tokens:
                chosen.append(i); used += cost
        chosen.sort(key=lambda i: -p[i])
        return {"ids": chosen, "p": [float(p[i]) for i in chosen], "complete": len(chosen) == len(admitted), "tokens": int(used)}

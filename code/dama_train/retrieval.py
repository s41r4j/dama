"""Candidate generation the way DAMA's read path does it: BM25 + dense embeddings, fused by reciprocal rank.

Used to build realistic candidate pools for evaluation and for the conformal threshold. The decision model's
rel head then re-scores the pool.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .data.bm25 import BM25
from .env import best_device

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def memory_key(text: str, previous: str = "", context_chars: int = 0) -> str:
    """What the retriever embeds for a stored turn (the value given to the models stays the turn itself).
    With context_chars > 0 the start of the previous turn is prepended (L1.2), so short replies such as
    "Yes! Last May." can be found by what they answer."""
    if context_chars <= 0 or not previous:
        return text
    return f"{previous[:context_chars]}\n{text}"


def memory_keys(texts: list[str], context_chars: int = 0) -> list[str]:
    return [memory_key(t, texts[i - 1] if i else "", context_chars) for i, t in enumerate(texts)]


class DenseEncoder:
    """bge-small-en-v1.5 (33M, 384-d, CLS pooling), or a fine-tuned copy: pass runs/retriever/best as `name`.
    Loaded once; runs on GPU if available. `model`/`tokenizer` reuse an already loaded model (used while training)."""
    def __init__(self, name="BAAI/bge-small-en-v1.5", device=None, max_length=256, model=None, tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or best_device()
        if model is None:
            root = Path(name)
            if (root / "meta.json").exists():  # a dama_train retriever run folder
                max_length = json.loads((root / "meta.json").read_text()).get("max_length", max_length)
                root = root / "model"
            tokenizer = AutoTokenizer.from_pretrained(str(root) if root.exists() else name)
            model = AutoModel.from_pretrained(str(root) if root.exists() else name).to(self.device).eval()
        self.tok, self.model, self.max_length = tokenizer, model, max_length

    def encode(self, texts, query=False, batch_size=128):
        torch = self.torch
        out = []
        prefix = BGE_QUERY_PREFIX if query else ""
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok([prefix + t for t in texts[i:i + batch_size]], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    h = self.model(**enc).last_hidden_state[:, 0].float()
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)


def rrf(rankings, k=60):
    score = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])


def rank_all(dialogue: dict, encoders: dict, depth: int = 128) -> list[dict]:
    """For each question of one dialogue, the first-stage ranking from every source: "bm25", each dense encoder
    alone ("<name>") and fused with BM25 ("bm25+<name>"). `encoders` maps a name to an encoder, or to
    (encoder, context_chars) when that encoder embeds neighbour keys (see memory_key)."""
    texts = dialogue["texts"]
    queries = [q["query"] for q in dialogue["questions"]]
    bm25 = BM25(texts)
    out = [{"bm25": bm25.top(q, depth)} for q in queries]
    for name, enc in encoders.items():
        if not queries:
            break
        enc, ctx = enc if isinstance(enc, tuple) else (enc, 0)
        sims = enc.encode(queries, query=True) @ enc.encode(memory_keys(texts, ctx)).T
        for j, ranks in enumerate(out):
            sem = [int(i) for i in np.argsort(-sims[j])[:depth]]
            ranks[name] = sem
            ranks[f"bm25+{name}"] = rrf([ranks["bm25"], sem])[:depth]
    return out

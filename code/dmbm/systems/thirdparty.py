"""Memory systems run from their official code, pinned to one commit and fetched at run time (licences: Mem0 Apache-2.0
via PyPI `mem0ai`; A-Mem MIT; MemoryOS Apache-2.0). Each library's LLM calls go to the same OpenAI-compatible endpoint
(our vLLM server, model name "m"); its own embedder defaults are kept unless noted. The packet they return is answered by
the shared reader, not by each library's own answer prompt.
"""
from __future__ import annotations
import importlib
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from .base import System, count_llm, lines

PINS = {
    "agentic_memory": ("agiresearch/A-mem", "ceffb860f0712bbae97b184d440df62bc910ca8d", "agentic_memory",
                       ["__init__.py", "llm_controller.py", "memory_system.py", "retrievers.py"]),
    "memoryos": ("BAI-LAB/MemoryOS", "587ed7755c7aed179965792830ff1b5ad9a6fa92", "memoryos-pypi",
                 ["__init__.py", "long_term.py", "memoryos.py", "mid_term.py", "prompts.py", "retriever.py", "short_term.py",
                  "updater.py", "utils.py"]),
}


def vendor(pkg: str, root: str) -> None:
    """Downloads the pinned files of `pkg` into root/pkg (once) and makes it importable."""
    repo, sha, sub, files = PINS[pkg]
    d = Path(root) / "vendor" / pkg
    if not (d / files[-1]).exists():
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{sha}/{sub}/{f}", d / f)
    if str(d.parent) not in sys.path:
        sys.path.insert(0, str(d.parent))


def _env(url):
    os.environ["OPENAI_BASE_URL"] = url
    os.environ.setdefault("OPENAI_API_KEY", "local")


def _speaker_lines(session):
    return [f"{t.speaker}: {t.text}" for t in session.turns]


class Mem0(System):
    """mem0ai (OSS): LLM fact extraction + ADD/UPDATE/DELETE per session; qdrant on disk; bge-small embedder on CPU."""
    name, needs_llm = "mem0", True
    TOP_K = 10

    def __init__(self, **kw):
        super().__init__(**kw)
        count_llm(); _env(self.llm_url)
        self.mem = {}

    def ingest(self, hid, sessions):
        from mem0 import Memory
        tag = "".join(c if c.isalnum() else "_" for c in hid)
        base = Path(self.workdir) / "mem0" / tag
        shutil.rmtree(base, ignore_errors=True); base.mkdir(parents=True)
        cfg = {"llm": {"provider": "openai", "config": {"model": self.llm_model, "openai_base_url": self.llm_url, "api_key": "local",
                                                         "temperature": 0, "max_tokens": 1500}},
               "embedder": {"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384,
                                                                   "model_kwargs": {"device": "cpu"}}},
               "vector_store": {"provider": "qdrant", "config": {"collection_name": f"c_{tag}"[:60], "path": str(base / "qdrant"),
                                                                 "embedding_model_dims": 384, "on_disk": True}},
               "history_db_path": str(base / "history.db")}
        m = Memory.from_config(cfg)
        for s in sessions:
            if self.out_of_time():
                break
            msgs = [{"role": "user", "content": l} for l in _speaker_lines(s)]
            if msgs:
                m.add(msgs, user_id=tag, metadata={"session_date": s.date, "session": s.session_id})
        self.mem[hid] = (m, tag)

    def context(self, hid, question, q_date):
        m, tag = self.mem[hid]
        try:
            r = m.search(question, top_k=self.TOP_K, filters={"user_id": tag})
        except TypeError:
            r = m.search(question, user_id=tag, limit=self.TOP_K)
        items = r.get("results", r) if isinstance(r, dict) else r
        return lines([((x.get("metadata") or {}).get("session_date", ""), x.get("memory", "")) for x in items])

    def close(self, hid):
        self.mem.pop(hid, None)


class AMem(System):
    """A-Mem (NeurIPS 2025): one note per turn; the LLM writes keywords, context and tags and evolves linked notes;
    ChromaDB + all-MiniLM-L6-v2 (the library default). Retrieval: search_agentic, k = 10."""
    name, needs_llm = "a_mem", True
    K = 10

    def __init__(self, **kw):
        super().__init__(**kw)
        count_llm(); _env(self.llm_url)
        vendor("agentic_memory", self.workdir)
        self.mem = {}

    def ingest(self, hid, sessions):
        ms = importlib.import_module("agentic_memory.memory_system")
        m = ms.AgenticMemorySystem(model_name="all-MiniLM-L6-v2", llm_backend="openai", llm_model=self.llm_model, api_key="local")
        for s in sessions:
            if self.out_of_time():
                break
            for l in _speaker_lines(s):
                m.add_note(l, time=s.date)
        self.mem[hid] = m

    def context(self, hid, question, q_date):
        res = self.mem[hid].search_agentic(question, k=self.K)
        return lines([(x.get("timestamp", ""), x.get("content", "")) for x in res])

    def close(self, hid):
        self.mem.pop(hid, None)


class MemoryOS(System):
    """MemoryOS (EMNLP 2025 Oral): short-term QA pairs -> mid-term pages (heat) -> long-term user profile and knowledge.
    Turns are paired as (first speaker = user, reply = agent). The packet is MemoryOS's own retrieval (pages, user profile,
    user and assistant knowledge, short-term history), formatted as in its get_response, without its answer call."""
    name, needs_llm = "memoryos", True

    def __init__(self, **kw):
        super().__init__(**kw)
        count_llm(); _env(self.llm_url)
        vendor("memoryos", self.workdir)
        self.mem = {}

    def ingest(self, hid, sessions):
        mo = importlib.import_module("memoryos.memoryos")
        tag = "".join(c if c.isalnum() else "_" for c in hid)
        path = Path(self.workdir) / "memoryos" / tag
        shutil.rmtree(path, ignore_errors=True)
        m = mo.Memoryos(user_id=tag, openai_api_key="local", openai_base_url=self.llm_url, data_storage_path=str(path),
                        llm_model=self.llm_model, assistant_id=f"a_{tag}")
        for s in sessions:
            ts = _speaker_lines(s)
            for i in range(0, len(ts), 2):
                m.add_memory(user_input=ts[i], agent_response=ts[i + 1] if i + 1 < len(ts) else "", timestamp=s.date)
            if self.out_of_time():
                break
        try:
            print(f"memoryos stats {hid}: {m.get_memory_stats()}", flush=True)
        except Exception as e:
            print(f"memoryos stats {hid}: unavailable ({type(e).__name__})", flush=True)
        self.mem[hid] = m

    def context(self, hid, question, q_date):
        m = self.mem[hid]
        r = m.retriever.retrieve_context(user_query=question, user_id=m.user_id)
        items = [(p.get("timestamp", ""), f"User: {p.get('user_input', '')} / Reply: {p.get('agent_response', '')} "
                                          f"(chain: {p.get('meta_info', '')})") for p in r["retrieved_pages"]]
        items += [(k.get("timestamp", ""), f"user knowledge: {k['knowledge']}") for k in r["retrieved_user_knowledge"]]
        items += [(k.get("timestamp", ""), f"assistant knowledge: {k['knowledge']}") for k in r["retrieved_assistant_knowledge"]]
        items += [(qa.get("timestamp", ""), f"User: {qa.get('user_input', '')} / Reply: {qa.get('agent_response', '')}")
                  for qa in m.short_term_memory.get_all()]
        prof = m.user_long_term_memory.get_raw_user_profile(m.user_id)
        head = f"USER PROFILE: {prof}\n" if prof and str(prof).lower() != "none" else ""
        return head + lines(items)

    def close(self, hid):
        self.mem.pop(hid, None)

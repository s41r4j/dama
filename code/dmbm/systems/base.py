"""DMBM system interface. Every memory system ingests one history (dated sessions) and returns a memory packet for a
question; the same reader (Qwen3.5-9B, prompt P1) answers from that packet, so only the memory differs.

LLM calls made inside third-party libraries (Mem0, A-Mem, MemoryOS, ...) go through the `openai` SDK; `count_llm()`
patches it once so every call is counted (calls, prompt and completion tokens, seconds) and decoded greedily
(temperature 0), whichever library makes it.
"""
from __future__ import annotations
import threading
import time
from collections import Counter

from ..schema import Session

NOT_FOUND = "Not mentioned in the conversation."
USAGE = Counter()                 # global, read and reset by the runner per system and history
_PATCHED = threading.Event()


def count_llm():
    """Counts every openai chat completion, forces temperature 0 and turns thinking off. Idempotent."""
    if _PATCHED.is_set():
        return
    from openai.resources.chat import completions as cc
    orig = cc.Completions.create

    def create(self, *a, **kw):
        kw["temperature"] = 0
        kw.pop("top_p", None)
        eb = dict(kw.get("extra_body") or {})            # Qwen3/3.5 think by default; the libraries cannot say otherwise
        eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
        kw["extra_body"] = eb
        t = time.time()
        r = orig(self, *a, **kw)
        USAGE["llm_calls"] += 1
        USAGE["llm_seconds"] += time.time() - t
        u = getattr(r, "usage", None)
        if u is not None:
            USAGE["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            USAGE["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
        return r

    cc.Completions.create = create
    _PATCHED.set()


class System:
    name = "base"
    needs_llm = False          # True: calls an LLM at write time (run on DMBM-lite)
    native = "official"        # official code | re-implemented | DAMA

    def __init__(self, llm_url: str | None = None, llm_model: str = "m", workdir: str = "/tmp/dmbm"):
        self.llm_url, self.llm_model, self.workdir = llm_url, llm_model, workdir
        self.deadline, self.truncated = None, False      # set per history by the runner

    def out_of_time(self) -> bool:
        """True once this history's share of the time-box is used: ingestion stops there (the memory is then partial,
        and the runner records it)."""
        if self.deadline is not None and time.time() > self.deadline:
            self.truncated = True
            return True
        return False

    def ingest(self, history_id: str, sessions: list[Session]) -> None:
        raise NotImplementedError

    def context(self, history_id: str, question: str, q_date: str | None) -> str:
        """The memory packet for the reader: one line per memory, dated where the system keeps dates."""
        raise NotImplementedError

    def close(self, history_id: str) -> None:
        """Frees one history's memory (systems keep one store per history)."""


def lines(items, start=1):
    """[(date, text)] -> numbered packet lines; empty -> NOT_FOUND."""
    out = [f"[m:{k}] ({d}) {t}" if d else f"[m:{k}] {t}" for k, (d, t) in enumerate(items, start)]
    return "\n".join(out) if out else NOT_FOUND


try:                                   # the tokenizer DAMA's packet budgets are measured with (v0.ntok)
    import tiktoken
    try:
        _TT = tiktoken.get_encoding("o200k_base")
    except Exception:
        _TT = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TT = None


def ntok(text: str) -> int:
    return len(_TT.encode(text, disallowed_special=())) if _TT else int(len(text.split()) * 1.3)

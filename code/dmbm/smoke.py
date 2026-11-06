"""Smoke test of the DMBM system adapters on CPU (ssa), with a small real LLM behind an OpenAI-compatible endpoint
(llama.cpp `llama-server`, Qwen3-0.6B). One LoCoMo dev session cut to a few turns, three questions. It checks that every
adapter ingests, retrieves and counts its LLM use; the numbers mean nothing.

Usage: python -m dmbm.smoke LOCOMO_JSON LLM_URL [system ...]
"""
from __future__ import annotations
import json
import sys
import time

from .run import build_contexts
from .schema import Item, Session, Turn
from .systems import REGISTRY


def tiny_history(path, n_turns=12):
    conv = json.load(open(path))[0]
    c = conv["conversation"]
    turns = [Turn(t["speaker"], t.get("text", ""), t["dia_id"]) for t in c["session_1"][:n_turns]]
    sessions = [Session("1", c["session_1_date_time"], turns)]
    qs = ["What did Caroline talk about?", "When did they talk?", "What is Melanie interested in?"]
    items = [Item(f"smoke:{k}", "locomo", "dev", "A1", "4", "locomo:0", q, None, "") for k, q in enumerate(qs)]
    return items, {"locomo:0": sessions}


if __name__ == "__main__":
    path, url = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(REGISTRY)
    items, hist = tiny_history(path)
    report = {}
    for n in names:
        t = time.time()
        try:
            sysm = REGISTRY[n](llm_url=url, llm_model="m", workdir="/tmp/dmbm_smoke")
            ctx, st = build_contexts(sysm, items, hist)
            st.pop("per_history", None) if not st.get("failed") else None
            report[n] = {"ok": st["ingested"] == 1 and st["items_with_context"] == len(items), "stats": st,
                         "example": (next(iter(ctx.values()))["context"] or "")[:400] if ctx else None, "minutes": round((time.time() - t) / 60, 1)}
        except Exception as e:
            import traceback
            report[n] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        print(f"== {n}: {json.dumps(report[n], default=str)[:1500]}", flush=True)
    print("SMOKE SUMMARY", {n: r["ok"] for n, r in report.items()})

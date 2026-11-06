"""DMBM runner, step 1: memory packets. For one system and a list of items, ingest each history once, build the packet
for each of its questions, and record write cost (LLM calls, tokens, seconds) and read latency. Reading and judging use
the DAMA engine (same reader and judge for every system), in the Kaggle notebook that calls this.
"""
from __future__ import annotations
import time
import traceback
from collections import defaultdict

from .systems.base import USAGE


def build_contexts(system, items, histories, log=print, timebox_s=None):
    """items: [Item]; histories: {history_id: [Session]}. -> ({item_id: {"context", "read_ms"}}, stats)"""
    by_h = defaultdict(list)
    for it in items:
        by_h[it.history_id].append(it)
    out, per_h, t0 = {}, [], time.time()
    for hid, its in by_h.items():
        if timebox_s and time.time() - t0 > timebox_s:
            log(f"{system.name}: time-box reached after {len(per_h)} of {len(by_h)} histories")
            break
        USAGE.clear()
        t = time.time()
        left = len(by_h) - len(per_h)                    # an even share of the remaining time-box for this history
        system.deadline = t + (t0 + timebox_s - t) / max(1, left) if timebox_s else None
        system.truncated = False
        try:
            system.ingest(hid, histories[hid])
        except Exception as e:
            per_h.append({"history": hid, "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]})
            log(f"{system.name}: ingest failed on {hid}: {type(e).__name__}: {str(e)[:200]}")
            continue
        ingest_s = time.time() - t
        write = dict(USAGE)
        USAGE.clear()
        for it in its:
            ts = time.perf_counter()
            try:
                ctx = system.context(hid, it.question, it.q_date)
            except Exception as e:
                ctx = None
                log(f"{system.name}: context failed on {it.item_id}: {type(e).__name__}: {str(e)[:200]}")
            out[it.item_id] = {"context": ctx, "read_ms": (time.perf_counter() - ts) * 1000}
        read = dict(USAGE)
        system.close(hid)
        n_turns = sum(len(s.turns) for s in histories[hid])
        per_h.append({"history": hid, "turns": n_turns, "ingest_s": round(ingest_s, 2), "write": write, "read_llm": read,
                      "truncated": system.truncated})
        log(f"{system.name}: {hid} ({n_turns} turns) ingested in {ingest_s:.1f} s, {write.get('llm_calls', 0)} LLM calls")
    ok = [h for h in per_h if "error" not in h]
    turns = sum(h["turns"] for h in ok) or 1
    stats = {"system": system.name, "native": system.native, "needs_llm": system.needs_llm, "histories": len(by_h),
             "ingested": len(ok), "failed": len(per_h) - len(ok), "truncated": sum(bool(h.get("truncated")) for h in ok), "items_with_context": sum(v["context"] is not None for v in out.values()),
             "llm_calls_per_turn": sum(h["write"].get("llm_calls", 0) for h in ok) / turns,
             "write_tokens_per_turn": sum(h["write"].get("prompt_tokens", 0) + h["write"].get("completion_tokens", 0) for h in ok) / turns,
             "ingest_s_per_turn": sum(h["ingest_s"] for h in ok) / turns,
             "read_ms_p50": _pct([v["read_ms"] for v in out.values()], 50), "read_ms_p95": _pct([v["read_ms"] for v in out.values()], 95),
             "per_history": per_h}
    return out, stats


def _worker(args):
    """One process: build the system once and run its share of histories (libraries such as A-Mem keep global state,
    so histories are split across processes, not threads)."""
    name, kw, items, histories, timebox_s = args
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""             # run 1: workers put their embedders on GPU 0 beside vLLM (OOM)
    from .systems import REGISTRY
    sysm = REGISTRY[name](**kw)
    return build_contexts(sysm, items, histories, log=lambda *a: print(*a, flush=True), timebox_s=timebox_s)


def build_contexts_parallel(name, kw, items, histories, workers=8, log=print, timebox_s=None):
    """build_contexts over `workers` processes (spawn), histories dealt round-robin. LLM-heavy libraries call the
    endpoint one request at a time; parallel histories let vLLM batch them."""
    import multiprocessing as mp
    hids = sorted(histories, key=lambda h: -sum(len(s.turns) for s in histories[h]))      # largest first
    shares = [hids[k::workers] for k in range(min(workers, len(hids)))]
    jobs = [(name, kw, [it for it in items if it.history_id in set(sh)], {h: histories[h] for h in sh}, timebox_s) for sh in shares]
    out, per_h = {}, []
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        for ctx, st in pool.imap_unordered(_worker, jobs):
            out.update(ctx); per_h += st["per_history"]
            log(f"{name}: a worker finished {st['ingested']} histories ({st['failed']} failed)")
    ok = [h for h in per_h if "error" not in h]
    turns = sum(h["turns"] for h in ok) or 1
    stats = {"system": name, "histories": len(histories), "ingested": len(ok), "failed": len(per_h) - len(ok), "workers": len(jobs),
             "truncated": sum(bool(h.get("truncated")) for h in ok),
             "items_with_context": sum(v["context"] is not None for v in out.values()),
             "llm_calls_per_turn": sum(h["write"].get("llm_calls", 0) for h in ok) / turns,
             "write_tokens_per_turn": sum(h["write"].get("prompt_tokens", 0) + h["write"].get("completion_tokens", 0) for h in ok) / turns,
             "ingest_s_per_turn": sum(h["ingest_s"] for h in ok) / turns,
             "read_ms_p50": _pct([v["read_ms"] for v in out.values()], 50), "read_ms_p95": _pct([v["read_ms"] for v in out.values()], 95),
             "per_history": per_h}
    return out, stats


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

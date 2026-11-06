"""DMBM run 1 (2026-11-01): DMBM-dev-lite. Every baseline and memory system under one writer, one reader and one judge,
on dev questions only. Private Kaggle notebook `s41r4j/dama-dmbm1` (T4 x2). Spec: docs/research/DMBM_spec.md.

Items: v2c's LoCoMo dev sample (conversations 0-3: 250 questions in categories 1-4 + 50 adversarial) and LME-lite (3 dev
questions per LongMemEval type, seeded; 18). Holdout, LoCoMo test, BEAM and MemoryAgentBench are not loaded.

Systems (code/dmbm/systems): floors (no memory, most recent 3,000 tokens); retrieval and RAG (BM25, dense bge-small,
chunk RAG, hybrid BM25+dense+cross-encoder RAG, exchange-level vector memory); LLM-written memories (session-summary
memory, Mem0, MemoryOS, LightMem, A-Mem, ENGRAM, Graphiti/Zep); DAMA A0 (v2c C3_m0) and the best DAMA v3 arm (their
packets re-read here); oracle = gold evidence turns. Every packet fits the same ~3,000-token budget or is smaller.

GPU use: ONE model (MODEL) is the writer of every LLM-based memory and the reader of every packet, on one vLLM server
(max length 16,384: Mem0's extraction prompt is ~8.3k tokens; up to 128 sequences batched). Ingestion runs in 12
processes; a reader thread reads each finished system while the next one ingests, so the GPUs are never idle waiting
on a library's one-call-at-a-time loop. Judge: Qwen2.5-7B. Scores follow the DMBM taxonomy (A1-A8).
"""
from __future__ import annotations
import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SMOKE = False
HERE = Path(globals().get("__file__", "dama_dmbm1.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v3b_2026-11-01/dama_paper_v3b.py")))
import dama_paper_v3b as B3      # engine (Srv, Cli on any GPU/port), v2c setup, reading and judging
Cm, Bm, V, L, v0 = B3.Cm, B3.Bm, B3.V, B3.L, B3.v0
import numpy as np
from dmbm import schema as S
from dmbm import sources
from dmbm.run import build_contexts, build_contexts_parallel

B3.SMOKE = Cm.SMOKE = Bm.SMOKE = V.SMOKE = v0.SMOKE = Cm.v1.SMOKE = SMOKE
Cm.N_LOCO, Cm.N_ADV = (12, 3) if SMOKE else (250, 50)
v0.TIME_LIMIT_H = 1.5 if SMOKE else 10.0
log, stage, OUT, D, STATE = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE
JUDGE_ID = B3.JUDGE_ID
MODEL = B3.READER              # ONE model is the writer of every LLM-based memory AND the reader of every packet, so
                               # reading overlaps with writing on the same vLLM server (GPUs stay busy). v3b: the 35B-A3B
                               # MoE ties the 9B as a reader and its 4,608-token limit cannot fit Mem0's ~8.3k prompt.
READER = WRITER = MODEL
DAMA_V3_ARM = None             # v3b: every ledger arm failed its rule (KILL), so DAMA enters as A0 (v2c C3_m0)
LME_PER_TYPE = 1 if SMOKE else 3
CHEAP = ["no_memory", "recent", "bm25", "dense_bge_small", "rag_chunk", "rag_hybrid_rerank", "vector_memory"]
HEAVY = ["summary_memory", "mem0", "memoryos", "lightmem", "a_mem", "engram", "graphiti_zep"]   # cheapest first
HEAVY = [h for h in HEAVY if h not in os.environ.get("DMBM_SKIP", "").split(",")]   # dry runs on a small box only
HEAVY_TIMEBOX_S = 1800 if SMOKE else 50 * 60               # per system (partial systems are reported as such)
WORKERS = 1 if SMOKE else 12
WORKERS_BY = {"graphiti_zep": 4, "lightmem": 6}     # their rerankers / compressors use ~1-2.5 GB of RAM per process
PORT = 8011
R = {"settings": {"model": MODEL, "judge": JUDGE_ID}, "systems": {}, "scores": {}, "notes": []}


def save_r():
    (OUT / "results_dmbm1.json").write_text(json.dumps(R, indent=1, default=str))


def pip_install():
    """Libraries the Kaggle image lacks (mem0ai is preinstalled)."""
    if B3.FAKE:
        return
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chromadb", "litellm", "rank_bm25", "nltk", "faiss-cpu",
                        "qdrant-client", "graphiti-core[falkordb]", "falkordblite", "llmlingua"], capture_output=True, text=True, timeout=1500)
    log("pip:", "ok" if r.returncode == 0 else r.stderr[-800:])
    import nltk
    for p in ("punkt", "punkt_tab"):
        nltk.download(p, quiet=True)


# ---------------------------------------------------------------------------------------------- items
@stage("dmbm1_items")
def stage_items():
    rng = random.Random(Cm.SEED)
    lq = D["lq"]
    loco = rng.sample([q for q in lq if q["cat"] != 5], min(Cm.N_LOCO, sum(q["cat"] != 5 for q in lq)))
    loco += rng.sample([q for q in lq if q["cat"] == 5], min(Cm.N_ADV, sum(q["cat"] == 5 for q in lq)))
    by = defaultdict(list)
    for q in D["lme"]:
        by[q["type"]].append(q)
    rng2 = random.Random(Cm.SEED + 1)
    lme = [q for t in sorted(by) for q in rng2.sample(by[t], min(LME_PER_TYPE, len(by[t])))]
    convs = [c for c in D["convs"] if any(q["ci"] == c["ci"] for q in loco)]
    i1, h1 = sources.locomo(convs, loco)
    i2, h2 = sources.lme(lme)
    D["items"], D["hist"] = i1 + i2, {**h1, **h2}
    D["qs_by_item"] = {f"locomo:{q['qid']}": q for q in loco} | {f"lme:{q['qid']}": q for q in lme}
    R["settings"]["items"] = {"locomo": len(i1), "lme": len(i2), "histories": len(D["hist"]),
                              "turns": sum(len(t.turns) for ss in D["hist"].values() for t in ss),
                              "abilities": dict(Counter(it.ability for it in D["items"]))}
    log("DMBM-dev-lite items:", R["settings"]["items"])
    D.pop("bge", None)
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- packets
def oracle(it):
    q = D["qs_by_item"][it.item_id]
    turns = V.reader_of(q)[1]
    ev = set(q["evidence"])
    chosen = [t for t in turns if t["dia"] in ev]
    return L.render_many(chosen, True) if chosen else v0.NOT_FOUND


@stage("dmbm1_cheap")
def stage_cheap():
    for name in CHEAP:
        kw = {"budget": 3000} if name == "recent" else {}
        sysm = S_REG()[name](**kw)
        ctx, st = build_contexts(sysm, D["items"], D["hist"], log=log)
        D.setdefault("packets", {})[name] = ctx
        st.pop("per_history", None)
        R["systems"][name] = st
        log(f"{name}:", st)
    D["packets"]["oracle"] = {it.item_id: {"context": oracle(it), "read_ms": 0.0} for it in D["items"]}
    save_r()


def S_REG():
    from dmbm.systems import REGISTRY
    return REGISTRY


@stage("dmbm1_heavy")
def stage_heavy(url):
    for name in HEAVY:
        if B3.hours() > v0.TIME_LIMIT_H - (0.3 if SMOKE else 3.0):
            R["notes"].append(f"{name} skipped: time"); continue
        t = time.time()
        kw = {"llm_url": url, "llm_model": "m", "workdir": "/tmp/dmbm"}
        ctx, st = build_contexts_parallel(name, kw, D["items"], D["hist"], workers=1 if SMOKE else WORKERS_BY.get(name, WORKERS),
                                            log=log, timebox_s=HEAVY_TIMEBOX_S)
        D["packets"][name] = ctx
        per = st.pop("per_history", [])
        st["minutes"] = round((time.time() - t) / 60, 1)
        st["errors"] = [h for h in per if "error" in h][:5]
        R["systems"][name] = st
        (OUT / f"dmbm1_{name}_per_history.json").write_text(json.dumps(per, default=str))
        log(f"{name}:", {k: v for k, v in st.items() if k != "errors"})
        save_r()
        READ_QUEUE.append(name)                   # read while the next system ingests (same vLLM server)


# ---------------------------------------------------------------------------------------------- reading + judging
ROWS = {}


READ_QUEUE = []


def read_one(cli, name):
    pk = D["packets"][name]
    jobs = [(name, D["qs_by_item"][it.item_id], {"context": pk[it.item_id]["context"], "read_ms": pk[it.item_id]["read_ms"]})
            for it in D["items"] if it.item_id in pk and pk[it.item_id]["context"] is not None]
    if not jobs:
        return
    rows, sec = Bm.read(cli, jobs, "P1", f"{B3.short(READER)}|dmbm1_{name}")
    R["systems"].setdefault(name, {})["reader_s_per_question"] = round(sec, 3)
    ROWS[name] = rows
    V.write_rows(f"dmbm1_{name}", rows)


def reader_loop(cli, stop):
    """Background thread: reads every finished system's packets while later systems are still ingesting."""
    while not (stop.is_set() and not READ_QUEUE):
        if READ_QUEUE:
            name = READ_QUEUE.pop(0)
            try:
                read_one(cli, name)
            except Exception as e:
                STATE["errors"][f"read_{name}"] = f"{type(e).__name__}: {str(e)[:150]}"
        else:
            time.sleep(5)


def dama_packets(arm_file, name):
    """DAMA's packets (context field of an attached earlier run's rows) for this run's items, re-read by READER."""
    hits = sorted(glob.glob(f"/kaggle/input/**/methods/{arm_file}.jsonl", recursive=True), key=len)
    if not hits:
        R["notes"].append(f"{arm_file} not attached"); return
    want = {it.item_id.split(":", 1)[1]: it.item_id for it in D["items"]}
    D["packets"][name] = {want[r["qid"]]: {"context": r["context"], "read_ms": r.get("read_ms") or 0.0}
                          for r in map(json.loads, open(hits[0])) if r["qid"] in want}


def judge_all():
    j = B3.Srv(JUDGE_ID, "0,1", PORT)
    if not j.start():
        raise RuntimeError("judge could not be served")
    try:
        jc = B3.Cli(JUDGE_ID, PORT, conc=64)
        for name, rs in ROWS.items():
            if rs and "qwen_correct" not in rs[0]:
                Bm.judge(jc, rs)
                V.write_rows(f"dmbm1_{name}", rs)
    finally:
        j.stop()


# ---------------------------------------------------------------------------------------------- scores (DMBM abilities)
def ability(r):
    if r["bench"] == "lme":
        return S.lme_ability(r["type"], r["qid"])
    return S.LOCOMO[r["cat"]]


def scores():
    for name, rs in ROWS.items():
        if not rs or "qwen_correct" not in rs[0]:
            continue
        by = defaultdict(list)
        for r in rs:
            by[ability(r)].append(bool(r["qwen_correct"]))
        ab = {a: (float(np.mean(v)), len(v)) for a, v in sorted(by.items())}
        head = [ab[a][0] for a in S.HEADLINE if a in ab]
        src = {b: float(np.mean([r["qwen_correct"] for r in rs if r["bench"] == b and (b != "locomo" or r["cat"] != 5)]))
               for b in ("locomo", "lme") if any(r["bench"] == b for r in rs)}
        R["scores"][name] = {"macro_headline": float(np.mean(head)) if head else None, "abilities_present": [a for a in S.HEADLINE if a in ab],
                             "by_ability": ab, "by_source": src, "n": len(rs),
                             "mean_packet_tokens": float(np.mean([r["input_tokens"] for r in rs]))}
    save_r()


def write_report():
    f = lambda x: "n/a" if x is None else f"{100 * x:.1f}"
    out = [f"# DMBM-dev-lite, run 1 ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           f"Items: `{json.dumps(R['settings'].get('items'))}`. One model as writer and reader: {MODEL} (P1); judge {JUDGE_ID}. Dev only.", "",
           "| system | macro A1-A8 | LoCoMo | LME | " + " | ".join(S.HEADLINE) + " | packet tokens | LLM calls / turn | write tokens / turn | read ms p50 |",
           "|---|---|---|---|" + "---|" * len(S.HEADLINE) + "---|---|---|---|"]
    for name, sc in sorted(R["scores"].items(), key=lambda kv: -(kv[1]["macro_headline"] or 0)):
        st = R["systems"].get(name, {})
        ab = sc["by_ability"]
        out.append(f"| {name} | {f(sc['macro_headline'])} | {f(sc['by_source'].get('locomo'))} | {f(sc['by_source'].get('lme'))} | "
                   + " | ".join(f(ab[a][0]) if a in ab else "—" for a in S.HEADLINE)
                   + f" | {sc['mean_packet_tokens']:.0f} | {st.get('llm_calls_per_turn', 0):.2f} | {st.get('write_tokens_per_turn', 0):.0f} | "
                   f"{st.get('read_ms_p50') if st.get('read_ms_p50') is None else round(st['read_ms_p50'], 1)} |")
    out += ["", "Ability n: `" + json.dumps({k: {a: v[1] for a, v in s["by_ability"].items()} for k, s in R["scores"].items()}) + "`", "",
            "## Systems", "```", json.dumps(R["systems"], indent=1, default=str)[:20000], "```", "",
            f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_DMBM1.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


if __name__ == "__main__":
    import threading
    log("environment", v0.environment())
    pip_install()
    Cm.stage_setup()
    if "v2c_setup" not in STATE["errors"]:
        V.stage_select_prior()
        stage_items()
        stage_cheap()
        dama_packets("dev_A0", "dama_A0")
        if DAMA_V3_ARM:
            dama_packets(f"dev_{DAMA_V3_ARM}", f"dama_{DAMA_V3_ARM}")
        READ_QUEUE.extend(D["packets"])               # cheap systems, oracle and DAMA: read at once
        if Bm.install_vllm(B3.VLLM):
            srv = B3.Srv(MODEL, "0,1", PORT, max_len=16384, util=0.90, max_seqs=128)
            if srv.start():
                stop = threading.Event()
                rt = threading.Thread(target=reader_loop, args=(B3.Cli(READER, PORT, conc=48), stop))
                rt.start()
                try:
                    stage_heavy(f"http://127.0.0.1:{PORT}/v1")
                finally:
                    stop.set(); rt.join()
                    srv.stop()
                judge_all()
    try:
        scores()
    except Exception as e:
        STATE["errors"]["dmbm1_scores"] = f"{type(e).__name__}: {str(e)[:150]}"
    save_r()
    write_report()
    log(f"dmbm1 done; errors: {STATE['errors'] or 'none'}")

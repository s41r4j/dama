"""DMBM run 2 (2026-11-02): the LLM-written memories that run 1 could not finish. Private Kaggle notebook
`s41r4j/dama-dmbm2` (T4 x2). Same items, model, reader prompt and judge as run 1 (experiments/dmbm1_2026-11-01).

Run 1 (8.3 h) finished the floors, retrieval/RAG/vector baselines, summary memory, Mem0, DAMA A0 and the oracle; it
lost MemoryOS to CUDA out-of-memory (its worker processes put their embedders on GPU 0 beside vLLM) and never reached
LightMem, A-Mem, ENGRAM and Graphiti (its time-box was checked only between histories, and reading every cheap packet
shared the GPU). Fixes here: workers are CPU-only; each history gets an even share of the remaining time-box and every
ingest loop stops at it (the memory is then partial and flagged "truncated"); run 1's judged rows are carried, so the
GPU serves only these five systems' writing and reading.
"""
from __future__ import annotations
import glob
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(globals().get("__file__", "dama_dmbm2.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("dmbm1_2026-11-01/dama_dmbm1.py")))
import dama_dmbm1 as M1          # run 1's items, systems runner, reader thread, judge and scores
B3, Bm, V, v0 = M1.B3, M1.Bm, M1.V, M1.v0
import numpy as np

SMOKE = M1.SMOKE
M1.HEAVY = [h for h in ["memoryos", "lightmem", "a_mem", "engram", "graphiti_zep"] if h not in M1.os.environ.get("DMBM_SKIP", "").split(",")]
M1.HEAVY_TIMEBOX_S = 600 if SMOKE else 65 * 60
v0.TIME_LIMIT_H = 1.5 if SMOKE else 9.0
log, OUT, D, STATE, R, ROWS = v0.log, v0.OUT, M1.D, v0.STATE, M1.R, M1.ROWS
CARRY_SKIP = {"memoryos"}        # run 1's MemoryOS rows are the OOM failure, not a result


def carry_run1():
    """Run 1's judged rows and system statistics (attached as kernel source s41r4j/dama-dmbm1)."""
    hits = glob.glob("/kaggle/input/**/methods/dmbm1_*.jsonl", recursive=True)
    for f in sorted(hits, key=len):
        name = Path(f).stem[len("dmbm1_"):]
        if name in CARRY_SKIP or name in ROWS:
            continue
        rs = [json.loads(l) for l in open(f)]
        if rs and "qwen_correct" in rs[0]:
            ROWS[name] = rs
    prev = sorted(glob.glob("/kaggle/input/**/paper/results_dmbm1.json", recursive=True), key=len)
    if prev:
        for k, v in json.load(open(prev[0])).get("systems", {}).items():
            if k not in CARRY_SKIP:
                R["systems"].setdefault(k, v).setdefault("from", "run 1")
    log("carried from run 1:", sorted(ROWS))


def table():
    """Per-benchmark accuracies with n (the macro over abilities in run 1 weighted abilities with n = 1-3 equally)."""
    def acc(rs, f):
        xs = [bool(r["qwen_correct"]) for r in rs if f(r)]
        return (100 * float(np.mean(xs)), len(xs)) if xs else (None, 0)
    out = []
    for name, rs in ROWS.items():
        if not rs or "qwen_correct" not in rs[0]:
            continue
        st = R["systems"].get(name, {})
        out.append({"system": name,
                    "locomo": acc(rs, lambda r: r["bench"] == "locomo" and r["cat"] != 5),
                    "multi_hop": acc(rs, lambda r: r["bench"] == "locomo" and r["cat"] == 1),
                    "temporal": acc(rs, lambda r: r["bench"] == "locomo" and r["cat"] == 2),
                    "single_hop": acc(rs, lambda r: r["bench"] == "locomo" and r["cat"] == 4),
                    "adversarial": acc(rs, lambda r: r["bench"] == "locomo" and r["cat"] == 5),
                    "lme_lite": acc(rs, lambda r: r["bench"] == "lme"),
                    "packet_tokens": float(np.mean([r["input_tokens"] for r in rs])),
                    "llm_calls_per_turn": st.get("llm_calls_per_turn"), "write_tokens_per_turn": st.get("write_tokens_per_turn"),
                    "read_ms_p50": st.get("read_ms_p50"), "histories": f"{st.get('ingested', '—')}/{st.get('histories', '—')}",
                    "truncated": st.get("truncated")})
    R["table"] = sorted(out, key=lambda x: -(x["locomo"][0] or -1))


def write_report():
    f = lambda x: "—" if x[0] is None else f"{x[0]:.1f} ({x[1]})"
    g = lambda x, d=2: "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    out = [f"# DMBM-dev-lite, runs 1 + 2 ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           f"Items: `{json.dumps(R['settings'].get('items'))}`. Writer and reader {M1.MODEL} (P1); judge {M1.JUDGE_ID}. Dev only. "
           "Accuracy % (n). LoCoMo = categories 1-4; adversarial = should abstain; LME-lite n is too small to rank on.", "",
           "| system | LoCoMo | multi-hop | temporal | single-hop | adversarial | LME-lite | packet tokens | LLM calls/turn | write tokens/turn | read ms p50 | histories | truncated |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in R.get("table", []):
        out.append(f"| {x['system']} | {f(x['locomo'])} | {f(x['multi_hop'])} | {f(x['temporal'])} | {f(x['single_hop'])} | {f(x['adversarial'])} | "
                   f"{f(x['lme_lite'])} | {x['packet_tokens']:.0f} | {g(x['llm_calls_per_turn'])} | {g(x['write_tokens_per_turn'], 0)} | "
                   f"{g(x['read_ms_p50'], 1)} | {x['histories']} | {g(x['truncated'])} |")
    out += ["", "## Systems (run 2)", "```", json.dumps({k: v for k, v in R["systems"].items() if v.get("from") != "run 1"}, indent=1, default=str)[:15000],
            "```", "", f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_DMBM2.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


if __name__ == "__main__":
    log("environment", v0.environment())
    M1.pip_install()
    M1.Cm.stage_setup()
    if "v2c_setup" not in STATE["errors"]:
        V.stage_select_prior()
        M1.stage_items()
        carry_run1()
        D["packets"] = {}
        if Bm.install_vllm(B3.VLLM):
            srv = B3.Srv(M1.MODEL, "0,1", M1.PORT, max_len=16384, util=0.90, max_seqs=128)
            if srv.start():
                stop = threading.Event()
                rt = threading.Thread(target=M1.reader_loop, args=(B3.Cli(M1.READER, M1.PORT, conc=48), stop))
                rt.start()
                try:
                    M1.stage_heavy(f"http://127.0.0.1:{M1.PORT}/v1")
                finally:
                    stop.set(); rt.join()
                    srv.stop()
                M1.judge_all()
    try:
        table()
    except Exception as e:
        STATE["errors"]["dmbm2_table"] = f"{type(e).__name__}: {str(e)[:150]}"
    (OUT / "results_dmbm2.json").write_text(json.dumps(R, indent=1, default=str))
    write_report()
    log(f"dmbm2 done; errors: {STATE['errors'] or 'none'}")

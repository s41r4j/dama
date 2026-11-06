"""DAMA paper v3a (2026-10-31): candidate A of DAMA v3, dev only. Private Kaggle notebook `s41r4j/dama-paper-v3a` (T4 x2).

v2c showed the memory is near this reader's ceiling on dev (LME 82.1% overall, 83.8% when the packet holds all evidence)
and that the reader's thinking mode costs 12-17 points. The remaining errors are reasoning over evidence already in the
packet: counting, two-fact arithmetic, dates, non-latest values. Candidate A (docs/research/DAMA_v3_research_and_
architecture.md §5.1) changes only how the same packet is shown and how the reader is asked:

  A0   V2_fast, no neighbours, B = 3000, time notes, v2's layout, P1     = v2c's C3_m0 (its cached calls are reused)
  A1   the same turns in layout R1 (session number, weekday + ISO date, days before the question, offsets on time
       notes; code/dama_agent/render.py), P1
  A2   the same turns, R1, prompt P2 (evidence table, ITEM lines, one CALC line), up to 768 new tokens
  A2c  A2's outputs with code/dama_agent/postcheck.py applied (no extra reader call)

Same dev questions as v2c (250 LoCoMo dev + 50 adversarial, all 218 LME dev); holdout and test never loaded. Reader
Qwen3.5-9B, thinking off; Qwen2.5-7B judge. No DeepSeek. Decision rules are fixed in README.md before the run.
"""
from __future__ import annotations
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SMOKE = False
HERE = Path(globals().get("__file__", "dama_paper_v3a.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v2c_2026-10-30/dama_paper_v2c.py")))
import dama_paper_v2c as Cm      # v2c's dev-only setup and packet; through it v2b's engine, reading and judging
Bm, V, L, v0 = Cm.Bm, Cm.V, Cm.L, Cm.v0
import numpy as np
from dama_agent.postcheck import postcheck
from dama_agent.render import messages_p2, render_packet, session_order

Cm.SMOKE = Bm.SMOKE = V.SMOKE = v0.SMOKE = Cm.v1.SMOKE = SMOKE
Cm.N_LOCO, Cm.N_ADV = (12, 3) if SMOKE else (250, 50)
Cm.save_r = lambda: None                                  # v2c's results file is not written by this run
v0.TIME_LIMIT_H, v0.FINAL_LIMIT_H = (1.5, 2.0) if SMOKE else (5.0, 5.5)
log, stage, OUT, D, STATE = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE
READER, LABEL, JUDGE_ID = Cm.READER, Cm.LABEL, Bm.JUDGE_ID
P2_MAX_NEW = 768
ARMS = ("A0", "A1", "A2", "A2c")
HARD = {"lme": {"temporal-reasoning", "multi-session"}, "locomo": {"temporal", "multi-hop"}}
R = {"settings": {}, "arms": {}, "paired": {}, "postcheck": {}, "decisions": {}, "notes": []}


def save_r():
    (OUT / "results_v3a.json").write_text(json.dumps(R, indent=1, default=str))


# ---------------------------------------------------------------------------------------------- P2 through v2b's read()
Bm.PROMPTS["P2"] = {"locomo": (None, P2_MAX_NEW), "lme": (None, P2_MAX_NEW)}
_messages_p1 = Bm.messages


def _messages(q, ctx, prompt):
    if prompt != "P2":
        return _messages_p1(q, ctx, prompt)
    lines = ctx.splitlines() if ctx != v0.NOT_FOUND else []
    return messages_p2(q["bench"], q["question"], q.get("date") if q["bench"] == "lme" else None, lines)


Bm.messages = _messages


# ---------------------------------------------------------------------------------------------- packets
@stage("v3a_contexts")
def stage_contexts():
    """v2c's question sample and C3_m0 packet; A1/A2 render the same turns in layout R1."""
    import random
    rng = random.Random(Cm.SEED)
    lq = D["lq"]
    loco = rng.sample([q for q in lq if q["cat"] != 5], min(Cm.N_LOCO, sum(q["cat"] != 5 for q in lq)))
    loco += rng.sample([q for q in lq if q["cat"] == 5], min(Cm.N_ADV, sum(q["cat"] == 5 for q in lq)))
    qs = loco + D["lme"]
    J = {a: [] for a in ("A0", "A1")}
    for q in qs:
        c = Cm.ctx(q, neighbours=0)                       # = v2c C3_m0
        turns = V.reader_of(q)[1]
        by_dia = {t["dia"]: t for t in turns}
        lines, _ = render_packet([by_dia[d] for d in c["dia"]], q.get("date") if q["bench"] == "lme" else None, session_order(turns))
        J["A0"].append(c)
        J["A1"].append({**c, "context": "\n".join(lines) if lines else v0.NOT_FOUND})
    D["qs"], D["ctx"] = qs, J
    D.pop("bge", None)                                    # as v2c: free GPU 0 for vLLM (its start-up checks free memory)
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()
    R["settings"]["questions"] = {"locomo": len(loco), "lme": len(D["lme"])}
    R["settings"]["mean_context_tokens"] = {a: float(np.mean([v0.ntok(c["context"]) for c in J[a]])) for a in J}
    log("v3a contexts:", R["settings"])
    save_r()


def jobs(arm, ctx_arm):
    return [(arm, q, c) for q, c in zip(D["qs"], D["ctx"][ctx_arm])]


def with_postcheck(rows):
    """A2c: A2's rows with the CALC check applied; unchanged answers stay byte-identical (their judge calls are cached)."""
    out = []
    for r in rows:
        pc = postcheck(r["raw"] or "")
        out.append({**r, "method": "A2c", "answer": pc.answer if pc.changed else r["answer"],
                    "postcheck": {"changed": pc.changed, "reason": pc.reason, "kind": pc.kind, "value": pc.value}})
    return out


# ---------------------------------------------------------------------------------------------- reading + judging
def gpu_free_mib():
    """Through nvidia-smi, so this process does not open a CUDA context on GPU 1."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split()]
    except Exception as e:
        return repr(e)


def keep_failed_log(srv, ver):
    """The server's 3000-character tail is mostly traceback; log its error lines and keep the whole log per version."""
    if not srv.log_path.exists():
        return
    text = srv.log_path.read_text(errors="replace")
    errs = [l for l in text.splitlines() if "Error" in l and not l.lstrip().startswith(("File ", "raise "))]
    log(f"vLLM {ver} error lines:", "\n".join(errs[:15]) or "(none)")
    srv.log_path.with_name(f"{srv.log_path.stem}_{ver}.log").write_text(text)


@stage("v3a_read")
def stage_read():
    ok = False
    for ver in Bm.VLLM_VERSIONS:
        if Bm.install_vllm(ver):
            log("GPU free MiB before vLLM:", gpu_free_mib())
            srv = Bm.Server(READER)
            if srv.start():
                ok = True; R["settings"]["vllm"] = ver
                break
            keep_failed_log(srv, ver)
    if not ok:
        raise RuntimeError("vLLM could not serve the reader")
    rows = {}
    for arm, ctx_arm, prompt in (("A0", "A0", "P1"), ("A1", "A1", "P1"), ("A2", "A1", "P2")):
        if (time.time() - v0.T_START) / 3600 > v0.TIME_LIMIT_H:
            STATE["errors"][arm] = "time guard"; continue
        client = Bm.Client(READER)
        rows[arm], sec = Bm.read(client, jobs(arm, ctx_arm), prompt, f"{LABEL}|v3a_{arm}")   # A0: same calls as v2c, cached
        R["arms"].setdefault(arm, {})["s_per_question"] = round(sec, 3)
        V.write_rows(f"dev_{arm}", rows[arm])
        save_r()
    srv.stop(drop_weights=True)
    if "A2" in rows:
        rows["A2c"] = with_postcheck(rows["A2"])
        R["arms"].setdefault("A2c", {})["s_per_question"] = R["arms"]["A2"]["s_per_question"]
    j = Bm.Server(JUDGE_ID)
    if not j.start():
        raise RuntimeError("judge could not be served")
    jc = Bm.Client(JUDGE_ID)
    for arm, rs in rows.items():
        Bm.judge(jc, rs)
        V.write_rows(f"dev_{arm}", rs)
    j.stop()
    D["rows"] = rows


# ---------------------------------------------------------------------------------------------- statistics + decisions
def acc(rs, bench=None, groups=None):
    """Qwen-judged accuracy, LoCoMo categories 1-4 only; groups: a type / category set per bench."""
    xs = [r["qwen_correct"] for r in rs if r["bench"] != "locomo" or r["cat"] != 5
          if (bench is None or r["bench"] == bench)
          and (groups is None or (r.get("type") if r["bench"] == "lme" else L.CATS.get(r.get("cat"))) in groups[r["bench"]])]
    return float(np.mean(xs)) if xs else None


def mean_acc(rs):
    return float(np.mean([acc(rs, "locomo"), acc(rs, "lme")]))


def summary(rs):
    a = Bm.dev_acc(rs)
    by_cat = defaultdict(list)
    for r in rs:
        if r["bench"] == "locomo" and r["cat"] != 5:
            by_cat[L.CATS[r["cat"]]].append(r["qwen_correct"])
    lm = [r for r in rs if r["bench"] == "lme"]
    return {**a, "hard": acc(rs, groups=HARD), "locomo_by_category": {k: float(np.mean(v)) for k, v in by_cat.items()},
            "lme_acc_if_all_evidence": float(np.mean([r["qwen_correct"] for r in lm if r.get("evidence_all")])) if lm else None,
            "mean_tokens": float(np.mean([r["input_tokens"] for r in rs])), "no_answer_line": float(np.mean([not r["parsed"] for r in rs]))}


def paired(a, b):
    """a - b, both benchmarks pooled (LoCoMo cat. 1-4 + LME), exact McNemar and item bootstrap."""
    keep = lambda rs: [r for r in rs if r["bench"] != "locomo" or r["cat"] != 5]
    return L.paired(keep(a), keep(b), key="qwen_correct", cluster=None)


def decide(rows):
    """The rules in README.md, applied mechanically."""
    s, out = R["arms"], {}
    if "A1" in rows and "A0" in rows:
        d, dh = s["A1"]["mean"] - s["A0"]["mean"], s["A1"]["hard"] - s["A0"]["hard"]
        worst = min(s["A1"][b] - s["A0"][b] for b in ("locomo", "lme"))
        out["R1"] = {"mean_gain": d, "hard_gain": dh, "worst_bench": worst,
                     "keep": d >= 0.01 and dh >= 0.02 and worst >= -0.02}
    if "A2" in rows and "A1" in rows:
        d, dh = s["A2"]["mean"] - s["A1"]["mean"], s["A2"]["hard"] - s["A1"]["hard"]
        ds = s["A2"]["s_per_question"] - s["A1"]["s_per_question"]
        out["P2"] = {"mean_gain": d, "hard_gain": dh, "extra_s_per_question": ds,
                     "keep": (d >= 0.02 or dh >= 0.04) and ds <= 2.5}
    if "A2c" in rows:
        ch = [(a, b) for a, b in zip(rows["A2"], rows["A2c"]) if b["postcheck"]["changed"]]
        good = sum(b["qwen_correct"] and not a["qwen_correct"] for a, b in ch)
        bad = sum(a["qwen_correct"] and not b["qwen_correct"] for a, b in ch)
        rate = len(ch) / max(1, len(rows["A2"]))
        R["postcheck"] = {"changed": len(ch), "rate": rate, "good": good, "bad": bad,
                          "reasons": dict(Counter(r["postcheck"]["reason"] for r in rows["A2c"]))}
        out["postcheck"] = {"keep": bool(ch) and rate <= 0.10 and good >= 2 * max(bad, 1), "rate": rate, "good": good, "bad": bad}
    R["decisions"] = out


def stage_stats():
    rows = D.get("rows", {})
    for arm, rs in rows.items():
        R["arms"].setdefault(arm, {}).update(summary(rs))
    for a, b in (("A1", "A0"), ("A2", "A1"), ("A2", "A0"), ("A2c", "A2"), ("A2c", "A0")):
        if a in rows and b in rows:
            R["paired"][f"{a}-{b}"] = paired(rows[a], rows[b])
    decide(rows)
    save_r()


def write_report():
    f = lambda x: "n/a" if x is None else f"{100 * x:.1f}"
    out = [f"# DAMA paper v3a: candidate A on dev ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           "Dev only (v2c's sample: 250 LoCoMo dev + 50 adversarial, 218 LME dev). Reader Qwen3.5-9B, thinking off; Qwen2.5-7B judge. "
           "A0 = v2c C3_m0; A1 = layout R1 + P1; A2 = R1 + P2; A2c = A2 + CALC post-check. Hard = LME temporal + multi-session, "
           "LoCoMo temporal + multi-hop.", "", f"Settings: `{json.dumps(R['settings'], default=str)}`", "",
           "| arm | mean | LoCoMo | LME | hard | LME if all evidence | LoCoMo abstention | tokens | s/question | no ANSWER line |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ARMS:
        v = R["arms"].get(k, {})
        if "mean" in v:
            out.append(f"| {k} | {f(v['mean'])} | {f(v['locomo'])} | {f(v['lme'])} | {f(v['hard'])} | {f(v['lme_acc_if_all_evidence'])} | "
                       f"{f(v['locomo_abstention'])} | {v['mean_tokens']:.0f} | {v.get('s_per_question')} | {f(v['no_answer_line'])} |")
    out += ["", "By type / category: `" + json.dumps({k: {"lme": v.get("lme_by_type"), "locomo": v.get("locomo_by_category")}
                                                      for k, v in R["arms"].items()}, default=str) + "`", "",
            "## Decisions (rules in README.md)", "```", json.dumps(R["decisions"], indent=1, default=str), "```", "",
            "## Post-check", "```", json.dumps(R["postcheck"], indent=1, default=str), "```", "",
            "## Paired (pooled LoCoMo cat. 1-4 + LME; exact McNemar, item bootstrap)", "```", json.dumps(R["paired"], indent=1, default=str), "```", "",
            f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_V3A.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


if __name__ == "__main__":
    log("environment", v0.environment())
    Cm.stage_setup()
    if "v2c_setup" not in STATE["errors"]:
        V.stage_select_prior()                 # v2's prior, re-fitted on dev exactly as v2/v2b/v2c did
        D["ce_name"], D["ce_n"] = None, 0
        R["settings"]["prior"] = V.V2["settings"].get("prior", {}).get("choice")
        stage_contexts()
        if "v3a_contexts" not in STATE["errors"]:
            stage_read()
    try:
        stage_stats()
    except Exception as e:
        STATE["errors"]["v3a_stats"] = f"{type(e).__name__}: {str(e)[:150]}"
    save_r()
    write_report()
    log(f"v3a done; errors: {STATE['errors'] or 'none'}")

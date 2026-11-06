"""DAMA v5 dev run. Private Kaggle notebook `s41r4j/dama-v5` (T4 x2). Follows the v4 dev run
(experiments/dama_v4_2026-11-02/README.md); packets built on CPU (build_packets.py), the GPU only reads and judges.

Questions: LoCoMo dev, DMBM run 1's 300 items. Reader Qwen3.5-9B, judge Qwen2.5-7B (as in runs 1 and v4).

Arms:
  dama_A0 x P4          prompt change alone (A0 x P3 and A0 x P1 are carried from v4 and run 1)
  dama_v5_s25 x P4      A0 lines + session summaries in 750 tokens (same ~3k budget)
  dama_v5_add x P4      all A0 lines + summaries in 1,000 tokens (~4k; the candidate: trimming A0 lines to make room
                        loses gold evidence, 91.2% -> 84.7% at 2,250 tokens, whatever the ranking; s50 dropped: 79.1%)
  summary_memory x P4   fairness control
  full_context x P3     ONLY the 39 questions whose v4 calls timed out, at concurrency 4; merged with v4's other 261
                        rows, this completes the raw-LLM baseline (goal 1)

P4 = P3 with three changes from the v4 error analysis: "how many" lists distinct occurrences once each; "would / likely"
questions never get "Not mentioned"; the decline rule is narrowed to "the question names one person but the memories
show the other person". It also tells the reader how to use session summaries.

Time: a hard limit checked inside every arm (v4 checked only between arms and ran 6.4 h instead of 3).
"""
from __future__ import annotations
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

SMOKE = False
HERE = Path(globals().get("__file__", "dama_v5.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v3b_2026-11-01/dama_paper_v3b.py")))
import dama_paper_v3b as B3      # engine (Srv, Cli), reading (Bm.read) and judging (Bm.judge), as in DMBM run 1 and v4
Bm, V, v0 = B3.Bm, B3.V, B3.v0
import numpy as np

B3.SMOKE = Bm.SMOKE = V.SMOKE = v0.SMOKE = SMOKE
v0.TIME_LIMIT_H = 1.0 if SMOKE else 2.5
JUDGE_RESERVE_H = 0.2 if SMOKE else 0.4
log, OUT, STATE = v0.log, v0.OUT, v0.STATE
MODEL, JUDGE_ID, PORT = B3.READER, B3.JUDGE_ID, 8011
R = {"settings": {"model": MODEL, "judge": JUDGE_ID}, "arms": {}, "scores": {}, "paired": {}, "notes": []}

LOCO_P4 = """You answer questions about a long conversation between two people, using only the memory provided.
Rules:
- Use only the MEMORY section. Do not use outside knowledge.
- Session summaries, if given, are overviews of whole sessions; memory lines are the exact words, each with its date and speaker. Use the summaries to see where to look and the lines for details. In a line, "I", "me" and "my" mean the speaker; "you" and "your" mean the other person.
- First write NOTES: list the lines and summaries that matter, each with its date and whom it is about. If the question asks for several things (which, what kinds, both), go through every date and list each matching item. For "how many", list each distinct occurrence with its date (the same event mentioned twice counts once), then count. If it needs date arithmetic, work it out step by step. Keep the notes under 150 words.
- Then write one final line that starts with "ANSWER:" and gives a short answer (at most 20 words).
- For time questions, give absolute dates (for example "7 May 2023"), using the session dates shown and any resolved dates in [= ...].
- If the question asks what is likely, or what someone would do, might be or would enjoy, never reply "Not mentioned": give your best guess from the related memories (for example "Likely yes, because ...").
- If the question names one person but the memories show that the other person did or said it, the final line is exactly: ANSWER: Not mentioned in the conversation. Do the same if nothing in the memory is about the question's topic.
- Otherwise answer from the closest memories, even when they are worded differently from the question."""
LOCO_P3 = """You answer questions about a long conversation between two people, using only the memory provided.
Rules:
- Use only the MEMORY section. Do not use outside knowledge.
- Each memory line shows its date and who is speaking. In a line, "I", "me" and "my" mean the speaker; "you" and "your" mean the other person.
- If there is a CHECK line, read it first: it tells whom the matching memories are about.
- First write NOTES: list the memory lines that matter for the question, each with its date and whom it is about. If the question asks for several things (which, what kinds, both, how many), go through every date and list each matching item. If it needs counting or date arithmetic, work it out step by step. Keep the notes under 150 words.
- Then write one final line that starts with "ANSWER:" and gives a short answer (at most 20 words).
- For time questions, give absolute dates (for example "7 May 2023"), using the session dates shown and any resolved dates in [= ...].
- If the question asks what is likely, or what someone would do or might be, give your best guess from the related memories (for example "Likely yes, because ...").
- If the memories show that it was someone else, not the person the question names, or no memory is about the question's topic, the final line is exactly: ANSWER: Not mentioned in the conversation.
- Otherwise answer from the closest memories, even when they are worded differently from the question."""      # v4's P3, verbatim
Bm.PROMPTS["P3"] = {"locomo": (LOCO_P3, 600), "lme": Bm.PROMPTS["P1"]["lme"]}
Bm.PROMPTS["P4"] = {"locomo": (LOCO_P4, 600), "lme": Bm.PROMPTS["P1"]["lme"]}
ARMS = [("dama_A0", "P4", 48), ("dama_v5_add", "P4", 48), ("dama_v5_s25", "P4", 48),
        ("summary_memory", "P4", 48), ("full_context_retry", "P3", 4)]
MAX_LEN = 40960


def save_r():
    (OUT / "results_v5.json").write_text(json.dumps(R, indent=1, default=str))


def time_left_h():
    return v0.TIME_LIMIT_H - JUDGE_RESERVE_H - B3.hours()


def carried(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/methods/{pattern}", recursive=True), key=len)
    return [r for r in map(json.loads, open(hits[0])) if r.get("bench") == "locomo"] if hits else []


def load():
    v4 = _exp_dir("dama_v4_2026-11-02/dama_v4.py") / "packets"
    v5 = _exp_dir("dama_v5_2026-11-03/dama_v5.py") / "packets"
    items = [json.loads(l) for l in open(v4 / "items.jsonl")]
    if SMOKE:
        items = items[:10] + [q for q in items if q["cat"] == 5][:3]
    keep = {q["qid"] for q in items}
    P = defaultdict(dict)
    for f in (v4 / "packets.jsonl", v5 / "packets.jsonl"):
        for p in map(json.loads, open(f)):
            if p["qid"] in keep:
                P[p["arm"]][p["qid"]] = p["context"]
    full = json.loads((v4 / "full_context.json").read_text())
    failed = {r["qid"] for r in carried("v4_full_context_P3.jsonl") if not (r.get("raw") or "").strip()}
    if SMOKE and not failed:
        failed = set(sorted(keep)[:3])
    P["full_context_retry"] = {q["qid"]: full[str(q["ci"])] for q in items if q["qid"] in failed}
    R["settings"].update(items=len(items), build=json.loads((v5 / "build_summary.json").read_text()),
                         full_context_retry=len(P["full_context_retry"]))
    log("v5 items:", len(items), "arms:", {k: len(v) for k, v in P.items()})
    return items, P


def acc(rs, f):
    xs = [bool(r["qwen_correct"]) for r in rs if f(r)]
    return (round(100 * float(np.mean(xs)), 1), len(xs)) if xs else (None, 0)


def score(rs):
    return {"locomo_1_4": acc(rs, lambda r: r["cat"] != 5), "multi_hop": acc(rs, lambda r: r["cat"] == 1),
            "temporal": acc(rs, lambda r: r["cat"] == 2), "open_domain": acc(rs, lambda r: r["cat"] == 3),
            "single_hop": acc(rs, lambda r: r["cat"] == 4), "adversarial": acc(rs, lambda r: r["cat"] == 5),
            "all_300": acc(rs, lambda r: True), "reader_input_tokens": round(float(np.mean([r["input_tokens"] for r in rs]))),
            "abstained_on_1_4": sum("not mentioned" in (r["answer"] or "").lower() for r in rs if r["cat"] != 5),
            "unparsed": sum(not r["parsed"] for r in rs), "empty": sum(not (r.get("raw") or "").strip() for r in rs)}


def paired(a, b, f=lambda r: r["cat"] != 5, n_boot=10000):
    A = {r["qid"]: bool(r["qwen_correct"]) for r in a if f(r)}
    B = {r["qid"]: bool(r["qwen_correct"]) for r in b if f(r)}
    q = sorted(set(A) & set(B))
    if not q:
        return None
    d = np.array([B[k] - A[k] for k in q], dtype=float)
    boots = d[np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"n": len(q), "diff_pts": round(100 * d.mean(), 1), "ci95": [round(100 * np.percentile(boots, 2.5), 1),
            round(100 * np.percentile(boots, 97.5), 1)], "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def write_report():
    f = lambda x: "—" if x[0] is None else f"{x[0]:.1f} ({x[1]})"
    out = [f"# DAMA v5 dev run ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           f"Reader {MODEL}, judge {JUDGE_ID}; LoCoMo dev ({R['settings'].get('items')} questions). Accuracy % (n).", "",
           "| arm | LoCoMo 1-4 | multi-hop | temporal | open-domain | single-hop | adversarial | all | reader input tokens | abstained 1-4 | empty |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(R["scores"].items(), key=lambda kv: -(kv[1]["locomo_1_4"][0] or 0)):
        out.append(f"| {k} | {f(s['locomo_1_4'])} | {f(s['multi_hop'])} | {f(s['temporal'])} | {f(s['open_domain'])} | "
                   f"{f(s['single_hop'])} | {f(s['adversarial'])} | {f(s['all_300'])} | {s['reader_input_tokens']} | "
                   f"{s['abstained_on_1_4']} | {s['empty']} |")
    out += ["", "## Paired differences (same questions; 95% bootstrap CI)", "```", json.dumps(R["paired"], indent=1), "```", "",
            f"- settings: `{json.dumps(R['settings'])}`", f"- arms: `{json.dumps(R['arms'])}`",
            f"- errors: `{json.dumps(STATE['errors'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_V5.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


def read_arm(arm, prompt, conc, items, P):
    jobs = [(arm, q, {"context": P[arm][q["qid"]], "read_ms": 0.0}) for q in items if q["qid"] in P[arm]]
    cli, rows, t = B3.Cli(MODEL, PORT, conc=conc), [], time.time()
    step = max(conc * 2, 8)
    for k in range(0, len(jobs), step):
        if time_left_h() <= 0:
            R["notes"].append(f"{arm}|{prompt} stopped at {len(rows)} of {len(jobs)}: time limit"); break
        rs, _ = Bm.read(cli, jobs[k:k + step], prompt, f"{B3.short(MODEL)}|v5_{arm}")
        rows += rs
    R["arms"][f"{arm}|{prompt}"] = {"n": len(rows), "of": len(jobs), "minutes": round((time.time() - t) / 60, 1)}
    return rows


def main():
    log("environment", v0.environment())
    items, P = load()
    ROWS = {"dama_A0|P1 (run 1)": carried("dmbm1_dama_A0.jsonl"), "dama_A0|P3 (v4)": carried("v4_dama_A0_P3.jsonl"),
            "summary_memory|P3 (v4)": carried("v4_summary_memory_P3.jsonl")}
    v4_full = carried("v4_full_context_P3.jsonl")
    keep = {q["qid"] for q in items}
    ROWS = {k: [r for r in v if r["qid"] in keep] for k, v in ROWS.items() if v}
    if Bm.install_vllm(B3.VLLM):
        srv = B3.Srv(MODEL, "0,1", PORT, max_len=MAX_LEN, util=0.92, max_seqs=64)
        if srv.start():
            try:
                for arm, prompt, conc in ARMS:
                    if time_left_h() <= 0:
                        R["notes"].append(f"{arm}|{prompt} skipped: time limit"); continue
                    rows = read_arm(arm, prompt, conc, items, P)
                    if rows:
                        ROWS[f"{arm}|{prompt}"] = rows
                        V.write_rows(f"v5_{arm}_{prompt}", rows)
                    save_r()
            finally:
                srv.stop()
    j = B3.Srv(JUDGE_ID, "0,1", PORT)
    if j.start():
        try:
            jc = B3.Cli(JUDGE_ID, PORT, conc=64)
            for k, rs in ROWS.items():
                if rs and "qwen_correct" not in rs[0]:
                    Bm.judge(jc, rs)
                    V.write_rows(f"v5_{k.split(' ')[0].replace('|', '_')}", rs)
        finally:
            j.stop()
    retry = ROWS.pop("full_context_retry|P3", None)
    if v4_full and retry and "qwen_correct" in retry[0]:
        fixed = {r["qid"]: r for r in retry if (r.get("raw") or "").strip()}
        merged = [fixed.get(r["qid"], r) for r in v4_full if r["qid"] in keep]
        ROWS["full_context|P3 (v4 + retry)"] = merged
        R["notes"].append(f"full context: {len(fixed)} of {len(retry)} timed-out v4 questions answered on retry")
    for k, rs in ROWS.items():
        if rs and "qwen_correct" in rs[0]:
            R["scores"][k] = score(rs)
    ref = ROWS.get("dama_A0|P1 (run 1)")
    full = ROWS.get("full_context|P3 (v4 + retry)")
    for k, rs in ROWS.items():
        if not rs or "qwen_correct" not in rs[0]:
            continue
        for name, base in (("vs dama_A0|P1", ref), ("vs full_context", full)):
            if base and rs is not base and base is not None:
                R["paired"][f"{k} {name}"] = {"locomo_1_4": paired(base, rs), "adversarial": paired(base, rs, lambda r: r["cat"] == 5)}
    save_r()
    write_report()
    log(f"v5 done; errors: {STATE['errors'] or 'none'}")


if __name__ == "__main__":
    main()

"""DAMA v4 dev run (2026-11-02). Private Kaggle notebook `s41r4j/dama-v4` (T4 x2). Plan: docs/research/DAMA_v4_plan_2026-11-02.md.

Reads packets built on CPU beforehand (build_packets.py; no setup stage on the GPU), with Qwen3.5-9B, then judges with
Qwen2.5-7B, exactly as DMBM run 1 (same reader model, judge, items and A0 packets). Questions: LoCoMo dev, 300.

Arms (packet x reader prompt):
  dama_A0 x P3          prompt effect alone (A0 x P1 is run 1's judged result, carried)
  dama_v4a x P3         + wrong-person CHECK line and list coverage
  dama_v4b x P3         + KEY LINES block                         <- the v4 candidate
  oracle x P3           ceiling under P3
  rag_hybrid_rerank x P3, summary_memory x P3   does P3 help other systems too (fairness control)
  full_context x P3     the raw LLM reading the whole conversation (goal 1)
  dama_v4b x P1, full_context x P1   if time is left
"""
from __future__ import annotations
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

SMOKE = False
HERE = Path(globals().get("__file__", "dama_v4.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v3b_2026-11-01/dama_paper_v3b.py")))
import dama_paper_v3b as B3      # engine (Srv, Cli), reading (Bm.read) and judging (Bm.judge), as in DMBM run 1
Bm, V, v0 = B3.Bm, B3.V, B3.v0
import numpy as np

B3.SMOKE = Bm.SMOKE = V.SMOKE = v0.SMOKE = SMOKE
v0.TIME_LIMIT_H = 1.0 if SMOKE else 3.0           # hard stop for the whole run (the week has ~4.6 GPU hours left)
log, OUT, STATE = v0.log, v0.OUT, v0.STATE
MODEL, JUDGE_ID, PORT = B3.READER, B3.JUDGE_ID, 8011
R = {"settings": {"model": MODEL, "judge": JUDGE_ID}, "arms": {}, "scores": {}, "paired": {}, "notes": []}

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
- Otherwise answer from the closest memories, even when they are worded differently from the question."""
Bm.PROMPTS["P3"] = {"locomo": (LOCO_P3, 600), "lme": Bm.PROMPTS["P1"]["lme"]}
ARMS = [("dama_A0", "P3"), ("dama_v4a", "P3"), ("dama_v4b", "P3"), ("oracle", "P3"), ("rag_hybrid_rerank", "P3"),
        ("summary_memory", "P3"), ("full_context", "P3"), ("dama_v4b", "P1"), ("full_context", "P1")]
MAX_LEN = 40960                                    # the longest dev conversation is ~29k tokens rendered


def save_r():
    (OUT / "results_v4.json").write_text(json.dumps(R, indent=1, default=str))


def load():
    d = _exp_dir("dama_v4_2026-11-02/dama_v4.py") / "packets"
    items = [json.loads(l) for l in open(d / "items.jsonl")]
    if SMOKE:
        items = items[:10] + [q for q in items if q["cat"] == 5][:3]
    keep = {q["qid"] for q in items}
    P = defaultdict(dict)
    for p in map(json.loads, open(d / "packets.jsonl")):
        if p["qid"] in keep:
            P[p["arm"]][p["qid"]] = p["context"]
    full = json.loads((d / "full_context.json").read_text())
    P["full_context"] = {q["qid"]: full[str(q["ci"])] for q in items}
    R["settings"]["items"] = len(items)
    R["settings"]["build"] = json.loads((d / "build_summary.json").read_text())
    log("v4 items:", len(items), "arms:", {k: len(v) for k, v in P.items()})
    return items, P


def carried_a0():
    """Run 1's judged A0 x P1 rows (the baseline), from the attached dama-dmbm1 output."""
    hits = sorted(glob.glob("/kaggle/input/**/methods/dmbm1_dama_A0.jsonl", recursive=True), key=len)
    if not hits:
        R["notes"].append("run 1 A0 rows not attached: no carried baseline"); return []
    return [r for r in map(json.loads, open(hits[0])) if r["bench"] == "locomo"]


def acc(rs, f):
    xs = [bool(r["qwen_correct"]) for r in rs if f(r)]
    return (round(100 * float(np.mean(xs)), 1), len(xs)) if xs else (None, 0)


def score(rs):
    return {"locomo_1_4": acc(rs, lambda r: r["cat"] != 5), "multi_hop": acc(rs, lambda r: r["cat"] == 1),
            "temporal": acc(rs, lambda r: r["cat"] == 2), "open_domain": acc(rs, lambda r: r["cat"] == 3),
            "single_hop": acc(rs, lambda r: r["cat"] == 4), "adversarial": acc(rs, lambda r: r["cat"] == 5),
            "all_300": acc(rs, lambda r: True),
            "reader_input_tokens": round(float(np.mean([r["input_tokens"] for r in rs]))),
            "abstained_on_1_4": sum("not mentioned" in (r["answer"] or "").lower() for r in rs if r["cat"] != 5),
            "unparsed": sum(not r["parsed"] for r in rs)}


def paired(a, b, f=lambda r: r["cat"] != 5, n_boot=10000):
    """b minus a on the same questions: accuracy difference, its 95% bootstrap CI, and wins / losses."""
    A = {r["qid"]: bool(r["qwen_correct"]) for r in a if f(r)}
    B = {r["qid"]: bool(r["qwen_correct"]) for r in b if f(r)}
    q = sorted(set(A) & set(B))
    if not q:
        return None
    d = np.array([B[k] - A[k] for k in q], dtype=float)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"n": len(q), "diff_pts": round(100 * d.mean(), 1), "ci95": [round(100 * np.percentile(boots, 2.5), 1),
            round(100 * np.percentile(boots, 97.5), 1)], "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def write_report(rows):
    f = lambda x: "—" if x[0] is None else f"{x[0]:.1f} ({x[1]})"
    out = [f"# DAMA v4 dev run ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           f"Reader {MODEL}, judge {JUDGE_ID}; LoCoMo dev ({R['settings'].get('items')} questions; DMBM run 1's items). "
           "Accuracy % (n). LoCoMo 1-4 is the headline; adversarial = should answer 'Not mentioned'.", "",
           "| arm | prompt | LoCoMo 1-4 | multi-hop | temporal | open-domain | single-hop | adversarial | all 300 | reader input tokens | abstained on 1-4 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(R["scores"].items(), key=lambda kv: -(kv[1]["locomo_1_4"][0] or 0)):
        arm, p = k.rsplit("|", 1)
        out.append(f"| {arm} | {p} | {f(s['locomo_1_4'])} | {f(s['multi_hop'])} | {f(s['temporal'])} | {f(s['open_domain'])} | "
                   f"{f(s['single_hop'])} | {f(s['adversarial'])} | {f(s['all_300'])} | {s['reader_input_tokens']} | {s['abstained_on_1_4']} |")
    out += ["", "## Paired differences (same questions; 95% bootstrap CI)", "```", json.dumps(R["paired"], indent=1), "```", "",
            f"- build: `{json.dumps(R['settings'].get('build'))}`", f"- arms: `{json.dumps(R['arms'])}`",
            f"- errors: `{json.dumps(STATE['errors'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_V4.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


def main():
    log("environment", v0.environment())
    items, P = load()
    ROWS = {}
    base = carried_a0()
    if base:
        ROWS["dama_A0|P1"] = [r for r in base if r["qid"] in {q["qid"] for q in items}]
    if Bm.install_vllm(B3.VLLM):
        srv = B3.Srv(MODEL, "0,1", PORT, max_len=MAX_LEN, util=0.92, max_seqs=64)
        if srv.start():
            try:
                cli = B3.Cli(MODEL, PORT, conc=48)
                for arm, prompt in ARMS:
                    if B3.hours() > v0.TIME_LIMIT_H - (0.2 if SMOKE else 0.6):     # keep time for the judge
                        R["notes"].append(f"{arm}|{prompt} skipped: time"); continue
                    jobs = [(arm, q, {"context": P[arm][q["qid"]], "read_ms": 0.0}) for q in items if q["qid"] in P[arm]]
                    t = time.time()
                    # run 1 of v4: the limit was checked only between arms and full context ran 280 min past it.
                    # Now reading goes in chunks and stops at the limit (the arm is then reported as partial).
                    rows, chunk = [], 16 if arm == "full_context" else 96
                    for k in range(0, len(jobs), chunk):
                        if B3.hours() > v0.TIME_LIMIT_H - (0.2 if SMOKE else 0.6):
                            R["notes"].append(f"{arm}|{prompt} stopped at {len(rows)} of {len(jobs)}: time"); break
                        rs, _ = Bm.read(cli, jobs[k:k + chunk], prompt, f"{B3.short(MODEL)}|v4_{arm}")
                        rows += rs
                    sec = (time.time() - t) / max(1, len(rows))
                    if not rows:
                        continue
                    ROWS[f"{arm}|{prompt}"] = rows
                    R["arms"][f"{arm}|{prompt}"] = {"n": len(rows), "s_per_question": round(sec, 3), "minutes": round((time.time() - t) / 60, 1)}
                    V.write_rows(f"v4_{arm}_{prompt}", rows)
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
                    V.write_rows(f"v4_{k.replace('|', '_')}", rs)
        finally:
            j.stop()
    for k, rs in ROWS.items():
        if rs and "qwen_correct" in rs[0]:
            R["scores"][k] = score(rs)
    ref = ROWS.get("dama_A0|P1")
    for k, rs in ROWS.items():
        if ref and k != "dama_A0|P1" and rs and "qwen_correct" in rs[0]:
            R["paired"][f"{k} vs dama_A0|P1"] = {"locomo_1_4": paired(ref, rs), "adversarial": paired(ref, rs, lambda r: r["cat"] == 5)}
    for a, b in (("full_context|P3", "dama_v4b|P3"), ("full_context|P1", "dama_A0|P1"), ("dama_A0|P3", "dama_v4b|P3")):
        if a in R["scores"] and b in R["scores"]:
            R["paired"][f"{b} vs {a}"] = {"locomo_1_4": paired(ROWS[a], ROWS[b]), "adversarial": paired(ROWS[a], ROWS[b], lambda r: r["cat"] == 5)}
    save_r()
    write_report(ROWS)
    log(f"v4 done; errors: {STATE['errors'] or 'none'}")


if __name__ == "__main__":
    main()

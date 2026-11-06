"""DAMA v3b probe P0 (2026-11-01): how much is left in today's packet? CPU only, no GPU. Private Kaggle notebook
`s41r4j/dama-ds-probe`. Must be started from the website (Save Version), because pushed versions cannot read secrets.

DeepSeek-flash (thinking off) reads the exact A0 packets of v3a (v2c C3_m0, 518 dev questions) with the same prompt P1
the Qwen3.5-9B reader got. DeepSeek then judges both its own answers and the 9B's A0 answers, so the comparison is
same-packet, same-judge. The Qwen2.5-7B judge re-labels the DeepSeek answers in the next GPU run.

Budget: hard cap USD 0.50 for this notebook (the whole project cap is USD 1.00 until the user raises it). The client
checks the spend before every chunk of 200 calls, so the worst case is the cap plus one chunk (~USD 0.11 off-peak).
Order: reads first, then judging the DeepSeek answers, then judging A0's answers; anything over the cap is skipped.

Pre-registered reading (docs/research/DAMA_v3_new_arch_research_2026-10-31.md, P0): headroom H = DeepSeek mean - A0
mean under the same judge. H >= 6 points: the packet holds answers the 9B cannot extract (move computation out of the
reader: V3-L). H <= 2: today's architecture is at the benchmark's practical ceiling for this packet.
"""
from __future__ import annotations
import glob
import json
import sys
import time
from pathlib import Path

CAP_USD = 0.50
HERE = Path(globals().get("__file__", "dama_ds_probe.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v2b_2026-10-29/dama_paper_v2b.py")))
import dama_paper_v2b as Bm      # P1 prompt, answer parsing; through it v1's DeepSeek client and judges, v0's cache
V, L, v0, v1 = Bm.V, Bm.L, Bm.v0, Bm.v1
import numpy as np

v1.DS_CAP_USD = CAP_USD
log, OUT = v0.log, v0.OUT
P = {"cap_usd": CAP_USD, "notes": []}


def a0_rows():
    hits = sorted(glob.glob("/kaggle/input/**/methods/dev_A0.jsonl", recursive=True), key=len)
    if not hits:
        raise SystemExit("v3a's dev_A0.jsonl is not attached (kernel source s41r4j/dama-paper-v3a)")
    return [json.loads(l) for l in open(hits[0])]


def lme_dates():
    path = Path("/tmp/longmemeval_s_cleaned.json")
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(L.LME_URL, path)
    return {it["question_id"]: it.get("question_date", "") for it in json.loads(path.read_text())}


def ds_judge(ds, rows):
    """DeepSeek labels in r['ds_correct'] with the same rules as the Qwen judge (v2b's ds_judge_all)."""
    loco = [r for r in rows if r["bench"] == "locomo"]
    if loco:
        v1.ds_judge_rows(loco)
    lme = [r for r in rows if r["bench"] == "lme"]
    outs = ds.many([[{"role": "user", "content": v1.lme_judge_prompt({"question_id": r["qid"], "question_type": r["type"],
                                                                      "question": r["question"], "answer": r["gold"]}, r["answer"])}]
                    for r in lme], max_tokens=6)
    for r, o in zip(lme, outs):
        r["ds_correct"] = None if o is None else ("yes" in o.lower())


def acc(rows, key, bench=None):
    xs = [r[key] for r in rows if (r["bench"] != "locomo" or r["cat"] != 5) and (bench is None or r["bench"] == bench) and r.get(key) is not None]
    return (float(np.mean(xs)), len(xs)) if xs else (None, 0)


if __name__ == "__main__":
    ds = v1.ds_client()
    if ds is None:
        log("no DeepSeek key (start this notebook from the website: Save Version > Save & Run All)"); sys.exit(1)
    base = a0_rows()
    dates = lme_dates()
    rows = []
    msgs = []
    for r in base:
        q = {"bench": r["bench"], "question": r["question"], "date": dates.get(r["qid"], "")}
        msgs.append(Bm.messages(q, r["context"], "P1"))
    t = time.time()
    outs = ds.many(msgs, max_tokens=400)
    for r, o in zip(base, outs):
        ans, ok = Bm.answer_of(o or "", "P1")
        rows.append({**{k: r[k] for k in ("qid", "bench", "question", "gold", "context", "evidence_all") if k in r},
                     **({"type": r["type"], "abs": r.get("abs")} if r["bench"] == "lme" else {"ci": r["ci"], "cat": r["cat"]}),
                     "method": "DS_A0", "reader": "deepseek-flash|P1", "prompt": "P1", "raw": o, "answer": ans if o is not None else "",
                     "parsed": ok, "read_failed": o is None})
    P["read_minutes"] = round((time.time() - t) / 60, 1)
    P["after_reads"] = ds.summary()
    log("DeepSeek reads:", P["after_reads"])
    ds_judge(ds, rows)
    a0 = [dict(r) for r in base]
    ds_judge(ds, a0)
    P["final"] = ds.summary()
    (OUT / "methods").mkdir(parents=True, exist_ok=True)
    V.write_rows("dev_DS_A0", rows)
    V.write_rows("dev_A0_dsjudged", a0)
    res = {}
    for name, rs in (("DS_A0", rows), ("A0_qwen9b", a0)):
        res[name] = {b or "all": acc(rs, "ds_correct", b) for b in (None, "locomo", "lme")}
    mean = lambda name: np.mean([res[name]["locomo"][0], res[name]["lme"][0]]) if None not in (res[name]["locomo"][0], res[name]["lme"][0]) else None
    for name in res:
        res[name]["mean"] = mean(name)
    if res["DS_A0"]["mean"] is not None and res["A0_qwen9b"]["mean"] is not None:
        h = res["DS_A0"]["mean"] - res["A0_qwen9b"]["mean"]
        res["headroom_points"] = round(100 * h, 1)
        res["reading"] = ("packet holds answers the 9B misses: move computation out of the reader (V3-L)" if h >= 0.06 else
                          "near the practical ceiling for this packet" if h <= 0.02 else "in between")
    P["results"] = res
    (OUT / "results_ds_probe.json").write_text(json.dumps(P, indent=1, default=str))
    print("=====BEGIN REPORT=====\n" + json.dumps(P, indent=1, default=str) + "\n=====END REPORT=====", flush=True)

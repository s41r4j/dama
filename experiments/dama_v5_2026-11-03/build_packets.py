"""DAMA v5, step 1 (CPU, run on ssa): raw lines + session summaries in one packet.

Why (v4 dev run, LoCoMo dev, reader Qwen3.5-9B, prompt P3): summary memory (half session summaries, half cosine turns)
and DAMA A0 (raw turns only) are right on different questions. Either one right: 88.0 on categories 1-4 and 78.2 on
multi-hop, against A0's 80.8 / 63.6, while two near-identical DAMA packets reach only 86.8 together (judge noise).
Summaries help most on lists and counts spread over sessions; DAMA's turn selection beats plain cosine turns
(80.8 vs 78.0). v5 keeps DAMA's turns and adds the most related session summaries, inside the same budget.

Summaries: run 1's summary_memory packets hold every session's summary (one Qwen3.5-9B call per session, 220 tokens
max; all 99 dev sessions recovered), so no new LLM calls are needed here. In DAMA they are written once per session,
after it ends (write path, off the critical path); the read path stays generative-free.

Arms (all within ~3,000 tokens unless noted):
- dama_v5_s25: top summaries in 750 tokens + A0's most related lines (bge-small) in the rest.
- dama_v5_s50: top summaries in 1,500 tokens + A0's lines in the rest.
- dama_v5_add: all of A0 + top summaries in 1,000 tokens (~4,000 tokens; still ~6x below full context).
Packet: "SESSION SUMMARIES" block (by date), then "MEMORY LINES" (by date).

Usage: python build_packets.py IN_DIR OUT_DIR   (IN_DIR: run 1's dmbm1_dama_A0 / dmbm1_summary_memory rows, locomo10.json)
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "experiments/v2_2026-10-28")]
import v2lib as L                                        # noqa: E402  (same turn ids, dates and time notes as A0)
from dmbm.systems.base import ntok                       # noqa: E402

BUDGET = 3000
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
ARMS = {"dama_v5_s25": (750, BUDGET - 750), "dama_v5_s50": (1500, BUDGET - 1500), "dama_v5_add": (1000, None)}
_SUM = re.compile(r"^\[m:\d+\] \((.*?)\) \[session summary\] (.*)$", re.M)


def pick(order, cost, budget):
    out, used = [], 0
    for i in order:
        if used + cost[i] <= budget:
            out.append(i); used += cost[i]
    return out


def main(in_dir, out_dir):
    ind, out = Path(in_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    convs, qs = L.load_locomo(ind / "locomo10.json")
    Q = {q["qid"]: q for q in qs}
    a0 = {r["qid"]: r["context"] for r in map(json.loads, open(ind / "dmbm1_dama_A0.jsonl")) if r["bench"] == "locomo"}
    sums = {}                                            # sample id -> {session date: summary}, first-seen order
    for r in map(json.loads, open(ind / "dmbm1_summary_memory.jsonl")):
        if r["bench"] == "locomo":
            for m in _SUM.finditer(r["context"]):
                sums.setdefault(r["qid"].split(":")[0], {})[m.group(1)] = m.group(2)
    items = [Q[k] for k in a0]
    if any(q["split"] != "dev" for q in items):
        raise ValueError("non-dev item")
    by_ci = {c["ci"]: c for c in convs}

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    enc = lambda xs: bge.encode(xs, normalize_embeddings=True, batch_size=64)
    tvec, svec, slist = {}, {}, {}
    for ci in {q["ci"] for q in items}:
        c = by_ci[ci]
        tvec[ci] = dict(zip([t["dia"] for t in c["turns"]], enc([t["text"] for t in c["turns"]])))
        dates = [t["date_raw"] for t in c["turns"]]                       # session order of the summaries
        first = {d: dates.index(d) for d in dict.fromkeys(dates)}
        slist[ci] = sorted(sums[c["sid"]].items(), key=lambda kv: first.get(kv[0], 10**9))
        svec[ci] = enc([s for _, s in slist[ci]])
    qvec = dict(zip([q["qid"] for q in items], enc([BGE_PREFIX + q["question"] for q in items])))
    print("embedded", sum(len(v) for v in tvec.values()), "turns,", sum(len(v) for v in slist.values()), "summaries", flush=True)

    rows, stats = [], {a: [0, 0] for a in ARMS}
    for q in items:
        ci, qv = q["ci"], qvec[q["qid"]]
        pos = {t["dia"]: t for t in by_ci[ci]["turns"]}
        ids = [d for d in re.findall(r"^\[m:(D\d+:\d+)\]", a0[q["qid"]], flags=re.M) if d in pos]
        lines = {d: L.render(pos[d], True) for d in ids}
        lcost = {d: ntok(lines[d]) + 1 for d in ids}
        s_text = [f"[s:{k + 1}] ({L.fmt_date(d)}) {s}" for k, (d, s) in enumerate(slist[ci])]
        scost = [ntok(x) + 1 for x in s_text]
        s_order = sorted(range(len(s_text)), key=lambda i: -float(svec[ci][i] @ qv))
        l_order = sorted(ids, key=lambda d: -float(tvec[ci][d] @ qv))
        for arm, (s_budget, l_budget) in ARMS.items():
            si = sorted(pick(s_order, scost, s_budget))
            keep = set(ids) if l_budget is None else set(pick(l_order, lcost, l_budget))
            ctx = ("SESSION SUMMARIES (overviews of whole sessions, most related to the question; check details in the "
                   "memory lines):\n" + "\n".join(s_text[i] for i in si) + "\n\nMEMORY LINES (exact words, by date):\n"
                   + "\n".join(lines[d] for d in ids if d in keep))
            rows.append({"qid": q["qid"], "arm": arm, "context": ctx})
            stats[arm][0] += ntok(ctx); stats[arm][1] += len(si)
    with open(out / "packets.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(items)
    summary = {"items": n, "mean_tokens": {a: round(v[0] / n) for a, v in stats.items()},
               "mean_summaries": {a: round(v[1] / n, 1) for a, v in stats.items()},
               "mean_tokens_a0": round(sum(ntok(a0[q["qid"]]) for q in items) / n),
               "summaries_per_conv": {str(ci): len(v) for ci, v in slist.items()}}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])

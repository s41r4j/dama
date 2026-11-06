"""DAMA v4, step 1 (CPU, run on ssa): every packet the GPU run reads, built ahead of time so the GPU only reads and judges.

Items: DMBM run 1's LoCoMo dev items (conversations 0-3: 250 in categories 1-4 + 50 adversarial). LME-lite is left out:
its 18 questions cannot rank systems, and its question dates are not in run 1's rows.

Arms (packet = what the reader sees):
- dama_A0: run 1's A0 packets, unchanged.
- dama_v4a: A0 + checks.py (wrong-person CHECK line; list-question coverage, same token size as A0 or smaller).
- dama_v4b: v4a with its 6 lines most related to the question (bge-small cosine) moved into a KEY LINES block at the top;
  the rest follow by date. No line is repeated, so the size is unchanged.
- oracle, rag_hybrid_rerank, summary_memory: run 1's packets (the reader-prompt control: does P3 help everyone?).
- full_context: the whole conversation, every turn rendered as in A0 (the raw-LLM baseline, goal 1).

Usage: python build_packets.py RUN1_METHODS_DIR LOCOMO_JSON OUT_DIR
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "experiments/v2_2026-10-28")]
import v2lib as L                                        # noqa: E402  (same turn ids, dates and time notes as A0)
from dama_agent.checks import History, check             # noqa: E402
from dmbm.systems.base import ntok                       # noqa: E402

K_KEY = 6
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
CARRY = {"dama_A0": "dmbm1_dama_A0", "oracle": "dmbm1_oracle", "rag_hybrid_rerank": "dmbm1_rag_hybrid_rerank",
         "summary_memory": "dmbm1_summary_memory"}


def main(methods_dir, locomo_json, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    convs, qs = L.load_locomo(locomo_json)
    Q = {q["qid"]: q for q in qs}
    rows = {arm: {r["qid"]: r for r in map(json.loads, open(Path(methods_dir) / f"{f}.jsonl")) if r["bench"] == "locomo"}
            for arm, f in CARRY.items()}
    items = [Q[qid] for qid in rows["dama_A0"]]
    if any(q["split"] != "dev" for q in items):
        raise ValueError("non-dev LoCoMo item")
    by_ci = {c["ci"]: c for c in convs}
    hist = {}
    for ci in {q["ci"] for q in items}:
        turns = by_ci[ci]["turns"]
        hist[ci] = History(turns, speakers=[s for s, _ in Counter(t["speaker"] for t in turns).most_common(2)])

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(max(1, torch.get_num_threads()))
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    tvec = {ci: dict(zip([t["dia"] for t in h.turns], bge.encode([t["text"] for t in h.turns], normalize_embeddings=True,
                                                                  batch_size=64))) for ci, h in hist.items()}
    qvec = dict(zip([q["qid"] for q in items], bge.encode([BGE_PREFIX + q["question"] for q in items], normalize_embeddings=True,
                                                           batch_size=64)))
    print("embedded", sum(len(v) for v in tvec.values()), "turns", flush=True)

    packets, stats = [], Counter()
    for q in items:
        h, a0 = hist[q["ci"]], rows["dama_A0"][q["qid"]]["context"]
        pos = {t["dia"]: t for t in h.turns}
        for arm in CARRY:
            if q["qid"] in rows[arm]:
                packets.append({"qid": q["qid"], "arm": arm, "context": rows[arm][q["qid"]]["context"]})
        ids = re.findall(r"^\[m:(D\d+:\d+)\]", a0, flags=re.M)
        res = check(q["question"], h, ids, ntok=ntok)
        head = res.header + "\n" if res.header else ""
        v4a = head + L.render_many([pos[d] for d in res.ids], True)
        if res.ids:
            sim = {d: float(tvec[q["ci"]][d] @ qvec[q["qid"]]) for d in res.ids}
            key = set(sorted(res.ids, key=lambda d: -sim[d])[:K_KEY])
        else:
            key = set()
        rest = [d for d in res.ids if d not in key]
        v4b = (head + "KEY LINES (the memories most related to the question):\n"
               + L.render_many([pos[d] for d in res.ids if d in key], True)
               + "\n\nOTHER LINES (by date):\n" + L.render_many([pos[d] for d in rest], True)) if key else v4a
        info = {"attribution": res.attribution, "list_q": res.list_q, "added": res.added, "dropped": res.dropped}
        packets.append({"qid": q["qid"], "arm": "dama_v4a", "context": v4a, "info": info})
        packets.append({"qid": q["qid"], "arm": "dama_v4b", "context": v4b, "info": info})
        stats["items"] += 1; stats["attribution"] += res.attribution is not None; stats["list_q"] += res.list_q
        stats["added"] += len(res.added); stats["dropped"] += len(res.dropped)
        stats["tok_a0"] += ntok(a0); stats["tok_v4a"] += ntok(v4a); stats["tok_v4b"] += ntok(v4b)

    full = {str(ci): L.render_many(hist[ci].turns, True) for ci in hist}
    with open(out / "packets.jsonl", "w") as f:
        for p in packets:
            f.write(json.dumps(p) + "\n")
    with open(out / "items.jsonl", "w") as f:
        for q in items:
            f.write(json.dumps({k: q[k] for k in ("qid", "ci", "split", "cat", "question", "gold", "evidence", "bench")}) + "\n")
    (out / "full_context.json").write_text(json.dumps(full))
    n = max(1, stats["items"])
    summary = {"items": stats["items"], "attribution_fires": stats["attribution"], "list_questions": stats["list_q"],
               "lines_added": stats["added"], "lines_dropped": stats["dropped"],
               "mean_tokens": {k: round(stats[f"tok_{k}"] / n) for k in ("a0", "v4a", "v4b")},
               "full_context_tokens": {ci: ntok(t) for ci, t in full.items()},
               "by_cat": dict(Counter(q["cat"] for q in items))}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])

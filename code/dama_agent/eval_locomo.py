"""Level 2 evaluation of the Memory Agent on held-out LoCoMo conversations, measured on the 4 pillars.

Each conversation is replayed turn by turn through the write path (observe), then every question goes through the
read path (packet):
  1 speed           read-path p50/p95 (wall clock, need gate included) per stage; write-path latency
  2 accuracy        share of questions whose packet holds every gold evidence turn, gold-turn recall (a proxy:
                    answer accuracy is Level 3, dama_agent.answer, run over the files saved here)
  3 context saving  packet tokens vs. the whole conversation history
  4 multi-agent     N agents (threads) sharing one memory send requests at the same moment: latency and throughput,
                    for distinct questions and for a team asking the same few questions (dedup)
Saved for Level 3 and for checking the write path:
  packets.jsonl   per question: packet, gold evidence turns (oracle context), answer, metrics
  histories.json  per conversation: the full rendered history (full-context baseline)
  writes.jsonl    every UPDATE / CONFLICT / DELETE / NOOP decision with the old fact and the new turn

    python -m dama_agent.eval_locomo --bundle DIR --locomo locomo10.json --out results/
    python -m dama_agent.eval_locomo --fake --locomo locomo10.json   # plumbing check, no models
Default conversations are LoCoMo 7, 8, 9: the Level-1 test split, never seen in training or calibration.
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .agent import AgentConfig, MemoryAgent, render
from .store import Store


def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, q)), 1) if xs else None


def concurrency(agent, scope, questions, ns=(1, 4, 8, 16), rounds=8) -> dict:
    """Pillar 4: N agents send a packet request at the same moment (a barrier), `rounds` times per N.
    distinct: every agent asks its own question. team: the N agents ask only 2 different questions (dedup applies)."""
    out = {}
    for scenario in ("distinct", "team"):
        out[scenario] = {}
        for n in ns:
            lat, wall = [], []
            for r in range(rounds):
                if scenario == "distinct":
                    qs = [questions[(r * n + i) % len(questions)] for i in range(n)]
                else:
                    qs = [questions[(r * 2 + i % 2) % len(questions)] for i in range(n)]
                agent._cache.clear()
                barrier, res = threading.Barrier(n), [None] * n

                def worker(i):
                    barrier.wait()
                    res[i] = agent.packet(scope, qs[i])["ms"]

                t0 = time.perf_counter()
                threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall.append(time.perf_counter() - t0)
                lat += res
            out[scenario][str(n)] = {"p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "late_share": float(np.mean([x > agent.cfg.deadline_ms for x in lat])),
                                     "requests_per_s": round(n * rounds / sum(wall), 1)}
    return out


def run(tier1, locomo_path, convs=(7, 8, 9), out_dir=None, config=None, max_questions=None, concurrency_test=True, log=print) -> dict:
    from dama_train.data.locomo import load_locomo
    cfg = config or AgentConfig()
    dialogues, questions = load_locomo(locomo_path)
    by_did = defaultdict(list)
    for q in questions:
        by_did[q.did].append(q)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    packets = (out / "packets.jsonl").open("w") if out else None
    writes_f = (out / "writes.jsonl").open("w") if out else None
    rows, writes, ops, histories, conc = [], [], defaultdict(int), {}, {}
    for ci in convs:
        d = dialogues[ci]
        agent = MemoryAgent(tier1, Store(), cfg)
        scope = d.did
        by_tid = {t.tid: t for t in d.turns}
        t_conv = time.perf_counter()
        for t in d.turns:
            t0 = time.perf_counter()
            r = agent.observe(scope, t.text, speaker=t.speaker, session=t.session, ts=t.date, source_id=t.tid)
            writes.append((time.perf_counter() - t0) * 1000)
            ops[r["op"]] += 1
        histories[scope] = "\n".join(render(t.date, t.speaker, t.text) for t in d.turns)
        if writes_f:   # every non-trivial write decision, with the texts it compared, for checking the write path
            facts = {f["id"]: f for f in agent.store.all_facts(scope)}
            for a in agent.store.audit_rows(scope, ("UPDATE", "CONFLICT", "DELETE", "NOOP")):
                old_id = a.get("replaces") or a.get("with") or (a.get("fact") if a["action"] in ("DELETE", "NOOP") else None)
                new_ep = agent.store.episodes([a["episode"]]).get(a["episode"])
                writes_f.write(json.dumps({"scope": scope, "action": a["action"], "old": facts[old_id]["text"] if old_id in facts else None,
                                           "old_speaker": facts[old_id]["speaker"] if old_id in facts else None,
                                           "new": new_ep["text"] if new_ep else None, "new_speaker": new_ep["speaker"] if new_ep else None,
                                           "p": a.get("p")}) + "\n")
        log(f"{scope}: {len(d.turns)} turns written in {time.perf_counter() - t_conv:.1f}s, ops so far {dict(ops)}")
        qs = [q for q in by_did[d.did] if q.evidence][:max_questions]
        n_before = len(rows)
        for q in qs:
            t0 = time.perf_counter()
            need = tier1.need(q.text)   # timed with the packet: production pays for the gate on every request
            need_ms = (time.perf_counter() - t0) * 1000
            # forced so coverage is measured even where the gate would skip; the gate's time still counts against the deadline
            p = agent.packet(scope, q.text, force=True, deadline_ms=max(0.0, cfg.deadline_ms - need_ms))
            p["ms"] += need_ms
            p["stages_ms"]["need"] = need_ms
            p["late"] = p["ms"] > cfg.deadline_ms
            eps = agent.store.episodes(p["ids"])
            got = {eps[e]["source_id"] for e in p["ids"] if e in eps}
            first = agent.store.episodes(p.get("first_stage", [])[:cfg.rerank_k])
            got_first = {r["source_id"] for r in first.values()}
            gold = set(q.evidence)
            row = {"qid": q.qid, "scope": scope, "category": q.qtype, "need": need, "n_gold": len(gold),
                   "all_gold": gold <= got, "gold_recall": len(gold & got) / len(gold),
                   "all_gold_first_stage": gold <= got_first, "gold_recall_first_stage": len(gold & got_first) / len(gold),
                   "tokens": p["tokens"], "memories": len(p["ids"]), "ms": p["ms"], "late": p["late"],
                   "degraded": p["degraded"], "k": p.get("reranked_k"), "stages_ms": p["stages_ms"], "history_tokens": len(histories[scope]) / 4}
            rows.append(row)
            if packets:
                gold_ctx = "\n".join(render(by_tid[e].date, by_tid[e].speaker, by_tid[e].text) for e in q.evidence if e in by_tid)
                packets.write(json.dumps({**row, "question": q.text, "answer": q.answer, "packet": p["text"], "gold_context": gold_ctx}) + "\n")
        log(f"{scope}: {len(qs)} questions, all-gold in packet {np.mean([r['all_gold'] for r in rows[n_before:]]):.3f}")
        if concurrency_test and not conc and qs:   # once, on the first conversation's filled store
            conc = concurrency(agent, scope, [q.text for q in qs])
            log(f"{scope}: concurrency {json.dumps(conc)}")
    for f in (packets, writes_f):
        if f:
            f.close()
    if out:
        (out / "histories.json").write_text(json.dumps(histories))

    def summary(rs):
        return {"questions": len(rs),
                "all_gold_in_packet": float(np.mean([r["all_gold"] for r in rs])),
                "gold_recall_in_packet": float(np.mean([r["gold_recall"] for r in rs])),
                "all_gold_in_first_stage_top_k": float(np.mean([r["all_gold_first_stage"] for r in rs])),
                "mean_packet_tokens": float(np.mean([r["tokens"] for r in rs])),
                "mean_memories": float(np.mean([r["memories"] for r in rs])),
                "need_below_threshold": float(np.mean([r["need"] < cfg.need_threshold for r in rs]))}

    report = {
        "conversations": [dialogues[c].did for c in convs],
        "overall": summary(rows),
        "by_category": {c: summary([r for r in rows if r["category"] == c]) for c in sorted({r["category"] for r in rows})},
        "context": {"mean_history_tokens": float(np.mean([len(h) / 4 for h in histories.values()])),
                    "packet_share_of_history": float(np.mean([r["tokens"] / r["history_tokens"] for r in rows]))},
        "latency_ms": {"read_p50": pct([r["ms"] for r in rows], 50), "read_p95": pct([r["ms"] for r in rows], 95),
                       "read_late_share": float(np.mean([r["late"] for r in rows])), "degraded_share": float(np.mean([r["degraded"] for r in rows])),
                       "write_p50": pct(writes, 50), "write_p95": pct(writes, 95),
                       "stages_p50": {st: pct([r["stages_ms"].get(st) for r in rows], 50) for st in ("need", "first_stage", "rerank", "render")},
                       "rerank_k_used": {str(kk): int(sum(r["k"] == kk for r in rows)) for kk in sorted({r["k"] for r in rows if r["k"] is not None})}},
        "concurrency": conc,
        "write_ops": dict(ops),
        "config": vars(cfg),
    }
    if out:
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def print_report(r):
    o = r["overall"]
    print(f"\nDAMA on {', '.join(r['conversations'])}: {o['questions']} questions")
    l = r["latency_ms"]
    print(f"  1 speed     read p50 {l['read_p50']} ms, p95 {l['read_p95']} ms, late {100 * l['read_late_share']:.1f}%, degraded {100 * l['degraded_share']:.1f}%; "
          f"write p50 {l['write_p50']} ms, p95 {l['write_p95']} ms")
    print(f"              stages p50 (ms) {l['stages_p50']}  rerank depth used {l['rerank_k_used']}")
    print(f"  2 accuracy  all gold turns in packet {o['all_gold_in_packet']:.3f} (first-stage top-k alone {o['all_gold_in_first_stage_top_k']:.3f}), "
          f"gold-turn recall {o['gold_recall_in_packet']:.3f}; need gate would skip {100 * o['need_below_threshold']:.1f}%")
    print(f"  3 context   packet {o['mean_packet_tokens']:.0f} tokens / {o['mean_memories']:.1f} memories vs history {r['context']['mean_history_tokens']:.0f} tokens "
          f"({100 * r['context']['packet_share_of_history']:.1f}%)")
    for scenario, by_n in r.get("concurrency", {}).items():
        print(f"  4 agents    {scenario:8s} " + "  ".join(f"N={n}: p95 {v['p95_ms']} ms, {v['requests_per_s']} req/s" for n, v in by_n.items()))
    print(f"  write ops {r['write_ops']}")
    for c, s in r["by_category"].items():
        print(f"    {c:12s} n={s['questions']:<4} all-gold {s['all_gold_in_packet']:.3f}  first-stage {s['all_gold_in_first_stage_top_k']:.3f}  "
              f"recall {s['gold_recall_in_packet']:.3f}  tokens {s['mean_packet_tokens']:.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="Tier-1 bundle folder (dama_tier1.json inside)")
    ap.add_argument("--fake", action="store_true", help="use FakeTier1 (plumbing check, no models)")
    ap.add_argument("--locomo", required=True)
    ap.add_argument("--convs", type=int, nargs="+", default=[7, 8, 9])
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--packet-k", type=int, default=8, help="memories per packet")
    ap.add_argument("--rerank", action="store_true", help="let rel re-score the packet candidates (off by default)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.fake:
        from .fake import FakeTier1
        tier1 = FakeTier1()
    else:
        from . import load_tier1
        tier1 = load_tier1(a.bundle, threads=a.threads)
    cfg = AgentConfig(rerank_k=a.packet_k, use_rerank=a.rerank)
    print_report(run(tier1, a.locomo, a.convs, a.out, config=cfg, max_questions=a.max_questions))


if __name__ == "__main__":
    main()

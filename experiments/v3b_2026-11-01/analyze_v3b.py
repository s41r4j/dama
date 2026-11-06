"""Offline analysis of v3b's judged rows (no GPU, no model). Usage: python analyze_v3b.py out/paper/methods

1. Ledger blocks: for routed questions, does the computed value match a number in the gold answer (COUNT / SUM)?
   Wins and losses of each ledger arm vs A0, by route.
2. Disagreement signal: when the ledger's count differs from the number in A0's answer, how often is A0 wrong?
3. Cascade (simulated from the zoo): answer with a small reader; escalate to a larger one when two small readers
   disagree. Accuracy and reader cost (s/question measured in v3b) per policy, against each reader alone.
All accuracies: Qwen-judged, LoCoMo categories 1-4 + LME, mean of the two benchmarks (as in v3a/v3b).
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def load(d):
    rows = {}
    for f in Path(d).glob("dev_*.jsonl"):
        rs = [json.loads(l) for l in open(f)]
        if rs and "qwen_correct" in rs[0]:
            rows[f.stem[4:]] = {r["qid"]: r for r in rs}
    return rows


def main_set(r):
    return r["bench"] != "locomo" or r.get("cat") != 5


def mean_acc(rs):
    by = defaultdict(list)
    for r in rs:
        if main_set(r):
            by[r["bench"]].append(bool(r["qwen_correct"]))
    return sum(sum(v) / len(v) for v in by.values()) / max(1, len(by))


def numbers(text):
    t = (text or "").lower()
    out = [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", t)]
    out += [float(NUM_WORDS[w]) for w in re.findall(r"\b(" + "|".join(NUM_WORDS) + r")\b", t)]
    return out


def ledger_report(rows):
    a0 = rows.get("A0", {})
    for arm in sorted(k for k in rows if k.startswith("L")):
        by_route = defaultdict(Counter)
        val_ok = Counter()
        for qid, r in rows[arm].items():
            led = r.get("ledger")
            if not led or qid not in a0 or not main_set(r):
                continue
            rt = led["route"]
            x, y = a0[qid]["qwen_correct"], r["qwen_correct"]
            by_route[rt]["n"] += 1
            by_route[rt]["win"] += (not x) and y
            by_route[rt]["loss"] += x and not y
            if rt in ("COUNT", "SUM") and led.get("value") is not None:
                g = numbers(r["gold"])
                val_ok["n"] += 1
                val_ok["match"] += bool(g) and any(abs(led["value"] - v) < 1e-6 for v in g)
        print(f"\n{arm}: routes {dict((k, dict(v)) for k, v in by_route.items())}")
        if val_ok["n"]:
            print(f"  COUNT/SUM block value equals a gold number: {val_ok['match']}/{val_ok['n']}")


def disagreement(rows, arm="L9P"):
    if arm not in rows or "A0" not in rows:
        return
    c = Counter()
    for qid, r in rows[arm].items():
        led = r.get("ledger")
        if not led or led["route"] not in ("COUNT", "SUM") or led.get("value") is None:
            continue
        a = rows["A0"][qid]
        na = numbers(a["answer"])
        if not na:
            continue
        agree = any(abs(led["value"] - v) < 1e-6 for v in na)
        c[("agree" if agree else "disagree", "A0 right" if a["qwen_correct"] else "A0 wrong")] += 1
    print(f"\nledger ({arm}) vs A0 numeric answer: {dict(c)}")


def cascade(rows, speeds):
    """Escalate when the two small readers disagree on the judge label proxy: their answers differ (normalised)."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
    zoo = [k for k in rows if k.startswith("Z")] + ["A0"]
    print("\nreaders alone:", {k: round(100 * mean_acc(rows[k].values()), 1) for k in zoo if k in rows})
    small = [k for k in zoo if k not in ("A0", "Z35_A0", "Z35_L35P")]
    big = [k for k in ("Z35_A0", "A0") if k in rows]
    for s1 in small:
        for s2 in small:
            if s1 >= s2:
                continue
            for b in big:
                qids = set(rows[s1]) & set(rows[s2]) & set(rows[b])
                out, esc = [], 0
                for q in qids:
                    if norm(rows[s1][q]["answer"]) == norm(rows[s2][q]["answer"]):
                        out.append(rows[s1][q])
                    else:
                        out.append(rows[b][q]); esc += 1
                cost = speeds.get(s1, 0) + speeds.get(s2, 0) + esc / max(1, len(qids)) * speeds.get(b, 0)
                print(f"  cascade {s1}+{s2} -> {b}: acc {100 * mean_acc(out):.1f}, escalated {esc / max(1, len(qids)):.0%}, "
                      f"reader s/question ~{cost:.2f} (vs {b} alone {speeds.get(b, 0):.2f})")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "out/paper/methods"
    rows = load(d)
    print("arms:", {k: round(100 * mean_acc(v.values()), 1) for k, v in sorted(rows.items())})
    res = json.load(open(Path(d).parent / "results_v3b.json")) if (Path(d).parent / "results_v3b.json").exists() else {}
    speeds = {k: v.get("s_per_question") or 0 for k, v in res.get("arms", {}).items()}
    speeds["A0"] = 3.6                                    # measured in v3a (cached here)
    ledger_report(rows)
    disagreement(rows)
    cascade(rows, speeds)

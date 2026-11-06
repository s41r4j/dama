"""Offline replay of the v4 checks (code/dama_agent/checks.py) on DAMA A0's stored dev packets (DMBM run 1). No model.
Reports, per LoCoMo category: how often the wrong-person check fires, list-question detection, and gold-evidence
recall before / after coverage. Usage: python replay_checks.py [rows.jsonl] [locomo10.json]"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import importlib.util  # noqa: E402  (checks.py alone: the package __init__ pulls numpy and the models)
_spec = importlib.util.spec_from_file_location("checks", ROOT / "code/dama_agent/checks.py")
C = importlib.util.module_from_spec(_spec); sys.modules["checks"] = C; _spec.loader.exec_module(C)
History, check, is_list_question = C.History, C.check, C.is_list_question

rows_f = sys.argv[1] if len(sys.argv) > 1 else ROOT / "experiments/dmbm1_2026-11-01/out/paper/methods/dmbm1_dama_A0.jsonl"
loc_f = sys.argv[2] if len(sys.argv) > 2 else ROOT / "experiments/dmbm2_2026-11-02/out/paper/locomo10.json"
ntok = lambda s: int(len(s.split()) * 1.3) + 1

H, EV = {}, {}
for c in json.load(open(loc_f)):
    conv, turns = c["conversation"], []
    for k in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])):
        for t in conv[k]:
            cap = f" [shares a photo: {t['blip_caption']}]" if t.get("blip_caption") else ""
            turns.append({"dia": t["dia_id"], "speaker": t["speaker"], "text": t["text"] + cap})
    H[c["sample_id"]] = History(turns, speakers=[conv["speaker_a"], conv["speaker_b"]])
    for qa in c["qa"]:
        EV[(c["sample_id"], qa["question"].strip())] = [e.strip() for x in qa.get("evidence", []) for e in x.split(";")]

st = defaultdict(Counter)
fired_examples = []
for r in map(json.loads, open(rows_f)):
    if r["bench"] != "locomo":
        continue
    cid = r["qid"].split(":")[0]
    h, cat, ok = H[cid], r["cat"], bool(r["qwen_correct"])
    ids = re.findall(r"^\[m:(D\d+:\d+)\]", r["context"], flags=re.M)
    res = check(r["question"], h, ids, ntok=ntok)
    ev = set(EV.get((cid, r["question"].strip()), []))
    s = st[cat]
    s["n"] += 1
    s["fire"] += res.attribution is not None
    s[f"fire_when_A0_{'ok' if ok else 'wrong'}"] += res.attribution is not None
    s["list"] += res.list_q
    if ev and cat != 5:
        b, a = len(ev & set(ids)), len(ev & set(res.ids))
        s["ev_full_before"] += b == len(ev); s["ev_full_after"] += a == len(ev)
        s["ev_gain_q"] += a > b; s["ev_loss_q"] += a < b
    s["added"] += len(res.added); s["dropped"] += len(res.dropped)
    if res.attribution is not None and len(fired_examples) < 40:
        fired_examples.append((cat, ok, r["question"], res.header[:160]))

print("cat | n | attribution fires | fires when A0 ok / wrong | list-q | evidence full before -> after | q gained / lost | lines added / dropped")
for cat in sorted(st):
    s = st[cat]
    print(f"{cat} | {s['n']} | {s['fire']} | {s['fire_when_A0_ok']} / {s['fire_when_A0_wrong']} | {s['list']} | "
          f"{s['ev_full_before']} -> {s['ev_full_after']} | {s['ev_gain_q']} / {s['ev_loss_q']} | {s['added']} / {s['dropped']}")
print()
for e in fired_examples:
    print(e)


def all_questions(variants):
    """Fire rates of the wrong-person check over every LoCoMo question (no packets, no answers): dev = conversations
    0-3 (tuned on), test = 4-9 (held out). variants: {name: dict of module constants}."""
    data = json.load(open(loc_f))
    print("\nvariant | split | cat5 fires | cat1-4 fires (false alarms)")
    for name, kv in variants.items():
        old = {k: getattr(C, k) for k in kv}
        for k, v in kv.items():
            setattr(C, k, v)
        for split, convs in (("dev", data[:4]), ("test", data[4:])):
            f5 = n5 = f14 = n14 = 0
            for c in convs:
                h = H[c["sample_id"]]
                for qa in c["qa"]:
                    fired = check(qa["question"], h, [], coverage=False).attribution is not None
                    if qa["category"] == 5:
                        n5 += 1; f5 += fired
                    else:
                        n14 += 1; f14 += fired
            print(f"{name} | {split} | {f5}/{n5} ({100*f5/n5:.0f}%) | {f14}/{n14} ({100*f14/n14:.1f}%)")
        for k, v in old.items():
            setattr(C, k, v)


if "--all" in sys.argv:
    all_questions({"strict": {}, "no_support_veto": {"ATTR_SUPPORT": 9.0}, "min0.5": {"ATTR_MIN": 0.5},
                   "support0.7": {"ATTR_SUPPORT": 0.7}})

"""Small fake LongMemEval / MNLI / Dolly files in the documented formats, for offline smoke tests."""
from __future__ import annotations
import json
import random
from pathlib import Path

TYPES = ["single-session-user", "single-session-assistant", "single-session-preference", "temporal-reasoning",
         "knowledge-update", "multi-session"]
FILLER = ["Can you suggest a recipe for dinner?", "Here is a simple pasta recipe with garlic and olive oil.",
          "What is the tallest mountain in Europe?", "Mount Elbrus is the tallest mountain in Europe.",
          "Help me write an email to my landlord.", "Sure, here is a polite draft for your landlord.",
          "Any tips for sleeping better?", "Keep a regular schedule and avoid screens before bed."]


def fake_longmemeval(path, n=40, seed=0):
    rng = random.Random(seed)
    items = []
    for k in range(n):
        qtype = TYPES[k % len(TYPES)]
        qid = f"q{k:04d}" + ("_abs" if k % 13 == 12 else "")
        sessions, ids, dates = [], [], []
        city_old, city_new = rng.sample(["Pune", "Oslo", "Lima", "Kyiv", "Quito", "Perth"], 2)
        for s in range(6):
            turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": rng.choice(FILLER)} for i in range(6)]
            if s == 1:
                turns[2] = {"role": "user", "content": f"By the way I live in {city_old} and work as a nurse.", "has_answer": True}
            if s == 4 and qtype == "knowledge-update":
                turns[0] = {"role": "user", "content": f"Big news, I moved to {city_new} last week.", "has_answer": True}
            sessions.append(turns); ids.append(f"s{k}_{s}"); dates.append(f"2023/0{s + 1}/1{s} (Mon) 10:00")
        items.append({"question_id": qid, "question_type": qtype, "question": "Where do I live now?",
                      "answer": city_new if qtype == "knowledge-update" else city_old, "question_date": "2023/09/01 (Fri) 09:00",
                      "haystack_session_ids": ids, "haystack_dates": dates, "haystack_sessions": sessions,
                      "answer_session_ids": [ids[1]]})
    Path(path).write_text(json.dumps(items))


def fake_mnli(path, n=300, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a = f"The team uses {rng.choice(['Go', 'Rust', 'Java'])} for service {i}."
        rows.append({"premise": a, "hypothesis": rng.choice([a, f"Service {i} is written in COBOL.", "The office has a cafe."]),
                     "label": rng.choice([0, 1, 2])})
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake_dolly(path, n=200):
    rows = [{"instruction": f"Explain topic number {i} in simple words.", "category": "open_qa"} for i in range(n)]
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake_questions(path, n=200):
    rows = [{"text": f"How do I fix problem number {i} on my laptop?", "src": "quora"} for i in range(n // 2)]
    rows += [{"text": f"What is the population of city number {i}?", "src": "squad"} for i in range(n // 2)]
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_sources(directory, locomo_path=None) -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    fake_longmemeval(d / "longmemeval_s_cleaned.json")
    fake_mnli(d / "mnli_sample.jsonl")
    fake_dolly(d / "dolly_sample.jsonl")
    fake_questions(d / "general_questions.jsonl")
    if locomo_path is None:
        locomo_path = d / "locomo10.json"
        fake_locomo(locomo_path)
    return {"locomo": str(locomo_path), "longmemeval_s_cleaned.json": str(d / "longmemeval_s_cleaned.json"),
            "mnli": str(d / "mnli_sample.jsonl"), "dolly": str(d / "dolly_sample.jsonl"),
            "questions": str(d / "general_questions.jsonl")}


def fake_locomo(path, n=10, seed=0):
    rng = random.Random(seed)
    data = []
    for c in range(n):
        conv = {"speaker_a": "Ana", "speaker_b": "Ben"}
        qa = []
        for s in range(1, 4):
            conv[f"session_{s}_date_time"] = f"1:00 pm on {s} May, 2023"
            conv[f"session_{s}"] = [{"speaker": "Ana" if i % 2 == 0 else "Ben", "dia_id": f"D{s}:{i + 1}",
                                     "text": rng.choice(FILLER)} for i in range(8)]
        conv["session_2"][3]["text"] = "I adopted a beagle named Max yesterday!"
        for cat in (1, 2, 3, 4, 5):
            qa.append({"question": "What pet did Ana adopt?", "answer": "a beagle", "evidence": ["D2:4"], "category": cat})
        data.append({"sample_id": f"conv-{c}", "conversation": conv, "qa": qa})
    Path(path).write_text(json.dumps(data))

"""LongMemEval (Wu et al., ICLR 2025): 500 questions, each with its own chat history; evidence turns carry has_answer."""
from __future__ import annotations
import json
from pathlib import Path
from .schema import Dialogue, Question, Turn


def load_longmemeval(path, max_questions: int | None = None) -> tuple[list[Dialogue], list[Question]]:
    items = json.loads(Path(path).read_text())
    if max_questions:
        items = items[:max_questions]
    dialogues, questions = [], []
    for item in items:
        qid = item["question_id"]
        did = f"lme:{qid}"
        turns, evidence = [], []
        for sid, date, session in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
            for i, t in enumerate(session):
                text = (t.get("content") or "").strip()
                if not text:
                    continue
                tid = f"{qid}:{sid}:{i}"
                role = t.get("role", "user")
                turns.append(Turn(tid=tid, speaker=role, text=text, date=date, session=f"{qid}:{sid}", role=role))
                if t.get("has_answer"):
                    evidence.append(tid)
        dialogues.append(Dialogue(did=did, source="lme", turns=turns))
        questions.append(Question(qid=qid, did=did, text=item["question"].strip(), date=item.get("question_date"),
                                  qtype=item["question_type"], evidence=evidence, answer=str(item.get("answer", "")),
                                  abstention=qid.endswith("_abs")))
    return dialogues, questions

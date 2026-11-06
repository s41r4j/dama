"""LoCoMo (Maharana et al., ACL 2024): 10 long conversations with QA and evidence turn ids."""
from __future__ import annotations
import json
import re
from pathlib import Path
from .schema import Dialogue, Question, Turn

# Category ids as used by the LoCoMo release and the Mem0 / Memory-R1 / MemRouter evaluations.
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
_EVIDENCE = re.compile(r"D:?\s*(\d+)\s*:\s*0*(\d+)")


def parse_evidence(items) -> list[str]:
    """Evidence strings are sometimes malformed ('D8:6; D9:17', 'D:11:26', 'D30:05'); recover every id."""
    out = []
    for item in items or []:
        for session, turn in _EVIDENCE.findall(str(item)):
            out.append(f"D{int(session)}:{int(turn)}")
    return list(dict.fromkeys(out))


def load_locomo(path) -> tuple[list[Dialogue], list[Question]]:
    data = json.loads(Path(path).read_text())
    dialogues, questions = [], []
    for sample in data:
        sid, conv = sample["sample_id"], sample["conversation"]
        sessions = sorted((int(m.group(1)), k) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)))
        turns, known = [], set()
        for n, key in sessions:
            date = conv.get(f"session_{n}_date_time")
            for t in conv[key]:
                text = t.get("text", "").strip()
                if t.get("blip_caption"):
                    text = f"{text} [shares a photo: {t['blip_caption']}]".strip()
                if not text:
                    continue
                dia = f"D{int(t['dia_id'].split(':')[0][1:])}:{int(t['dia_id'].split(':')[1])}"
                known.add(dia)
                turns.append(Turn(tid=f"{sid}:{dia}", speaker=t["speaker"], text=text, date=date, session=f"{sid}:s{n}"))
        dialogues.append(Dialogue(did=f"locomo:{sid}", source="locomo", turns=turns))
        for i, qa in enumerate(sample["qa"]):
            evidence = [f"{sid}:{e}" for e in parse_evidence(qa.get("evidence")) if e in known]
            category = CATEGORY.get(qa.get("category"), "unknown")
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            questions.append(Question(qid=f"{sid}:q{i}", did=f"locomo:{sid}", text=qa["question"].strip(), date=None,
                                      qtype=category, evidence=evidence, answer=str(answer),
                                      abstention=category == "adversarial"))
    return dialogues, questions

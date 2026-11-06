"""Hand-written behaviour probes (CheckList-style minimum functionality tests), never used for training or for any
model choice. They catch shortcut learning that a benchmark test split cannot: in L1.0 `need` scored AUC 0.999 on
test and 0.049 on "Where do I live these days?", and the op model said ADD for Berlin -> Pune.

Wording, names and values are kept out of the training templates (build_decision.NEED_TEMPLATES, build_ops), so a
pass means the model generalised, not that it memorised a frame. Reported in report.json["probes"].
"""
from __future__ import annotations
import numpy as np
from .config import OPS

# (request, needs memory?)
NEED = [
    ("Where do I live these days?", 1),
    ("What's my sister's name again?", 1),
    ("Which database did we pick for the project?", 1),
    ("What was the name of the restaurant I liked last month?", 1),
    ("How long have I been learning Spanish?", 1),
    ("When is my dentist appointment?", 1),
    ("What did my doctor say about my knee?", 1),
    ("Is my flight on Friday or Saturday?", 1),
    ("Which laptop did I end up buying?", 1),
    ("What gift did I get my mom for her birthday?", 1),
    ("Suggest a dinner recipe that fits my diet.", 1),
    ("Remind me of the deadline we agreed on.", 1),
    ("How many times have I gone hiking this year?", 1),
    ("Who is my manager now?", 1),
    ("What did Caroline say about the adoption agency?", 1),
    ("Where do I find the Downloads folder on a Mac?", 0),
    ("How do I boil an egg?", 0),
    ("What is the capital of Australia?", 0),
    ("Write a haiku about autumn.", 0),
    ("Translate 'good morning' into French.", 0),
    ("How do I center a div in CSS?", 0),
    ("Who wrote Pride and Prejudice?", 0),
    ("Summarize this: the mitochondria is the powerhouse of the cell.", 0),
    ("What's 17 times 23?", 0),
    ("Explain how vaccines work.", 0),
    ("Give me three tips for better sleep.", 0),
    ("How do I change my password on Gmail?", 0),
    ("What time zone is Tokyo in?", 0),
    ("Is a tomato a fruit or a vegetable?", 0),
    ("Write a Python function that reverses a string.", 0),
]

# (existing memory, new statement, expected op)
OP = [
    ("The user lives in Berlin.", "I moved to Pune last month", "UPDATE"),
    ("The user works as a teacher.", "I quit teaching and now I'm a software developer", "UPDATE"),
    ("The project database is MySQL.", "We migrated everything to Postgres last sprint", "UPDATE"),
    ("The user's phone is a Pixel 7.", "Just got an iPhone 16, loving it so far", "UPDATE"),
    ("The user's manager is Priya.", "My new manager is Tom", "UPDATE"),
    ("The user is vegetarian.", "I started eating fish again, so pescatarian now", "UPDATE"),
    ("The client meeting is on Monday.", "The client meeting got pushed to Thursday", "UPDATE"),
    ("The user drives a Honda Civic.", "I sold the Civic and bought a Tesla", "UPDATE"),
    ("The user lives in Berlin.", "I have a dog named Bruno", "ADD"),
    ("The user enjoys hiking.", "I've also started doing pottery on weekends", "ADD"),
    ("The user speaks Hindi.", "I can speak Japanese too", "ADD"),
    ("The user works as a nurse.", "My favourite band is Coldplay", "ADD"),
    ("The user is allergic to peanuts.", "Shellfish is a problem for me as well", "ADD"),
    ("The user has a cat.", "My daughter starts school in September", "ADD"),
    ("The user has a parrot.", "Please forget everything about my parrot", "DELETE"),
    ("The user enjoys chess.", "I don't play chess anymore, take it off my list", "DELETE"),
    ("The user's address is 12 Elm Street.", "Delete my address from your memory", "DELETE"),
    ("The user is learning guitar.", "I gave up on guitar, you can drop that", "DELETE"),
    ("The user lives in Berlin.", "Yeah, Berlin is still home", "NOOP"),
    ("The user works as a teacher.", "Thanks, that helps a lot!", "NOOP"),
    ("The project database is MySQL.", "Someone on the team thinks we might switch to Mongo, who knows", "NOOP"),
    ("The user lives in Berlin.", "I might move to Pune next year, not sure yet", "NOOP"),
    ("The user enjoys hiking.", "Hiking is still my favourite thing", "NOOP"),
    ("The user's phone is an iPhone 15.", "Can you help me write an email?", "NOOP"),
]


def need_rows():
    return [{"task": "need", "a": t, "b": "", "y": float(y)} for t, y in NEED]


def op_rows():
    return [{"task": "op", "a": a, "b": b, "y": OPS.index(y)} for a, b, y in OP]


def score_need(p) -> dict:
    """p: calibrated need probabilities in NEED order."""
    y = np.array([y for _, y in NEED]); p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {"n": len(NEED), "accuracy": float((pred == y).mean()),
            "positives_found": float(pred[y == 1].mean()), "negatives_rejected": float(1 - pred[y == 0].mean()),
            "failures": [{"request": t, "expected": int(yy), "p": round(float(pp), 3)} for (t, yy), pp in zip(NEED, p) if int(pp >= 0.5) != yy]}


def score_op(probs) -> dict:
    """probs: [n, len(OPS)] calibrated op probabilities in OP order."""
    probs = np.asarray(probs, dtype=float)
    pred = [OPS[i] for i in probs.argmax(-1)]
    per = {}
    for op in OPS:
        idx = [i for i, (_, _, y) in enumerate(OP) if y == op]
        per[op] = float(np.mean([pred[i] == op for i in idx])) if idx else None
    return {"n": len(OP), "accuracy": float(np.mean([p == y for p, (_, _, y) in zip(pred, OP)])), "accuracy_per_op": per,
            "failures": [{"existing": a, "new": b, "expected": y, "got": p, "p": round(float(pr.max()), 3)}
                         for (a, b, y), p, pr in zip(OP, pred, probs) if p != y]}


def run_runtime(t1) -> dict:
    """The same probes through the exported CPU bundle (runtime.Tier1), i.e. what DAMA will actually ship."""
    from .config import OPS as labels
    need = score_need([t1.need(t) for t, _ in NEED])
    op = score_op([[t1.op(a, b)[k] for k in labels] for a, b, _ in OP])
    return {"need": need, "op": op}

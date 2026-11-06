"""DAMA v4 read-path checks (code/dama_agent/checks.py)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("checks", Path(__file__).resolve().parents[1] / "dama_agent/checks.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

TURNS = [
    {"dia": "D1:1", "speaker": "Caroline", "text": "Hey Mel! Good to see you! How have you been?"},
    {"dia": "D1:2", "speaker": "Melanie", "text": "Great, thanks! I just signed up for a pottery class yesterday."},
    {"dia": "D1:3", "speaker": "Caroline", "text": "Thanks, Mel! I made this stained glass window for a local church."},
    {"dia": "D1:4", "speaker": "Melanie", "text": "Wow, your new hair color looks amazing! What made you choose that shade?"},
    {"dia": "D1:5", "speaker": "Caroline", "text": "Purple felt bold. I read Charlotte's Web as a kid and loved it."},
    {"dia": "D2:1", "speaker": "Melanie", "text": "My kids loved the beach. We went camping and saw the meteor shower."},
    {"dia": "D2:2", "speaker": "Caroline", "text": "I also read Becoming Nicole, it is my favorite book now."},
    {"dia": "D2:3", "speaker": "Melanie", "text": "I took the kids to a museum and we painted a sunset together."},
]
H = C.History(TURNS, speakers=["Caroline", "Melanie"])


def test_stem_and_words():
    assert C.stem("turtles") == C.stem("turtle") and C.stem("hiking") == C.stem("hike") and C.stem("stories") == C.stem("story")
    assert C.stem("dances") == C.stem("dance") == C.stem("danced") and C.stem("class") == "class"
    assert C.words("I'm here, Mel!") == ["i'm", "here", "mel"]


def test_named_speakers_in_order():
    assert C.named_speakers("What did Melanie give Caroline?", H) == ["Melanie", "Caroline"]
    assert C.named_speakers("What did Caroline's kids do?", H) == ["Caroline"]


def test_list_question_detection():
    assert C.is_list_question("What books has Caroline read?")
    assert C.is_list_question("What do Caroline and Melanie both like?")
    assert C.is_list_question("How many kids does Melanie have?")
    assert not C.is_list_question("When did Caroline go to the church?")


def test_about_whom_first_and_second_person():
    assert C.about_whom("What did Caroline make for a local church?", H, 2) == "Caroline"     # "I made"
    assert C.about_whom("What color did Caroline choose for her hair?", H, 3) == "Caroline"   # Melanie: "your hair"


def test_wrong_person_fires_on_swapped_name():
    r = C.check("What did Melanie make for a local church?", H, ["D1:1"])
    assert r.attribution == "Caroline"
    assert "D1:3" in r.ids and "D1:3" in r.added           # the reader sees the line the check names
    assert r.header.startswith("CHECK:") and "not Melanie" in r.header


def test_wrong_person_silent_on_true_question():
    assert C.check("What did Caroline make for a local church?", H, ["D1:3"]).attribution is None
    assert C.check("What color did Caroline choose for her hair?", H, ["D1:4"]).attribution is None


def test_wrong_person_silent_when_both_named():
    assert C.check("What did Melanie and Caroline make for a church?", H, []).attribution is None


def test_list_coverage_adds_and_keeps_budget():
    ntok = lambda s: len(s.split())
    base = ["D1:5", "D2:1", "D2:3"]
    r = C.check("What books has Caroline read?", H, base, ntok=ntok)
    assert r.list_q and "D2:2" in r.ids                    # the second book, from another session
    size = lambda ids: sum(ntok(TURNS[[t["dia"] for t in TURNS].index(d)]["text"]) + 12 for d in ids)
    assert size(r.ids) <= size(base)
    assert r.ids == sorted(r.ids, key=lambda d: [t["dia"] for t in TURNS].index(d))

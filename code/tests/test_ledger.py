"""Unit tests for the event ledger (no models). Run: python tests/test_ledger.py"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dama_agent.ledger import (finish_rows, match, norm_item, operate, parse_rows, render, resolve_when, route,
                               session_prompts)

REF = "2023/05/20 (Sat) 02:21"


def w(text, ref=REF):
    x = resolve_when(text, ref)
    return x.start, x.end, x.gran


def test_resolve_when_absolute_and_relative():
    assert w("") == (date(2023, 5, 20), date(2023, 5, 20), "mention")
    assert w("on May 3") == (date(2023, 5, 3), date(2023, 5, 3), "day")
    assert w("3rd of March 2022")[0] == date(2022, 3, 3)
    assert w("in June")[0] == date(2022, 6, 1) and w("in June")[2] == "month"      # a later month: last year
    assert w("2023-01-09")[0] == date(2023, 1, 9)
    assert w("yesterday")[0] == date(2023, 5, 19)
    assert w("two weeks ago")[0] == date(2023, 5, 6)
    assert w("3 days ago")[0] == date(2023, 5, 17)
    assert w("last Saturday")[0] == date(2023, 5, 13)                                 # ref is a Saturday
    assert w("last Tuesday")[0] == date(2023, 5, 16)
    assert w("last weekend")[:2] == (date(2023, 5, 13), date(2023, 5, 14))
    assert w("last month")[:2] == (date(2023, 4, 1), date(2023, 4, 30))
    assert w("a few days ago")[2] == "approx"
    assert w("over the past month")[2] == "mention"
    assert w("in 2021")[:2] == (date(2021, 1, 1), date(2021, 12, 31))


def test_parse_rows_tolerates_noise():
    out = ('Here are the lines:\n```json\n{"t": 3, "who": "user", "kind": "event", "what": "attended", "item": "AFI Fest", '
           '"cat": "film festival", "qty": null, "unit": "", "when": "last Saturday"}\n'
           "{'t': '#5', 'who': 'user', 'kind': 'pref', 'what': 'likes', 'item': 'sushi', 'cat': 'food', 'qty': '', 'unit': '', 'when': ''}\n"
           '{"t": 99, "what": "x", "item": "y"}\n```')
    rows = parse_rows(out, valid_turns={3, 5})
    assert [r["t"] for r in rows] == [3, 5] and rows[1]["kind"] == "pref" and rows[0]["cat"] == "film festival"
    assert parse_rows('[{"t": 1, "what": "bought", "item": "bike", "qty": "$1,200", "unit": "USD"}]')[0]["qty"] == 1200.0


def test_session_prompts_filter_and_split():
    turns = [(0, "user", "a" * 30), (1, "assistant", "b" * 30), (2, "user", "c" * 30)]
    p = session_prompts("2023/05/20", turns, speaker_filter={"user"}, max_chars=50)
    assert [ts for _, ts in p] == [[0], [2]] and "[#2] user:" in p[1][0][1]["content"]


def test_route():
    assert route("How many film festivals did I attend?") == "COUNT"
    assert route("How many days passed between my visit to the museum and the concert?") == "DATEDIFF"
    assert route("How many weeks ago did I start using Ibotta?") == "DATEDIFF"
    assert route("How many hours have I spent on my sculpture?") == "SUM"
    assert route("How much money did I spend on bike parts in total?") == "SUM"
    assert route("Which did I visit first, Rome or Paris?") == "ORDER"
    assert route("Where do I currently keep my bike?") == "LATEST"
    assert route("Can you recommend a recipe for my dinner party?") == "PREF"
    assert route("What did John say about Lebron James?") is None


def rows_fixture():
    raw = [
        {"t": 1, "who": "user", "kind": "event", "what": "attended", "item": "AFI Fest", "cat": "film festival", "qty": None, "unit": "", "when": "last Saturday"},
        {"t": 2, "who": "user", "kind": "event", "what": "attended", "item": "the AFI Fest", "cat": "film festival", "qty": None, "unit": "", "when": ""},
        {"t": 3, "who": "user", "kind": "event", "what": "attended", "item": "Sundance", "cat": "film festival", "qty": None, "unit": "", "when": "in January"},
        {"t": 4, "who": "user", "kind": "event", "what": "spent", "item": "sculpture", "cat": "art project", "qty": 4, "unit": "hours", "when": ""},
        {"t": 5, "who": "user", "kind": "event", "what": "spent", "item": "sculpture", "cat": "art project", "qty": 6, "unit": "hours", "when": "yesterday"},
        {"t": 6, "who": "user", "kind": "state", "what": "keeps bike in", "item": "garage", "cat": "storage", "qty": None, "unit": "", "when": ""},
    ]
    return finish_rows(raw, "s1", REF, lambda t: f"s1:{t}", "h1")


def test_count_dedupes_and_sorts():
    rows = rows_fixture()
    b = operate("COUNT", rows[:3], "How many festivals?", REF)
    assert b.value == 2 and "Sundance" in b.text.splitlines()[1]          # January before May
    assert render(b).startswith("LEDGER")


def test_sum_and_datediff_and_latest():
    rows = rows_fixture()
    assert operate("SUM", rows[3:5], "How many hours have I spent?", REF).value == 10
    b = operate("DATEDIFF", [rows[0]], "How many days ago did I attend AFI Fest?", "2023/05/30 (Tue) 10:00")
    assert "17 days" in b.text                                            # 13 May -> 30 May
    assert "turn s1:6" in operate("LATEST", rows, "Where do I keep my bike now?", REF).text


def test_match_relative_threshold_and_pref_filter():
    rows = rows_fixture()
    got = match(rows, [0.9, 0.88, 0.85, 0.3, 0.3, 0.2], "COUNT", rel=0.08)
    assert [r["t"] for r in got] == [1, 2, 3]
    assert match(rows, [0.9] * 6, "PREF") == []                           # no pref rows in the fixture
    assert norm_item("The AFI Fests") == "afi fest"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

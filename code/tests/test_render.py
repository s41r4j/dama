"""Unit tests for packet rendering R1 and prompt P2 (no models). Run: python tests/test_render.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dama_agent.render import (NOT_FOUND, annotate, days_before, messages_p2, render_packet, route_thinking,
                               session_order)
from dama_agent.timeparse import parse_date

LME_Q = "2023/11/20 (Mon) 10:00"


def _t(i, session, date, speaker, text):
    return {"i": i, "dia": f"{session}:{i}", "session": session, "date": date, "date_raw": date, "speaker": speaker, "text": text}


HIST = [_t(0, "s_a", "2023/05/20 (Sat) 02:21", "user", "I went to AFI Fest yesterday."),
        _t(1, "s_a", "2023/05/20 (Sat) 02:21", "assistant", "Sounds fun!"),
        _t(2, "s_b", "2023/11/17 (Fri) 09:00", "user", "Back from Sundance."),
        _t(3, "s_c", "2024/02/29 (Thu) 09:00", "user", "Leap day.")]


def test_lines_are_chronological_with_offsets():
    lines, ids = render_packet([HIST[2], HIST[0]], LME_Q, session_order(HIST))
    assert ids == ["s_a:0", "s_b:2"]
    assert lines[0] == "#1 [S1 | Sat 2023-05-20 | 184 days before question | user] I went to AFI Fest yesterday [= 19 May 2023, 185 days before question]."
    assert lines[1].startswith("#2 [S2 | Fri 2023-11-17 | 3 days before question | user] Back from Sundance.")


def test_session_numbers_come_from_the_history_not_the_packet():
    lines, _ = render_packet([HIST[3]], "2024/03/01 (Fri) 08:00", session_order(HIST))
    assert lines[0].startswith("#1 [S3 | Thu 2024-02-29 | 1 day before question | user]")


def test_offsets_across_month_and_leap_year():
    q = parse_date("2024-03-01")
    assert days_before(parse_date("2024-02-28"), q) == "2 days before question"
    assert days_before(parse_date("2023-03-01"), q) == "366 days before question"
    assert days_before(parse_date("2024-03-01"), q) == "same day as question"
    assert days_before(None, q) == "" and days_before(q, None) == ""


def test_time_notes_only_exact_days_get_offsets():
    ref, q = "1:56 pm on 17 November, 2023", parse_date("2023-11-20")
    assert annotate("I started four months ago.", ref, q) == "I started four months ago [= July 2023]."
    assert "[= about 3 November 2023]" in annotate("2 weeks ago", ref, q)
    assert "[= 16 November 2023, 4 days before question]" in annotate("yesterday", ref, q)


def test_no_question_date_drops_offsets_only():
    locomo = [{"i": 0, "dia": "D1:1", "session": 1, "date": "8 May 2023", "date_raw": "1:56 pm on 8 May, 2023",
               "speaker": "Caroline", "text": "I went to a support group yesterday."}]
    lines, _ = render_packet(locomo, None, session_order(locomo))
    assert lines == ["#1 [S1 | Mon 2023-05-08 | Caroline] I went to a support group yesterday [= 7 May 2023]."]


def test_without_time_notes():
    lines, _ = render_packet([HIST[0]], LME_Q, time_notes=False)
    assert lines[0] == "#1 [Sat 2023-05-20 | 184 days before question | user] I went to AFI Fest yesterday."


def test_p2_messages():
    lines, _ = render_packet(HIST[:2], LME_Q, session_order(HIST))
    sys_m, user_m = messages_p2("lme", "How many festivals did I attend?", LME_Q, lines)
    assert sys_m["role"] == "system" and "DIGEST" not in sys_m["content"] and "recommendation" in sys_m["content"]
    u = user_m["content"]
    assert u.startswith(f"QUESTION DATE: {LME_Q}\nQUESTION: How many festivals did I attend?\n\nMEMORY (oldest first):\n#1 [")
    assert "DIGEST" not in u and "EVIDENCE_STATUS" not in u and NOT_FOUND in u and "at most 30 words" in u
    assert u.rstrip().endswith("then answer fully.")


def test_p2_locomo_digest_and_status():
    sys_m, user_m = messages_p2("locomo", "When did Caroline go?", None, ["#1 [x] y"], digest=[],
                                evidence_status="possibly incomplete")
    assert "trust the raw line" in sys_m["content"] and "two people" in sys_m["content"]
    u = user_m["content"]
    assert u.startswith("QUESTION: When did Caroline go?") and "DIGEST:\n(none)" in u
    assert "EVIDENCE_STATUS: possibly incomplete" in u and "7 May 2023" in u and "at most 15 words" in u
    assert "per MEMORY or DIGEST line" in u


def test_route_thinking():
    assert route_thinking("How many festivals have I been to?")
    assert route_thinking("What is my current job title?")
    assert route_thinking("How much older is my brother than me?")
    assert not route_thinking("What did I name my cat?")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("ok")

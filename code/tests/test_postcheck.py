"""Unit tests for the CALC post-check (no models). Run: python tests/test_postcheck.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dama_agent.postcheck import date_diff, first_number, items, parse_calc, postcheck, safe_eval


def out(*lines):
    return "\n".join(["EVIDENCE:", "| # | date | session | fact |", "| 1 | 2023-05-20 | 1 | AFI Fest |", *lines])


def test_count_uses_deduplicated_items():
    r = postcheck(out("ITEM: AFI Fest (#1)", "ITEM: Sundance (#4)", "ITEM: the AFI Fest (#7)", "ITEM: Tribeca",
                      "ITEM: Telluride", "CALC: COUNT", "ANSWER: You attended 3 festivals."))
    assert r.changed and r.answer == "You attended 4 festivals." and r.reason == "count_mismatch"
    r = postcheck(out("ITEM: A", "ITEM: B", "CALC: COUNT", "ANSWER: two"))
    assert not r.changed and r.reason == "consistent"


def test_count_word_number_is_replaced_with_digits():
    r = postcheck(out("ITEM: A", "ITEM: B", "ITEM: C", "CALC: COUNT", "ANSWER: Two times."))
    assert r.changed and r.answer == "3 times."


def test_sum_and_arith():
    assert parse_calc("SUM(3, 5.5)") == ("SUM", [3.0, 5.5])
    r = postcheck(out("CALC: SUM($120, $45)", "ANSWER: $155 in total"))
    assert r.changed and r.answer == "$165 in total"
    r = postcheck(out("CALC: 32 - 25 = 7", "ANSWER: 7 years"))
    assert not r.changed and r.value == 7


def test_datediff():
    assert date_diff("2024-01-31", "2024-03-01", "days") == 30
    assert date_diff("2024-03-01", "2024-01-31", "days") == 30
    assert date_diff("2023-01-15", "2023-04-15", "months") == 3
    assert date_diff("1990-06-02", "2023-06-01", "years") == 32
    assert date_diff("2023-01-01", "2023-01-22", "weeks") == 3
    r = postcheck(out("CALC: DATEDIFF(2023-05-20, 2023-06-19, days)", "ANSWER: 31 days"))
    assert not r.changed                                     # within one day: inclusive counting is accepted
    r = postcheck(out("CALC: DATEDIFF(2023-05-20, 2023-06-19, days)", "ANSWER: 20 days"))
    assert r.changed and r.answer == "30 days"
    r = postcheck(out("CALC: DATEDIFF(2023-01-01, 2023-01-22, weeks)", "ANSWER: 2 weeks"))
    assert r.changed and r.answer == "3 weeks"


def test_never_changes_latest_none_or_abstention():
    assert not postcheck(out("CALC: LATEST(#3)", "ANSWER: Senior analyst, since 2 March")).changed
    assert not postcheck(out("CALC: NONE", "ANSWER: 5")).changed
    r = postcheck(out("ITEM: A", "CALC: COUNT", "ANSWER: Not mentioned in the conversation."))
    assert not r.changed and r.reason == "abstained"
    assert postcheck("no answer line").reason == "no_answer"


def test_first_number_skips_years_and_dates():
    a = "On 7 May 2023 you had 3 cats"
    assert first_number(a).group() == "3"
    assert first_number("In 2023 you had 3 cats").group() == "3"
    assert first_number("no numbers") is None


def test_markdown_tags_and_last_answer_wins():
    r = postcheck("**CALC:** COUNT\nITEM: a\nITEM: b\nANSWER: 1\n**ANSWER:** 1 thing")
    assert r.changed and r.answer == "2 thing"


def test_numbered_steps_as_written_by_the_reader():
    # v3a run 2: Qwen3.5-9B numbers P2's steps and puts CALC after the RESOLVE label
    r = postcheck("1. EVIDENCE:\n| 1 | 2023-05-20 | S1 | a |\n- ITEM: a\n- ITEM: b\n- ITEM: c\n"
                  "2. RESOLVE: CALC: COUNT\n\n4. ANSWER: 2 things")
    assert r.changed and r.answer == "3 things"
    assert postcheck("2. **RESOLVE:** CALC: NONE\n4. **ANSWER:** Paris").reason == "none"


def test_safe_eval_rejects_code():
    assert safe_eval("(3 + 5) * 2 / 4") == 4
    for bad in ("__import__('os')", "2 ** 999999", "a + 1", "len([1])"):
        try:
            safe_eval(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert parse_calc("open('x')") == ("NONE", None)


def test_items_dedupe():
    assert items("ITEM: The Sundance (#2)\nITEM: sundance\n- ITEM: Tribeca.") == ["The Sundance (#2)", "Tribeca."]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("ok")

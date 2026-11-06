"""Unit tests for write-time time resolution (no models). Run: python tests/test_timeparse.py"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dama_agent.timeparse import TEMPORAL_QUERY, annotate, find, parse_date

REF = "1:56 pm on 17 November, 2023"          # a Friday


def test_parse_date_formats():
    assert parse_date(REF) == date(2023, 11, 17)
    assert parse_date("2023/05/20 (Sat) 02:21") == date(2023, 5, 20)
    assert parse_date("2023-05-02") == date(2023, 5, 2)
    assert parse_date("August 7, 2023") == date(2023, 8, 7)
    assert parse_date("sometime") is None and parse_date(None) is None


def test_days():
    assert annotate("I went to a gala yesterday!", REF) == "I went to a gala yesterday [= 16 November 2023]!"
    assert "[= 15 November 2023]" in annotate("the day before yesterday it rained", REF)
    assert "[= 17 November 2023]" in annotate("Today was great", REF)
    assert "[= 18 November 2023]" in annotate("see you tomorrow", REF)
    assert "last night [= 16 November 2023]" in annotate("We watched it last night.", REF)


def test_counts_ago():
    assert "[= 14 November 2023]" in annotate("three days ago", REF)
    assert "[= about 3 November 2023]" in annotate("2 weeks ago", REF)
    assert "[= July 2023]" in annotate("I started four months ago.", REF)
    assert "[= 2020]" in annotate("I visited Brazil three years ago", REF)
    assert "[= a few days before 17 November 2023]" in annotate("a few days ago", REF)
    assert "[= 2022]" in annotate("a year ago", REF)


def test_calendar_units():
    assert "[= the week before 17 November 2023]" in annotate("I met her last week", REF)
    assert "[= December 2023]" in annotate("I leave next month", REF)
    assert "[= October 2023]" in annotate("last month we moved", REF)
    assert "[= 2024]" in annotate("next year", REF)
    assert "[= January 2024]" in annotate("next month", "5 December, 2023")     # year rolls over


def test_weekdays_and_weekends():
    assert "[= 10 November 2023]" in annotate("last Friday", REF)                # said on a Friday: a week back
    assert "[= 13 November 2023]" in annotate("this past Monday", REF)
    assert "[= the weekend of 11-12 November 2023]" in annotate("last weekend", REF)
    assert "[= the weekend of 18-19 November 2023]" in annotate("this weekend", REF)


def test_seasons():
    assert "[= summer 2023]" in annotate("last summer", REF)
    assert "[= summer 2022]" in annotate("last summer", "10 July, 2023")
    assert "[= winter 2022/2023]" in annotate("last winter", "10 January, 2024")
    assert annotate("this summer", REF) == "this summer"                          # outside summer: past or coming?


def test_ambiguous_left_alone():
    for s in ["On Friday we met", "next Friday", "next weekend", "I lost weight in two weeks", "the other day",
              "over the last month", "the last week of school", "in the past year", "recently"]:
        assert annotate(s, REF) == s, s


def test_unknown_reference_and_idempotence():
    assert annotate("yesterday", "no date here") == "yesterday"
    once = annotate("yesterday", REF)
    assert annotate(once, REF) == once
    assert [h[2] for h in find("yesterday and last week", REF)] == ["16 November 2023", "the week before 17 November 2023"]


def test_temporal_query():
    assert TEMPORAL_QUERY.search("When did Calvin attend a gala?")
    assert TEMPORAL_QUERY.search("How long did they date before marrying?")
    assert not TEMPORAL_QUERY.search("What is Nate's favourite game?")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

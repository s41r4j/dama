"""Unit tests for the Level 3 scorer (no LLM). Run: python tests/test_answer.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dama_agent.answer import f1, normalize, score


def test_normalize_and_f1():
    assert normalize("The Beach!") == "beach"
    assert f1("the beach", "Beach") == 1.0
    assert abs(f1("7 May 2023", "May 7, 2023") - 1.0) < 1e-9          # word order does not matter
    assert f1("Paris", "Berlin") == 0.0
    assert 0 < f1("a red bike", "red bike and a helmet") < 1


def test_adversarial_rule():
    assert score("Not mentioned.", "anything", "adversarial") == 1.0
    assert score("In Paris", "anything", "adversarial") == 0.0
    assert score("In Paris", "Paris", "single-hop") == f1("In Paris", "Paris")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all answer tests passed")

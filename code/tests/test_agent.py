"""Unit tests for the v0 Memory Agent with FakeTier1 (no models, no onnxruntime). Run: python tests/test_agent.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dama_agent import AgentConfig, MemoryAgent, Store
from dama_agent.fake import FakeTier1
from dama_agent.tools import TOOLS, dispatch


def agent(**cfg):
    cfg.setdefault("op_min_similarity", 0.1)   # bag-of-words fake embeddings give low cosines
    return MemoryAgent(FakeTier1(), Store(), AgentConfig(**cfg))


def test_update_supersedes_and_packet_shows_both():
    a = agent()
    a.observe("u", "I live in Berlin with my partner", speaker="Ana", ts="2023-01-01")
    a.observe("u", "lol nice", speaker="Ana", ts="2023-01-02")                       # not salient: no fact
    r = a.observe("u", "I moved to Pune last month, I live in Pune now", speaker="Ana", ts="2023-05-01")
    assert r["op"] == "UPDATE", r
    facts = {f["id"]: f for f in a.store.all_facts("u")}
    assert [f["status"] for f in facts.values()] == ["superseded", "current"]
    p = a.packet("u", "Where does Ana live?", force=True)
    assert "outdated: see m:" in p["text"] and "Pune" in p["text"], p["text"]
    assert a.store.audit_counts("u") == {"ADD": 1, "UPDATE": 1}


def test_delete_noop_and_speakers():
    a = agent()
    a.observe("u", "My cat is called Miso and she is grey", speaker="Ana")
    assert a.observe("u", "My cat is called Miso and she is grey", speaker="Ana")["op"] == "NOOP"
    assert a.observe("u", "My cat is called Miso and she is grey", speaker="Ben")["op"] == "ADD"   # another person
    assert a.observe("u", "Please forget my cat Miso grey", speaker="Ana")["op"] == "DELETE"
    statuses = sorted(f["status"] for f in a.store.all_facts("u"))
    assert statuses == ["current", "deleted"], statuses


def test_need_gate_scopes_and_not_found():
    a = agent()
    assert a.packet("u", "Write a haiku about rain")["skipped"]
    assert a.packet("u", "Where do I live?")["text"] == "NOT_FOUND"
    a.observe("other", "I live in Oslo near the fjord", speaker="Ana")
    assert a.packet("u", "Where do I live?", force=True)["text"] == "NOT_FOUND"     # scopes do not leak


def test_deadline_degrades_but_answers():
    a = agent(deadline_ms=0.0, use_rerank=True)
    a.observe("u", "The project database is PostgreSQL version 16", speaker="Ana")
    p = a.packet("u", "Which database does the project use?", force=True)
    assert p["degraded"] and "PostgreSQL" in p["text"], p


def test_tools_and_deferred_commit():
    a = agent()
    assert {t["name"] for t in TOOLS} == {"recall", "timeline", "note", "observe", "commit"}
    dispatch(a, "u", "observe", {"text": "I started learning the violin this spring", "speaker": "Ana"})
    assert a.store.episode_matrix("u")[0] == []                                       # queued, not written yet
    assert dispatch(a, "u", "commit", {})[0]["op"] == "ADD"
    assert "violin" in dispatch(a, "u", "recall", {"query": "what instrument is Ana learning violin"})["text"]
    assert dispatch(a, "u", "note", {"text": "ok", "speaker": "Ana"})["op"] == "ADD"   # notes skip the salience gate
    assert any("violin" in f["text"] for f in dispatch(a, "u", "timeline", {"query": "violin"}))


def test_cache_dedups_until_the_scope_changes():
    a = agent()
    a.observe("u", "Our launch is planned for March this year", speaker="Ana")
    first = a.packet("u", "When is the launch planned?", force=True)
    again = a.packet("u", "When is the launch planned?", force=True)
    assert not first["cached"] and again["cached"] and again["ids"] == first["ids"]
    a.observe("u", "We moved the launch to April now", speaker="Ana")             # a write invalidates
    assert not a.packet("u", "When is the launch planned?", force=True)["cached"]


def test_store_grows_and_recall_goes_deeper():
    a = agent(sal_threshold=1.1)                                                     # store episodes only, fast
    for i in range(150):                                                             # past the first 64-row buffer
        a.observe("u", f"Note number {i} about the garden project and tomatoes", speaker="Ana")
    ids, mat = a.store.episode_matrix("u")
    assert len(ids) == 150 and mat.shape[0] == 150
    assert len(a.packet("u", "garden project tomatoes", force=True)["ids"]) == 8
    assert len(a.recall("u", "garden project tomatoes", k=16)["ids"]) == 16


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all agent tests passed")

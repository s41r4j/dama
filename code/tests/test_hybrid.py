"""Unit tests for the hybrid read path (numpy only, fake models). Run: python tests/test_hybrid.py"""
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dama_agent import AgentConfig, MemoryAgent, Store
from dama_agent.fake import FakeTier1
from dama_agent.hybrid import HybridConfig, HybridReader, SparseIndex, _top
from dama_train.data.bm25 import BM25

DOCS = ["Caroline went to the LGBTQ support group on 7 May 2023.",
        "Melanie painted a sunrise over the lake last year.",
        "Hey Melanie! How are you doing today?",
        "I moved to Pune last month and I love it.",
        "My cat Miso is grey and very lazy.",
        "Caroline is researching adoption agencies for her future family.",
        "We went camping near the mountains with the kids in June.",
        "Did you see the new pottery class schedule?"]
FAKE = FakeTier1()


def reader(**cfg):
    r = HybridReader(lambda qs: FAKE.embed(qs, query=True), config=HybridConfig(**cfg))
    r.add(DOCS, FAKE.embed(DOCS), prev=[-1, -1, 1, -1, -1, 0, -1, -1])
    return r


def test_sparse_bm25_matches_dama_train():
    s = SparseIndex(); s.add(DOCS[:5]); s.add(DOCS[5:])     # appended in two steps
    ref = BM25(DOCS)
    for q in ["When did Caroline go to the support group?", "Melanie lake painting", "nothing matches xyzzy"]:
        assert np.allclose(s.bm25(q), ref.scores(q), atol=1e-4), q


def test_entity_overlap_counts_capitalised_words():
    s = SparseIndex(); s.add(DOCS)
    e = s.entity_overlap("What did Caroline and Melanie do in 2023?")
    assert e[0] == 2 and e[1] == 1 and e[3] == 0, e      # Caroline + 2023; Melanie


def test_top_breaks_ties_with_second_key():
    scores = np.array([1, 3, 3, 0, 3], dtype=float)
    tie = np.array([0, 0.1, 0.9, 0, 0.5])
    assert _top(scores, 2, tie=tie).tolist() == [2, 4]
    assert _top(scores, 10, positive=True).tolist()[:3] == [1, 2, 4] and 3 not in _top(scores, 10, positive=True)


def test_read_packs_by_probability_within_budget():
    r = reader(budget_tokens=30, rerank_n=0)
    out = r.read("When did Caroline go to the LGBTQ support group?")
    assert out["order"][0] == 0, out
    assert out["tokens"] <= 30 and out["budget_limited"]
    assert out["ids"] == sorted(out["ids"])


def test_scorer_reorders_the_top_n():
    def scorer(qs, ds):   # prefers anything about pottery
        return np.array([0.99 if "pottery" in d else 0.01 for d in ds])
    r = reader(rerank_n=8, budget_tokens=1000)
    r.scorer = scorer
    out = r.read("What class did they talk about?")
    assert out["order"][0] == 7 and out["rescored"] == min(8, out["pool"]), out


def test_rescored_candidates_pack_before_unrescored():
    # DAMA-HT bug (2026-10-28): low calibrated probabilities on the re-scored top N sorted below max-scaled RRF values
    # of candidates the cross-encoder never saw. Re-scored candidates must come first.
    def low(qs, ds):
        return np.full(len(qs), 0.01)
    r = reader(rerank_n=2, budget_tokens=1000)          # no prior: the rest of the pool keeps scaled RRF (up to 1.0)
    r.scorer = low
    out = r.read("Caroline support group adoption")
    assert out["rescored"] == 2 and out["pool"] > 2, out
    first_two = set(out["order"][:2])
    r.cfg = HybridConfig(rerank_n=2, budget_tokens=1000, rescored_first=False)
    old = r.read("Caroline support group adoption")
    assert set(old["order"][:2]) != first_two, (old, out)  # the old order let unrescored candidates jump ahead
    pool, feats, _ = r.pool("Caroline support group adoption", FAKE.embed(["Caroline support group adoption"], query=True)[0])
    top2 = set(pool[np.argsort(-feats[:, 2], kind="stable")[:2]].tolist())
    assert first_two == top2, (first_two, top2)


def test_v2_features():
    from dama_agent.hybrid import FEATURES, FEATURES_V2
    r = HybridReader(lambda qs: FAKE.embed(qs, query=True), config=HybridConfig(features="v2", rerank_n=0))
    spk = ["Caroline", "Melanie", "Caroline", "Melanie", "Caroline", "Melanie", "Caroline", "Melanie"]
    r.add(DOCS, FAKE.embed(DOCS), prev=[-1, -1, 1, -1, -1, 0, -1, -1], speakers=spk)
    assert r.next[0] == 5 and r.next[1] == 2 and r.next[3] == -1
    q = "When did Caroline go to the support group?"
    idx, feats, _ = r.pool(q, FAKE.embed([q], query=True)[0])
    assert feats.shape == (len(idx), len(FEATURES_V2)) and len(FEATURES_V2) == len(FEATURES) + 8
    col = {n: feats[:, k] for k, n in enumerate(FEATURES_V2)}
    row = {int(i): j for j, i in enumerate(idx.tolist())}
    assert col["speaker_in_query"][row[0]] == 1 and col["speaker_in_query"][row[1]] == 0
    assert col["turn_has_time"][row[0]] == 1 and col["temporal_query_x_time"][row[0]] == 1    # "7 May 2023"; "When ..."
    assert 0 < col["query_term_coverage"][row[0]] <= 1
    assert col["position"].min() >= 0 and col["position"].max() <= 1
    v1 = reader(rerank_n=0)
    i1, f1, _ = v1.pool(q, FAKE.embed([q], query=True)[0])
    assert f1.shape[1] == len(FEATURES) and np.allclose(f1, feats[:, :len(FEATURES)])   # v2 extends v1, never changes it


def test_neighbours_bring_previous_turn():
    r = reader(rerank_n=0, neighbours=1, budget_tokens=1000, dense_k=1, bm25_k=1, entity_k=0)
    out = r.read("adoption agencies")
    assert out["order"][:2] == [5, 0], out          # hit 5, then its previous turn 0


def test_lambda_admission_and_read_many():
    r = reader(rerank_n=0, lam=0.999)
    assert r.read("Pune")["admitted"] <= 1
    r.cfg = HybridConfig(rerank_n=0)
    many = r.read_many(["Pune", "cat Miso", "camping"])
    one = [r.read(q) for q in ["Pune", "cat Miso", "camping"]]
    assert [m["order"] for m in many] == [o["order"] for o in one]


def test_deadline_shrinks_rerank():
    def slow(qs, ds):
        time.sleep(0.002 * len(qs)); return np.full(len(qs), 0.5)
    r = reader(rerank_n=8, deadline_ms=10.0)
    r.scorer = slow
    r.read("Caroline")                                # learns ~2 ms per pair
    out = r.read("Melanie")
    assert out["rescored"] < 8, out


def test_agent_hybrid_path():
    a = MemoryAgent(FakeTier1(), Store(), AgentConfig(hybrid=HybridConfig(rerank_n=4, budget_tokens=200), op_min_similarity=0.1))
    for i, t in enumerate(DOCS):
        a.observe("u", t, speaker="Ana", session="s1", ts=f"2023-05-0{i + 1}")
    p = a.packet("u", "Where did Caroline go for support?", force=True)
    assert "support group" in p["text"] and p["pool"] >= 1, p
    a.observe("u", "Caroline joined a second support group in Oslo.", speaker="Ana", session="s2")
    p2 = a.packet("u", "Which support group did Caroline join in Oslo?", force=True)
    assert "Oslo" in p2["text"], p2["text"]           # the reader picked up the new episode


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

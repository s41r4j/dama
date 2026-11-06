"""DAMA paper v2: the CPU part of the experiment. No GPU, no LLM, no side effects at import; runs locally and on Kaggle.

Rules this module enforces in code (plan and pre-registration: README.md next to this file):
- Data splits are fixed here and checked: LoCoMo dev = conversations 0-3, test = 4-9 (regression only: inspected in
  rounds 1-2). LongMemEval-S: dev = the 218 questions v1 already sampled and reported (seed 13, replicated exactly and
  checked against v1's rows); holdout = the other questions, never run by any DAMA experiment before v2. Excluded from
  the holdout: a question whose base id (without "_abs") has a sibling in dev, and a question whose answer session
  appears in any dev history. Filler sessions shared with dev histories (LongMemEval-S draws them from one pool) are
  counted in the leakage audit, not excluded: they carry no answer, and v2 learns only generic ranking features.
- Everything that fits or selects (`fit_prior`, `select_*`, `sufficiency_probe`) refuses non-dev questions.
- Holdout rows are produced once, by the Kaggle runner, after the frozen config is written.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 13
DEV_CONVS, TEST_CONVS = (0, 1, 2, 3), (4, 5, 6, 7, 8, 9)
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"   # the file every earlier round used
LME_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"      # longmemeval_s_cleaned.json, 2026-10-28
LME_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LME_PER_TYPE, LME_EXTRA, LME_TURN_CHARS = 20, {"knowledge-update": 78, "temporal-reasoning": 60}, 2000   # v1's sampler
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
POOL = {"dense_k": 256, "bm25_k": 32, "entity_k": 16}       # chosen on LoCoMo dev in v1; kept fixed (see README)
NEIGHBOURS, BUDGET, BUDGET_COMPACT = 4, 3000, 1500           # v1 dev choices; 1500 is reported for the frontier only


def code_path():
    """dama_agent / dama_train: the attached code dataset on Kaggle, else this repo's code/ folder."""
    import glob
    hits = glob.glob("/kaggle/input/**/dama_train/__init__.py", recursive=True)
    return str(Path(min(hits, key=len)).parents[1]) if hits else str(Path(__file__).resolve().parents[2] / "code")


sys.path.insert(0, code_path())
from dama_agent.hybrid import FEATURES, FEATURES_V2, HybridConfig, HybridReader   # noqa: E402
from dama_agent.timeparse import annotate                                       # noqa: E402
from dama_train.data.splits import hashed_split                                 # noqa: E402

_ENC = None


def ntok(s: str) -> int:
    """Token count used for budgets and reporting: tiktoken o200k_base, as in rounds 1-2. Text that looks like a special
    token ("<|endoftext|>" occurs in LongMemEval chats) is counted as plain text instead of raising."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(s, disallowed_special=()))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:                     # streamed: LongMemEval-S is 277 MB
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- data
MONTHS = "January February March April May June July August September October November December".split()


def fmt_date(raw):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", raw or "")
    return f"{int(m.group(1))} {m.group(2)} {m.group(3)}" if m else (raw or "")


def load_locomo(path):
    """Identical to dama_paper_v0.load_data (same qids, evidence repair and turn ids), plus the split check."""
    from dama_train.data.locomo import parse_evidence
    if sha256(path) != LOCOMO_SHA256:
        raise ValueError("locomo10.json differs from the file used in rounds 1-2")
    raw = json.loads(Path(path).read_text())
    convs, questions = [], []
    for ci, s in enumerate(raw):
        conv = s["conversation"]
        sess = sorted((int(m.group(1)), k) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)))
        turns = []
        for n, key in sess:
            date = fmt_date(conv.get(f"session_{n}_date_time", ""))
            for t in conv[key]:
                text = (t.get("text") or "").strip()
                if t.get("blip_caption"):
                    text = f"{text} [shares a photo: {t['blip_caption']}]".strip()
                if not text:
                    continue
                dia = f"D{int(t['dia_id'].split(':')[0][1:])}:{int(t['dia_id'].split(':')[1])}"
                turns.append({"i": len(turns), "dia": dia, "speaker": t["speaker"], "text": text, "date": date,
                              "date_raw": conv.get(f"session_{n}_date_time", ""), "session": n})
        known = {t["dia"] for t in turns}
        convs.append({"ci": ci, "sid": s["sample_id"], "turns": turns})
        for qi, qa in enumerate(s["qa"]):
            kept = [e for e in parse_evidence(qa.get("evidence") or []) if e in known]
            questions.append({"qid": f"{s['sample_id']}:q{qi}", "ci": ci, "split": "dev" if ci in DEV_CONVS else "test",
                              "cat": int(qa.get("category", 0)), "question": qa["question"].strip(),
                              "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))), "evidence": kept, "bench": "locomo"})
    return convs, questions


def _lme_date_key(d):
    return re.sub(r"\s*\([A-Za-z]+\)\s*", " ", d or "").strip()


def lme_base(qid: str) -> str:
    return qid.removesuffix("_abs")


def _iter_lme(path):
    """The questions one at a time (ijson, if installed: a 3.8 GB machine cannot hold the parsed 277 MB file), else all."""
    try:
        import ijson
    except ImportError:
        yield from json.loads(Path(path).read_text())
        return
    with open(path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def lme_splits(path):
    """Light pass (ids, types, session ids): each question's split. dev = exactly v1's seeded sample; holdout = the
    rest, minus _abs siblings of dev questions and questions whose answer session sits in a dev history."""
    if sha256(path) != LME_SHA256:
        raise ValueError("longmemeval_s_cleaned.json differs from the pinned file")
    meta = [{"qid": it["question_id"], "type": it["question_type"], "sessions": list(it["haystack_session_ids"]),
             "answers": list(it.get("answer_session_ids") or [])} for it in _iter_lme(path)]
    by = defaultdict(list)
    for m in meta:
        by[m["type"]].append(m)
    rng, dev = random.Random(SEED), set()
    for t in sorted(by):
        dev |= {m["qid"] for m in rng.sample(by[t], min(max(LME_PER_TYPE, LME_EXTRA.get(t, 0)), len(by[t])))}
    dev_bases = {lme_base(q) for q in dev}
    dev_sessions = {s for m in meta if m["qid"] in dev for s in m["sessions"]}
    for m in meta:
        q = m["qid"]
        m["split"] = "dev" if q in dev else ("excluded_sibling" if lme_base(q) in dev_bases else
                                            "excluded_answer_in_dev" if set(m["answers"]) & dev_sessions else "holdout")
        m["l12_split"] = hashed_split(f"lme:{lme_base(q)}")
    return meta


def load_lme(path, want=("dev", "holdout"), limit=None, meta=None):
    """The questions whose split is in `want` (at most `limit` per split, in file order) in v1's turn format."""
    meta = meta or lme_splits(path)
    split = {m["qid"]: m for m in meta}
    keep, per = set(), Counter()
    for m in meta:
        if m["split"] in want and (limit is None or per[m["split"]] < limit):
            keep.add(m["qid"]); per[m["split"]] += 1
    items = []
    for it in _iter_lme(path):
        qid = it["question_id"]
        if qid not in keep:
            continue
        sess = sorted(zip(it["haystack_dates"], it["haystack_session_ids"], it["haystack_sessions"]), key=lambda x: _lme_date_key(x[0]))
        turns = []
        for date, sid, ts in sess:
            for j, t in enumerate(ts):
                turns.append({"i": len(turns), "dia": f"{sid}:{j}", "date": date, "date_raw": date, "speaker": t.get("role", "user"),
                              "text": str(t.get("content", ""))[:LME_TURN_CHARS], "session": sid, "has": bool(t.get("has_answer"))})
        items.append({"qid": qid, "type": it["question_type"], "question": it["question"], "gold": str(it["answer"]),
                      "date": it.get("question_date", ""), "turns": turns, "evidence": [t["dia"] for t in turns if t["has"]],
                      "abs": qid.endswith("_abs"), "split": split[qid]["split"], "l12_split": split[qid]["l12_split"],
                      "answer_sessions": split[qid]["answers"], "raw": {k: it[k] for k in ("question_id", "question_type", "question", "answer")},
                      "bench": "lme"})
    return items


def leakage_audit(meta, v1_dev_qids=None):
    """What the holdout shares with everything fitted or inspected before it (from lme_splits' light records)."""
    dev = [m for m in meta if m["split"] == "dev"]
    hold = [m for m in meta if m["split"] == "holdout"]
    dev_sess = {s for m in dev for s in m["sessions"]}
    dev_ans = {s for m in dev for s in m["answers"]}
    out = {"n": Counter(m["split"] for m in meta),
           "holdout_types": Counter(m["type"] for m in hold),
           "holdout_l12_partition": Counter(m["l12_split"] for m in hold),
           "holdout_share_of_history_sessions_also_in_dev_histories": float(np.mean([np.mean([s in dev_sess for s in m["sessions"]]) for m in hold])) if hold else None,
           "holdout_questions_whose_answer_session_is_in_a_dev_history": int(sum(any(s in dev_sess for s in m["answers"]) for m in hold)),
           "holdout_questions_whose_answer_session_is_a_dev_answer_session": int(sum(any(s in dev_ans for s in m["answers"]) for m in hold))}
    if v1_dev_qids is not None:
        mine = {it["qid"] for it in dev}
        out["dev_matches_v1_sample"] = mine == set(v1_dev_qids)
        if not out["dev_matches_v1_sample"]:
            raise ValueError(f"replicated LME dev sample differs from v1's: {len(mine ^ set(v1_dev_qids))} questions")
    out["n"], out["holdout_types"], out["holdout_l12_partition"] = dict(out["n"]), dict(out["holdout_types"]), dict(out["holdout_l12_partition"])
    return out


# ----------------------------------------------------------------------------------------------- rendering + readers
def render(t, time_notes=False):
    """v0's line format; with time_notes, relative time expressions carry their resolved date (write-time, Tier 0)."""
    text = annotate(t["text"], t["date_raw"]) if time_notes else t["text"]
    return f"[m:{t['dia']}] ({t['date']}) {t['speaker']}: {text}"


def render_many(turns, time_notes=False):
    return "\n".join(render(t, time_notes) for t in sorted(turns, key=lambda t: t["i"]))


def pair_text(t):
    return f"{t['speaker']}: {t['text']}"


def prev_index(turns):
    out, last = [], {}
    for j, t in enumerate(turns):
        s = t.get("session") or t["dia"].split(":")[0]
        out.append(last.get(s, -1)); last[s] = j
    return out


def key_texts(turns):
    """v1's dense key "prev" (chosen on LoCoMo dev): start of the previous turn of the session, then speaker: text."""
    prev = prev_index(turns)
    return [(turns[p]["text"][:200] + "\n" if p >= 0 else "") + pair_text(t) for t, p in zip(turns, prev)]


def build_reader(turns, vecs, encode_query, features="v1", prior=None, scorer=None, **cfg):
    """One HybridReader per history, on plain lines. Time notes are added when the chosen turns are rendered, after
    packing (`render_many(..., time_notes=True)`), so they cannot change what is retrieved or packed: an arm with and
    one without notes differ only in the packet text. Their few extra tokens are counted in each row's input_tokens."""
    r = HybridReader(encode_query, scorer=scorer, prior=prior,
                     config=HybridConfig(**{**POOL, "neighbours": NEIGHBOURS, "budget_tokens": BUDGET, "rerank_n": 0, "features": features, **cfg}))
    r.add([render(t) for t in turns], vecs, pair_texts=[pair_text(t) for t in turns],
          ntok=[ntok(render(t)) + 1 for t in turns], prev=prev_index(turns), speakers=[t["speaker"] for t in turns])
    return r


def covered(q, turns, ids):
    return set(q["evidence"]) <= {turns[int(i)]["dia"] for i in ids}


def labelled(q):
    """Questions with a retrieval target: LoCoMo categories 1-4 with evidence; LongMemEval non-abstention with evidence."""
    return bool(q["evidence"]) and (q.get("cat") != 5 if q["bench"] == "locomo" else not q["abs"])


def _dev_only(qs):
    bad = [q["qid"] for q in qs if q["split"] != "dev"]
    if bad:
        raise PermissionError(f"{len(bad)} non-dev questions passed to a fitting/selection step (e.g. {bad[:3]})")


# ----------------------------------------------------------------------------------------------- priors (fit on dev only)
def fold_of(q, k=4):
    """Group folds: LoCoMo by conversation, LongMemEval by hashed base question id (an _abs twin stays with its base)."""
    if q["bench"] == "locomo":
        return DEV_CONVS.index(q["ci"]) % k
    return int(hashlib.sha256(lme_base(q["qid"]).encode()).hexdigest()[:8], 16) % k


def pool_pairs(qs, reader_of, qvec_of, features):
    X, y, g = [], [], []
    for q in qs:
        r, turns = reader_of(q)
        idx, feats, _ = r.pool(q["question"], qvec_of(q), r.cfg if r.cfg.features == features else HybridConfig(**{**r.cfg.__dict__, "features": features}))
        ev = set(q["evidence"])
        X.append(feats); y += [int(turns[int(i)]["dia"] in ev) for i in idx]; g += [q["qid"]] * len(idx)
    return np.concatenate(X) if X else np.zeros((0, len(FEATURES_V2 if features == "v2" else FEATURES))), np.array(y), g


def make_model(kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=SEED)
    raise ValueError(kind)


def fit_prior(qs, reader_of, qvec_of, features, kind):
    _dev_only(qs)
    X, y, _ = pool_pairs([q for q in qs if labelled(q)], reader_of, qvec_of, features)
    return make_model(kind).fit(X, y)


def packet_ids(r, q, qvec, **kw):
    return r.read_many([q["question"]], qvecs=[qvec], **kw)[0]["ids"]


def oof_coverage(qs, reader_of, qvec_of, features, kind, budgets=(BUDGET_COMPACT, BUDGET), k=4):
    """Out-of-fold all-evidence coverage of the packed packet: each fold's questions are packed by a prior fitted on
    the other folds only. Returns {budget: {bench: coverage}} and per-question hits."""
    _dev_only(qs)
    qs = [q for q in qs if labelled(q)]
    hits = {B: {} for B in budgets}
    for f in range(k):
        train = [q for q in qs if fold_of(q, k) != f]
        held = [q for q in qs if fold_of(q, k) == f]
        if not held:
            continue
        model = fit_prior(train, reader_of, qvec_of, features, kind)
        for q in held:
            r, turns = reader_of(q)
            old = (r.prior, r.cfg)
            r.prior, r.cfg = model, HybridConfig(**{**r.cfg.__dict__, "features": features})
            for B in budgets:
                hits[B][q["qid"]] = covered(q, turns, packet_ids(r, q, qvec_of(q), budget_tokens=B))
            r.prior, r.cfg = old
    by = {B: {b: float(np.mean([hits[B][q["qid"]] for q in qs if q["bench"] == b and q["qid"] in hits[B]]))
              for b in sorted({q["bench"] for q in qs})} for B in budgets}
    return by, hits


def pool_coverage(qs, reader_of, qvec_of):
    return {b: float(np.mean([covered(q, reader_of(q)[1], reader_of(q)[0].pool(q["question"], qvec_of(q))[0]) for q in qs if labelled(q) and q["bench"] == b]))
            for b in sorted({q["bench"] for q in qs if labelled(q)})}


def select_prior(qs, reader_of, qvec_of, candidates=(("v1", "lr"), ("v2", "lr"), ("v2", "gbm")), log=print):
    """Pre-registered rule: highest mean (over benchmarks) out-of-fold all-evidence coverage at B=3000; ties (within
    0.5 point) go to the earlier, simpler candidate. Returns (choice, table)."""
    _dev_only(qs)
    table = {}
    for feats, kind in candidates:
        by, _ = oof_coverage(qs, reader_of, qvec_of, feats, kind)
        table[f"{feats}|{kind}"] = by
        log(f"prior {feats}|{kind}: OOF coverage {json.dumps(by)}")
    score = {c: float(np.mean(list(v[BUDGET].values()))) for c, v in table.items()}
    best = max(score.values())
    choice = next(c for c in table if score[c] >= best - 0.005)
    return choice, {"oof_coverage": table, "mean_at_B": score, "rule": "highest mean OOF all-evidence coverage at B=3000 over benchmarks; ties within 0.5 pt -> simpler"}


# ----------------------------------------------------------------------------------------------- E4: sufficiency probe
_WH = ("what", "when", "where", "who", "which", "why", "how", "did", "does", "is", "was", "do", "are", "has", "can", "would")


def query_features(q):
    w = q["question"].lower().split()
    first = w[0] if w else ""
    return [float(first == x) for x in _WH] + [len(w), sum(c.isdigit() for c in q["question"]), sum(x[:1].isupper() for x in q["question"].split()),
                                               float("how many" in q["question"].lower()), float("how long" in q["question"].lower())]


def packet_features(p_sorted, packed_mask, ntoks, budget):
    """Shape of the pool's probabilities and what the packet kept: top values, mass inside and outside the packet."""
    top = list(p_sorted[:10]) + [0.0] * max(0, 10 - len(p_sorted))
    inside, outside = float(np.sum(p_sorted[packed_mask])), float(np.sum(p_sorted[~packed_mask]))
    return top + [inside, outside, outside / (inside + outside + 1e-9), float(np.sum(ntoks[packed_mask]) / budget), float(len(p_sorted))]


def sufficiency_probe(rows):
    """rows: dev dicts {qid, bench, fold, label, qf, pf}. Grouped-CV AUROC of (i) query-only, (ii) query + packet features.
    If (ii) barely beats (i), a learned 'sufficiency' head would mostly read question style (review §7, E4)."""
    from sklearn.metrics import roc_auc_score
    _dev_only([{"qid": r["qid"], "split": r["split"]} for r in rows])
    y = np.array([r["label"] for r in rows]); folds = np.array([r["fold"] for r in rows])
    out = {"n": len(rows), "positive_rate": float(y.mean()) if len(y) else None}
    for name, X in (("query_only", np.array([r["qf"] for r in rows])), ("query_plus_packet", np.array([r["qf"] + r["pf"] for r in rows])),
                    ("packet_only", np.array([r["pf"] for r in rows]))):
        p = np.zeros(len(y))
        for f in sorted(set(folds)):
            tr, te = folds != f, folds == f
            if len(set(y[tr])) < 2:
                continue
            p[te] = make_model("lr").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out[name] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    if out.get("query_only") is not None and out.get("query_plus_packet") is not None:
        out["packet_gain"] = out["query_plus_packet"] - out["query_only"]
        out["decision"] = ("build a sufficiency controller: packet features add >= 0.03 AUROC" if out["packet_gain"] >= 0.03
                           else "do not build: packet features add < 0.03 AUROC over the question alone")
    return out


# ----------------------------------------------------------------------------------------------- statistics
def boot_ci(x, n=10000, seed=SEED):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return (None, None)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def mcnemar_p(a, b):
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    from math import comb
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    if not n:
        return 1.0
    k = min(n01, n10)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired(a_rows, b_rows, key="ds_correct", cluster="ci", seed=SEED):
    """A - B on shared judged questions: mean, item bootstrap CI, cluster bootstrap CI (by conversation for LoCoMo),
    exact McNemar p, per-cluster means and a one-sided sign-flip p over clusters (its floor is 1/2^clusters)."""
    b = {r["qid"]: r for r in b_rows}
    pairs = [(float(r[key]), float(b[r["qid"]][key]), r.get(cluster)) for r in a_rows
             if r.get(key) is not None and r["qid"] in b and b[r["qid"]].get(key) is not None and not r.get("excluded")]
    if not pairs:
        return None
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs]); d = x - y
    out = {"n": len(d), "a": float(x.mean()), "b": float(y.mean()), "diff": float(d.mean()), "ci": boot_ci(d, seed=seed),
           "discordant": float(np.mean(d != 0)), "mcnemar_p": mcnemar_p(x.astype(bool), y.astype(bool))}
    groups = [p[2] for p in pairs]
    if cluster and all(g is not None for g in groups) and len(set(groups)) > 1:
        cl = sorted(set(groups)); by = {c: d[np.array([g == c for g in groups])] for c in cl}
        rng = np.random.default_rng(seed)
        boots = [np.concatenate([by[c] for c in rng.choice(cl, len(cl))]).mean() for _ in range(10000)]
        means = [float(by[c].mean()) for c in cl]
        flips = [np.mean([(1 if (s >> i) & 1 else -1) * m for i, m in enumerate(means)]) for s in range(1 << len(cl))] if len(cl) <= 16 else []
        out.update({"cluster_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                    "per_cluster": dict(zip([str(c) for c in cl], means)), "clusters_positive": int(sum(m > 0 for m in means)), "clusters": len(cl),
                    "signflip_p_one_sided": float(np.mean(np.array(flips) >= np.mean(means) - 1e-12)) if flips else None})
    return out


def power_n(discordant, delta, z_a=1.96, z_b=0.8416):
    """Approximate questions needed for a paired McNemar test (alpha 0.05 two-sided, power 0.8)."""
    return math.ceil((z_a + z_b) ** 2 * discordant / delta ** 2) if delta else None


def audit_sheet(rows_by_method, per_method=24, seed=SEED):
    """Blind human-audit items, stratified by method (and category inside it); returns (sheet rows, key rows)."""
    rng, items = random.Random(seed), []
    for m in sorted(rows_by_method):
        rows = [r for r in rows_by_method[m] if r.get("cat", 0) != 5]
        by = defaultdict(list)
        for r in rows:
            by[r.get("cat") or r.get("type")].append(r)
        k = max(1, per_method // max(1, len(by)))
        for c in sorted(by, key=str):
            items += [(m, r) for r in rng.sample(by[c], min(k, len(by[c])))]
    rng.shuffle(items)
    sheet = [{"id": f"A{i + 1:03d}", "method": "hidden", "question": r["question"], "gold": r["gold"], "answer": r["answer"], "human_label": ""}
             for i, (m, r) in enumerate(items)]
    key = [{"id": f"A{i + 1:03d}", "method": m, "qid": r["qid"], "qwen_label": r.get("correct", r.get("qwen_correct")), "deepseek_label": r.get("ds_correct")}
           for i, (m, r) in enumerate(items)]
    return sheet, key

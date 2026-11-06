"""DAMA paper v2 (2026-10-28): a new private Kaggle notebook `s41r4j/dama-paper-v2` (T4 x2). Plan, pre-registered
hypotheses and cost: README.md next to this file. The CPU logic (data, splits, leakage audit, selection, statistics) is
in v2lib.py; this script adds the GPU work (embeddings, cross-encoders, the reader) and the report.

What is new against round 2 (DAMA-H):
- V2_fast: DAMA-H's union pool + a prior chosen on dev among {v1 features + LR, v2 features + LR, v2 features + GBM},
  re-scored candidates packed first (the DAMA-HT fix), and write-time time notes ("yesterday [= 16 November 2023]").
- V2_acc: V2_fast + a cross-encoder over the prior's top N (model and N chosen on dev): the accuracy track.
- A sealed LongMemEval-S holdout (282 questions no DAMA run has touched) is the confirmatory test; LoCoMo test is
  regression only (it shaped DAMA-H). Settings come from dev only; v2lib refuses non-dev data in fitting/selection.
- Diagnostics: an oracle ladder (gold evidence, with/without time notes), attribution arms, the DAMA-HT 2x2 repair
  (retrieval only), the E4 sufficiency probe, conversation-cluster statistics and a blind human-audit sheet.

Rules kept from rounds 1-2: every number comes from this run and is logged; one reader (Qwen2.5-7B-Instruct, fp16,
greedy) and one prompt per benchmark for every method; every LLM call cached in SQLite (round 1-2 caches are attached,
so repeated prompts cost nothing); seed 13. MODE "judge" re-judges saved answers with DeepSeek (website run, no GPU).
"""
from __future__ import annotations
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODE = "full"   # "full": GPU run. "judge": no GPU; carry this notebook's previous output forward, DeepSeek-judge every saved
                # answer, rewrite the report (needs the DEEPSEEK_API_KEY secret: start it from the website, see README)
SMOKE = False   # True: a few questions in every stage, to test the whole run cheaply first
HERE = Path(globals().get("__file__", "dama_paper_v2.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v1_2026-10-24/dama_paper_v1.py")))
sys.path.insert(0, str(_exp_dir("v2_2026-10-28/v2lib.py")))
import dama_paper_v1 as v1      # imports v0 (reader, LLM cache, Qwen judge) and reads the DeepSeek secret if it can
import v2lib as L
v0 = v1.v0
import numpy as np
import torch
from dama_agent.hybrid import HybridConfig, HybridReader
from dama_agent.fastmodels import OnnxEncoder

v0.SMOKE = v1.SMOKE = SMOKE
v0.ntok = L.ntok    # same tokenizer and counts; also accepts chat text containing "<|endoftext|>" (full run v2, 2026-10-28)
v0.TIME_LIMIT_H, v0.FINAL_LIMIT_H = (1.5, 2.0) if SMOKE else (9.0, 10.5)
v1.DS_CAP_USD = (0.05 if MODE == "judge" else 0.10) if SMOKE else 2.00
log, stage, OUT, D, STATE, SEED = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE, v0.SEED
FINAL_H = v0.FINAL_LIMIT_H
CE_CANDIDATES = [("cross-encoder/ms-marco-MiniLM-L6-v2", 256), ("BAAI/bge-reranker-base", 512), ("BAAI/bge-reranker-v2-m3", 512)]
CE_N = (64, 128)
CE_GPU_MINUTES = 25          # stop trying larger cross-encoders on dev once this much GPU time is spent
B3_CE = ("cross-encoder/ms-marco-MiniLM-L6-v2", 256)      # round 2's LongMemEval B3, unchanged
NOTES_DEV_SAMPLE = 16 if SMOKE else 150
NOTES_RULE_DROP = 0.03       # pre-registered: time notes stay on unless they cost >= 3 points on the dev sample
V2 = {"settings": {}, "dev": {}, "leakage": {}, "lme": {}, "locomo": {}, "ht": {}, "latency": {}, "stats": {}, "notes": []}
H = {}                       # histories: key -> {"reader", "turns"}
EXPORT = Path("/tmp/dama_onnx_v2")


def save_v2():
    (OUT / "results_v2.json").write_text(json.dumps(V2, indent=1, default=str))


def file_sha(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def hkey(q):
    return ("locomo", q["ci"]) if q["bench"] == "locomo" else ("lme", q["qid"])


def reader_of(q):
    h = H[hkey(q)]
    return h["reader"], h["turns"]


def qvec_of(q):
    return D["qv"][q["qid"]]


# ----------------------------------------------------------------------------------------------- setup
@stage("v2_setup")
def stage_setup():
    from sentence_transformers import SentenceTransformer
    v0.stage_data()                                           # downloads locomo10.json into OUT, logs categories
    convs, lq = L.load_locomo(OUT / "locomo10.json")
    old = {q["qid"]: q["evidence"] for q in D["questions"]}
    if {q["qid"]: q["evidence"] for q in lq if q["qid"] in old} != old:
        raise ValueError("v2lib.load_locomo disagrees with dama_paper_v0.load_data")
    lme_path = Path(os.environ.get("DAMA_LME_PATH", "/tmp/longmemeval_s_cleaned.json"))   # 277 MB: kept out of the output
    if not lme_path.exists():
        import urllib.request
        urllib.request.urlretrieve(L.LME_URL, lme_path)
    meta = L.lme_splits(lme_path)
    v1_rows = sorted(glob.glob("/kaggle/input/**/paper/methods/LME_DAMA_H.jsonl", recursive=True))
    v1_dev = [json.loads(l)["qid"] for l in open(v1_rows[0])] if v1_rows else None
    V2["leakage"] = L.leakage_audit(meta, v1_dev)
    lme = L.load_lme(lme_path, limit=10 if SMOKE else None, meta=meta)
    if v1_dev is None:
        V2["notes"].append("v1's LongMemEval rows not attached: the dev sample was replicated but not checked against them")
    log("v2 leakage audit", V2["leakage"])
    if SMOKE:
        keep = []
        for ci in (0, 1, 4, 7):
            keep += [q for q in lq if q["ci"] == ci][:8]
        lq = keep
    D["convs"], D["lq"], D["lme"] = convs, lq, [it for it in lme if it["split"] in ("dev", "holdout")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D["bge"] = SentenceTransformer(v0.EMBED_ID, device=dev)
    try:
        sample = [q["question"] for q in lq[:50]]
        D["qenc"] = OnnxEncoder.export(v0.EMBED_ID, EXPORT / "bge", prefix=L.BGE_PREFIX, sample=sample, log=log)
        enc, kind = D["qenc"].encode, f"onnx {D['qenc'].variant}"
    except Exception as e:
        log("ONNX query encoder failed; sentence-transformers on CPU", type(e).__name__, str(e)[:300])
        cpu = SentenceTransformer(v0.EMBED_ID, device="cpu")
        enc, kind = (lambda qs: cpu.encode([L.BGE_PREFIX + q for q in qs], normalize_embeddings=True, convert_to_numpy=True)), "sentence-transformers cpu"
    D["encode_query"] = enc
    allq = lq + D["lme"]
    D["qv"] = dict(zip([q["qid"] for q in allq], enc([q["question"] for q in allq])))
    V2["settings"]["query_encoder"] = kind
    t = time.time()
    for c in convs:
        if any(q["ci"] == c["ci"] for q in lq):
            vec = D["bge"].encode(L.key_texts(c["turns"]), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
            H[("locomo", c["ci"])] = {"reader": L.build_reader(c["turns"], vec, enc), "turns": c["turns"]}
    for it in D["lme"]:
        vec = D["bge"].encode(L.key_texts(it["turns"]), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        H[("lme", it["qid"])] = {"reader": L.build_reader(it["turns"], vec, enc), "turns": it["turns"]}
    V2["settings"]["embedding_minutes"] = round((time.time() - t) / 60, 1)
    V2["settings"]["data"] = {"locomo_sha256": L.LOCOMO_SHA256, "lme_sha256": L.LME_SHA256, "locomo_questions": len(lq),
                              "lme": dict(Counter(it["split"] for it in D["lme"]))}
    log("v2 setup:", V2["settings"]["data"], "query encoder", kind)
    save_v2()


def dev_questions():
    return [q for q in D["lq"] if q["split"] == "dev"] + [it for it in D["lme"] if it["split"] == "dev"]


# ----------------------------------------------------------------------------------------------- selection (dev only)
@stage("v2_select_prior")
def stage_select_prior():
    dq = dev_questions()
    V2["dev"]["pool_coverage"] = L.pool_coverage(dq, reader_of, qvec_of)
    choice, table = L.select_prior(dq, reader_of, qvec_of, log=log)
    feats, kind = choice.split("|")
    D["feats"], D["prior"] = feats, L.fit_prior(dq, reader_of, qvec_of, feats, kind)
    # round 2's DAMA-H prior, refitted exactly as v1 did (v1 features, LR, all LoCoMo dev pairs): the LongMemEval DAMA-H arm
    D["prior_h"] = L.fit_prior([q for q in dq if q["bench"] == "locomo"], reader_of, qvec_of, "v1", "lr")
    V2["settings"]["prior"] = {"choice": choice, **table, "pool": L.POOL, "B": L.BUDGET, "neighbours": L.NEIGHBOURS,
                               "prior_h_coef": [round(float(c), 4) for c in D["prior_h"][-1].coef_[0]]}
    log("v2 prior:", choice, table["mean_at_B"], "DAMA-H refit coef", V2["settings"]["prior"]["prior_h_coef"])
    save_v2()


def ce_model(name, max_len):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device="cuda" if torch.cuda.is_available() else "cpu", max_length=max_len)


def ce_scorer(model):
    def score(qs, ds):
        return np.asarray(model.predict(list(zip(qs, ds)), batch_size=128, show_progress_bar=False), dtype=np.float64)
    return score


@stage("v2_select_ce")
def stage_select_ce():
    """Cross-encoder model and N for V2_acc on dev: highest mean (over benchmarks) all-evidence coverage at B=3000 with
    the dev prior ordering the rest of the pool (in-sample for every candidate alike); ties within 0.5 pt -> cheaper."""
    dq = [q for q in dev_questions() if L.labelled(q)]
    if SMOKE:
        dq = dq[:12]
    grid, t_gpu = {}, 0.0
    base = {}
    for q in dq:
        r, turns = reader_of(q)
        r.prior = D["prior"]
        base.setdefault(q["bench"], []).append(L.covered(q, turns, L.packet_ids(r, q, qvec_of(q), features=D["feats"])))
    grid["none|0"] = {b: float(np.mean(v)) for b, v in base.items()}
    for name, max_len in CE_CANDIDATES[:1] if SMOKE else CE_CANDIDATES:
        if t_gpu / 60 > CE_GPU_MINUTES:
            V2["notes"].append(f"{name} not tried on dev: CE GPU budget ({CE_GPU_MINUTES} min) spent"); continue
        try:
            ce = ce_model(name, max_len); sc = ce_scorer(ce)
            t = time.time(); look, pairs = {}, 0
            for q in dq:
                r, turns = reader_of(q)
                idx, feats, _ = r.pool(q["question"], qvec_of(q), HybridConfig(**{**r.cfg.__dict__, "features": D["feats"]}))
                p = r.prior_p(feats)
                top = idx[np.argsort(-p, kind="stable")[:max(CE_N)]]
                z = sc([q["question"]] * len(top), [r.pair_texts[int(i)] for i in top])
                look.update({(q["qid"], r.pair_texts[int(i)]): float(v) for i, v in zip(top, z)}); pairs += len(top)
            dt = time.time() - t; t_gpu += dt
            for n in CE_N:
                hits = defaultdict(list)
                for q in dq:
                    r, turns = reader_of(q)
                    r.scorer = (lambda qs, ds, qid=q["qid"]: np.array([look.get((qid, d), -1e9) for d in ds]))
                    hits[q["bench"]].append(L.covered(q, turns, L.packet_ids(r, q, qvec_of(q), rerank_n=n, features=D["feats"])))
                    r.scorer = None
                grid[f"{name}|{n}"] = {b: float(np.mean(v)) for b, v in hits.items()}
            V2["dev"].setdefault("ce_gpu_ms_per_pair", {})[name] = 1000 * dt / max(1, pairs)
            del ce; torch.cuda.empty_cache()
            log(f"CE {name}: dev coverage", {k: v for k, v in grid.items() if k.startswith(name)}, f"{1000 * dt / max(1, pairs):.2f} ms/pair")
        except Exception as e:
            V2["notes"].append(f"CE {name} failed on dev: {type(e).__name__}: {str(e)[:150]}"); log("CE failed", name, e)
    mean = {k: float(np.mean(list(v.values()))) for k, v in grid.items() if k != "none|0"}
    if not mean:
        raise RuntimeError("no cross-encoder could be scored on dev")
    best = max(mean.values())
    order = [f"{n}|{k}" for n, _ in CE_CANDIDATES for k in CE_N]             # cheaper model and smaller N first
    choice = next(k for k in order if k in mean and mean[k] >= best - 0.005)
    name, n = choice.split("|")
    none = float(np.mean(list(grid["none|0"].values())))
    if mean[choice] < none - 0.005:      # pre-registered: no cross-encoder beat the prior alone -> V2_acc is not run
        D["ce_name"] = None
        V2["notes"].append(f"V2_acc not run: best cross-encoder ({choice}, {mean[choice]:.3f}) is below the prior alone ({none:.3f}) on dev; H2 becomes V2_fast vs B3")
    else:
        D["ce_name"], D["ce_len"], D["ce_n"] = name, dict(CE_CANDIDATES)[name], int(n)
    V2["settings"]["ce"] = {"choice": choice if D["ce_name"] else None, "best_ce": choice, "dev_grid": grid, "mean": mean, "prior_alone": none,
                            "rule": "highest mean dev coverage at B=3000; ties within 0.5 pt -> cheaper model, smaller N; not used if below the prior alone"}
    log("v2 CE choice", choice, mean)
    save_v2()


@stage("v2_sufficiency")
def stage_sufficiency():
    """E4 on dev, out of fold: can the packet's shape predict 'all evidence is in the packet' beyond the question alone?"""
    dq = [q for q in dev_questions() if L.labelled(q)]
    kind = V2["settings"]["prior"]["choice"].split("|")[1]
    rows = []
    for f in range(4):
        model = L.fit_prior([q for q in dq if L.fold_of(q) != f], reader_of, qvec_of, D["feats"], kind)
        for q in [q for q in dq if L.fold_of(q) == f]:
            r, turns = reader_of(q)
            r.prior = model
            out = r.read_many([q["question"]], qvecs=[qvec_of(q)], features=D["feats"])[0]
            idx, feats, _ = r.pool(q["question"], qvec_of(q), HybridConfig(**{**r.cfg.__dict__, "features": D["feats"]}))
            p = r.prior_p(feats); o = np.argsort(-p)
            packed = np.array([int(i) in set(out["ids"]) for i in idx[o]])
            rows.append({"qid": q["qid"], "split": q["split"], "bench": q["bench"], "fold": f, "label": int(L.covered(q, turns, out["ids"])),
                         "qf": L.query_features(q), "pf": L.packet_features(p[o], packed, np.array([r.ntok[int(i)] for i in idx[o]]), L.BUDGET)})
    for q in dq:
        reader_of(q)[0].prior = D["prior"]
    V2["dev"]["sufficiency_probe"] = {"all": L.sufficiency_probe(rows),
                                      **{b: L.sufficiency_probe([r for r in rows if r["bench"] == b]) for b in ("locomo", "lme")}}
    log("v2 E4 sufficiency probe", V2["dev"]["sufficiency_probe"])
    save_v2()


@stage("v2_freeze")
def stage_freeze():
    code = Path(L.code_path()) / "dama_agent"
    frozen = {"prior": V2["settings"]["prior"]["choice"], "ce": V2["settings"].get("ce", {}).get("choice"), "pool": L.POOL,
              "B": L.BUDGET, "neighbours": L.NEIGHBOURS, "time_notes": D.get("time_notes", True), "reader": v0.READER_ID,
              "sha256": {"v2lib.py": file_sha(L.__file__), "dama_paper_v2.py": file_sha(globals().get("__file__", "")),
                         "hybrid.py": file_sha(code / "hybrid.py"), "timeparse.py": file_sha(code / "timeparse.py")},
              "frozen_at_min": round((time.time() - v0.T_START) / 60, 1)}
    (OUT / "FROZEN.json").write_text(json.dumps(frozen, indent=1))
    V2["settings"]["frozen"] = frozen
    log("v2 frozen before any test or holdout row:", frozen)


# ----------------------------------------------------------------------------------------------- contexts
def v2_ctx(q, acc=False, notes=True, budget=L.BUDGET):
    r, turns = reader_of(q)
    r.prior = D["prior"]
    r.scorer = D["ce_scorer"] if acc else None
    t = time.perf_counter()
    out = r.read_many([q["question"]], qvecs=[qvec_of(q)], budget_tokens=budget, features=D["feats"], rerank_n=D["ce_n"] if acc else 0)[0]
    ms = (time.perf_counter() - t) * 1000
    r.scorer = None
    chosen = [turns[i] for i in out["ids"]]
    return {"context": L.render_many(chosen, notes) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen], "read_ms": ms,
            "budget_limited": out["budget_limited"]}


def h_ctx_v1(q):
    """Round 2's DAMA-H, as run on LongMemEval in v1: v1 features, the LoCoMo-dev LR prior, no re-ranking, no notes."""
    r, turns = reader_of(q)
    r.prior = D["prior_h"]
    t = time.perf_counter()
    out = r.read_many([q["question"]], qvecs=[qvec_of(q)], features="v1", rerank_n=0)[0]
    chosen = [turns[i] for i in out["ids"]]
    return {"context": L.render_many(chosen) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen],
            "read_ms": (time.perf_counter() - t) * 1000}


def oracle_ctx(q, notes):
    r, turns = reader_of(q)
    ev = set(q["evidence"])
    chosen = [t for t in turns if t["dia"] in ev]
    return {"context": L.render_many(chosen, notes) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen]}


# ----------------------------------------------------------------------------------------------- reader runs
@stage("v2_load_reader")
def stage_load_reader():
    v1.load_reader()
    if D.get("ce_name"):
        D["ce"] = ce_model(D["ce_name"], D["ce_len"]); D["ce_scorer"] = ce_scorer(D["ce"])


@stage("v2_dev_notes")
def stage_dev_notes():
    """Pre-registered check on a fixed LoCoMo dev sample: time notes stay on unless they cost >= 3 points (Qwen judge)."""
    dev = [q for q in D["lq"] if q["split"] == "dev" and q["cat"] != 5]
    sample = random.Random(SEED).sample(dev, min(NOTES_DEV_SAMPLE, len(dev)))
    acc = {}
    for notes in (True, False):
        name = f"dev_V2_fast_notes{int(notes)}"
        v0.evaluate(name, sample, lambda q, n=notes: v2_ctx(q, notes=n))
        acc[notes] = v0.RESULTS[name]["acc"]
        V2["dev"].setdefault("notes_check", {})[name] = {k: v0.RESULTS[name][k] for k in ("acc", "acc_ci", "by_category", "mean_tokens")}
    D["time_notes"] = not (acc[True] <= acc[False] - NOTES_RULE_DROP)
    V2["dev"]["notes_check"]["decision"] = {"time_notes": D["time_notes"], "rule": "on unless with-notes accuracy is >= 3 points below without"}
    log("v2 dev time-notes check", acc, "->", D["time_notes"])
    save_v2()


def lme_eval(name, items, ctx_fn):
    t0 = time.time()
    ctxs = [ctx_fn(it) for it in items]
    msgs = [v1.lme_messages(it, c["context"]) for it, c in zip(items, ctxs)]
    outs, ptoks, ms = v0.READER(msgs, max_new=64)
    rows = []
    for it, c, m, o, pt in zip(items, ctxs, msgs, outs, ptoks):
        ev = set(it["evidence"])
        rows.append({"method": name, "qid": it["qid"], "type": it["type"], "abs": it["abs"], "split": it["split"], "question": it["question"],
                     "gold": it["gold"], "answer": o, "context": c["context"], "input_tokens": v0.ntok(m[0]["content"]) + v0.ntok(m[1]["content"]),
                     "reader_prompt_tokens": pt, "read_ms": c.get("read_ms"),
                     "evidence_all": (ev <= set(c.get("dia") or [])) if ev and c.get("dia") is not None else None})
    jq, _, _ = v0.READER([[{"role": "user", "content": v1.lme_judge_prompt(it["raw"], r["answer"])}] for it, r in zip(items, rows)], max_new=6)
    for r, o in zip(rows, jq):
        r["qwen_correct"] = "yes" in (o or "").lower()
    ds = v1.ds_client()
    if ds is not None:
        jd = ds.many([[{"role": "user", "content": v1.lme_judge_prompt(it["raw"], r["answer"])}] for it, r in zip(items, rows)], max_tokens=6)
        for r, o in zip(rows, jd):
            r["ds_correct"] = None if o is None else ("yes" in o.lower())
    write_rows(name, rows)
    D.setdefault("lme_rows", {})[name] = rows
    V2["lme"][name] = {**lme_summary(rows), "minutes": round((time.time() - t0) / 60, 1)}
    log(f"LME {name}:", {k: V2["lme"][name][k] for k in ("n", "acc_ds", "acc_qwen", "mean_tokens", "evidence_recall")})
    save_v2()


def write_rows(name, rows):
    with (OUT / "methods" / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _acc(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def lme_summary(rows):
    ev = [r["evidence_all"] for r in rows if r.get("evidence_all") is not None]
    rd = [r["read_ms"] for r in rows if r.get("read_ms") is not None]
    return {"n": len(rows), "acc_ds": _acc(rows, "ds_correct"), "acc_qwen": _acc(rows, "qwen_correct"),
            "ci_ds": L.boot_ci([float(r["ds_correct"]) for r in rows if r.get("ds_correct") is not None]),
            "ci_qwen": L.boot_ci([float(r["qwen_correct"]) for r in rows]),
            "by_type": {t: {"n": sum(r["type"] == t for r in rows), "ds": _acc([r for r in rows if r["type"] == t], "ds_correct"),
                            "qwen": _acc([r for r in rows if r["type"] == t], "qwen_correct")} for t in sorted({r["type"] for r in rows})},
            "mean_tokens": float(np.mean([r["input_tokens"] for r in rows])) if rows else None,
            "evidence_recall": float(np.mean(ev)) if ev else None, "read_p50": v0.pct(rd, 50)}


@stage("v2_lme_holdout")
def stage_lme_holdout():
    """The confirmatory test, first in the run so a time guard can never drop it."""
    hold = [it for it in D["lme"] if it["split"] == "holdout"]
    notes = D.get("time_notes", True)
    lme_eval("LME_V2_fast", hold, lambda it: v2_ctx(it, notes=notes))
    if D.get("ce_name"):
        lme_eval("LME_V2_acc", hold, lambda it: v2_ctx(it, acc=True, notes=notes))
    lme_eval("LME_DAMA_H", hold, h_ctx_v1)
    ce = ce_model(*B3_CE)
    def b3(it):
        turns = it["turns"]
        z = ce.predict([(it["question"], L.pair_text(t)) for t in turns], batch_size=256, show_progress_bar=False)
        chosen, used = [], 0
        for j in np.argsort(-np.asarray(z)):
            n = L.ntok(L.render(turns[j])) + 1
            if used + n <= L.BUDGET:
                chosen.append(turns[j]); used += n
        return {"context": L.render_many(chosen) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen]}
    lme_eval("LME_B3", hold, b3)
    del ce; torch.cuda.empty_cache()


def loco_eval(name, qs, ctx_fn):
    rows = v0.evaluate(name, qs, ctx_fn)
    V2["locomo"][name] = {k: v0.RESULTS[name].get(k) for k in ("n", "acc", "acc_ci", "by_category", "mean_tokens", "evidence_recall", "read_p50", "read_p95", "abstention")}
    save_v2()
    return rows


@stage("v2_locomo_main")
def stage_locomo_main():
    test = [q for q in D["lq"] if q["split"] == "test"]
    notes = D.get("time_notes", True)
    loco_eval("V2_fast", test, lambda q: v2_ctx(q, notes=notes))
    if D.get("ce_name"):
        loco_eval("V2_acc", test, lambda q: v2_ctx(q, acc=True, notes=notes))


def test_subset():
    test = [q for q in D["lq"] if q["split"] == "test"]
    D["questions"] = D["lq"]
    return v0.test_subset() if len(test) > 500 else test


@stage("v2_locomo_attribution")
def stage_attribution():
    """On round 1-2's fixed 501-question subset: what the time notes and the v2 prior each add."""
    sub = test_subset()
    loco_eval("V2_fast_no_notes", sub, lambda q: v2_ctx(q, notes=False))
    if D.get("ce_name"):
        loco_eval("V2_acc_no_notes", sub, lambda q: v2_ctx(q, acc=True, notes=False))


@stage("v2_oracle")
def stage_oracle():
    """Gold evidence only, with and without time notes (categories 1-4 with evidence): the reader's ceiling."""
    qs = [q for q in D["lq"] if q["split"] == "test" and q["cat"] != 5 and q["evidence"]]
    loco_eval("ORACLE_gold", qs, lambda q: oracle_ctx(q, False))
    loco_eval("ORACLE_gold_notes", qs, lambda q: oracle_ctx(q, True))


# ----------------------------------------------------------------------------------------------- DAMA-HT repair (CPU)
@stage("v2_ht")
def stage_ht():
    """Retrieval only, conversations 7-9 (unseen by L1.2). (a) round 2's DAMA-HT exactly, (b) the same with re-scored
    candidates first, (c) encoder x reranker 2x2 with a prior per arm fitted on conversations 5-6 (L1.2 val/calib: not
    trained on, and not scored here; they belong to the paper's test split, so this is a diagnostic, not a result)."""
    from dama_train.runtime import Tier1
    hits = glob.glob("/kaggle/input/**/dama_tier1.json", recursive=True)
    if not hits:
        raise FileNotFoundError("Tier-1 bundle not attached")
    t1 = Tier1(os.path.dirname(min(hits, key=len)), threads=4)
    def dated(t):
        return f"[{t['date_raw']}] {t['speaker']}: {t['text'][:1000]}"
    def rel(qs, ds):
        out = np.zeros(len(qs)); by = defaultdict(list)
        for i, q in enumerate(qs):
            by[q].append(i)
        for q, ii in by.items():
            out[ii] = t1.relevance(q, [ds[i] for i in ii])
        return out
    tuned_q = lambda qs: t1.embed(qs, query=True)
    convs = {c["ci"]: c for c in D["convs"]}
    all_q = L.load_locomo(OUT / "locomo10.json")[1]
    vec = {}
    for ci in (5, 6, 7, 8, 9):
        turns = convs[ci]["turns"]; texts = [dated(t) for t in turns]
        vec[("tuned", ci)] = np.concatenate([t1.embed([t1.memory_key(x, p) for x, p in zip(texts[s:s + 64], ([""] + texts)[s:s + 64])]) for s in range(0, len(texts), 64)])
        vec[("base", ci)] = D["bge"].encode(L.key_texts(turns), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    qs = {ci: [q for q in all_q if q["ci"] == ci and L.labelled(q)] for ci in (5, 6, 7, 8, 9)}
    if SMOKE:
        qs = {ci: v[:6] for ci, v in qs.items()}
    qv = {"base": lambda q: D["encode_query"]([q["question"]])[0], "tuned": lambda q: tuned_q([q["question"]])[0]}
    def readers(enc, orig=False):
        out = {}
        for ci in (5, 6, 7, 8, 9):
            turns = convs[ci]["turns"]
            r = HybridReader(tuned_q if enc == "tuned" else D["encode_query"], config=HybridConfig(**{**L.POOL, "neighbours": L.NEIGHBOURS, "budget_tokens": L.BUDGET}))
            pts = [dated(t) for t in turns] if orig else [L.pair_text(t) for t in turns]      # round 2 indexed dated text
            r.add([L.render(t) for t in turns], vec[(enc, ci)], pair_texts=pts, ntok=[L.ntok(L.render(t)) + 1 for t in turns],
                  prev=L.prev_index(turns), speakers=[t["speaker"] for t in turns])
            out[ci] = (r, turns)
        return out
    def rel_native(ci):
        """Tier-1 rel always reads its training format ([date] speaker: text), whatever the index text is."""
        native = {L.pair_text(t): dated(t) for t in convs[ci]["turns"]}
        return lambda qs, ds: rel(qs, [native.get(d, d) for d in ds])
    def coverage(rs, enc, prior, rerank, first=True):
        hits = []
        for ci in (7, 8, 9):
            r, turns = rs[ci]
            r.prior, r.scorer = prior, (rel_native(ci) if rerank else None)
            for q in qs[ci]:
                out = r.read_many([q["question"]], qvecs=[qv[enc](q)], rerank_n=32 if rerank else 0, rescored_first=first)[0]
                hits.append(L.covered(q, turns, out["ids"]))
        return float(np.mean(hits)) if hits else None
    res = {}
    orig = readers("tuned", orig=True)
    res["a_round2_DAMA_HT"] = coverage(orig, "tuned", None, True, first=False)
    res["b_round2_DAMA_HT_rescored_first"] = coverage(orig, "tuned", None, True, first=True)
    for enc in ("base", "tuned"):
        rs = readers(enc)
        X, y = [], []
        for ci in (5, 6):
            r, turns = rs[ci]
            for q in qs[ci]:
                idx, feats, _ = r.pool(q["question"], qv[enc](q))
                X.append(feats); y += [int(turns[int(i)]["dia"] in set(q["evidence"])) for i in idx]
        prior = L.make_model("lr").fit(np.concatenate(X), np.array(y))
        for rerank in (False, True):
            res[f"c_{enc}_encoder_reranker_{'on' if rerank else 'off'}"] = coverage(rs, enc, prior, rerank)
    V2["ht"] = {"packet_all_evidence_coverage_conv7_9": res, "n_questions": sum(len(qs[c]) for c in (7, 8, 9)),
                "round2_reported": {"DAMA_HT": 0.495, "DAMA_H": 0.805}}
    log("v2 DAMA-HT repair:", res)
    save_v2()


# ----------------------------------------------------------------------------------------------- latency (CPU)
@stage("v2_latency", limit_h=FINAL_H)
def stage_latency():
    """V2_fast on CPU as deployed (ONNX query encoder, numpy pool, sklearn prior): LoCoMo stores, a 10^5-turn store, and
    16 agents at once. V2_acc's cross-encoder is timed on the GPU in its rows (read_ms); it is the accuracy track."""
    torch.set_num_threads(4)
    test = [q for q in D["lq"] if q["split"] == "test"][: (40 if SMOKE else 400)]
    def run(get, qs, **kw):
        wall, stages = [], defaultdict(list)
        for q in qs:
            r = get(q); r.prior = D["prior"]
            t = time.perf_counter(); out = r.read(q["question"], features=D["feats"], **kw); wall.append((time.perf_counter() - t) * 1000)
            for k, v in out["stages_ms"].items():
                stages[k].append(v)
        return {"p50": v0.pct(wall, 50), "p95": v0.pct(wall, 95), "over_100ms": float(np.mean([w > 100 for w in wall])) if wall else None,
                "stages_p50": {k: v0.pct(v, 50) for k, v in stages.items()}, "n": len(qs)}
    V2["latency"]["locomo"] = run(lambda q: reader_of(q)[0], test)
    size = 10000 if SMOKE else 100000
    src = [(t, H[("locomo", c["ci"])]["reader"].vectors[j]) for c in D["convs"] if ("locomo", c["ci"]) in H for j, t in enumerate(c["turns"])]
    reps = size // max(1, len(src)) + 1
    rows = [(t, v, k) for k in range(reps) for t, v in src][:size]
    big = HybridReader(D["encode_query"], prior=D["prior"], config=HybridConfig(**{**L.POOL, "neighbours": 0, "budget_tokens": L.BUDGET, "features": D["feats"]}))
    big.add([L.render(t) + f" #{k}" for t, _, k in rows], np.stack([v for _, v, _ in rows]), pair_texts=[L.pair_text(t) for t, _, _ in rows],
            ntok=[L.ntok(L.render(t)) + 1 for t, _, _ in rows], speakers=[t["speaker"] for t, _, _ in rows])
    qs = random.Random(SEED).sample(test, min(len(test), 300))
    V2["latency"]["store_1e5"] = {**run(lambda q: big, qs), "size": size}
    lat, t_all = [], time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        for b in [qs[i:i + 16] for i in range(0, min(len(qs), 160), 16)]:
            def one(q):
                t = time.perf_counter(); big.read(q["question"], features=D["feats"]); return (time.perf_counter() - t) * 1000
            lat += list(ex.map(one, b))
    V2["latency"]["agents16"] = {"p50": v0.pct(lat, 50), "p95": v0.pct(lat, 95), "requests_per_s": len(lat) / (time.perf_counter() - t_all)}
    log("v2 latency", json.dumps(V2["latency"], default=str)[:1500])
    save_v2()


# ----------------------------------------------------------------------------------------------- statistics + report
def load_round2_rows():
    """Round 2's LoCoMo test rows (answers, Qwen and DeepSeek labels), from the attached dama-paper-v0 output."""
    out = {}
    for name in ("DAMA_H", "B3", "B1"):
        hits = sorted(glob.glob(f"/kaggle/input/**/paper/methods/{name}.jsonl", recursive=True))
        if hits:
            out[name] = [json.loads(l) for l in open(hits[0])]
    return out


def stage_stats():
    r2 = load_round2_rows()
    rows = {**{k: v for k, v in v0.ROWS.items() if k.startswith(("V2_", "ORACLE_"))}, **r2}
    st = {}
    for key in ("ds_correct", "correct"):
        pairs = [("V2_fast", "DAMA_H"), ("V2_acc", "B3"), ("V2_acc", "V2_fast"), ("V2_fast", "B1"), ("V2_fast", "V2_fast_no_notes"),
                 ("V2_acc", "V2_acc_no_notes"), ("ORACLE_gold_notes", "ORACLE_gold")]
        for a, b in pairs:
            if a in rows and b in rows:
                st.setdefault("locomo", {}).setdefault(key, {})[f"{a} - {b}"] = L.paired([r for r in rows[a] if r["cat"] != 5], [r for r in rows[b] if r["cat"] != 5], key=key)
    lr = D.get("lme_rows", {})
    for key in ("ds_correct", "qwen_correct"):
        for a, b in (("LME_V2_fast", "LME_DAMA_H"), ("LME_V2_acc", "LME_B3"), ("LME_V2_acc", "LME_V2_fast"), ("LME_V2_fast", "LME_B3")):
            if a in lr and b in lr:
                st.setdefault("lme_holdout", {}).setdefault(key, {})[f"{a} - {b}"] = L.paired(lr[a], lr[b], key=key, cluster=None)
    # Holm over the two pre-registered primary comparisons (DeepSeek judge; Qwen if DeepSeek is absent)
    prim = st.get("lme_holdout", {}).get("ds_correct") or st.get("lme_holdout", {}).get("qwen_correct") or {}
    h2 = "LME_V2_acc - LME_B3" if "LME_V2_acc" in lr else "LME_V2_fast - LME_B3"      # pre-registered fallback
    ps = sorted((v["mcnemar_p"], k) for k, v in prim.items() if k in ("LME_V2_fast - LME_DAMA_H", h2) and v)
    st["primary_holm"] = {k: {"p": p, "holm_threshold": 0.05 / (len(ps) - i), "significant": all(pp <= 0.05 / (len(ps) - j) for j, (pp, _) in enumerate(ps[:i + 1]))}
                          for i, (p, k) in enumerate(ps)}
    disc = (st.get("lme_holdout", {}).get("ds_correct") or st.get("lme_holdout", {}).get("qwen_correct") or {}).get("LME_V2_fast - LME_DAMA_H")
    if disc:
        st["power_note"] = {"discordant": disc["discordant"], "n": disc["n"],
                            "questions_needed_for_3pt": L.power_n(disc["discordant"], 0.03), "questions_needed_for_5pt": L.power_n(disc["discordant"], 0.05)}
    # evidence -> accuracy conversion for the LoCoMo arms
    conv = {}
    for m, rs in rows.items():
        k = "ds_correct" if any(r.get("ds_correct") is not None for r in rs) else "correct"
        for c in (1, 2, 3, 4):
            a = [r for r in rs if r["cat"] == c and r.get("evidence_all") is not None and r.get(k) is not None]
            conv.setdefault(m, {})[L.CATS[c]] = {"n": len(a), "acc_if_all_evidence": _acc([r for r in a if r["evidence_all"]], k),
                                                 "acc_if_missing": _acc([r for r in a if not r["evidence_all"]], k)}
    st["evidence_to_accuracy"] = conv
    V2["stats"] = st
    sheet_src = {m: rows[m] for m in ("V2_fast", "V2_acc", "DAMA_H", "B3", "B1") if m in rows}
    if sheet_src:
        import csv
        sheet, key = L.audit_sheet(sheet_src)
        for fn, data in (("human_audit_v2_sheet.csv", sheet), ("human_audit_v2_key.csv", key)):
            with (OUT / fn).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        V2["audit"] = {"items": len(sheet), "methods": sorted(sheet_src)}
    save_v2()


def ds_judge_all():
    """DeepSeek labels for every v2 row (LoCoMo: round 2's rules; LongMemEval: the benchmark's prompts)."""
    for name, rs in list(v0.ROWS.items()):
        if name.startswith(("V2_", "ORACLE_", "dev_V2")):
            v1.ds_judge_rows(rs); write_rows(name, rs)
            V2["locomo"].setdefault(name, {})["acc_ds"] = _acc([r for r in rs if r["cat"] != 5], "ds_correct")
            V2["locomo"][name]["by_category_ds"] = {L.CATS[c]: _acc([r for r in rs if r["cat"] == c], "ds_correct") for c in L.CATS}
    ds = v1.ds_client()
    for name, rs in D.get("lme_rows", {}).items():
        if ds is not None:
            jd = ds.many([[{"role": "user", "content": v1.lme_judge_prompt({"question_id": r["qid"], "question_type": r["type"], "question": r["question"], "answer": r["gold"]}, r["answer"])}]
                          for r in rs], max_tokens=6)
            for r, o in zip(rs, jd):
                r["ds_correct"] = None if o is None else ("yes" in o.lower())
            write_rows(name, rs)
        V2["lme"][name] = {**V2["lme"].get(name, {}), **lme_summary(rs)}
    if D.get("ds"):
        V2["ds_cost"] = D["ds"].summary()


def fmt(x, d=1, pct=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def write_report():
    s = V2["settings"]; L_ = []
    L_ += [f"# DAMA paper v2 results ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, mode {MODE}{', SMOKE' if SMOKE else ''})", "",
           "Plan and pre-registration: `experiments/v2_2026-10-28/README.md`. Primary = LongMemEval-S holdout (never run before v2); "
           "LoCoMo test = regression only (it shaped DAMA-H). Judge: DeepSeek V4.1-Flash where present, else Qwen2.5-7B (marked).", "",
           "## 1. Data, leakage audit, frozen settings", "```", json.dumps(V2.get("leakage"), indent=1, default=str), "```",
           f"- Frozen before any test/holdout row: `{json.dumps(s.get('frozen'), default=str)}`", ""]
    L_ += ["## 2. Dev selection (out of fold)", f"- Pool coverage on dev: {json.dumps(V2['dev'].get('pool_coverage'))}",
           f"- Prior: **{s.get('prior', {}).get('choice')}**; OOF all-evidence coverage: `{json.dumps(s.get('prior', {}).get('oof_coverage'))}`",
           f"- Cross-encoder for V2_acc: **{s.get('ce', {}).get('choice')}**; dev grid: `{json.dumps(s.get('ce', {}).get('dev_grid'))}`; GPU ms/pair `{json.dumps(V2['dev'].get('ce_gpu_ms_per_pair'))}`",
           f"- Time notes dev check: `{json.dumps(V2['dev'].get('notes_check'), default=str)}`",
           f"- E4 sufficiency probe (AUROC, OOF): `{json.dumps(V2['dev'].get('sufficiency_probe'), default=str)}`", ""]
    L_ += ["## 3. PRIMARY: LongMemEval-S holdout", "| method | n | DeepSeek acc % [95% CI] | Qwen acc % | mean tokens | evidence recall % | read p50 ms |", "|---|---|---|---|---|---|---|"]
    for m in ("LME_V2_fast", "LME_V2_acc", "LME_DAMA_H", "LME_B3"):
        v = V2["lme"].get(m)
        if v:
            ci = v.get("ci_ds") or (None, None)
            L_.append(f"| {m} | {v['n']} | {fmt(v.get('acc_ds'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(v.get('acc_qwen'))} | {fmt(v.get('mean_tokens'), 0, False)} | {fmt(v.get('evidence_recall'))} | {fmt(v.get('read_p50'), 1, False)} |")
        else:
            L_.append(f"| {m} | not run: {STATE['errors'].get('v2_lme_holdout', 'n/a')} | | | | | |")
    L_ += ["", "By type: `" + json.dumps({m: V2["lme"][m].get("by_type") for m in V2["lme"]}, default=str) + "`", "",
           "Paired (holdout, item bootstrap; histories are independent):", "```", json.dumps(V2["stats"].get("lme_holdout"), indent=1, default=str), "```",
           f"Holm over the two primary comparisons: `{json.dumps(V2['stats'].get('primary_holm'), default=str)}`; power: `{json.dumps(V2['stats'].get('power_note'), default=str)}`", ""]
    L_ += ["## 4. LoCoMo test (regression; inspected in rounds 1-2)", "| method | n | Qwen acc % [95% CI] | DeepSeek acc % | mean tokens | evidence recall % | read p50 / p95 ms |", "|---|---|---|---|---|---|---|"]
    for m, v in V2["locomo"].items():
        if v.get("n"):
            ci = v.get("acc_ci") or (None, None)
            L_.append(f"| {m} | {v['n']} | {fmt(v.get('acc'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(v.get('acc_ds'))} | {fmt(v.get('mean_tokens'), 0, False)} | {fmt(v.get('evidence_recall'))} | {fmt(v.get('read_p50'), 1, False)} / {fmt(v.get('read_p95'), 1, False)} |")
    L_ += ["", "By category (Qwen; DeepSeek where judged): `" + json.dumps({m: {"qwen": v.get("by_category"), "ds": v.get("by_category_ds")} for m, v in V2["locomo"].items()}, default=str) + "`", "",
           "Paired against round 2's rows (item and conversation-cluster bootstrap, sign-flip over 6 conversations):", "```", json.dumps(V2["stats"].get("locomo"), indent=1, default=str), "```",
           "Evidence -> accuracy conversion: `" + json.dumps(V2["stats"].get("evidence_to_accuracy"), default=str) + "`", ""]
    L_ += ["## 5. DAMA-HT repair (retrieval only, conversations 7-9)", "```", json.dumps(V2.get("ht"), indent=1, default=str), "```",
           "## 6. CPU latency (V2_fast)", "```", json.dumps(V2.get("latency"), indent=1, default=str), "```",
           "## 7. Human audit", f"- `human_audit_v2_sheet.csv` (blind) + `human_audit_v2_key.csv`: {json.dumps(V2.get('audit'))}. Not labelled yet.",
           "## 8. Errors, notes, cost", f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`",
           f"- notes: {V2['notes']}", f"- DeepSeek: `{json.dumps(V2.get('ds_cost'), default=str)}`"]
    text = "\n".join(L_) + "\n"
    (OUT / "RESULTS_V2.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)
    return text


@stage("v2_carry")
def stage_carry():
    """Judge mode: this notebook's previous output (rows, results), then reload it."""
    src = sorted(glob.glob("/kaggle/input/**/paper/results_v2.json", recursive=True))
    if not src:
        raise FileNotFoundError("no earlier results_v2.json attached")
    root = Path(src[0]).parent
    for f in root.rglob("*"):
        if f.is_file() and f.name != "llm_cache.sqlite":
            dst = OUT / f.relative_to(root); dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy(f, dst)
    V2.update(json.loads((root / "results_v2.json").read_text()))
    for f in sorted((OUT / "methods").glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        if f.stem.startswith("LME_"):
            D.setdefault("lme_rows", {})[f.stem] = rows
        else:
            v0.ROWS[f.stem] = rows
    log("v2 carried forward:", sorted(v0.ROWS), sorted(D.get("lme_rows", {})))


if __name__ == "__main__" and MODE == "judge":
    D["env"] = v0.environment(); log("environment (judge mode)", D["env"])
    stage_carry()
    if v1.ds_client() is None:
        log("no DeepSeek key in this run (versions pushed from the CLI cannot read secrets): start it from the website"); sys.exit(1)
    ds_judge_all()
    stage_stats()
    write_report()
    sys.exit(0)

if __name__ == "__main__":
    if v0.ON_KAGGLE and not SMOKE and not glob.glob("/kaggle/input/**/paper/methods/DAMA_H.jsonl", recursive=True):
        log("round 2's rows (dama-paper-v0 output) are not attached; stopping before using GPU time"); sys.exit(1)
    D["env"] = v0.environment(); log("environment", D["env"])
    stage_setup()
    if "v2_setup" in STATE["errors"]:            # nothing downstream can work: stop before loading the reader
        save_v2(); write_report(); log("v2 stopped after a failed setup"); sys.exit(1)
    stage_select_prior()
    stage_select_ce()
    stage_sufficiency()
    stage_load_reader()
    stage_dev_notes()
    stage_freeze()
    if "prior" in D and v0.READER is not None:
        stage_lme_holdout()
        stage_locomo_main()
        stage_attribution()
        stage_oracle()
    stage_ht()
    stage_latency()
    try:
        if v1.ds_client() is not None:
            ds_judge_all()
        stage_stats()
    except Exception as e:
        STATE["errors"]["v2_stats"] = f"{type(e).__name__}: {str(e)[:150]}"
    save_v2()
    write_report()
    log(f"v2 done; errors: {STATE['errors'] or 'none'}")

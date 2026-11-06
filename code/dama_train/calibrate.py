"""Stage 3: calibrate both models on the calib split, fit the conformal threshold, and report held-out test metrics.

Outputs (in out_dir):
  calibration.json : Platt (a, b) for rel/sal/need/route heads, temperature for the op model,
                     conformal lambda for alpha in {0.05, 0.1, 0.2} and each rerank depth k
  report.json      : test metrics (never used for any choice above)
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
from .config import OPS, ROUTES
from .data.io import read_jsonl
from .metrics import auc, best_f1, binary_ece, conformal_threshold, ece, fit_platt, fit_temperature, macro_f1, mrr, recall_at_k, sigmoid
from .env import best_device
from .probes import need_rows, op_rows, score_need, score_op
from .retrieval import DenseEncoder, rank_all

SWEEP = (4, 8, 16, 32, 64, 128)  # candidate-pool sizes for the "is the gold even retrievable?" table
RERANK_K = (4, 8, 16, 32, 64)    # how many first-stage candidates the rel head re-scores (its CPU cost grows with k)
DEFAULT_RERANK_K = 8             # ~16 ms per pair on 2 vCPUs, so k=8 is the most that fits a ~100 ms push


def min_gold_p(x, k):
    """Lowest calibrated score among a question's gold turns when only the first-stage top-k is re-scored;
    -inf if any gold turn is outside the top-k (a retrieval miss counts against the guarantee)."""
    pos = [j for j, f in enumerate(x["gold_flags_pool_order"]) if f]
    if not x["n_gold"] or len(pos) < x["n_gold"] or max(pos) >= k:
        return -math.inf
    return float(min(x["p"][j] for j in pos))


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def run_calibration(data_dir, decision_dir, ops_dir, out_dir, *, pool_size=64, alphas=(0.05, 0.1, 0.2),
                    dense_model="BAAI/bge-small-en-v1.5", use_dense=True, max_pool_questions=None, retriever_dir=None, log=print):
    import torch
    from .models.decision import load_decision
    from .train.decision import make_collate as dec_collate, predict_logits as dec_logits
    from .train.ops import make_collate as op_collate, predict_logits as op_logits
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .models.ops import OpLogits

    t0 = time.time()
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = best_device()
    dec, dec_tok, dec_meta = load_decision(decision_dir, device=device)
    dcollate = dec_collate(dec_tok, dec_meta.get("max_length", 256))
    op_model = OpLogits(AutoModelForSequenceClassification.from_pretrained(Path(ops_dir) / "model")).to(device).eval()
    op_tok = AutoTokenizer.from_pretrained(Path(ops_dir) / "tokenizer")
    op_meta = json.loads((Path(ops_dir) / "meta.json").read_text())
    ocollate = op_collate(op_tok, op_meta.get("max_length", 160))

    # ---- 1. calibration parameters (calib split only)
    cal_rows = read_jsonl(data / "decision" / "calib.jsonl")
    z = dec_logits(dec, cal_rows, dcollate)
    cal = {"platt": {}, "route_platt": {}}
    for name, col in (("rel", 0), ("sal", 1), ("need", 2)):
        idx = [i for i, r in enumerate(cal_rows) if r["task"] == name]
        y = [cal_rows[i]["y"] for i in idx]
        cal["platt"][name] = fit_platt(z[idx, col], y) if len(set(y)) == 2 else (1.0, 0.0)
    idx = [i for i, r in enumerate(cal_rows) if r["task"] == "route"]
    for k, route in enumerate(ROUTES):
        y = [cal_rows[i]["y"][k] for i in idx]
        cal["route_platt"][route] = fit_platt(z[idx, 3 + k], y) if len(set(y)) == 2 else (1.0, 0.0)
    op_cal = read_jsonl(data / "ops" / "calib.jsonl")
    zo = op_logits(op_model, op_cal, ocollate)
    cal["op_temperature"] = fit_temperature(zo, [r["y"] for r in op_cal])
    log(f"calibration fitted: {json.dumps(cal)}")

    def p_of(name, logits):
        a, b = cal["platt"][name]
        return sigmoid(a * logits + b)

    # ---- 2. first-stage source (chosen on calib), candidate pools and conformal threshold
    encoders = {}
    if use_dense:
        try:
            encoders["bge"] = DenseEncoder(dense_model, device=device)
        except Exception as e:  # offline or blocked: fall back to BM25 pools and say so
            log(f"dense encoder unavailable ({e}); pools use BM25 only")
    tuned_meta = Path(retriever_dir) / "meta.json" if retriever_dir else None
    # epoch 0 means fine-tuning did not beat the base model, so "tuned" would only duplicate "bge"
    if tuned_meta and tuned_meta.exists() and (json.loads(tuned_meta.read_text()).get("epoch", 0) > 0 or "bge" not in encoders):
        tuned = DenseEncoder(str(retriever_dir), device=device)
        ctx = json.loads(tuned_meta.read_text()).get("context_chars", 0)
        encoders["tuned"] = (tuned, ctx)
        if ctx:  # same model on bare turns, so the report shows what the neighbour keys add
            encoders["tuned_plain"] = (tuned, 0)

    def rank_split(split):
        out = []
        for d in read_jsonl(data / "dialogues" / f"{split}.jsonl"):
            qs = d["questions"][:max_pool_questions] if max_pool_questions else d["questions"]
            d = {**d, "questions": qs}
            for q, ranks in zip(qs, rank_all(d, encoders, depth=max(pool_size, *SWEEP))):
                out.append((d, q, ranks))
        return out

    def source_table(ranked):
        sources = ranked[0][2].keys() if ranked else []
        return {src: {str(k): float(np.mean([set(q["gold"]) <= set(r[src][:k]) for _, q, r in ranked if q["gold"]])) for k in SWEEP}
                for src in sources}

    ranked = {"calib": rank_split("calib"), "test": rank_split("test")}
    sources = {split: source_table(r) for split, r in ranked.items()}
    # the source is chosen on calib only, by mean coverage at the depths a CPU reranker can afford
    source = max(sources["calib"], key=lambda src: np.mean([sources["calib"][src][str(k)] for k in (8, 16, 32)])) if sources["calib"] else "bm25"
    cal["first_stage"] = source
    log(f"first-stage source chosen on calib: {source}")

    def score_pools(split):
        result = []
        for d, q, ranks in ranked[split]:
            ranking = ranks[source]
            pool = ranking[:pool_size]
            rows = [{"task": "rel", "a": q["query"], "b": d["texts"][i], "y": 0.0} for i in pool]
            logits = dec_logits(dec, rows, dcollate)[:, 0] if rows else np.zeros(0)
            gold = set(q["gold"])
            result.append({"qid": q["qid"], "qtype": q["qtype"], "n_gold": len(gold), "pool": pool, "z": logits,
                           "all_gold_at_k": {k: bool(gold) and gold <= set(ranking[:k]) for k in SWEEP},
                           "gold_flags_pool_order": [int(i in gold) for i in pool],
                           "chars": [len(d["texts"][i]) for i in pool]})
        return result

    cal_pools = score_pools("calib")
    test_pools = score_pools("test")
    # rel is calibrated on the retrieved pools it will actually re-score (hybrid top candidates), not on the
    # BM25-negative training-style rows, so its probabilities mean the same thing at runtime
    z_pool = np.concatenate([x["z"] for x in cal_pools]) if cal_pools else np.zeros(0)
    y_pool = [f for x in cal_pools for f in x["gold_flags_pool_order"]]
    if len(set(y_pool)) == 2:
        cal["platt"]["rel"] = fit_platt(z_pool, y_pool)
        cal["rel_platt_fitted_on"] = "calib candidate pools"
    for x in cal_pools + test_pools:
        p = p_of("rel", x.pop("z"))
        x["p"] = p.tolist()
        x["gold_flags_model_order"] = [x["gold_flags_pool_order"][j] for j in np.argsort(-p)]
    # one threshold per rerank depth k: only the first-stage top-k is re-scored, so lambda depends on k.
    # probabilities live in [0, 1], so 0.0 means "admit every re-scored candidate": the top-k itself misses gold
    # turns for more than alpha of the calib questions and no threshold can reach the target coverage
    ks = [k for k in RERANK_K if k <= pool_size]
    cal["conformal_by_k"] = {str(k): {str(a): max(0.0, float(conformal_threshold([min_gold_p(x, k) for x in cal_pools], a)))
                                      for a in alphas} for k in ks}
    cal["default_rerank_k"] = DEFAULT_RERANK_K if DEFAULT_RERANK_K in ks else ks[-1]
    cal["conformal"] = cal["conformal_by_k"][str(cal["default_rerank_k"])]
    (out / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n")

    # ---- 3. held-out test report
    report = {"pool_source": source, "pool_size": pool_size, "first_stage_sources": sources}
    flags_pool = [x["gold_flags_pool_order"] for x in test_pools]
    flags_model = [x["gold_flags_model_order"] for x in test_pools]
    n_gold = [x["n_gold"] for x in test_pools]
    report["retrieval"] = {
        "questions": len(test_pools),
        "gold_in_pool": float(np.mean([sum(f) / g for f, g in zip(flags_pool, n_gold) if g])),
        "all_gold_in_pool": float(np.mean([sum(f) == g for f, g in zip(flags_pool, n_gold) if g])),
        "all_gold_in_top_k": {str(k): float(np.mean([x["all_gold_at_k"][k] for x in test_pools if x["n_gold"]])) for k in SWEEP},
        **{f"recall@{k}_retriever": recall_at_k(flags_pool, n_gold, k) for k in (5, 10, 20)},
        **{f"recall@{k}_dama_rel": recall_at_k(flags_model, n_gold, k) for k in (5, 10, 20)},
        "mrr_retriever": mrr(flags_pool), "mrr_dama_rel": mrr(flags_model),
        "rel_ece_on_pools": binary_ece([f for x in test_pools for f in x["gold_flags_pool_order"]], [v for x in test_pools for v in x["p"]])}
    report["conformal_test_by_k"] = {}
    for k, lams in cal["conformal_by_k"].items():
        k_int = int(k)
        reranked = [[x["gold_flags_pool_order"][j] for j in sorted(range(min(k_int, len(x["p"]))), key=lambda j: -x["p"][j])]
                    for x in test_pools]
        per_k = {"all_gold_in_top_k": float(np.mean([min_gold_p(x, k_int) > -math.inf for x in test_pools if x["n_gold"]])),
                 "recall@5_after_rerank": recall_at_k(reranked, n_gold, 5), "alphas": {}}
        for a, lam in lams.items():
            covered, sizes, tokens = [], [], []
            for x in test_pools:
                admitted = [j for j, p in enumerate(x["p"][:k_int]) if p >= lam]
                gold_pos = [j for j, f in enumerate(x["gold_flags_pool_order"]) if f]
                covered.append(int(len(gold_pos) == x["n_gold"] and all(j in admitted for j in gold_pos)))
                sizes.append(len(admitted))
                tokens.append(sum(x["chars"][j] for j in admitted) / 4)
            per_k["alphas"][a] = {"lambda": lam, "target_coverage": 1 - float(a), "test_coverage": float(np.mean(covered)),
                                  "mean_admitted": float(np.mean(sizes)), "mean_admitted_tokens": float(np.mean(tokens))}
        report["conformal_test_by_k"][k] = per_k
    report["default_rerank_k"] = cal["default_rerank_k"]

    test_rows = read_jsonl(data / "decision" / "test.jsonl")
    zt = dec_logits(dec, test_rows, dcollate)
    heads = {}
    for name, col in (("rel", 0), ("sal", 1), ("need", 2)):
        idx = [i for i, r in enumerate(test_rows) if r["task"] == name]
        if not idx:
            continue
        y = [test_rows[i]["y"] for i in idx]
        heads[name] = {"n": len(idx), "auc": auc(y, zt[idx, col]), "best_f1": best_f1(y, p_of(name, zt[idx, col]))["f1"],
                       "ece_raw": binary_ece(y, sigmoid(zt[idx, col])), "ece_calibrated": binary_ece(y, p_of(name, zt[idx, col]))}
    idx = [i for i, r in enumerate(test_rows) if r["task"] == "route"]
    if idx:
        y = np.array([test_rows[i]["y"] for i in idx])
        per = {}
        for k, route in enumerate(ROUTES):
            a, b = cal["route_platt"][route]
            pred = sigmoid(a * zt[idx, 3 + k] + b) >= 0.5
            tp = (pred & (y[:, k] == 1)).sum()
            per[route] = float(2 * tp / max(1, pred.sum() + (y[:, k] == 1).sum()))
        heads["route"] = {"n": len(idx), "f1_per_route": per}
    report["decision_heads"] = heads

    op_test = read_jsonl(data / "ops" / "test.jsonl")
    zo = op_logits(op_model, op_test, ocollate)
    y = np.array([r["y"] for r in op_test]); probs = _softmax(zo / cal["op_temperature"]); pred = probs.argmax(-1)
    confusion = {g: {p: int(((y == i) & (pred == j)).sum()) for j, p in enumerate(OPS)} for i, g in enumerate(OPS)}
    by_src = {}
    for src in sorted({r["src"] for r in op_test}):
        m = np.array([r["src"] == src for r in op_test])
        by_src[src] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()), "macro_f1": macro_f1(y[m], pred[m], len(OPS))}
    report["ops"] = {"n": len(op_test), "accuracy": float((pred == y).mean()), "macro_f1": macro_f1(y, pred, len(OPS)),
                     "ece_raw": ece(_softmax(zo).max(-1), pred == y), "ece_calibrated": ece(probs.max(-1), pred == y),
                     "by_source": by_src, "confusion": confusion}
    # behaviour probes (probes.py): out-of-distribution checks, never used for any choice above
    zn = dec_logits(dec, need_rows(), dcollate)[:, 2]
    zp = op_logits(op_model, op_rows(), ocollate)
    report["probes"] = {"need": score_need(p_of("need", zn)), "op": score_op(_softmax(zp / cal["op_temperature"]))}
    report["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return cal, report


def print_report(report):
    r = report["retrieval"]
    print("\nFIRST-STAGE SOURCES: questions with every gold turn in the top-k (source chosen on calib)")
    for split, table in report.get("first_stage_sources", {}).items():
        for src, row in table.items():
            mark = "  <- chosen" if src == report["pool_source"] else ""
            print(f"  {split:5s} {src:12s} " + "  ".join(f"k={k}: {v:.3f}" for k, v in row.items()) + mark)
    print(f"\nRETRIEVAL on {r['questions']} held-out questions (pool: {report['pool_source']}, {report['pool_size']} candidates)")
    print(f"  gold turns inside the candidate pool: {r['gold_in_pool']:.3f}  "
          f"(questions with every gold turn in the pool: {r['all_gold_in_pool']:.3f} = ceiling for packet coverage)")
    print("  questions with every gold turn in the top-k candidates (pick pool_size from this): "
          + ", ".join(f"k={k}: {v:.3f}" for k, v in r.get("all_gold_in_top_k", {}).items()))
    for k in (5, 10, 20):
        print(f"  recall@{k:<2}  retriever order {r[f'recall@{k}_retriever']:.3f}  ->  DAMA rel head {r[f'recall@{k}_dama_rel']:.3f}")
    print(f"  MRR        retriever {r['mrr_retriever']:.3f}  ->  DAMA rel head {r['mrr_dama_rel']:.3f}")
    print("\nCONFORMAL PACKET SELECTION per rerank depth k (rel head re-scores the first-stage top-k; threshold fitted on calib, checked on test)")
    for k, per_k in report["conformal_test_by_k"].items():
        default = "  <- runtime default" if int(k) == report["default_rerank_k"] else ""
        print(f"  k={k:<3} all gold in top-k {per_k['all_gold_in_top_k']:.3f}  recall@5 after rerank {per_k['recall@5_after_rerank']:.3f}{default}")
        for a, c in per_k["alphas"].items():
            print(f"     alpha {a:<5} target coverage {c['target_coverage']:.2f}  test coverage {c['test_coverage']:.3f}  "
                  f"admitted {c['mean_admitted']:.1f} turns / ~{c['mean_admitted_tokens']:.0f} tokens"
                  + ("   <- top-k too small for this alpha: everything admitted" if c["lambda"] <= 0.0 else ""))
    print("\nDECISION HEADS (test)")
    for name, h in report["decision_heads"].items():
        if name == "route":
            print(f"  route  F1 per route: " + ", ".join(f"{k} {v:.3f}" for k, v in h["f1_per_route"].items()))
        else:
            print(f"  {name:5s} AUC {h['auc']:.3f}  best-F1 {h['best_f1']:.3f}  ECE {h['ece_raw']:.3f} -> {h['ece_calibrated']:.3f}  (n={h['n']})")
    o = report["ops"]
    print(f"\nOP MODEL (test)  accuracy {o['accuracy']:.3f}  macro-F1 {o['macro_f1']:.3f}  ECE {o['ece_raw']:.3f} -> {o['ece_calibrated']:.3f}")
    for src, s in o["by_source"].items():
        print(f"  {src:10s} n={s['n']:<5} accuracy {s['accuracy']:.3f}")
    print("  confusion (rows = truth):")
    for g, row in o["confusion"].items():
        print(f"   {g:7s} " + " ".join(f"{p}:{n:<5}" for p, n in row.items()))
    if report.get("probes"):
        n, o = report["probes"]["need"], report["probes"]["op"]
        print(f"\nPROBES (hand-written, never trained on)  need accuracy {n['accuracy']:.3f} "
              f"(memory requests found {n['positives_found']:.2f}, others rejected {n['negatives_rejected']:.2f})  "
              f"op accuracy {o['accuracy']:.3f} " + str({k: round(v, 2) for k, v in o["accuracy_per_op"].items() if v is not None}))
        for f in n["failures"]:
            print(f"   need  {'should need' if f['expected'] else 'should not need'} memory  p={f['p']:.3f}  {f['request']}")
        for f in o["failures"]:
            print(f"   op    expected {f['expected']:6s} got {f['got']:6s} p={f['p']:.2f}  {f['existing']} | {f['new']}")

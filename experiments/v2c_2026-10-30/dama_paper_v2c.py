"""DAMA paper v2c (2026-10-30): dev-only diagnostics for v3. Private Kaggle notebook `s41r4j/dama-paper-v2c` (T4 x2).

v2b's reader (Qwen3.5-9B, notes-first prompt P1) lifted the LongMemEval holdout from 52% to 78% (DeepSeek), but B3
stayed level (80.5%, difference -2.6 [-7.4, +2.2]). On LongMemEval dev the packet held all evidence for every
multi-session question, yet 61% were answered: miscounts, skipped arithmetic, date confusion. This run asks, on dev
only (no holdout or test question is loaded), which change helps the strong reader, and what each memory costs:

  C1 base      V2_fast (m=4, B=3000, time notes), P1, thinking off        = v2b's frozen setting
  C2 think     C1 with the reader's thinking mode on (P1 system prompt, up to 3000 new tokens)
  C3 m0        C1 without neighbour turns
  C4 b3        B3 context (MiniLM-L6 over the whole history), P1
  C5 b4500     C1 with B = 4500
  C6 v2acc     V2_fast + MiniLM-L6 re-scoring the prior's top 64 (V2_acc), P1

Plus the cost of one read on the same machine: DAMA V2_fast on CPU vs B3's cross-encoder on CPU and on GPU.
Qwen2.5-7B judge (as v2b). No DeepSeek. Results feed the v3 design; nothing here is a confirmatory claim.
"""
from __future__ import annotations
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SMOKE = False
HERE = Path(globals().get("__file__", "dama_paper_v2c.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v2b_2026-10-29/dama_paper_v2b.py")))
import dama_paper_v2b as Bm      # v2b's engine (vLLM server, client), reading, judging; imports v2, v1, v0, v2lib
V, L, v0, v1 = Bm.V, Bm.L, Bm.v0, Bm.v1
import numpy as np

Bm.SMOKE = V.SMOKE = v0.SMOKE = v1.SMOKE = SMOKE
v0.TIME_LIMIT_H, v0.FINAL_LIMIT_H = (1.5, 2.0) if SMOKE else (8.0, 8.5)
log, stage, OUT, D, STATE, SEED = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE, v0.SEED
READER, LABEL = "Qwen/Qwen3.5-9B", "qwen3.5-9b"        # v2b's frozen choice
JUDGE_ID = Bm.JUDGE_ID
N_LOCO, N_ADV = (12, 3) if SMOKE else (250, 50)        # LoCoMo dev sample (categories 1-4, adversarial); LME: all dev
THINK_MAX = 3000
R = {"settings": {}, "configs": {}, "paired": {}, "cost": {}, "notes": []}
H = V.H


def save_r():
    (OUT / "results_v2c.json").write_text(json.dumps(R, indent=1, default=str))


class ThinkClient(Bm.Client):
    """The reader with its thinking mode on; cached under its own key."""
    def _post(self, messages, max_new):
        import urllib.request
        body = {"model": "m", "messages": messages, "max_tokens": max_new, "temperature": 0, "seed": SEED,
                "chat_template_kwargs": {"enable_thinking": True}}
        req = urllib.request.Request(f"http://127.0.0.1:{Bm.PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            t = time.time()
            try:
                j = json.loads(urllib.request.urlopen(req, timeout=3600).read())
                raw = j["choices"][0]["message"].get("content") or ""
                if "<think>" in raw and "</think>" not in raw:      # cut off while thinking: no answer
                    raw = ""
                return Bm.clean(raw), int(j.get("usage", {}).get("prompt_tokens", 0)), (time.time() - t) * 1000
            except Exception:
                time.sleep(5 * (attempt + 1))
        return None, 0, 0.0

    def __call__(self, convs, max_new=40, tools=None, **kw):
        name, self.name = self.name, self.name + "|think"
        try:
            return super().__call__(convs, max_new=THINK_MAX)
        finally:
            self.name = name


# ---------------------------------------------------------------------------------------------- setup: dev only
@stage("v2c_setup")
def stage_setup():
    import torch
    from sentence_transformers import SentenceTransformer
    from dama_agent.fastmodels import OnnxEncoder
    v0.stage_data()
    convs, lq = L.load_locomo(OUT / "locomo10.json")
    lme_path = Path(os.environ.get("DAMA_LME_PATH", "/tmp/longmemeval_s_cleaned.json"))
    if not lme_path.exists():
        import urllib.request
        urllib.request.urlretrieve(L.LME_URL, lme_path)
    if L.sha256(lme_path) != L.LME_SHA256:
        raise ValueError("LongMemEval-S file differs from the pinned SHA-256")
    meta = L.lme_splits(lme_path)
    lme = L.load_lme(lme_path, want=("dev",), limit=12 if SMOKE else None, meta=meta)       # holdout never loaded
    lq = [q for q in lq if q["split"] == "dev"]
    if SMOKE:
        lq = [q for q in lq if q["ci"] in (0, 1)][:40]
    D["convs"], D["lq"], D["lme"] = convs, lq, lme
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D["bge"] = SentenceTransformer(v0.EMBED_ID, device=dev)
    try:
        D["qenc"] = OnnxEncoder.export(v0.EMBED_ID, V.EXPORT / "bge", prefix=L.BGE_PREFIX, sample=[q["question"] for q in lq[:50]], log=log)
        enc = D["qenc"].encode
    except Exception as e:
        log("ONNX query encoder failed; sentence-transformers on CPU", type(e).__name__, str(e)[:300])
        cpu = SentenceTransformer(v0.EMBED_ID, device="cpu")
        enc = lambda qs: cpu.encode([L.BGE_PREFIX + q for q in qs], normalize_embeddings=True, convert_to_numpy=True)
    D["encode_query"] = enc
    allq = lq + lme
    D["qv"] = dict(zip([q["qid"] for q in allq], enc([q["question"] for q in allq])))
    hist = [(("locomo", c["ci"]), c["turns"]) for c in convs if any(q["ci"] == c["ci"] for q in lq)]
    hist += [(("lme", it["qid"]), it["turns"]) for it in lme]
    texts = [x for _, turns in hist for x in L.key_texts(turns)]
    vec = D["bge"].encode(texts, batch_size=128 if dev == "cuda" else 16, normalize_embeddings=True, convert_to_numpy=True)
    s = 0
    for key, turns in hist:
        H[key] = {"reader": L.build_reader(turns, vec[s:s + len(turns)], enc), "turns": turns}
        s += len(turns)
    if any(q["split"] != "dev" for q in allq):
        raise ValueError("non-dev question loaded")
    R["settings"]["data"] = {"locomo_dev_questions": len(lq), "lme_dev": len(lme), "lme_types": dict(Counter(it["type"] for it in lme))}
    log("v2c setup (dev only):", R["settings"]["data"])
    save_r()


def ctx(q, neighbours=L.NEIGHBOURS, budget=L.BUDGET, rerank=False):
    r, turns = V.reader_of(q)
    r.prior, r.scorer = D["prior"], (D["ce_scorer"] if rerank else None)
    t = time.perf_counter()
    out = r.read_many([q["question"]], qvecs=[V.qvec_of(q)], budget_tokens=budget, features=D["feats"],
                      rerank_n=64 if rerank else 0, neighbours=neighbours)[0]
    ms = (time.perf_counter() - t) * 1000
    r.scorer = None
    chosen = [turns[i] for i in out["ids"]]
    return {"context": L.render_many(chosen, True) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen], "read_ms": ms,
            "budget_limited": out["budget_limited"]}


@stage("v2c_contexts")
def stage_contexts():
    import torch
    rng = random.Random(SEED)
    loco = rng.sample([q for q in D["lq"] if q["cat"] != 5], min(N_LOCO, sum(q["cat"] != 5 for q in D["lq"])))
    loco += rng.sample([q for q in D["lq"] if q["cat"] == 5], min(N_ADV, sum(q["cat"] == 5 for q in D["lq"])))
    qs = loco + D["lme"]
    b3, ce = Bm.b3_ctx_fn()
    D["ce_scorer"] = V.ce_scorer(ce)
    base = [ctx(q) for q in qs]
    J = {"C1_base": base, "C2_think": base, "C3_m0": [ctx(q, neighbours=0) for q in qs], "C4_b3": [b3(q) for q in qs],
         "C5_b4500": [ctx(q, budget=4500) for q in qs], "C6_v2acc": [ctx(q, rerank=True) for q in qs]}
    D["qs"], D["ctx"] = qs, J
    del b3; D.pop("ce_scorer"); del ce
    D.pop("bge", None)
    import gc
    gc.collect(); torch.cuda.empty_cache()
    R["settings"]["questions"] = {"locomo": len(loco), "lme": len(D["lme"])}
    log("v2c contexts:", R["settings"]["questions"])
    save_r()


@stage("v2c_cost")
def stage_cost():
    """Per-question cost on this machine for LongMemEval dev histories (~500 turns): DAMA V2_fast (query embedding on
    CPU via ONNX + pool + prior + pack) vs B3's cross-encoder over every turn, on CPU (4 threads) and on one T4."""
    import torch
    from sentence_transformers import CrossEncoder
    items = D["lme"][: (5 if SMOKE else 100)]
    torch.set_num_threads(4)
    dama = []
    for it in items:
        r, _ = V.reader_of(it); r.prior = D["prior"]
        t = time.perf_counter()
        r.read_many([it["question"]], features=D["feats"])           # embeds the query itself (ONNX, CPU)
        dama.append((time.perf_counter() - t) * 1000)
    out = {"questions": len(items), "turns_per_history_mean": float(np.mean([len(it["turns"]) for it in items])),
           "dama_cpu_ms": {"p50": v0.pct(dama, 50), "p95": v0.pct(dama, 95)}}
    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        ce = CrossEncoder(V.B3_CE[0], device=dev, max_length=V.B3_CE[1])
        ce.predict([("warm up", "warm up")] * 8, show_progress_bar=False)
        ms = []
        for it in items[: (5 if SMOKE else (20 if dev == "cpu" else 100))]:
            t = time.perf_counter()
            ce.predict([(it["question"], L.pair_text(x)) for x in it["turns"]], batch_size=64 if dev == "cpu" else 256, show_progress_bar=False)
            if dev == "cuda":
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000)
        out[f"b3_{dev}_ms"] = {"p50": v0.pct(ms, 50), "p95": v0.pct(ms, 95), "n": len(ms)}
        del ce; torch.cuda.empty_cache()
    out["b3_cpu_over_dama_cpu_p50"] = out["b3_cpu_ms"]["p50"] / max(1e-9, out["dama_cpu_ms"]["p50"])
    R["cost"] = out
    log("v2c cost:", out)
    save_r()


# ---------------------------------------------------------------------------------------------- reading + judging
def jobs(name):
    return [(name, q, c) for q, c in zip(D["qs"], D["ctx"][name])]


@stage("v2c_read")
def stage_read():
    ok = False
    for ver in Bm.VLLM_VERSIONS:
        if Bm.install_vllm(ver):
            srv = Bm.Server(READER)
            if srv.start():
                ok = True; R["settings"]["vllm"] = ver
                break
    if not ok:
        raise RuntimeError("vLLM could not serve the reader")
    rows = {}
    for name in ("C1_base", "C2_think", "C3_m0", "C4_b3", "C5_b4500", "C6_v2acc"):
        if (time.time() - v0.T_START) / 3600 > v0.TIME_LIMIT_H:
            STATE["errors"][name] = "time guard"; continue
        client = ThinkClient(READER) if name == "C2_think" else Bm.Client(READER)
        rows[name], sec = Bm.read(client, jobs(name), "P1", f"{LABEL}|{name}")
        R["configs"].setdefault(name, {})["s_per_question"] = round(sec, 3)
        V.write_rows(f"dev_{name}", rows[name])
        save_r()
    srv.stop(drop_weights=True)
    j = Bm.Server(JUDGE_ID)
    if not j.start():
        raise RuntimeError("judge could not be served")
    jc = Bm.Client(JUDGE_ID)
    for name, rs in rows.items():
        Bm.judge(jc, rs)
        V.write_rows(f"dev_{name}", rs)
    j.stop()
    D["rows"] = rows


def summary(rs):
    a = Bm.dev_acc(rs)
    lm = [r for r in rs if r["bench"] == "lme"]
    lo = [r for r in rs if r["bench"] == "locomo" and r["cat"] != 5]
    by_cat = defaultdict(list)
    for r in lo:
        by_cat[L.CATS[r["cat"]]].append(r["correct"])
    ev = [r["evidence_all"] for r in rs if r.get("evidence_all") is not None]
    return {**a, "locomo_by_category": {k: float(np.mean(v)) for k, v in by_cat.items()},
            "lme_acc_if_all_evidence": float(np.mean([r["qwen_correct"] for r in lm if r.get("evidence_all")])) if lm else None,
            "evidence_recall": float(np.mean(ev)) if ev else None, "mean_tokens": float(np.mean([r["input_tokens"] for r in rs])),
            "think_cut_off": float(np.mean([not r["raw"] for r in rs]))}


def stage_stats():
    rows = D.get("rows", {})
    for name, rs in rows.items():
        R["configs"].setdefault(name, {}).update(summary(rs))
    base = rows.get("C1_base")
    for name, rs in rows.items():
        if name != "C1_base" and base:
            for bench in ("lme", "locomo"):
                a = [r for r in rs if r["bench"] == bench and r.get("cat") != 5]
                b = [r for r in base if r["bench"] == bench and r.get("cat") != 5]
                R["paired"].setdefault(name, {})[bench] = L.paired(a, b, key="qwen_correct", cluster=None)
    save_r()


def write_report():
    out = [f"# DAMA paper v2c: dev-only diagnostics ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           "Dev questions only (LoCoMo conversations 0-3 sample, LongMemEval v1 dev). Reader Qwen3.5-9B, prompt P1; Qwen2.5-7B judge. "
           "Diagnostics for the v3 design, not confirmatory results.", "",
           f"Data: `{json.dumps(R['settings'], default=str)}`", "",
           "| config | mean | LoCoMo | LME | LME if all evidence | LoCoMo abstention | evidence % | tokens | s/question | no answer |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "n/a" if x is None else f"{100 * x:.1f}"
    for k, v in R["configs"].items():
        if "mean" in v:
            out.append(f"| {k} | {f(v['mean'])} | {f(v['locomo'])} | {f(v['lme'])} | {f(v['lme_acc_if_all_evidence'])} | {f(v['locomo_abstention'])} | "
                       f"{f(v['evidence_recall'])} | {v['mean_tokens']:.0f} | {v.get('s_per_question')} | {f(v['think_cut_off'])} |")
    out += ["", "By type / category: `" + json.dumps({k: {"lme": v.get("lme_by_type"), "locomo": v.get("locomo_by_category")} for k, v in R["configs"].items()}, default=str) + "`", "",
            "Paired vs C1_base (exact McNemar, item bootstrap):", "```", json.dumps(R["paired"], indent=1, default=str), "```", "",
            "## Cost of one read (same machine)", "```", json.dumps(R["cost"], indent=1, default=str), "```", "",
            f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_V2C.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


if __name__ == "__main__":
    log("environment", v0.environment())
    stage_setup()
    if "v2c_setup" not in STATE["errors"]:
        V.stage_select_prior()                 # v2's prior, re-fitted on dev exactly as v2/v2b did
        D["ce_name"], D["ce_n"] = None, 0
        R["settings"]["prior"] = V.V2["settings"].get("prior", {}).get("choice")
        stage_contexts()
        stage_cost()
        if "v2c_contexts" not in STATE["errors"]:
            stage_read()
    try:
        stage_stats()
    except Exception as e:
        STATE["errors"]["v2c_stats"] = f"{type(e).__name__}: {str(e)[:150]}"
    save_r()
    write_report()
    log(f"v2c done; errors: {STATE['errors'] or 'none'}")

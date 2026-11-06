"""DAMA paper experiments v1 (2026-10-24): the hybrid read path against v0's baselines, an outside judge, a second
benchmark, and CPU latency with 16 agents.

What is new against v0 (version 8 of this notebook, whose outputs are attached and reused, never re-run):
  - DAMA-H: code/dama_agent/hybrid.py (union pool, LR prior, a small cross-encoder on the top N, probability packing,
    optional previous-turn neighbours). Every setting is chosen on dev (LoCoMo conversations 0-3); test (4-9) runs once.
  - Outside judge: DeepSeek V4.1-Flash (`deepseek-flash`, thinking off, temperature 0) re-judges every main method.
    API key from the Kaggle secret DEEPSEEK_API_KEY only (never printed or saved); every call cached; hard spend cap.
  - Second benchmark: a stratified LongMemEval-S subset (official judge prompts); settings fixed from LoCoMo dev.
  - DAMA-HT: the same read path on the trained L1.2 models (fine-tuned retriever + rel), conversations 7-9 only.
  - Latency on CPU with the ONNX models actually used: per stage, a 10^5-turn store, and 16 concurrent agents.

Pillars: speed (CPU read p50/p95 against 100 ms), accuracy (judge accuracy against B1-B4), context saving (tokens),
multi-agent (16 agents on one memory: per-request latency and throughput).
"""
from __future__ import annotations
import glob
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODE = "full"  # "judge": no GPU; carry this notebook's previous outputs forward, restore missing round-1 files, run the
               # DeepSeek judge on every saved answer and rewrite the report (SMOKE then judges ~20 answers, cap 5 cents)
SMOKE = False  # True: a few questions everywhere, a tiny LongMemEval subset and a 10 cent judge cap, to test every stage
HERE = Path(__file__).resolve().parent


def _v0_dir():
    hits = glob.glob("/kaggle/input/**/experiments/v0_2026-10-22/dama_paper_v0.py", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / "v0_2026-10-22"


def _secret(label):
    """Read a Kaggle secret before anything is pip-installed: v0's installs upgrade protobuf, after which the secrets
    client (gRPC) fails with ConnectionError. Kept in memory only; never logged or written."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(label), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


_DS_KEY, _DS_KEY_ERR = _secret("DEEPSEEK_API_KEY") if os.path.isdir("/kaggle/working") else (os.environ.get("DEEPSEEK_API_KEY"), "")
sys.path.insert(0, str(_v0_dir()))
import dama_paper_v0 as v0     # data, reader (Qwen2.5-7B), Qwen judge, statistics, LLM cache shared with v0's runs
import subprocess
if v0.ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"], check=False)
import numpy as np
import torch
from dama_agent.hybrid import FEATURES, HybridConfig, HybridReader
from dama_agent.fastmodels import OnnxCrossEncoder, OnnxEncoder

v0.SMOKE = SMOKE
v0.TIME_LIMIT_H, v0.FINAL_LIMIT_H = (2.0, 2.5) if SMOKE else (9.0, 10.3)
FINAL_H = v0.FINAL_LIMIT_H
log, stage, OUT, D, STATE, SEED = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE, v0.SEED
CATS = v0.CATS

CE_CANDIDATES = ["cross-encoder/ms-marco-TinyBERT-L2-v2", "cross-encoder/ms-marco-MiniLM-L2-v2",
                 "cross-encoder/ms-marco-MiniLM-L4-v2", "cross-encoder/ms-marco-MiniLM-L6-v2"]
N_CHOICES = (16, 32, 64)
POOL_CHOICES = [(128, 0, 0), (128, 32, 0), (128, 32, 16), (128, 64, 32), (192, 32, 16), (256, 32, 16), (256, 64, 32)]
KEY_CHOICES = ("text", "speaker", "prev")
NEIGHBOUR_CHOICES = (0, 4)
B_CHOICES = (1000, 1500, 3000)
DEADLINE_MS = 100.0
CE_MAX_LEN = 128      # question + one turn; halves the CPU cost against 256 (LoCoMo pairs are almost always shorter)
DEV_SAMPLE = 16 if SMOKE else 150
LME_PER_TYPE = 2 if SMOKE else 20                     # the "core" subset, every method incl. B1
LME_B1_PER_TYPE = 2 if SMOKE else 10                  # B1 prompts are ~30k tokens: ~65 s each on 2x T4
LME_EXTRA = {} if SMOKE else {"knowledge-update": 78, "temporal-reasoning": 60}   # all of them for H4 (no B1)
LME_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LME_TURN_CHARS = 2000
EXPORT = Path("/tmp/dama_onnx")      # trial exports; the chosen CPU models are copied to the output at the end
DS_CAP_USD = (0.05 if MODE == "judge" else 0.10) if SMOKE else 2.00
MAIN_METHODS = ["B1", "B2", "B3", "B4", "DAMA_v0", "DAMA_H"]      # + DAMA_H_compact when chosen
CONV79_METHODS = ["DAMA_v1_conv7-9", "DAMA_HT_conv7-9"]
V1 = {"settings": {}, "dev": {}, "ds": {}, "lme": {}, "latency": {}, "notes": []}
R = {}   # per conversation: {"reader": HybridReader, "turns": [...]}


def save_v1():
    (OUT / "results_v1.json").write_text(json.dumps(V1, indent=1, default=str))


def render_turn(t):
    return v0.render(t)


def pair_text(t):
    return f"{t['speaker']}: {t['text']}"


def prev_index(turns):
    """Index of the previous turn of the same session (-1 for a session's first turn)."""
    out, last = [], {}
    for j, t in enumerate(turns):
        s = t.get("session") or t["dia"].split(":")[0]
        out.append(last.get(s, -1)); last[s] = j
    return out


def key_texts(turns, key):
    prev = prev_index(turns)
    if key == "text":
        return [t["text"] for t in turns]
    if key == "speaker":
        return [pair_text(t) for t in turns]
    return [(turns[p]["text"][:200] + "\n" if p >= 0 else "") + pair_text(t) for t, p in zip(turns, prev)]


# ----------------------------------------------------------------------------------------------- outside judge
class DeepSeek:
    """deepseek-flash (DeepSeek-V4.1-Flash), thinking off, temperature 0. Cached per (model, messages, max_tokens) in
    the shared SQLite cache; the spend is tracked from the API's usage and capped. Prices USD per 1M tokens (off-peak;
    peak = 01-04 and 06-10 UTC on weekdays costs double): cache hit 0.003, miss 0.15, output 0.60."""
    MODEL, URL = "deepseek-flash", "https://api.deepseek.com/chat/completions"

    def __init__(self, key, cap_usd):
        self._key, self.cap = key, cap_usd
        self.spent, self.calls, self.cached, self.failed, self.skipped_cap = 0.0, 0, 0, 0, 0
        self.usage = Counter()

    @staticmethod
    def peak(t=None):
        g = time.gmtime(t)
        return g.tm_wday < 5 and (1 <= g.tm_hour < 4 or 6 <= g.tm_hour < 10)

    def cost(self, u):
        m = 2.0 if self.peak() else 1.0
        miss = u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - u.get("prompt_cache_hit_tokens", 0))
        return m * (u.get("prompt_cache_hit_tokens", 0) * 0.003 + miss * 0.15 + u.get("completion_tokens", 0) * 0.60) / 1e6

    def _post(self, messages, max_tokens):
        import requests
        body = {"model": self.MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "thinking": {"type": "disabled"}, "stream": False}
        for attempt in range(4):
            try:
                r = requests.post(self.URL, json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=90)
                if r.status_code == 200:
                    j = r.json()
                    return (j["choices"][0]["message"].get("content") or ""), j.get("usage", {})
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt + random.random()); continue
                return None, {"http": r.status_code}
            except Exception:
                time.sleep(2 ** attempt)
        return None, {"http": "retries exhausted"}

    def many(self, msgs, max_tokens=12, workers=16):
        keys = [v0.Cache.key(self.MODEL, [m, max_tokens]) for m in msgs]
        out, todo = [None] * len(msgs), []
        for i, k in enumerate(keys):
            hit = v0.CACHE.get(k)
            if hit:
                out[i] = hit[0]; self.cached += 1
            else:
                todo.append(i)
        with ThreadPoolExecutor(workers) as ex:
            for s in range(0, len(todo), 200):
                chunk = todo[s:s + 200]
                if self.spent >= self.cap:
                    self.skipped_cap += len(todo) - s; break
                for i, (o, u) in zip(chunk, ex.map(lambda i: self._post(msgs[i], max_tokens), chunk)):
                    if o is None:
                        self.failed += 1; continue
                    self.spent += self.cost(u); self.calls += 1
                    self.usage.update({k: v for k, v in u.items() if isinstance(v, int)})
                    v0.CACHE.put(keys[i], self.MODEL, o, int(u.get("prompt_tokens", 0)), 0.0); out[i] = o
        return out

    def summary(self):
        return {"model": self.MODEL, "api_calls": self.calls, "cached": self.cached, "failed": self.failed,
                "skipped_by_cap": self.skipped_cap, "usd": round(self.spent, 4), "cap_usd": self.cap, "usage": dict(self.usage)}


JUDGE_SYS = ("You grade answers to questions about a conversation. Mark CORRECT if the model answer contains the same key "
             "information as the gold answer, even if worded differently or shorter. For dates, the same date in another format is "
             "CORRECT; an off-by-one-day error caused by relative wording (\"yesterday\") is CORRECT only if the gold answer is itself "
             "relative. Otherwise mark WRONG.\nReply with JSON only: {\"label\": \"CORRECT\"} or {\"label\": \"WRONG\"}")
DECLINE_SYS = ("Does the model answer decline to answer, i.e. say that the information is not available or not mentioned?\n"
               "Reply with JSON only: {\"declines\": true} or {\"declines\": false}")


def ds_client():
    if D.get("ds") is None:            # retried until found: a secret attached during the run is picked up later
        key, why = _DS_KEY, _DS_KEY_ERR
        for attempt in range(0 if key else 3):
            try:
                from kaggle_secrets import UserSecretsClient
                key = UserSecretsClient().get_secret("DEEPSEEK_API_KEY")
                break
            except Exception as e:   # the message names the problem (not attached, unreachable); it never contains the key
                why = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(5 * (attempt + 1))
        if not key:
            key = os.environ.get("DEEPSEEK_API_KEY")
            names = sorted(k for k in os.environ if k.startswith("KAGGLE") or "SECRET" in k or "PROXY" in k.upper())
            log("secret lookup failed; environment variable names present (values not shown):", names)
        D["ds"] = DeepSeek(key, DS_CAP_USD) if key else None
        log("DeepSeek judge:", "ready (key read at start-up)" if key else f"no DEEPSEEK_API_KEY secret ({why or 'empty'}); outside judge not run")
    return D["ds"]


def ds_judge_rows(rows):
    """Adds r['ds_correct'] (None if not judged): the same rules as the Qwen judge, graded by deepseek-flash."""
    ds = ds_client()
    if ds is None:
        return rows
    main = [r for r in rows if r["cat"] != 5]
    outs = ds.many([[{"role": "system", "content": JUDGE_SYS},
                     {"role": "user", "content": f"Question: {r['question']}\nGold answer: {r['gold']}\nModel answer: {r['answer']}"}] for r in main])
    for r, o in zip(main, outs):
        m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', o or "") or re.search(r"\b(CORRECT|WRONG)\b", o or "")
        r["ds_correct"] = (m.group(1) == "CORRECT") if m else None
    adv = [r for r in rows if r["cat"] == 5]
    need = [r for r in adv if "not mentioned" not in r["answer"].lower()]
    outs = ds.many([[{"role": "system", "content": DECLINE_SYS}, {"role": "user", "content": f"Question: {r['question']}\nModel answer: {r['answer']}"}] for r in need])
    dec = {id(r): (None if o is None else bool(re.search(r'"declines"\s*:\s*true', o))) for r, o in zip(need, outs)}
    for r in adv:
        r["ds_correct"] = True if "not mentioned" in r["answer"].lower() else dec.get(id(r))
    return rows


def ds_summary(rows):
    main = [r for r in rows if r["cat"] != 5 and r.get("ds_correct") is not None]
    adv = [r for r in rows if r["cat"] == 5 and r.get("ds_correct") is not None]
    acc = [float(r["ds_correct"]) for r in main]
    return {"n": len(rows), "judged": len(main) + len(adv), "acc": float(np.mean(acc)) if acc else None, "acc_ci": v0.boot_ci(acc),
            "abstention": float(np.mean([r["ds_correct"] for r in adv])) if adv else None,
            "by_category": {CATS[c]: float(np.mean([r["ds_correct"] for r in rows if r["cat"] == c and r.get("ds_correct") is not None]))
                            for c in CATS if any(r["cat"] == c and r.get("ds_correct") is not None for r in rows)},
            "qwen_agreement": float(np.mean([r["ds_correct"] == r["correct"] for r in main])) if main else None,
            "qwen_kappa": _kappa([r["ds_correct"] for r in main], [r["correct"] for r in main])}


def _kappa(a, b):
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None
    except Exception:
        return None


def paired(a_rows, b_rows, key="ds_correct"):
    """DAMA - baseline on shared category 1-4 questions: mean difference, bootstrap CI, McNemar p."""
    b = {r["qid"]: r.get(key) for r in b_rows if r["cat"] != 5}
    pairs = [(float(r[key]), float(b[r["qid"]])) for r in a_rows if r["cat"] != 5 and r.get(key) is not None and b.get(r["qid"]) is not None]
    if not pairs:
        return None
    x, y = np.array(pairs).T
    return {"n": len(pairs), "diff": float((x - y).mean()), "ci": v0.boot_ci(x - y), "mcnemar_p": v0.mcnemar_p(x.astype(bool), y.astype(bool))}


# ----------------------------------------------------------------------------------------------- stages: setup
@stage("v1_setup")
def stage_setup():
    from sentence_transformers import SentenceTransformer
    v0.stage_data()
    D["v1_bge"] = SentenceTransformer(v0.EMBED_ID, device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        sample = [q["question"] for q in D["questions"][:50]]
        D["qenc"] = OnnxEncoder.export(v0.EMBED_ID, EXPORT / "bge", prefix=v0.BGE_PREFIX, sample=sample, log=log)
        D["qenc_kind"] = f"onnx {D['qenc'].variant}"
        enc = D["qenc"].encode
    except Exception as e:
        log("ONNX query encoder failed; using sentence-transformers on CPU", type(e).__name__, str(e)[:300])
        cpu = SentenceTransformer(v0.EMBED_ID, device="cpu")
        enc = lambda qs: cpu.encode([v0.BGE_PREFIX + q for q in qs], normalize_embeddings=True, convert_to_numpy=True)
        D["qenc_kind"] = "sentence-transformers cpu"
    D["encode_query"] = enc
    D["qv"] = dict(zip([q["qid"] for q in D["questions"]], enc([q["question"] for q in D["questions"]])))
    V1["settings"]["query_encoder"] = D["qenc_kind"]
    log("v1 setup: query encoder", D["qenc_kind"], "questions", len(D["questions"]))


@stage("v1_baselines")
def stage_baselines():
    """v0's per-question rows (version 8's answers and Qwen labels), never re-run. Every run copies them, complete, to
    paper/baselines_v8/, so a later run (which mounts this run's output) still finds them, even after a smoke run."""
    import shutil
    keep = set(MAIN_METHODS + CONV79_METHODS) | {"topology_pull", "topology_sidecar"}
    files = sorted(glob.glob("/kaggle/input/**/paper/baselines_v8/*.jsonl", recursive=True)) or \
        sorted(glob.glob("/kaggle/input/**/paper/methods/*.jsonl", recursive=True))
    (OUT / "baselines_v8").mkdir(exist_ok=True)
    files = [f for f in files if Path(f).stem in keep or Path(f).stem.startswith("abl_")]
    for f in files + sorted(glob.glob("/kaggle/input/**/paper/baselines_v8/judge_audit.csv", recursive=True) or
                            glob.glob("/kaggle/input/**/paper/judge_audit.csv", recursive=True))[:1]:
        shutil.copy(f, OUT / "baselines_v8" / Path(f).name)        # everything first: a later failure must not lose files
    qids = {q["qid"] for q in D["questions"]}
    for f in files:
        rows = [r for r in (json.loads(l) for l in open(f)) if r["qid"] in qids]
        if rows:
            v0.ROWS[Path(f).stem] = rows; v0.RESULTS[Path(f).stem] = safe_summary(rows)
    log("v1 baselines loaded:", {k: len(v) for k, v in v0.ROWS.items()})


def safe_summary(rows):
    """v0.summarize, or the accuracy part only for rows without token counts (the topology runs)."""
    try:
        return v0.summarize(rows)
    except TypeError:
        main = [float(r["correct"]) for r in rows if r["cat"] != 5]
        return {"n": len(rows), "acc": float(np.mean(main)) if main else None, "acc_ci": v0.boot_ci(main), "mean_tokens": float("nan")}


@stage("v1_restore")
def stage_restore():
    """Rebuild round-1 files that a run did not carry forward, exactly, from the attached LLM cache: topology_sidecar
    (same prompts, so every reader and judge call is a cache hit) and judge_audit.csv (v0's own seeded sampler)."""
    missing = []
    if "topology_sidecar" not in v0.ROWS and "DAMA_v0" in v0.ROWS:
        load_reader(cache_only=MODE == "judge")
        test = v0.test_subset()
        pk = {r["qid"]: r for r in v0.ROWS["DAMA_v0"]}
        rows, hits0 = [], len(v0.CACHE.db.execute("SELECT 1 FROM llm").fetchall())
        msgs = [v0.reader_messages(q["question"], pk[q["qid"]]["context"], v0.SIDECAR_SYSTEM) for q in test]
        outs, ptoks, ms = v0.READER(msgs, max_new=96, tools=[v0.RECALL_TOOL])
        for q, o, pt, t in zip(test, outs, ptoks, ms):
            called = bool(re.search(r"<tool_call>", o))
            ev = set(q["evidence"])
            rows.append({"method": "topology_sidecar", "qid": q["qid"], "ci": q["ci"], "cat": q["cat"], "question": q["question"], "gold": q["gold"],
                         "answer": re.sub(r"<tool_call>.*?(</tool_call>|$)", "", o, flags=re.S).strip(), "context": None, "input_tokens": None,
                         "reader_prompt_tokens": pt, "reader_ms": t, "read_ms": None, "evidence_all": pk[q["qid"]].get("evidence_all"),
                         "calls": int(called), "pull_all": None, "budget_limited": None, "flags": "restored from cache" + ("; tool call not followed" if called else "")})
        v0.judge_rows(rows)
        new_calls = len(v0.CACHE.db.execute("SELECT 1 FROM llm").fetchall()) - hits0
        with (OUT / "baselines_v8" / "topology_sidecar.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        v0.ROWS["topology_sidecar"] = rows; v0.RESULTS["topology_sidecar"] = safe_summary(rows)
        missing.append(f"topology_sidecar ({len(rows)} rows, {new_calls} LLM calls not in the cache, {sum(r['calls'] for r in rows)} tool calls)")
    if not (OUT / "baselines_v8" / "judge_audit.csv").exists():
        v0.stage_audit()
        import shutil
        shutil.copy(OUT / "judge_audit.csv", OUT / "baselines_v8" / "judge_audit.csv")
        missing.append("judge_audit.csv (v0 sampler, seed 13)")
    V1["restored"] = missing
    log("v1 restored:", missing or "nothing needed")


def dev_m():
    return [q for q in D["questions"] if q["split"] == "dev" and q["cat"] != 5 and q["evidence"]]


def covered(q, turns, idx):
    return set(q["evidence"]) <= {turns[int(i)]["dia"] for i in idx}


@stage("v1_index")
def stage_index():
    """Dense key chosen on dev (all-evidence coverage of dense top-128), then one HybridReader per conversation."""
    vecs, cov = {}, {}
    for key in KEY_CHOICES:
        vecs[key] = {c["ci"]: D["v1_bge"].encode(key_texts(c["turns"], key), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
                     for c in D["convs"]}
        hits = []
        for q in dev_m():
            d = vecs[key][q["ci"]] @ D["qv"][q["qid"]]
            hits.append(covered(q, D["convs"][q["ci"]]["turns"], np.argsort(-d)[:128]))
        cov[key] = float(np.mean(hits)) if hits else 0.0
    best = max(KEY_CHOICES, key=lambda k: (round(cov[k], 4), -KEY_CHOICES.index(k)))
    V1["settings"]["key"] = {"value": best, "dev_dense128_coverage": cov, "rule": "highest dev all-evidence coverage of dense top-128"}
    D["key"] = best
    for c in D["convs"]:
        turns = c["turns"]
        r = HybridReader(D["encode_query"], config=HybridConfig())
        r.add([render_turn(t) for t in turns], vecs[best][c["ci"]], pair_texts=[pair_text(t) for t in turns],
              ntok=[v0.ntok(render_turn(t)) + 1 for t in turns], prev=prev_index(turns))
        R[c["ci"]] = {"reader": r, "turns": turns}
    log("v1 key", V1["settings"]["key"])

    # pool composition (coverage on dev; smallest mean pool within 1 point of the best)
    res = {}
    for dk, bk, ek in POOL_CHOICES:
        cfg = HybridConfig(dense_k=dk, bm25_k=bk, entity_k=ek)
        hits, sizes = [], []
        for q in dev_m():
            idx, _, _ = R[q["ci"]]["reader"].pool(q["question"], D["qv"][q["qid"]], cfg)
            hits.append(covered(q, R[q["ci"]]["turns"], idx)); sizes.append(len(idx))
        res[(dk, bk, ek)] = (float(np.mean(hits)), float(np.mean(sizes)))
    top = max(v[0] for v in res.values())
    choice = min((k for k, v in res.items() if v[0] >= top - 0.01), key=lambda k: res[k][1])
    for c in R.values():
        c["reader"].cfg = HybridConfig(dense_k=choice[0], bm25_k=choice[1], entity_k=choice[2])
    V1["settings"]["pool"] = {"value": {"dense_k": choice[0], "bm25_k": choice[1], "entity_k": choice[2]},
                              "dev": {f"{k[0]}/{k[1]}/{k[2]}": {"coverage": v[0], "mean_pool": v[1]} for k, v in res.items()},
                              "rule": "smallest mean pool within 1 point of the best dev all-evidence coverage"}
    # the same coverage on test, reported only
    tm = [q for q in D["questions"] if q["split"] == "test" and q["cat"] != 5 and q["evidence"]]
    V1["settings"]["pool"]["test_coverage"] = float(np.mean([covered(q, R[q["ci"]]["turns"], R[q["ci"]]["reader"].pool(q["question"], D["qv"][q["qid"]])[0]) for q in tm])) if tm else None
    log("v1 pool", V1["settings"]["pool"]["value"], {k: V1["settings"]["pool"]["dev"][k] for k in list(V1["settings"]["pool"]["dev"])}, "test", V1["settings"]["pool"]["test_coverage"])
    save_v1()


@stage("v1_prior")
def stage_prior():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    def pairs(qs):
        X, y = [], []
        for q in qs:
            r = R[q["ci"]]
            idx, feats, _ = r["reader"].pool(q["question"], D["qv"][q["qid"]])
            ev = set(q["evidence"])
            X += list(feats); y += [int(r["turns"][int(i)]["dia"] in ev) for i in idx]
        return np.array(X), np.array(y)
    X, y = pairs(dev_m())
    prior = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(X, y)
    for c in R.values():
        c["reader"].prior = prior
    D["prior"] = prior
    tm = [q for q in D["questions"] if q["split"] == "test" and q["cat"] != 5 and q["evidence"]]
    Xt, yt = pairs(tm)
    V1["settings"]["prior"] = {"features": FEATURES, "coef": [round(float(c), 4) for c in prior[-1].coef_[0]], "dev_pairs": int(len(y)),
                               "dev_positives": int(y.sum()), "test_ece": v0.ece(prior.predict_proba(Xt)[:, 1], yt) if len(yt) else None}
    log("v1 prior", V1["settings"]["prior"])


def final_p(r, q, qv, n, scorer):
    """Pool, prior and cross-encoder probabilities exactly as HybridReader.read_many computes them."""
    idx, feats, _ = r.pool(q, qv)
    p = r.prior_p(feats).astype(np.float64)
    if n and scorer is not None and len(idx):
        top = np.argsort(-p, kind="stable")[:n]
        p[top] = scorer([q] * len(top), [r.pair_texts[int(idx[j])] for j in top])
    return idx, p


@stage("v1_ranker")
def stage_ranker():
    """Cross-encoder and N: dev evidence recall of the packed packet (B 1500 and 3000) for each model and N, against the
    CPU cost of the whole read path; the best recall whose estimated p50 fits 100 ms wins."""
    from sentence_transformers import CrossEncoder
    from sklearn.linear_model import LogisticRegression
    qs = dev_m()
    tables, platt, ms = {}, {}, {}
    # CPU cost of everything but the cross-encoder (ONNX query encoder, pool, prior, packing), on dev questions
    t_base = []
    for q in qs[:80]:
        r = R[q["ci"]]["reader"]; t = time.perf_counter(); r.read(q["question"], rerank_n=0); t_base.append((time.perf_counter() - t) * 1000)
    base = v0.pct(t_base, 50)
    for name in CE_CANDIDATES:
        try:
            ce = CrossEncoder(name, device="cuda" if torch.cuda.is_available() else "cpu", max_length=CE_MAX_LEN)
            z_all, y_all, tops = [], [], {}
            for q in qs:
                r = R[q["ci"]]
                idx, p = final_p(r["reader"], q["question"], D["qv"][q["qid"]], 0, None)
                top = np.argsort(-p, kind="stable")[:max(N_CHOICES)]
                docs = [r["reader"].pair_texts[int(idx[j])] for j in top]
                z = ce.predict([(q["question"], d) for d in docs], batch_size=256, show_progress_bar=False)
                tops[q["qid"]] = (q["question"], docs, np.asarray(z, dtype=np.float64))
                ev = set(q["evidence"])
                z_all += list(z); y_all += [int(r["turns"][int(idx[j])]["dia"] in ev) for j in top]
            pl = LogisticRegression(random_state=SEED).fit(np.array(z_all).reshape(-1, 1), np.array(y_all))
            a, b = float(pl.coef_[0][0]), float(pl.intercept_[0])
            del ce; torch.cuda.empty_cache()
            onnx = OnnxCrossEncoder.export(name, EXPORT / name.split("/")[-1], platt=(a, b), max_length=CE_MAX_LEN,
                                           sample=[(q["question"], d) for q in qs[:4] for d in tops[q["qid"]][1][:8]], log=log)
            per = {}
            for n in N_CHOICES:
                tt = []
                for q in qs[:15]:
                    docs = tops[q["qid"]][1][:n]
                    t = time.perf_counter(); onnx.probs([q["question"]] * len(docs), docs); tt.append((time.perf_counter() - t) * 1000)
                per[n] = v0.pct(tt, 50)
            # only a model that exported and was timed on CPU can be chosen
            D.setdefault("ce_onnx", {})[name] = onnx
            platt[name], ms[name] = (a, b), per
            tables[name] = {(qt, d): 1 / (1 + math.exp(-(a * zz + b))) for qt, docs, z in tops.values() for d, zz in zip(docs, z)}
            log(f"ranker {name}: platt {a:.3f} {b:.3f}, CPU ms by N {per}, onnx {onnx.variant}")
        except Exception as e:
            log("ranker failed", name, type(e).__name__, str(e)[:300]); STATE["errors"][f"ranker {name}"] = str(e)[:200]
    # dev recall of packed packets
    grid = {}
    for name in ["none"] + list(tables):
        for n in ([0] if name == "none" else N_CHOICES):
            look = tables.get(name)
            scorer = (lambda qq, dd, look=look: np.array([look.get((a, b), 0.0) for a, b in zip(qq, dd)])) if look else None
            rec = {}
            for B in (1500, 3000):
                hits = []
                for q in qs:
                    r = R[q["ci"]]
                    r["reader"].scorer = scorer
                    out = r["reader"].read_many([q["question"]], qvecs=[D["qv"][q["qid"]]], budget_tokens=B, rerank_n=n, neighbours=0)[0]
                    hits.append(covered(q, r["turns"], out["ids"]))
                rec[B] = float(np.mean(hits)) if hits else 0.0
            est = base + (ms.get(name, {}).get(n, 0.0) if n else 0.0)
            grid[f"{name}|{n}"] = {"recall_1500": rec[1500], "recall_3000": rec[3000], "est_cpu_p50_ms": est}
    ok = {k: v for k, v in grid.items() if v["est_cpu_p50_ms"] <= DEADLINE_MS} or {"none|0": grid["none|0"]}
    best = max(ok, key=lambda k: ((ok[k]["recall_1500"] + ok[k]["recall_3000"]) / 2, -ok[k]["est_cpu_p50_ms"]))
    name, n = best.split("|"); n = int(n)
    D["ce_name"], D["N"] = (None if name == "none" else name), n
    D["scorer"] = D["ce_onnx"][name].probs if name != "none" else None
    for c in R.values():
        c["reader"].scorer = D["scorer"]
        c["reader"].cfg.rerank_n = n
    V1["settings"]["ranker"] = {"value": {"model": D["ce_name"], "N": n, "platt": platt.get(name), "onnx": D["ce_onnx"][name].variant if name != "none" else None},
                                "base_cpu_p50_ms": base, "ce_cpu_ms_by_N": ms, "dev_grid": grid,
                                "rule": "highest mean dev evidence recall at B 1500 and 3000 among settings whose estimated CPU p50 <= 100 ms"}
    # conformal lambda on the final probabilities (dev), for the admission ablation
    s = []
    for q in qs:
        r = R[q["ci"]]
        idx, p = final_p(r["reader"], q["question"], D["qv"][q["qid"]], n, D["scorer"])
        dias = {r["turns"][int(i)]["dia"]: float(pp) for i, pp in zip(idx, p)}
        s.append(min(dias[e] for e in q["evidence"]) if set(q["evidence"]) <= set(dias) else -math.inf)
    D["lam_h"] = {a: v0.fit_lambda(s, a) for a in v0.ALPHAS}
    V1["settings"]["lambda_h"] = {str(a): D["lam_h"][a] for a in v0.ALPHAS}
    V1["settings"]["lambda_h_dev_minus_inf_share"] = float(np.mean([x == -math.inf for x in s])) if s else None
    import shutil
    for src in [EXPORT / "bge"] + ([EXPORT / name.split("/")[-1]] if name != "none" else []):
        if src.exists():
            shutil.copytree(src, OUT / "onnx" / src.name, dirs_exist_ok=True)
    log("v1 ranker", V1["settings"]["ranker"]["value"], "base", base, "lambda", V1["settings"]["lambda_h"])
    save_v1()


# ----------------------------------------------------------------------------------------------- reading
def h_ctx(q, B, m, lam=-math.inf, n=None, pool=None):
    r = R[q["ci"]]
    over = {"budget_tokens": B, "neighbours": m, "lam": lam}
    if n is not None:
        over["rerank_n"] = n
    if pool is not None:
        over.update(pool)
    t = time.perf_counter()
    out = r["reader"].read(q["question"], **over)
    ms = (time.perf_counter() - t) * 1000
    turns = [r["turns"][i] for i in out["ids"]]
    return {"context": v0.render_many(turns) if turns else v0.NOT_FOUND, "dia": [t["dia"] for t in turns], "read_ms": ms,
            "budget_limited": out["budget_limited"]}


class CacheOnlyLLM(v0.LLM):
    """The reader without its weights (no GPU): answers every call from the shared cache and fails on a miss."""
    def __init__(self, name):
        self.name, self.model, self.max_ctx = name, None, 32768
        self.tok = lambda t, **kw: {"input_ids": []}

    def _batch(self, idx, texts, lens, keys, res, max_new):
        raise RuntimeError(f"{len(idx)} reader calls are not in the cache (judge mode has no GPU)")

    _long_group = _batch

    def text(self, messages, tools=None):
        return ""


def load_reader(cache_only=False):
    if v0.READER is None:
        v0.READER = CacheOnlyLLM(v0.READER_ID) if cache_only else v0.LLM(v0.READER_ID)
        log("reader loaded", "cache only" if cache_only else v0.READER.max_ctx)


@stage("v1_dev_reader")
def stage_dev_reader():
    """Neighbours and budget on a fixed dev sample (categories 1-4, seed 13): first m at B 3000, then B with the better m."""
    load_reader()
    dev = [q for q in D["questions"] if q["split"] == "dev" and q["cat"] != 5]
    sample = random.Random(SEED).sample(dev, min(DEV_SAMPLE, len(dev)))
    acc = {}
    for m in NEIGHBOUR_CHOICES:
        rows = v0.evaluate(f"dev_H_m{m}_B3000", sample, lambda q, m=m: h_ctx(q, 3000, m))
        acc[(m, 3000)] = v0.RESULTS[f"dev_H_m{m}_B3000"]["acc"]
    m_best = max(NEIGHBOUR_CHOICES, key=lambda m: (acc[(m, 3000)], -m))
    for B in B_CHOICES:
        if (m_best, B) not in acc:
            v0.evaluate(f"dev_H_m{m_best}_B{B}", sample, lambda q, B=B: h_ctx(q, B, m_best))
            acc[(m_best, B)] = v0.RESULTS[f"dev_H_m{m_best}_B{B}"]["acc"]
    by_b = {B: acc[(m_best, B)] for B in B_CHOICES}
    best = max(by_b.values())
    B_main = min(B for B in B_CHOICES if by_b[B] == best)
    compact = [B for B in B_CHOICES if B < B_main and by_b[B] >= best - 0.03]
    D["m"], D["B"], D["B_compact"] = m_best, B_main, (min(compact) if compact else None)
    V1["settings"]["neighbours"] = {"value": m_best, "dev_acc_B3000": {m: acc[(m, 3000)] for m in NEIGHBOUR_CHOICES}}
    V1["settings"]["B"] = {"value": B_main, "compact": D["B_compact"], "dev_acc": by_b, "n_dev_sample": len(sample),
                           "rule": "highest dev-sample accuracy (ties -> smaller B); compact = smallest smaller B within 3 points"}
    log("v1 dev choice: neighbours", m_best, "B", B_main, "compact", D["B_compact"], by_b)
    save_v1()


@stage("v1_test_main")
def stage_test_main():
    test = [q for q in D["questions"] if q["split"] == "test"]
    v0.evaluate("DAMA_H", test, lambda q: h_ctx(q, D["B"], D["m"]), extra={"B": D["B"], "neighbours": D["m"]})
    if D.get("B_compact"):
        v0.evaluate("DAMA_H_compact", test, lambda q: h_ctx(q, D["B_compact"], D["m"]), extra={"B": D["B_compact"]})


@stage("v1_judge_main", limit_h=FINAL_H)
def stage_judge_main():
    """DeepSeek re-judges every answer: the main methods first, then topology and all ablations (v0's and DAMA-H's)."""
    main = [m for m in MAIN_METHODS + ["DAMA_H_compact"] + CONV79_METHODS if m in v0.ROWS]
    extra = sorted(m for m in v0.ROWS if m.startswith(("topology_", "abl_")))
    for m in main + extra:
        ds_judge_rows(v0.ROWS[m])
        V1["ds"][m] = ds_summary(v0.ROWS[m])
        with (OUT / "methods" / f"{m}.jsonl").open("w") as fh:
            for r in v0.ROWS[m]:
                fh.write(json.dumps(r) + "\n")
    base = [b for b in ("B1", "B2", "B3", "B4") if b in v0.ROWS]
    for a in ("DAMA_v0", "DAMA_H", "DAMA_H_compact"):
        if a in v0.ROWS:
            for key, out in (("ds_correct", "ds_paired"), ("correct", "qwen_paired")):
                V1.setdefault(out, {})[a] = {b: paired(v0.ROWS[a], v0.ROWS[b], key=key) for b in base + (["DAMA_v0"] if a != "DAMA_v0" else [])}
    ids = {q["qid"] for q in v0.test_subset()}
    V1["ds_subset"] = {m: ds_summary([r for r in v0.ROWS[m] if r["qid"] in ids]) for m in ("DAMA_v0", "DAMA_H", "B3", "B1") if m in v0.ROWS}
    V1["qwen_subset"] = {m: v0.summarize([r for r in v0.ROWS[m] if r["qid"] in ids]) for m in ("DAMA_v0", "DAMA_H", "B3", "B1") if m in v0.ROWS}
    if "DAMA_HT_conv7-9" in v0.ROWS:
        ids79 = {r["qid"] for r in v0.ROWS["DAMA_HT_conv7-9"]}
        V1["ds_conv79"] = {m: ds_summary([r for r in v0.ROWS[m] if r["qid"] in ids79]) for m in main if m in v0.ROWS}
    if D.get("ds"):
        V1["ds_cost"] = D["ds"].summary()
    log("v1 DeepSeek judge", {m: (round(v["acc"], 3) if v.get("acc") is not None else None) for m, v in V1["ds"].items()}, V1.get("ds_cost"))
    save_v1()


@stage("v1_audit", limit_h=FINAL_H)
def stage_audit():
    """Blind sheet for the team's hand check (round 2, part 2): v0's 100 stratified judge_audit items, shuffled (seed 13),
    judge labels removed; a separate key holds each judge's label. Nobody here labels anything."""
    import csv
    src = [str(OUT / "baselines_v8" / "judge_audit.csv")] if (OUT / "baselines_v8" / "judge_audit.csv").exists() else \
        sorted(glob.glob("/kaggle/input/**/paper/judge_audit.csv", recursive=True))
    if not src:
        raise FileNotFoundError("judge_audit.csv from version 8 is not attached")
    items = list(csv.DictReader(open(src[0])))
    look = {(r["method"], r["question"], r["answer"]): r for m in ("B1", "B2", "B3", "B4", "DAMA_v0") for r in v0.ROWS.get(m, [])}
    order = list(range(len(items))); random.Random(SEED).shuffle(order)
    lab = lambda v: "n/a" if v is None else ("CORRECT" if v else "WRONG")
    with (OUT / "human_audit_sheet.csv").open("w", newline="") as fs, (OUT / "human_audit_key.csv").open("w", newline="") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["id", "method", "question", "gold", "answer", "human_label"])
        wk.writerow(["id", "method", "category", "qwen7b_label", "deepseek_label", "phi35_label"])
        for k, i in enumerate(order):
            it = items[i]; r = look.get((it["method"], it["question"], it["answer"]))
            ws.writerow([f"A{k + 1:03d}", it["method"], it["question"], it["gold"], it["answer"], ""])
            wk.writerow([f"A{k + 1:03d}", it["method"], it["category"], it["label"], lab(r.get("ds_correct") if r else None), "n/a"])
    V1["audit"] = {"items": len(items), "matched_to_rows": sum(1 for it in items if (it["method"], it["question"], it["answer"]) in look)}
    log("v1 human audit sheet", V1["audit"])


@stage("v1_loco", limit_h=FINAL_H)
def stage_loco():
    """Leave-one-conversation-out check of the conformal threshold on DAMA-H's final probabilities (round 2, part 3.4):
    lambda from 9 conversations, omission measured on the 10th."""
    allq = [q for q in D["questions"] if q["cat"] != 5 and q["evidence"]]
    sc = {}
    for q in allq:
        r = R[q["ci"]]
        idx, p = final_p(r["reader"], q["question"], D["qv"][q["qid"]], D["N"], D["scorer"])
        dias = {r["turns"][int(i)]["dia"]: float(pp) for i, pp in zip(idx, p)}
        sc[q["qid"]] = min(dias[e] for e in q["evidence"]) if set(q["evidence"]) <= set(dias) else -math.inf
    out = {}
    for a in v0.ALPHAS:
        folds, inf = [], 0
        for c in sorted({q["ci"] for q in allq}):
            lam = v0.fit_lambda([sc[q["qid"]] for q in allq if q["ci"] != c], a)
            inf += lam == -math.inf
            held = [sc[q["qid"]] for q in allq if q["ci"] == c]
            folds.append(float(np.mean([x < lam or x == -math.inf for x in held])))
        out[str(a)] = {"target": a, "mean": float(np.mean(folds)), "sd": float(np.std(folds)), "min": min(folds), "max": max(folds),
                       "folds_lambda_minus_inf": inf, "per_fold": [round(x, 3) for x in folds]}
    V1["loco_h"] = out
    log("v1 LOCO DAMA-H", out)
    save_v1()


@stage("v1_trained")
def stage_trained():
    """DAMA-HT: the hybrid read path on the trained L1.2 bundle (fine-tuned retriever keys, Tier-1 rel as the
    cross-encoder, bundle calibration), conversations 7-9 (never seen in its training, validation or calibration)."""
    from dama_train.runtime import Tier1
    bundle = os.path.dirname(min(glob.glob("/kaggle/input/**/dama_tier1.json", recursive=True), key=len))
    t1 = Tier1(bundle, threads=4)
    def rendered(t):
        return f"[{t['date_raw']}] {t['speaker']}: {t['text'][:1000]}"
    def rel(qs, ds):
        out = np.zeros(len(qs)); by = defaultdict(list)
        for i, q in enumerate(qs):
            by[q].append(i)
        for q, ii in by.items():
            out[ii] = t1.relevance(q, [ds[i] for i in ii])
        return out
    readers = {}
    for ci in v0.V1_CONVS:
        turns = D["convs"][ci]["turns"]
        texts = [rendered(t) for t in turns]
        vec = np.concatenate([t1.embed([t1.memory_key(x, p) for x, p in zip(texts[s:s + 64], ([""] + texts)[s:s + 64])]) for s in range(0, len(texts), 64)])
        r = HybridReader(lambda qs: t1.embed(qs, query=True), scorer=rel, config=R[ci]["reader"].cfg)
        r.add([render_turn(t) for t in turns], vec, pair_texts=texts, ntok=[v0.ntok(render_turn(t)) + 1 for t in turns], prev=prev_index(turns))
        readers[ci] = {"reader": r, "turns": turns}
    def ctx(q):
        r = readers[q["ci"]]; t = time.perf_counter()
        out = r["reader"].read(q["question"], budget_tokens=D["B"], neighbours=D["m"], rerank_n=D["N"] or 32)
        turns = [r["turns"][i] for i in out["ids"]]
        return {"context": v0.render_many(turns) if turns else v0.NOT_FOUND, "dia": [t["dia"] for t in turns],
                "read_ms": (time.perf_counter() - t) * 1000, "budget_limited": out["budget_limited"]}
    qs = [q for q in D["questions"] if q["ci"] in v0.V1_CONVS]
    v0.evaluate("DAMA_HT_conv7-9", qs, ctx, extra={"bundle": bundle, "checkpoint": t1.meta.get("checkpoint")})
    if "DAMA_H" in v0.ROWS:
        ids = {q["qid"] for q in qs}
        V1["conv79_qwen"] = {m: v0.summarize([r for r in v0.ROWS[m] if r["qid"] in ids]) for m in MAIN_METHODS + CONV79_METHODS if m in v0.ROWS}


@stage("v1_ablations")
def stage_ablations():
    test = v0.test_subset()
    lam = D["lam_h"].get(0.2, -math.inf)
    pc = R[test[0]["ci"]]["reader"].cfg if test else HybridConfig()
    runs = {"abl_H_no_rerank": lambda q: h_ctx(q, D["B"], D["m"], n=0),
            "abl_H_dense_pool": lambda q: h_ctx(q, D["B"], D["m"], pool={"bm25_k": 0, "entity_k": 0}),
            "abl_H_conformal_a0.2": lambda q: h_ctx(q, D["B"], D["m"], lam=lam)}
    if not math.isfinite(lam):
        runs.pop("abl_H_conformal_a0.2"); V1["notes"].append("abl_H_conformal_a0.2 not run: lambda at alpha 0.2 is -inf on dev")
    for name, fn in runs.items():
        if (time.time() - v0.T_START) / 3600 > v0.TIME_LIMIT_H:
            STATE["errors"][name] = "time guard"; continue
        try:
            v0.evaluate(name, test, fn)
        except Exception as e:
            STATE["errors"][name] = f"{type(e).__name__}: {str(e)[:120]}"; log("ablation failed", name, traceback.format_exc()[-1500:])
    ids = {q["qid"] for q in test}
    V1["subset_ref"] = {m: v0.summarize([r for r in v0.ROWS[m] if r["qid"] in ids]) for m in ("DAMA_H", "B3", "B1", "DAMA_v0") if m in v0.ROWS}


# ----------------------------------------------------------------------------------------------- LongMemEval-S
LME_SYSTEM = """You answer questions about a user's past conversations with an AI assistant, using only the memory provided.
Rules:
- Use only the MEMORY section. Do not use outside knowledge.
- If the memory does not contain the answer, reply exactly: Not mentioned in the conversation.
- Answer briefly (at most 30 words). No explanation.
- Use the session dates shown and the current date for time questions."""


def lme_judge_prompt(it, resp):
    q, a, qt = it["question"], it["answer"], it["question_type"]
    if "_abs" in it["question_id"]:
        return ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model "
                "correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other "
                "information is given but the asked information is not.\n\n"
                f"Question: {q}\n\nExplanation: {a}\n\nModel Response: {resp}\n\n"
                "Does the model correctly identify the question as unanswerable? Answer yes or no only.")
    if qt == "single-session-preference":
        return ("I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if "
                "the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the "
                "rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                f"Question: {q}\n\nRubric: {a}\n\nModel Response: {resp}\n\nIs the model response correct? Answer yes or no only.")
    head = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no.")
    if qt == "knowledge-update":
        rule = (" If the response contains some previous information along with an updated answer, the response should be considered as "
                "correct as long as the updated answer is the required answer.")
    else:
        rule = (" If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you "
                "should also answer yes. If the response only contains a subset of the information required by the answer, answer no.")
        if qt == "temporal-reasoning":
            rule += (" In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of "
                     "days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the "
                     "model's response is still correct.")
    return head + rule + f"\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {resp}\n\nIs the model response correct? Answer yes or no only."


def _lme_date_key(d):
    return re.sub(r"\s*\([A-Za-z]+\)\s*", " ", d or "").strip()


def load_lme():
    path = Path("/tmp/longmemeval_s_cleaned.json")   # 277 MB: kept out of the notebook output
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(LME_URL, path)
    data = json.loads(path.read_text())
    by = defaultdict(list)
    for it in data:
        by[it["question_type"]].append(it)
    rng = random.Random(SEED)
    sample, core = [], set()
    for t in sorted(by):
        pick = rng.sample(by[t], min(max(LME_PER_TYPE, LME_EXTRA.get(t, 0)), len(by[t])))
        core |= {it["question_id"] for it in pick[:LME_PER_TYPE]}
        sample += pick
    items = []
    for it in sample:
        sess = sorted(zip(it["haystack_dates"], it["haystack_session_ids"], it["haystack_sessions"]), key=lambda x: _lme_date_key(x[0]))
        turns = []
        for date, sid, ts in sess:
            for j, t in enumerate(ts):
                turns.append({"i": len(turns), "dia": f"{sid}:{j}", "date": date, "date_raw": date, "speaker": t.get("role", "user"),
                              "text": str(t.get("content", ""))[:LME_TURN_CHARS], "session": sid, "has": bool(t.get("has_answer"))})
        items.append({"qid": it["question_id"], "type": it["question_type"], "question": it["question"], "answer": str(it["answer"]),
                      "date": it.get("question_date", ""), "turns": turns, "evidence": [t["dia"] for t in turns if t["has"]], "raw": it,
                      "core": it["question_id"] in core})
    return items, {t: len(v) for t, v in by.items()}


def lme_stats(rows):
    def acc(key, rs):
        v = [float(r[key]) for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    key = "ds_correct" if any(r.get("ds_correct") is not None for r in rows) else "qwen_correct"
    ev = [r["evidence_all"] for r in rows if r["evidence_all"] is not None]
    rd = [r["read_ms"] for r in rows if r.get("read_ms") is not None]
    types = sorted({r["type"] for r in rows})
    return {"n": len(rows), "judge": key, "ds_acc": acc("ds_correct", rows), "ds_ci": v0.boot_ci([float(r["ds_correct"]) for r in rows if r.get("ds_correct") is not None]),
            "qwen_acc": acc("qwen_correct", rows), "mean_tokens": float(np.mean([r["input_tokens"] for r in rows])) if rows else None,
            "evidence_recall": float(np.mean(ev)) if ev else None, "read_p50": v0.pct(rd, 50),
            "abstention": acc(key, [r for r in rows if r["abs"]]),
            "by_type": {t: {"n": sum(r["type"] == t for r in rows), "acc": acc(key, [r for r in rows if r["type"] == t]),
                            "ci": v0.boot_ci([float(r[key]) for r in rows if r["type"] == t and r.get(key) is not None])} for t in types}}


def lme_messages(it, ctx):
    return [{"role": "system", "content": LME_SYSTEM},
            {"role": "user", "content": f"MEMORY:\n{ctx}\n\nCURRENT DATE: {it['date']}\nQUESTION: {it['question']}\nANSWER:"}]


def lme_eval(name, items, ctx_fn):
    t0 = time.time()
    ctxs = [ctx_fn(it) for it in items]
    msgs = [lme_messages(it, c["context"]) for it, c in zip(items, ctxs)]
    outs, ptoks, ms = v0.READER(msgs, max_new=64)
    rows = []
    for it, c, m, o, pt in zip(items, ctxs, msgs, outs, ptoks):
        ev = set(it["evidence"])
        rows.append({"method": name, "qid": it["qid"], "type": it["type"], "abs": "_abs" in it["qid"], "core": it["core"], "question": it["question"],
                     "gold": it["answer"], "answer": o, "input_tokens": v0.ntok(m[0]["content"]) + v0.ntok(m[1]["content"]),
                     "reader_prompt_tokens": pt, "read_ms": c.get("read_ms"),
                     "evidence_all": (ev <= set(c.get("dia") or [])) if ev and c.get("dia") is not None else None, "flags": c.get("flags")})
    jq, _, _ = v0.READER([[{"role": "user", "content": lme_judge_prompt(it["raw"], r["answer"])}] for it, r in zip(items, rows)], max_new=6)
    for r, o in zip(rows, jq):
        r["qwen_correct"] = "yes" in (o or "").lower()
    ds = ds_client()
    if ds is not None:
        jd = ds.many([[{"role": "user", "content": lme_judge_prompt(it["raw"], r["answer"])}] for it, r in zip(items, rows)], max_tokens=6)
        for r, o in zip(rows, jd):
            r["ds_correct"] = None if o is None else ("yes" in o.lower())
    with (OUT / "methods" / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    V1["lme"][name] = {"all": lme_stats(rows), "core": lme_stats([r for r in rows if r["core"]]), "minutes": round((time.time() - t0) / 60, 1)}
    D.setdefault("lme_rows", {})[name] = rows
    log(f"LME {name}:", {k: V1["lme"][name]["all"][k] for k in ("n", "ds_acc", "qwen_acc", "mean_tokens", "evidence_recall")})
    save_v1()


@stage("v1_lme")
def stage_lme():
    from sentence_transformers import CrossEncoder
    load_reader()
    items, counts = load_lme()
    V1["lme_meta"] = {"file": "longmemeval_s_cleaned.json", "type_counts_full": counts, "n_sample": len(items), "per_type": LME_PER_TYPE,
                      "mean_turns": float(np.mean([len(it["turns"]) for it in items])),
                      "mean_history_tokens": float(np.mean([sum(v0.ntok(render_turn(t)) for t in it["turns"]) for it in items]))}
    log("LME loaded", V1["lme_meta"])
    for it in items:
        turns = it["turns"]
        vec = D["v1_bge"].encode(key_texts(turns, D["key"]), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        r = HybridReader(D["encode_query"], scorer=D.get("scorer"), prior=D.get("prior"), config=HybridConfig(**{**R[0]["reader"].cfg.__dict__}))
        r.add([render_turn(t) for t in turns], vec, pair_texts=[pair_text(t) for t in turns], ntok=[v0.ntok(render_turn(t)) + 1 for t in turns],
              prev=prev_index(turns))
        it["reader"] = r
    D["lme_items"] = items
    def h(it):
        t = time.perf_counter()
        out = it["reader"].read(it["question"], budget_tokens=D["B"], neighbours=D["m"])
        ms = (time.perf_counter() - t) * 1000
        turns = [it["turns"][i] for i in out["ids"]]
        return {"context": v0.render_many(turns) if turns else v0.NOT_FOUND, "dia": [t["dia"] for t in turns], "read_ms": ms}
    lme_eval("LME_DAMA_H", items, h)
    ce = CrossEncoder(v0.CE_ID, device="cuda" if torch.cuda.is_available() else "cpu", max_length=256)
    def b3(it):
        z = ce.predict([(it["question"], pair_text(t)) for t in it["turns"]], batch_size=256, show_progress_bar=False)
        chosen, used = [], 0
        for j in np.argsort(-np.asarray(z)):
            n = v0.ntok(render_turn(it["turns"][j])) + 1
            if used + n <= D["B"]:
                chosen.append(it["turns"][j]); used += n
        return {"context": v0.render_many(chosen) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen]}
    lme_eval("LME_B3", items, b3)
    del ce; torch.cuda.empty_cache()
    def b2(it):
        chosen, used = [], 0
        for t in reversed(it["turns"]):
            n = v0.ntok(render_turn(t)) + 1
            if used + n > D["B"]:
                break
            chosen.append(t); used += n
        return {"context": v0.render_many(chosen), "dia": [t["dia"] for t in chosen]}
    lme_eval("LME_B2", items, b2)
    h, b = D["lme_rows"].get("LME_DAMA_H", []), {r["qid"]: r for r in D["lme_rows"].get("LME_B3", [])}
    for key in ("ds_correct", "qwen_correct"):
        pr = [(float(r[key]), float(b[r["qid"]][key])) for r in h if r["type"] == "knowledge-update" and r.get(key) is not None
              and r["qid"] in b and b[r["qid"]].get(key) is not None]
        if pr:
            x, y = np.array(pr).T
            V1.setdefault("h4", {})[key] = {"n": len(pr), "dama_h": float(x.mean()), "b3": float(y.mean()), "diff": float((x - y).mean()),
                                            "ci": v0.boot_ci(x - y), "rejected": bool(x.mean() < y.mean())}
    log("H4 (knowledge-update, DAMA-H vs B3)", V1.get("h4"))


@stage("v1_lme_b1")
def stage_lme_b1():
    """Full history, truncated to the newest turns that fit the reader's 32k window (last: long prompts are the ones that
    have hit 'CUDA error: unspecified launch failure' on T4s)."""
    items = D["lme_items"]
    limit = v0.READER.max_ctx - 700
    def b1(it):
        # reader-tokenizer length of each line, counted once; keep the newest turns that fit, then verify the whole
        n = [len(v0.READER.tok(render_turn(t) + "\n")["input_ids"]) for t in it["turns"]]
        keep, used = len(n), 0
        for j in range(len(n) - 1, -1, -1):
            if used + n[j] > limit:
                break
            used += n[j]; keep = j
        turns = it["turns"][keep:]; ctx = v0.render_many(turns)
        while len(v0.READER.tok(ctx)["input_ids"]) > limit:
            turns = turns[1:]; ctx = v0.render_many(turns)
        flags = f"truncated to the newest {len(turns)} of {len(it['turns'])} turns" if len(turns) < len(it["turns"]) else None
        return {"context": ctx, "dia": [t["dia"] for t in turns], "flags": flags}
    per = defaultdict(int); sub = []
    for it in items:
        if it["core"] and per[it["type"]] < LME_B1_PER_TYPE:
            sub.append(it); per[it["type"]] += 1
    lme_eval("LME_B1", sub, b1)
    V1["lme"]["LME_B1"]["truncated_share"] = float(np.mean([bool(r["flags"]) for r in D["lme_rows"]["LME_B1"]]))


# ----------------------------------------------------------------------------------------------- latency (CPU)
@stage("v1_latency", limit_h=FINAL_H)
def stage_latency():
    """CPU, ONNX models as deployed. (1) every stage on LoCoMo test stores with the 100 ms deadline on; (2) a 10^5-turn
    store (LoCoMo turns tiled); (3) 16 agents on one memory: concurrent threads, and one read_many batch."""
    torch.set_num_threads(4)
    test = [q for q in D["questions"] if q["split"] == "test"][: (40 if SMOKE else 400)]
    def run(reader, qs, **kw):
        stages, wall, resc, late = defaultdict(list), [], [], 0
        for q in qs:
            t = time.perf_counter(); out = reader(q).read(q["question"], **kw); w = (time.perf_counter() - t) * 1000
            wall.append(w); resc.append(out["rescored"]); late += w > DEADLINE_MS
            for k, v in out["stages_ms"].items():
                stages[k].append(v)
        return {"p50": v0.pct(wall, 50), "p95": v0.pct(wall, 95), "late_share": late / max(1, len(qs)), "mean_rescored": float(np.mean(resc)) if resc else None,
                "stages_p50": {k: v0.pct(v, 50) for k, v in stages.items()}, "n": len(qs)}
    V1["latency"]["locomo_no_deadline"] = run(lambda q: R[q["ci"]]["reader"], test, budget_tokens=D["B"], neighbours=D["m"])
    V1["latency"]["locomo_deadline_100"] = run(lambda q: R[q["ci"]]["reader"], test, budget_tokens=D["B"], neighbours=D["m"], deadline_ms=DEADLINE_MS)
    # 10^5-turn store
    size = 10000 if SMOKE else 100000
    src_turns, src_vecs = [], []
    for c in D["convs"]:
        r = R[c["ci"]]["reader"]
        src_turns += c["turns"]; src_vecs.append(r.vectors)
    src_vecs = np.concatenate(src_vecs)
    big = HybridReader(D["encode_query"], scorer=D.get("scorer"), prior=D.get("prior"), config=R[0]["reader"].cfg)
    reps = size // len(src_turns) + 1
    texts, vecs, pts, nt = [], [], [], []
    for k in range(reps):
        for t in src_turns:
            texts.append(render_turn(t) + f" #{k}"); pts.append(pair_text(t)); nt.append(v0.ntok(render_turn(t)) + 1)
        vecs.append(src_vecs)
    big.add(texts[:size], np.concatenate(vecs)[:size], pair_texts=pts[:size], ntok=nt[:size])
    qs = random.Random(SEED).sample(test, min(len(test), 300))
    V1["latency"]["store_1e5"] = run(lambda q: big, qs, budget_tokens=D["B"], neighbours=0)
    V1["latency"]["store_1e5"]["size"] = size
    V1["latency"]["store_1e5_deadline_100"] = run(lambda q: big, qs, budget_tokens=D["B"], neighbours=0, deadline_ms=DEADLINE_MS)
    # 16 agents on one memory (the 10^5 store): distinct questions at once
    agents = 16
    batches = [qs[i:i + agents] for i in range(0, min(len(qs), 160), agents)]
    lat, t_all = [], time.perf_counter()
    with ThreadPoolExecutor(agents) as ex:
        for b in batches:
            def one(q):
                t = time.perf_counter(); big.read(q["question"], budget_tokens=D["B"], deadline_ms=DEADLINE_MS); return (time.perf_counter() - t) * 1000
            lat += list(ex.map(one, b))
    span = time.perf_counter() - t_all
    bm = []
    for b in batches:
        t = time.perf_counter(); big.read_many([q["question"] for q in b], budget_tokens=D["B"]); bm.append((time.perf_counter() - t) * 1000)
    V1["latency"]["agents16"] = {"threads_p50": v0.pct(lat, 50), "threads_p95": v0.pct(lat, 95), "requests_per_s": len(lat) / span,
                                 "read_many_batch16_p50_ms": v0.pct(bm, 50), "read_many_per_request_ms": (v0.pct(bm, 50) or 0) / agents,
                                 "store": size, "requests": len(lat)}
    log("v1 latency", json.dumps(V1["latency"], default=str)[:1500])
    save_v1()


# ----------------------------------------------------------------------------------------------- report
def f(x, d=1, pct_=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{d}f}" if pct_ else f"{x:.{d}f}"


def ci_cell(v, acc="acc", ci="acc_ci"):
    a, c = v.get(acc), v.get(ci) or (None, None)
    if a is None:
        return "n/a"
    return f"{100 * a:.1f} [{f(c[0], 1, True)}, {f(c[1], 1, True)}]"


LABEL = {"B1": "B1 full context (32k window)", "B2": "B2 recent window", "B3": "B3 retrieval + CE top-64", "B4": "B4 Mem0",
         "DAMA_v0": "DAMA v0 (paper v0)", "DAMA_H": "**DAMA-H (hybrid, this run)**", "DAMA_H_compact": "DAMA-H compact",
         "DAMA_v1_conv7-9": "DAMA v1 (trained rel, v0 pipeline)", "DAMA_HT_conv7-9": "DAMA-HT (hybrid on trained L1.2)"}


def _main_table(L, names, src_ds, src_q):
    L += ["| method | n | DeepSeek acc % [95% CI] | Qwen-7B acc % | F1 | abstention % (DeepSeek / Qwen) | mean tokens | evidence recall % | read p50 / p95 ms |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        q, d = src_q.get(m), src_ds.get(m, {})
        if q:
            L.append(f"| {LABEL.get(m, m)} | {q['n']} | {ci_cell(d) if d.get('acc') is not None else 'not judged'} | {f(q['acc'], 1, True)} | {f(q['f1'], 3)} | "
                     f"{f(d.get('abstention'), 1, True)} / {f(q.get('abstention'), 1, True)} | {q['mean_tokens']:.0f} | {f(q['evidence_recall'], 1, True)} | {f(q.get('read_p50'))} / {f(q.get('read_p95'))} |")
        elif m == "DAMA_H_compact" and D.get("B") and not D.get("B_compact"):
            L.append(f"| {LABEL[m]} | not chosen: no smaller budget came within 3 points on dev | | | | | | | |")
        else:
            L.append(f"| {LABEL.get(m, m)} | not run: {STATE['errors'].get('v1_test_main' if m.startswith('DAMA_H') else 'v1_baselines', 'n/a')} | | | | | | | |")


def _paired_table(L, a):
    pd, pq = (V1.get("ds_paired") or {}).get(a, {}), (V1.get("qwen_paired") or {}).get(a, {})
    if not any(pd.values()):
        pd = {b: None for b in pq}          # no DeepSeek labels: the Qwen column still shows
    if not pd:
        return
    L += [f"Paired, {a} minus each method (categories 1-4, same questions; non-inferior if the CI's lower end is above -5 points):", "",
          "| vs | n | DeepSeek diff [95% CI] | McNemar p | non-inferior at -5 | Qwen-7B diff [95% CI] |", "|---|---|---|---|---|---|"]
    for b, v in pd.items():
        q = pq.get(b)
        if not v and not q:
            continue
        qs = f"{100 * q['diff']:+.1f} [{100 * q['ci'][0]:+.1f}, {100 * q['ci'][1]:+.1f}] (p {q['mcnemar_p']:.4f})" if q else "n/a"
        if v:
            L.append(f"| {b} | {v['n']} | {100 * v['diff']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] | {v['mcnemar_p']:.4f} | "
                     f"{'yes' if v['ci'][0] > -0.05 else 'no'} | {qs} |")
        else:
            L.append(f"| {b} | {q['n']} | not judged | | {'yes' if q['ci'][0] > -0.05 else 'no'} (Qwen) | {qs} |")
    L.append("")


def _cat_table(L, names):
    L += ["| method | multi-hop | temporal | open-domain | single-hop | adversarial |", "|---|---|---|---|---|---|"]
    for m in names:
        bc, who = V1["ds"].get(m, {}).get("by_category"), "DeepSeek"
        if not bc and m in v0.RESULTS:
            bc, who = v0.RESULTS[m].get("by_category"), "Qwen-7B"
        if bc:
            L.append(f"| {m} ({who}) | " + " | ".join(f(bc.get(CATS[c]), 1, True) for c in (1, 2, 3, 4, 5)) + " |")
    L.append("")


def _sec_env(L):
    env = D.get("env") or {}
    ds = V1.get("ds_cost") or (D["ds"].summary() if D.get("ds") else None)
    L += ["## 1. Environment and judge model",
          f"- Kaggle notebook `s41r4j/dama-paper-v0` (this version), {env.get('gpus')}, CPU {env.get('cpu')} ({env.get('cores')} cores), Python {env.get('python')}; "
          f"torch {(env.get('packages') or {}).get('torch')}, transformers {(env.get('packages') or {}).get('transformers')}.",
          f"- Reader (answers; unchanged from round 1): `{v0.READER_ID}`, fp16, greedy, 40 new tokens (64 on LongMemEval), the same prompt for every LoCoMo method. Round-1 answers are reused, never regenerated.",
          "- **New judge: DeepSeek `deepseek-flash` (DeepSeek-V4.1-Flash, API), thinking disabled, temperature 0, max 12 output tokens**, the round-1 judge rules (LoCoMo) or the benchmark's own prompts (LongMemEval). "
          "It replaces the suggested local Qwen2.5-14B/32B: a stronger judge from a different model family, at no GPU cost.",
          (f"- DeepSeek spend: **USD {ds['usd']}** ({ds['api_calls']} API calls, {ds['cached']} answered from the cache, {ds['failed']} failed, {ds['skipped_by_cap']} skipped by the USD {ds['cap_usd']} cap); "
           f"tokens {ds['usage'].get('prompt_tokens', 0)} in ({ds['usage'].get('prompt_cache_hit_tokens', 0)} cache hits), {ds['usage'].get('completion_tokens', 0)} out." if ds else
           "- DeepSeek judge: not run (no DEEPSEEK_API_KEY secret attached)."),
          "- Second judge: Qwen2.5-7B-Instruct (the reader, round 1's judge). Phi-3.5's per-item labels were not saved in round 1, so only its aggregate (72.5% agreement, kappa 0.399 on 200 items) exists.",
          f"- CPU models of the new read path: query encoder {V1['settings'].get('query_encoder')}; cross-encoder {((V1['settings'].get('ranker') or {}).get('value') or {})}.", ""]


def _sec_part1(L):
    L += ["## 2. Part 1: round-1 results with the new judge", "### Main table (LoCoMo test, conversations 4-9)"]
    _main_table(L, ["B1", "B2", "B3", "B4", "DAMA_v0", "DAMA_v1_conv7-9"], V1["ds"], v0.RESULTS)
    L.append("")
    L.append("### By category (DeepSeek)")
    _cat_table(L, ["B1", "B2", "B3", "B4", "DAMA_v0"])
    _paired_table(L, "DAMA_v0")
    sub_d, sub_q = V1.get("ds_subset") or {}, V1.get("qwen_subset") or {}
    L += ["### Topology (fixed stratified 501-question test subset, seed 13)", "| mode | DeepSeek acc % [95% CI] | Qwen-7B acc % |", "|---|---|---|"]
    if "DAMA_v0" in sub_d:
        L.append(f"| gateway (DAMA v0 packet) | {ci_cell(sub_d['DAMA_v0'])} | {f(sub_q.get('DAMA_v0', {}).get('acc'), 1, True)} |")
    for m in ("topology_pull", "topology_sidecar"):
        if m in V1["ds"]:
            L.append(f"| {m.split('_')[1]} | {ci_cell(V1['ds'][m])} | {f(v0.RESULTS[m]['acc'], 1, True)} |")
    L += ["", "### Round-1 ablations (same subset)", "| ablation | DeepSeek acc % [95% CI] | Qwen-7B acc % | mean tokens |", "|---|---|---|---|"]
    for m in ("DAMA_v0", "B3", "B1"):
        if m in sub_d:
            L.append(f"| {m} (reference) | {ci_cell(sub_d[m])} | {f(sub_q[m]['acc'], 1, True)} | {sub_q[m]['mean_tokens']:.0f} |")
    for m in sorted(k for k in V1["ds"] if k.startswith("abl_") and not k.startswith("abl_H")):
        L.append(f"| {m} | {ci_cell(V1['ds'][m])} | {f(v0.RESULTS[m]['acc'], 1, True)} | {v0.RESULTS[m]['mean_tokens']:.0f} |")
    L.append("")


def _sec_agree(L):
    L += ["## 3. Judge agreement; does any conclusion change?", "| method | items | DeepSeek vs Qwen-7B agreement % | Cohen's kappa |", "|---|---|---|---|"]
    for m, v in V1["ds"].items():
        L.append(f"| {m} | {v['judged']} | {f(v.get('qwen_agreement'), 1, True)} | {f(v.get('qwen_kappa'), 3)} |")
    pairs = [(r["ds_correct"], r["correct"]) for m in V1["ds"] for r in v0.ROWS.get(m, []) if r["cat"] != 5 and r.get("ds_correct") is not None]
    if pairs:
        a, b = zip(*pairs)
        L.append(f"All judged items (categories 1-4): agreement {100 * np.mean([x == y for x, y in pairs]):.1f}%, kappa {f(_kappa(list(a), list(b)), 3)} (n = {len(pairs)}). New judge vs Phi-3.5: not computable (no per-item Phi-3.5 labels).")
    d, q = ((V1.get("ds_paired") or {}).get("DAMA_v0") or {}).get("B3"), ((V1.get("qwen_paired") or {}).get("DAMA_v0") or {}).get("B3")
    if d and q:
        L.append(f"- DAMA v0 - B3: Qwen-7B {100 * q['diff']:+.1f} [{100 * q['ci'][0]:+.1f}, {100 * q['ci'][1]:+.1f}] vs DeepSeek {100 * d['diff']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]: "
                 f"sign {'unchanged' if (d['diff'] < 0) == (q['diff'] < 0) else 'CHANGED'}; non-inferior at -5: {'yes' if d['ci'][0] > -0.05 else 'no'} (DeepSeek).")
    for m in ("B1", "B3", "B4", "DAMA_v0"):
        if m in V1["ds"] and m in v0.RESULTS and V1["ds"][m].get("acc") is not None:
            L.append(f"- {m}: Qwen-7B {100 * v0.RESULTS[m]['acc']:.1f}% -> DeepSeek {100 * V1['ds'][m]['acc']:.1f}%.")
    L.append("")


def _sec_audit(L):
    L += ["## 4. Part 2: human audit (prepared, not labelled)"]
    a = V1.get("audit")
    if a:
        L += [f"- `paper/human_audit_sheet.csv`: {a['items']} items (round 1's stratified judge_audit.csv from B1-B4 and DAMA v0), shuffled with seed 13, columns id, method, question, gold, answer, human_label (empty). No judge labels in it.",
              f"- `paper/human_audit_key.csv`: id -> Qwen-7B label, DeepSeek label ({a['matched_to_rows']} of {a['items']} matched to judged rows), Phi-3.5 (n/a: not saved per item in round 1).",
              "- Score after the team fills human_label with CORRECT or WRONG: `python experiments/v1_2026-10-24/score_human_audit.py human_audit_sheet.csv human_audit_key.csv` (prints agreement and Cohen's kappa per judge)."]
    else:
        L.append(f"not run: {STATE['errors'].get('v1_audit', 'n/a')}")
    L.append("")


def _sec_part3(L):
    s = V1["settings"]
    L += ["## 5. Part 3: DAMA-H, a faster read path chosen on dev only (post hoc; does not replace DAMA v0)",
          "What changed against v0, motivated by round 1's test ablations (so DAMA-H is post hoc): a union pool instead of the reciprocal-rank cut, a small cross-encoder on the prior's top N, "
          "probability-ordered packing without MMR, optional previous-turn neighbours, ONNX models on CPU, numpy BM25/entity indexes (code/dama_agent/hybrid.py). Every setting below was chosen on dev (conversations 0-3) only.", "",
          "### Dev selection"]
    if s.get("key"):
        L.append(f"- Dense key: **{s['key']['value']}**; dev coverage of dense top-128: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["key"]["dev_dense128_coverage"].items()) + ".")
    if s.get("pool"):
        pv = s["pool"]["value"]
        L.append(f"- Pool: **dense {pv['dense_k']} + BM25 {pv['bm25_k']} + entity {pv['entity_k']}** ({s['pool']['rule']}). Dev coverage / mean pool: "
                 + "; ".join(f"{k}: {100 * v['coverage']:.1f}% / {v['mean_pool']:.0f}" for k, v in s["pool"]["dev"].items())
                 + f". Test coverage of the chosen pool: {f(s['pool'].get('test_coverage'), 1, True)}% (v0 hybrid@128: 74.9%).")
    if s.get("prior"):
        L.append(f"- Prior: logistic head on {', '.join(FEATURES)}; {s['prior']['dev_pairs']} dev pairs; test ECE {f(s['prior']['test_ece'], 4)}.")
    if s.get("ranker"):
        rk = s["ranker"]
        L += [f"- Ranker: **{rk['value']['model'] or 'none'}, N = {rk['value']['N']}** ({rk['value'].get('onnx')} ONNX); {rk['rule']}. CPU p50 without it: {f(rk['base_cpu_p50_ms'])} ms.", "",
              "| ranker | N | dev evidence recall @1500 % | @3000 % | est. CPU p50 ms |", "|---|---|---|---|---|"]
        for k, v in sorted(rk["dev_grid"].items(), key=lambda kv: -(kv[1]["recall_1500"] + kv[1]["recall_3000"])):
            L.append(f"| {k.split('|')[0].split('/')[-1]} | {k.split('|')[1]} | {f(v['recall_1500'], 1, True)} | {f(v['recall_3000'], 1, True)} | {f(v['est_cpu_p50_ms'])} |")
        L.append("")
    if s.get("B"):
        L.append(f"- Neighbours: **{s['neighbours']['value']}** (dev-sample accuracy at B 3000: " + ", ".join(f"m={k} {100 * v:.1f}%" for k, v in s["neighbours"]["dev_acc_B3000"].items())
                 + f"). Budget **B = {s['B']['value']}**, compact {s['B']['compact']} (dev-sample accuracy " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in s["B"]["dev_acc"].items())
                 + f"; n = {s['B']['n_dev_sample']}, Qwen-7B judge, no API spend on dev).")
    if s.get("lambda_h"):
        L.append(f"- Conformal lambda on DAMA-H's final probabilities (dev): {s['lambda_h']}; dev questions with evidence outside the pool: {f(s.get('lambda_h_dev_minus_inf_share'), 1, True)}%.")
    L += ["", "### Test results (LoCoMo 4-9, run once)"]
    _main_table(L, ["DAMA_H", "DAMA_H_compact", "B3", "B1", "DAMA_v0"], V1["ds"], v0.RESULTS)
    L.append("")
    _cat_table(L, ["DAMA_H", "DAMA_H_compact", "B3", "B1", "DAMA_v0"])
    _paired_table(L, "DAMA_H")
    if V1.get("loco_h"):
        L += ["### Leave-one-conversation-out conformal check (DAMA-H probabilities, all 10 conversations)", "| alpha | mean omission | SD | min | max | folds with lambda=-inf |", "|---|---|---|---|---|---|"]
        for a, v in V1["loco_h"].items():
            L.append(f"| {a} | {v['mean']:.3f} | {v['sd']:.3f} | {v['min']:.3f} | {v['max']:.3f} | {v['folds_lambda_minus_inf']} |")
        L.append("")
    L += ["### DAMA-H ablations (501-question subset; Qwen-7B and DeepSeek)", "| run | DeepSeek acc % [95% CI] | Qwen-7B acc % | mean tokens | evidence recall % |", "|---|---|---|---|---|"]
    sub_d, sub_q = V1.get("ds_subset") or {}, V1.get("qwen_subset") or {}
    if "DAMA_H" in sub_q:
        L.append(f"| DAMA-H (reference) | {ci_cell(sub_d.get('DAMA_H', {})) if sub_d.get('DAMA_H', {}).get('acc') is not None else 'n/a'} | {f(sub_q['DAMA_H']['acc'], 1, True)} | {sub_q['DAMA_H']['mean_tokens']:.0f} | {f(sub_q['DAMA_H']['evidence_recall'], 1, True)} |")
    for m in ("abl_H_no_rerank", "abl_H_dense_pool", "abl_H_conformal_a0.2"):
        q, d = v0.RESULTS.get(m), V1["ds"].get(m, {})
        L.append(f"| {m} | {ci_cell(d) if d.get('acc') is not None else 'n/a'} | {f(q['acc'], 1, True)} | {q['mean_tokens']:.0f} | {f(q['evidence_recall'], 1, True)} |" if q
                 else f"| {m} | not run: {STATE['errors'].get(m, STATE['errors'].get('v1_ablations', '; '.join(n for n in V1['notes'] if m in n) or 'n/a'))} | | | |")
    L += ["", "### Trained models on DAMA-H (conversations 7-9, unseen by the L1.2 bundle)", "| method | DeepSeek acc % [95% CI] | Qwen-7B acc % | mean tokens | evidence recall % | read p50 ms |", "|---|---|---|---|---|---|"]
    for m, v in (V1.get("ds_conv79") or {}).items():
        q = (V1.get("conv79_qwen") or {}).get(m, {})
        L.append(f"| {LABEL.get(m, m)} | {ci_cell(v) if v.get('acc') is not None else 'n/a'} | {f(q.get('acc'), 1, True)} | {f(q.get('mean_tokens'), 0)} | {f(q.get('evidence_recall'), 1, True)} | {f(q.get('read_p50'))} |")
    if "DAMA_HT_conv7-9" not in v0.RESULTS:
        L.append(f"DAMA-HT not run: {STATE['errors'].get('v1_trained', 'n/a')}")
    lat = V1.get("latency") or {}
    L += ["", "### Speed and multi-agent (CPU, the ONNX models as deployed)"]
    if lat:
        L += ["| setting | p50 ms | p95 ms | over 100 ms % | cross-encoder pairs (mean) |", "|---|---|---|---|---|"]
        for k, lab in (("locomo_no_deadline", "LoCoMo stores, full N"), ("locomo_deadline_100", "LoCoMo stores, 100 ms deadline"),
                       ("store_1e5", f"{(lat.get('store_1e5') or {}).get('size', '')}-turn store, full N"), ("store_1e5_deadline_100", "same store, 100 ms deadline")):
            v = lat.get(k)
            if v:
                L.append(f"| {lab} | {f(v['p50'])} | {f(v['p95'])} | {f(v['late_share'], 1, True)} | {f(v['mean_rescored'])} |")
        for k, lab in (("locomo_no_deadline", "LoCoMo stores"), ("store_1e5", "large store")):
            st = (lat.get(k) or {}).get("stages_p50")
            if st:
                L.append(f"Per-component p50 ({lab}, ms): " + ", ".join(f"{a} {b:.2f}" for a, b in st.items()) + ".")
        a = lat.get("agents16")
        if a:
            L.append(f"16 agents on one {a['store']}-turn memory, distinct questions at once ({a['requests']} requests, 100 ms deadline): p50 {f(a['threads_p50'])} ms, p95 {f(a['threads_p95'])} ms, "
                     f"{a['requests_per_s']:.0f} requests/s; one read_many batch of 16: {f(a['read_many_batch16_p50_ms'])} ms ({f(a['read_many_per_request_ms'])} ms per request). "
                     "Round 1 for comparison: whole read path p50 105 ms (LoCoMo), 158 ms (10^5 store); B3's cross-encoder read 1,697 ms.")
    else:
        L.append(f"not run: {STATE['errors'].get('v1_latency', 'n/a')}")
    L.append("")


def _sec_lme(L):
    L += ["## 6. Part 4: LongMemEval-S"]
    lm = V1.get("lme_meta")
    if not lm:
        L += [f"not run: {STATE['errors'].get('v1_lme', 'n/a')}", ""]
        return
    L += [f"`{lm['file']}` (Hugging Face xiaowu0162/longmemeval-cleaned). Sample (seed 13): {lm['n_sample']} questions = a core subset of {lm['per_type']} per type, plus all "
          f"{', '.join(f'{v} {k}' for k, v in LME_EXTRA.items()) or 'none'} questions for H4; mean {lm['mean_turns']:.0f} turns / {lm['mean_history_tokens']:.0f} tokens of history. "
          f"Settings fixed from LoCoMo dev (no tuning and no calibration sample here); turns cut to {LME_TURN_CHARS} characters for every method. Judge: the benchmark's prompts, graded by DeepSeek (Qwen-7B alongside). Mem0 not run.", "",
          "| method | set | n | DeepSeek acc % [95% CI] | Qwen-7B acc % | abstention % | mean tokens | evidence recall % | read p50 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for m in ("LME_DAMA_H", "LME_B3", "LME_B2", "LME_B1"):
        v = V1["lme"].get(m)
        if not v:
            L.append(f"| {m[4:]} | | not run: {STATE['errors'].get('v1_lme_b1' if m == 'LME_B1' else 'v1_lme', 'n/a')} | | | | | | |"); continue
        for part in ("core", "all"):
            x = v[part]
            if x["n"] and not (part == "all" and m == "LME_B1"):
                L.append(f"| {m[4:]} | {part} | {x['n']} | {ci_cell(x, 'ds_acc', 'ds_ci')} | {f(x['qwen_acc'], 1, True)} | {f(x['abstention'], 1, True)} | {x['mean_tokens']:.0f} | {f(x['evidence_recall'], 1, True)} | {f(x['read_p50'])} |")
    types = sorted({t for v in V1["lme"].values() if isinstance(v, dict) and "all" in v for t in v["all"]["by_type"]})
    L += ["", "By type (all sampled questions; DeepSeek when available), acc % [95% CI] (n):", "", "| method | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for m, v in V1["lme"].items():
        if isinstance(v, dict) and "all" in v:
            bt = v["all"]["by_type"]
            L.append(f"| {m[4:]} | " + " | ".join((f"{100 * bt[t]['acc']:.1f} [{f(bt[t]['ci'][0], 0, True)}, {f(bt[t]['ci'][1], 0, True)}] ({bt[t]['n']})" if t in bt and bt[t]["acc"] is not None else "n/a") for t in types) + " |")
    if "LME_B1" in V1["lme"]:
        L.append(f"B1 ran on {LME_B1_PER_TYPE} questions per type of the core subset; truncated to the 32k window for {f(V1['lme']['LME_B1'].get('truncated_share'), 0, True)}% of them.")
        ids = {r["qid"] for r in D.get("lme_rows", {}).get("LME_B1", [])}
        same = {m: lme_stats([r for r in rows if r["qid"] in ids]) for m, rows in D.get("lme_rows", {}).items()}
        L.append("Same questions as B1 (DeepSeek, else Qwen-7B): " + "; ".join(f"{m[4:]} {f(v['ds_acc'] if v['ds_acc'] is not None else v['qwen_acc'], 1, True)}% ({v['mean_tokens']:.0f} tokens)" for m, v in same.items() if v["n"]) + ".")
    h4 = V1.get("h4") or {}
    for key, v in h4.items():
        L.append(f"- **H4** ({'DeepSeek' if key == 'ds_correct' else 'Qwen-7B'}): knowledge-update accuracy DAMA-H {100 * v['dama_h']:.1f}% vs B3 {100 * v['b3']:.1f}% (n = {v['n']}, paired diff {100 * v['diff']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]): "
                 f"{'rejected' if v['rejected'] else 'not rejected'} (rule: rejected if DAMA is below B3).")
    L.append("")


def _sec_problems(L):
    L += ["## 7. Problems, deviations and caveats"]
    for k, v in STATE["errors"].items():
        L.append(f"- Error in `{k}`: {str(v)[:200]}")
    L += [f"- {n}" for n in V1["notes"]]
    L += ["- Judge: DeepSeek-V4.1-Flash (API) instead of a local Qwen2.5-14B/32B. It is not open-weights; its answers are cached per item (paper/llm_cache.sqlite) so the labels are reproducible from this run.",
          "- DAMA-H is post hoc: round 1's test ablations (MMR hurt, threshold inactive, entity list lowered pool recall) motivated which components to change; only the settings were then chosen on dev. MMR was not re-tested on dev (it is not in the new read path); round 1's test ablation is the only evidence against it.",
          "- The dev-sample reader checks (neighbours, budget) use the Qwen-7B judge on 150 questions, so differences under ~5 points there are within noise.",
          "- Round-1 answers (B1-B4, DAMA v0, DAMA v1, topology, ablations) are reused from version 8; only the judging is new. DAMA-H's answers are new, with the same reader and prompt.",
          "- DAMA-HT (trained bundle) only on conversations 7-9: conversations 0-6 were used for its training, validation and calibration.",
          "- LongMemEval-S: a subset, settings carried over from LoCoMo without a calibration sample; B3 there ranks the whole history with the cross-encoder and fills DAMA-H's budget; B1 only on the core subset.",
          "- The 100-item human audit is prepared but not labelled; until the team labels it, judge validity rests on judge agreement only."]
    L.append("")


def write_report():
    run_id = time.strftime("v1-%Y%m%d-%H%M%S", time.gmtime())
    L = [f"# DAMA round 2 results (generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, run id {run_id})", ""]
    for sec in (_sec_env, _sec_part1, _sec_agree, _sec_audit, _sec_part3, _sec_lme, _sec_problems):
        try:
            sec(L)
        except Exception as e:
            L += [f"(section {sec.__name__} failed: {type(e).__name__}: {str(e)[:200]})", ""]
            log("report section failed", sec.__name__, traceback.format_exc()[-1500:])
    L += ["## 8. Files", "Kaggle notebook output (`s41r4j/dama-paper-v0`, this version): `paper/ROUND2_RESULTS_FOR_PAPER.md`, `paper/results_v1.json` (every number), "
          "`paper/methods/*.jsonl` (per question: context, answer, Qwen-7B and DeepSeek labels), `paper/human_audit_sheet.csv` + `paper/human_audit_key.csv`, `paper/baselines_v8/` (round 1's rows), "
          "`paper/onnx/` (the CPU models used), `paper/llm_cache.sqlite` (every reader and judge call), `paper/run.log`. Code: `experiments/v1_2026-10-24/dama_paper_v1.py`, `code/dama_agent/hybrid.py`, `code/dama_agent/fastmodels.py`."]
    text = "\n".join(L) + "\n"
    (OUT / "ROUND2_RESULTS_FOR_PAPER.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)
    return text


@stage("v1_carry")
def stage_carry():
    """Judge mode: copy the previous version's outputs (rows, results, ONNX models, round-1 files) into this version's
    output so nothing is lost, then reload rows, settings and results."""
    import shutil
    src = sorted(glob.glob("/kaggle/input/**/paper/results_v1.json", recursive=True))
    if not src:
        raise FileNotFoundError("no earlier results_v1.json attached")
    root = Path(src[0]).parent
    for f in root.rglob("*"):
        if f.is_file() and f.name != "llm_cache.sqlite":          # the cache is merged by v0's Cache instead
            dst = OUT / f.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy(f, dst)
    old = json.loads((root / "results_v1.json").read_text())
    for k, v in old.items():
        V1[k] = v
    V1["ds"], V1["notes"] = {}, old.get("notes", [])
    s = old.get("settings", {})
    D["B"], D["m"], D["B_compact"] = s.get("B", {}).get("value"), s.get("neighbours", {}).get("value"), s.get("B", {}).get("compact")
    for f in sorted((OUT / "methods").glob("*.jsonl")):
        if f.stem.startswith("LME_"):
            rows = [json.loads(l) for l in open(f)]
            D.setdefault("lme_rows", {})[f.stem] = rows
        elif f.stem.startswith(("DAMA_H", "abl_H")):
            rows = [json.loads(l) for l in open(f)]
            v0.ROWS[f.stem] = rows; v0.RESULTS[f.stem] = safe_summary(rows)
    log("v1 carried forward:", {"rows": sorted(v0.ROWS), "lme": sorted(D.get("lme_rows", {}))})


@stage("v1_judge_lme", limit_h=FINAL_H)
def stage_judge_lme():
    """DeepSeek on the saved LongMemEval answers (the benchmark's own prompts), then the tables and H4 again."""
    ds = ds_client()
    for name, rows in D.get("lme_rows", {}).items():
        if ds is not None:
            raw = [{"question_id": r["qid"], "question_type": r["type"], "question": r["question"], "answer": r["gold"]} for r in rows]
            jd = ds.many([[{"role": "user", "content": lme_judge_prompt(it, r["answer"])}] for it, r in zip(raw, rows)], max_tokens=6)
            for r, o in zip(rows, jd):
                r["ds_correct"] = None if o is None else ("yes" in o.lower())
            with (OUT / "methods" / f"{name}.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        V1["lme"][name] = {**V1["lme"].get(name, {}), "all": lme_stats(rows), "core": lme_stats([r for r in rows if r.get("core")])}
    h, b = D.get("lme_rows", {}).get("LME_DAMA_H", []), {r["qid"]: r for r in D.get("lme_rows", {}).get("LME_B3", [])}
    for key in ("ds_correct", "qwen_correct"):
        pr = [(float(r[key]), float(b[r["qid"]][key])) for r in h if r["type"] == "knowledge-update" and r.get(key) is not None
              and r["qid"] in b and b[r["qid"]].get(key) is not None]
        if pr:
            x, y = np.array(pr).T
            V1.setdefault("h4", {})[key] = {"n": len(pr), "dama_h": float(x.mean()), "b3": float(y.mean()), "diff": float((x - y).mean()),
                                            "ci": v0.boot_ci(x - y), "rejected": bool(x.mean() < y.mean())}
    save_v1()


@stage("v1_judge_smoke")
def stage_judge_smoke():
    """Smoke test of the outside judge: ~20 saved DAMA-H answers, logged one by one (no key, ever)."""
    ds = ds_client()
    if ds is None:
        raise RuntimeError("DeepSeek key not available")
    rows = [dict(r) for r in v0.ROWS.get("DAMA_H", [])[:20]]
    ds_judge_rows(rows)
    for r in rows:
        log(f"  judge smoke: cat {r['cat']} gold {str(r['gold'])[:40]!r} answer {r['answer'][:50]!r} -> DeepSeek {r.get('ds_correct')} / Qwen {r['correct']}")
    log("judge smoke:", ds.summary())


if __name__ == "__main__" and MODE == "judge":
    D["env"] = v0.environment(); log("environment (judge mode, no GPU)", D["env"])
    v0.stage_data()
    stage_baselines()
    stage_carry()
    if not SMOKE:                 # smoke mode keeps only a few questions: restored files would be truncated
        stage_restore()
    if SMOKE:
        stage_judge_smoke()
    else:
        stage_judge_main()
        stage_audit()
        stage_judge_lme()
    save_v1()
    text = write_report()
    log(f"v1 judge-mode report written ({len(text)} chars); errors: {STATE['errors'] or 'none'}; DeepSeek {D['ds'].summary() if D.get('ds') else None}")
    sys.exit(0)

if __name__ == "__main__":
    if v0.ON_KAGGLE and not glob.glob("/kaggle/input/**/paper/baselines_v8/B3.jsonl", recursive=True) + glob.glob("/kaggle/input/**/paper/methods/B3.jsonl", recursive=True):
        log("version 8's per-question rows are not attached; stopping before using GPU time"); sys.exit(1)
    D["env"] = v0.environment(); log("environment", D["env"])
    stage_setup()
    stage_baselines()
    stage_index()
    stage_prior()
    stage_ranker()
    if "B" not in D:
        stage_dev_reader()
    stage_restore()
    if "B" in D:
        stage_test_main()
        stage_judge_main()
        stage_audit()
        stage_loco()
        stage_lme()
        stage_trained()
        stage_ablations()
        stage_lme_b1()
        stage_judge_main()      # judges the rows added since (cached calls are free)
    stage_latency()
    save_v1()
    text = write_report()
    log(f"v1 report written ({len(text)} chars); errors: {STATE['errors'] or 'none'}; DeepSeek {V1.get('ds_cost')}")

"""DAMA paper v3b (2026-11-01): event ledger (V3-L) and a model zoo, dev only. Private Kaggle notebook
`s41r4j/dama-paper-v3b` (T4 x2).

v3a showed the reader cannot be pushed further by instructions (structured prompt -6, layout -2). The research note
docs/research/DAMA_v3_new_arch_research_2026-10-31.md proposes moving counting, sums, date differences, order and latest
values out of the reader: an extractor writes dated, typed rows per session (code/dama_agent/ledger.py); at question
time fixed rules route the question to a Python operator, which adds a short LEDGER block to the unchanged A0 packet.
Unrouted questions keep byte-identical packets (their reader and judge calls are cached from v3a).

Arms (all on v2c's dev sample: 250 LoCoMo dev + 50 adversarial, 218 LongMemEval dev; reader prompt P1, thinking off):
  A0        v2c C3_m0 packet, Qwen3.5-9B (cached)
  L9        A0 + LEDGER (writer Qwen3.5-9B), routes COUNT/SUM/DATEDIFF/ORDER/LATEST
  L9P       L9 + the PREF route (user preferences for recommendation questions)
  L35P      as L9P, writer Qwen3.5-35B-A3B (GPTQ-Int4, 3B active; the strongest open writer that fits)
  L2P/L08P  as L9P, writers Qwen3.5-2B / Qwen3.5-0.8B (can a small CPU-class writer do the job?)
  Z_<m>     model zoo: the A0 packets read by other models, two single-GPU models side by side
  Z35_L35P  the 35B-A3B MoE reading its own L35P packets (open top-line reader)
Judge Qwen2.5-7B for every arm. Decision rules: README.md (fixed before the run). No DeepSeek.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SMOKE = False
HERE = Path(globals().get("__file__", "dama_paper_v3b.py")).resolve().parent

if __name__ == "__main__" and "--stub-server" in sys.argv:      # local dry run only (DAMA_FAKE_ENGINE=1): fixed answers
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sys_text, text = body["messages"][0]["content"], body["messages"][-1]["content"]
            if sys_text.startswith("You extract a memory ledger"):
                ts = re.findall(r"\[#(\d+)\]", text)
                out = "\n".join(json.dumps({"t": int(t), "who": "user", "kind": k, "what": "attended", "item": f"event {t}",
                                            "cat": "festival", "qty": 2 if k == "event" else None, "unit": "hours",
                                            "when": "last Saturday"}) for t, k in zip(ts[:3], ("event", "state", "pref")))
            elif "Answer yes or no" in text:
                out = "yes"
            elif '"label"' in text:
                out = '{"label": "CORRECT"}'
            elif '"declines"' in text:
                out = '{"declines": false}'
            elif "NOTES" in sys_text:
                out = "NOTES: [m:1] says something.\nANSWER: 7 May 2023"
            else:
                out = "7 May 2023"
            time.sleep(0.005)
            msg = json.dumps({"choices": [{"message": {"content": out}}], "usage": {"prompt_tokens": len(text) // 4}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(msg)

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[-1])), Stub).serve_forever()


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v2c_2026-10-30/dama_paper_v2c.py")))
import dama_paper_v2c as Cm      # v2c's dev-only setup and packet; through it v2b's engine, reading and judging
Bm, V, L, v0 = Cm.Bm, Cm.V, Cm.L, Cm.v0
import numpy as np
from dama_agent import ledger as LG

Cm.SMOKE = Bm.SMOKE = V.SMOKE = v0.SMOKE = Cm.v1.SMOKE = SMOKE
Cm.N_LOCO, Cm.N_ADV = (12, 3) if SMOKE else (250, 50)
Cm.save_r = lambda: None
v0.TIME_LIMIT_H, v0.FINAL_LIMIT_H = (1.5, 2.0) if SMOKE else (9.0, 10.5)
log, stage, OUT, D, STATE = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE
FAKE = Bm.FAKE_ENGINE
READER, LABEL, JUDGE_ID = Cm.READER, Cm.LABEL, Bm.JUDGE_ID
VLLM = "0.32.0"
OPTIONAL_STOP_H = 1.0 if SMOKE else 7.5      # no new optional job (zoo, extra writers) after this many hours
HARD = {"lme": {"temporal-reasoning", "multi-session"}, "locomo": {"temporal", "multi-hop"}}
SAFE = {"lme": {"single-session-user", "single-session-assistant"}, "locomo": {"single-hop"}}
MOE = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
WRITER_SMALL = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B"]                       # one per GPU, side by side
ZOO_PAIRS = [("Qwen/Qwen3.5-4B", "Qwen/Qwen3-4B-Instruct-2507"),               # (GPU 0, GPU 1); A0 packets, prompt P1
             ("Qwen/Qwen3-8B-AWQ", "microsoft/Phi-4-mini-instruct"),
             ("Qwen/Qwen3.5-2B", "HuggingFaceTB/SmolLM3-3B")]
SHORT = {READER: "9", MOE: "35", "Qwen/Qwen3.5-2B": "2", "Qwen/Qwen3.5-0.8B": "08"}
MATCH = {"rel": 0.15, "floor": 0.58, "top": 12}       # fixed before the run from synthetic pairs, not dev labels (README)
WRITE_MAX_NEW = 700
A1_S_PER_Q = 3.6           # Qwen3.5-9B on A0-size packets, measured in v3a (A0 itself is cached here)
R = {"settings": {"match": MATCH, "vllm": VLLM}, "arms": {}, "paired": {}, "ledgers": {}, "gates": {}, "decisions": {},
     "zoo": {}, "notes": []}

# the shared LLM cache is used from two GPU threads: one connection, one lock
_DB_LOCK = threading.Lock()
v0.CACHE.db = sqlite3.connect(str(OUT / "llm_cache.sqlite"), check_same_thread=False)
_get, _put = v0.Cache.get, v0.Cache.put
v0.Cache.get = lambda self, k: (_DB_LOCK.acquire(), _get(self, k), _DB_LOCK.release())[1]
v0.Cache.put = lambda self, *a: (_DB_LOCK.acquire(), _put(self, *a), _DB_LOCK.release())[1]


def save_r():
    (OUT / "results_v3b.json").write_text(json.dumps(R, indent=1, default=str))


def hours():
    return (time.time() - v0.T_START) / 3600


# ---------------------------------------------------------------------------------------------- engine: any GPU, any port
def _tmpdir():
    """vLLM's lock files: a private directory (v3b version 1 lost the judge to 'Read-only file system: /tmp/...lock')."""
    d = Path(os.environ.get("DAMA_TMP", "/kaggle/working/tmp" if Path("/kaggle/working").exists() else "/tmp/dama_tmp"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class Srv:
    """`vllm serve` on the given GPUs ("0", "1" or "0,1") and port; one model per server."""
    def __init__(self, model, gpus, port, max_len=8192, util=0.88, max_seqs=64, extra=()):
        self.model, self.gpus, self.port, self.max_len, self.util, self.max_seqs = model, gpus, port, max_len, util, max_seqs
        self.extra = list(extra)
        self.proc = None
        self.log_path = OUT / f"vllm_{model.split('/')[-1]}_{gpus.replace(',', '')}.log"

    def start(self, timeout_min=40):
        tp = len(self.gpus.split(","))
        if FAKE:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--stub-server", str(self.port)]
        else:
            cmd = [str(Bm.VENV / "bin/vllm"), "serve", self.model, "--served-model-name", "m", "--port", str(self.port),
                   "--dtype", "half", "--tensor-parallel-size", str(tp), "--max-model-len", str(self.max_len),
                   "--gpu-memory-utilization", str(self.util), "--max-num-seqs", str(self.max_seqs), "--seed", str(v0.SEED),
                   "--disable-custom-all-reduce", "--trust-remote-code", *self.extra]
        env = {**os.environ, "HF_HOME": str(Bm.HF_HOME), "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
               "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": self.gpus, "TMPDIR": str(_tmpdir())}
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        t = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.log_path.open("w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
        Bm.RUNNING.append(self)
        while time.time() - t < timeout_min * 60:
            if self.proc.poll() is not None:
                break
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5).status == 200:
                    self.load_min = round((time.time() - t) / 60, 1)
                    log(f"served {self.model} on GPU {self.gpus} in {self.load_min} min")
                    return True
            except Exception:
                pass
            time.sleep(5)
        text = self.log_path.read_text(errors="replace") if self.log_path.exists() else ""
        errs = [l for l in text.splitlines() if ("Error" in l or "error" in l) and not l.lstrip().startswith(("File ", "raise "))]
        log(f"could not serve {self.model} on GPU {self.gpus}:", "\n".join(errs[-12:]) or text[-1500:])
        self.stop()
        return False

    def stop(self, drop_weights=False):
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, 15); self.proc.wait(60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, 9)
                except Exception:
                    pass
            self.proc = None
        if self in Bm.RUNNING:
            Bm.RUNNING.remove(self)
        time.sleep(2 if FAKE else 10)
        if drop_weights and not FAKE:                    # v3b version 1 kept every model on disk
            import shutil
            shutil.rmtree(Bm.HF_HOME / "hub" / ("models--" + self.model.replace("/", "--")), ignore_errors=True)


class Cli(Bm.Client):
    """v2b's client on a given port with its own concurrency; same cache key (model, messages, max_new, "vllm")."""
    def __init__(self, model, port, conc=32):
        super().__init__(model)
        self.port, self.conc = port, conc

    def _post(self, messages, max_new):
        body = {"model": "m", "messages": messages, "max_tokens": max_new, "temperature": 0, "seed": v0.SEED,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            t = time.time()
            try:
                j = json.loads(urllib.request.urlopen(req, timeout=1800).read())
                return Bm.clean(j["choices"][0]["message"].get("content")), int(j.get("usage", {}).get("prompt_tokens", 0)), (time.time() - t) * 1000
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    return None, 0, 0.0
                time.sleep(5 * (attempt + 1))
            except Exception:
                time.sleep(5 * (attempt + 1))
        return None, 0, 0.0

    def __call__(self, convs, max_new=40, tools=None, **kw):
        keys = [v0.Cache.key(self.name, [c, None, max_new, "vllm"]) for c in convs]
        res, todo = [None] * len(convs), []
        for i, k in enumerate(keys):
            hit = v0.CACHE.get(k)
            if hit:
                res[i] = hit
            else:
                todo.append(i)
        with ThreadPoolExecutor(self.conc) as ex:
            for i, (o, pt, ms) in zip(todo, ex.map(lambda i: self._post(convs[i], max_new), todo)):
                if o is None:
                    self.failed += 1; res[i] = ("", pt, ms)
                else:
                    v0.CACHE.put(keys[i], self.name, o, pt, ms); res[i] = (o, pt, ms)
        return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]


# ---------------------------------------------------------------------------------------------- packets (as v3a)
@stage("v3b_contexts")
def stage_contexts():
    import random
    rng = random.Random(Cm.SEED)
    lq = D["lq"]
    loco = rng.sample([q for q in lq if q["cat"] != 5], min(Cm.N_LOCO, sum(q["cat"] != 5 for q in lq)))
    loco += rng.sample([q for q in lq if q["cat"] == 5], min(Cm.N_ADV, sum(q["cat"] == 5 for q in lq)))
    D["qs"] = loco + D["lme"]
    D["ctx"] = {"A0": [Cm.ctx(q, neighbours=0) for q in D["qs"]]}
    D.pop("bge", None)
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()
    R["settings"]["questions"] = {"locomo": len(loco), "lme": len(D["lme"])}
    log("v3b contexts:", R["settings"]["questions"])


# ---------------------------------------------------------------------------------------------- write: sessions -> ledger
def sessions():
    """Every session the dev questions can see, once: LME user turns only (deduplicated by session id), LoCoMo dev
    conversations with both speakers. -> list of (session key, date, [(t, speaker, text)], {t: dia})"""
    out, seen = [], set()
    for it in D["lme"]:
        by = defaultdict(list)
        for t in it["turns"]:
            by[t["session"]].append(t)
        for sid, ts in by.items():
            if sid in seen:
                continue
            seen.add(sid)
            items = [(int(t["dia"].rsplit(":", 1)[1]), "user", t["text"]) for t in ts if t["speaker"] == "user"]
            out.append((f"lme|{sid}", ts[0]["date"], items, {j: f"{sid}:{j}" for j, _, _ in items}))
    cis = {q["ci"] for q in D["lq"]}
    for c in D["convs"]:
        if c["ci"] not in cis:
            continue
        by = defaultdict(list)
        for t in c["turns"]:
            by[t["session"]].append(t)
        for n, ts in by.items():
            items = [(k, t["speaker"], t["text"]) for k, t in enumerate(ts)]
            out.append((f"locomo{c['ci']}|{n}", ts[0].get("date_raw") or ts[0]["date"], items, {k: t["dia"] for k, t in enumerate(ts)}))
    return out


def write_ledger(model, cli):
    """One extractor pass over every session -> {session key: rows}. Cached per call, so a re-run costs nothing."""
    if "sessions" not in D:
        D["sessions"] = sessions()
    ss = D["sessions"]
    jobs = []
    for key, date, items, dia_of in ss:
        for msgs, ts in LG.session_prompts(date, items):
            jobs.append((key, date, msgs, set(ts), dia_of))
    t = time.time()
    outs, ptoks, _ = cli([m for _, _, m, _, _ in jobs], max_new=WRITE_MAX_NEW)
    led = defaultdict(list)
    n_raw = 0
    for (key, date, _, ts, dia_of), o in zip(jobs, outs):
        rows = LG.parse_rows(o, valid_turns=ts)
        n_raw += len(rows)
        led[key] += LG.finish_rows(rows, key, date, lambda x, d=dia_of: d.get(x), key)
    mins = (time.time() - t) / 60
    st = {"sessions": len(ss), "calls": len(jobs), "rows": n_raw, "minutes": round(mins, 1), "failed_calls": cli.failed,
          "prompt_tokens": int(sum(ptoks)), "kinds": dict(Counter(r["kind"] for rs in led.values() for r in rs)),
          "gran": dict(Counter(r["gran"] for rs in led.values() for r in rs)), "empty_outputs": sum(1 for o in outs if not (o or "").strip())}
    R["ledgers"][model] = st
    log(f"ledger by {model}:", st)
    (OUT / f"ledger_{SHORT.get(model, model.split('/')[-1])}.jsonl").write_text(
        "\n".join(json.dumps(r) for rs in led.values() for r in rs) + "\n")
    save_r()
    return dict(led)


# ---------------------------------------------------------------------------------------------- read: route + match + operate
_ENC = {}
_ENC_LOCK = threading.Lock()


def encoder():
    if "st" not in _ENC:
        from sentence_transformers import SentenceTransformer
        _ENC["st"] = SentenceTransformer(v0.EMBED_ID, device="cpu")
    return _ENC["st"]


def history_rows(q, led):
    if q["bench"] == "lme":
        sids = {t["session"] for t in q_turns(q)}
        return [r for s in sids for r in led.get(f"lme|{s}", [])]
    return [r for k, rs in led.items() if k.startswith(f"locomo{q['ci']}|") for r in rs]


def q_turns(q):
    return V.reader_of(q)[1]


def ledger_contexts(led, prefs, tag):
    """A0 packets + a LEDGER block where the router fires and rows match; else the A0 context unchanged."""
    with _ENC_LOCK:
        return _ledger_contexts(encoder(), led, prefs, tag)


def _ledger_contexts(enc, led, prefs, tag):
    texts, idx = [], {}
    for rs in led.values():
        for r in rs:
            k = LG.row_text(r)
            if k not in idx:
                idx[k] = len(texts); texts.append(k)
    t = time.time()
    vec = enc.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True) if texts else np.zeros((0, 384))
    qv = enc.encode([L.BGE_PREFIX + q["question"] for q in D["qs"]], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    out, info = [], Counter()
    for q, c, v in zip(D["qs"], D["ctx"]["A0"], qv):
        rname = LG.route(q["question"], prefs=prefs)
        block = None
        if rname:
            info[f"routed_{rname}"] += 1
            rows = history_rows(q, led)
            if rows:
                sc = vec[[idx[LG.row_text(r)] for r in rows]] @ v
                m = LG.match(rows, sc.tolist(), rname, **MATCH)
                block = LG.operate(rname, m, q["question"], q.get("date") if q["bench"] == "lme" else None)
        if block is None:
            out.append(c)
        else:
            info["with_block"] += 1
            base = c["context"] if c["context"] != v0.NOT_FOUND else ""
            out.append({**c, "context": (base + "\n\n" if base else "") + LG.render(block),
                        "ledger": {"route": block.route, "value": block.value, "dias": [r["dia"] for r in block.rows]}})
    R.setdefault("contexts", {})[tag] = {**info, "embed_minutes": round((time.time() - t) / 60, 1)}
    log(f"ledger contexts {tag}:", dict(info))
    return out


def gates(led, model):
    """G1: share of hard-type evidence turns that have >= 1 ledger row. G2: share of hard-pool questions routed."""
    by_dia = set(r["dia"] for rs in led.values() for r in rs)
    ev_h, cov, routed, nh = 0, 0, 0, 0
    for q in D["qs"]:
        grp = q.get("type") if q["bench"] == "lme" else L.CATS.get(q.get("cat"))
        if grp not in HARD[q["bench"]]:
            continue
        nh += 1
        routed += LG.route(q["question"]) is not None
        for d in q["evidence"]:
            if q["bench"] == "lme" and not any(t["dia"] == d and t["speaker"] == "user" for t in q_turns(q)):
                continue
            ev_h += 1; cov += d in by_dia
    g = {"G1_evidence_turn_coverage": cov / max(1, ev_h), "evidence_turns": ev_h, "G2_hard_routed": routed / max(1, nh), "hard_n": nh}
    R["gates"][model] = g
    log(f"gates {model}:", g)
    return g


# ---------------------------------------------------------------------------------------------- phases
ROWS = {}


def jobs(arm, ctx):
    return [(arm, q, c) for q, c in zip(D["qs"], ctx)]


def read_arm(arm, ctx, cli, label):
    rows, sec = Bm.read(cli, jobs(arm, ctx), "P1", label)
    for r, c in zip(rows, ctx):
        r["ledger"] = c.get("ledger")
    ROWS[arm] = rows
    R["arms"].setdefault(arm, {})["s_per_question"] = round(sec, 3)
    V.write_rows(f"dev_{arm}", rows)
    save_r()


def short(m):
    return m.split("/")[-1].replace("Instruct-", "I").replace("-instruct", "").replace("Qwen", "Q")


def gpu_worker(gpu, port, tasks, out):
    """Runs (kind, model) tasks one after another on one GPU. kind: write | read."""
    for kind, model in tasks:
        if hours() > OPTIONAL_STOP_H:
            R["notes"].append(f"skipped {kind} {model}: time"); continue
        srv = Srv(model, gpu, port)
        if not srv.start():
            R["notes"].append(f"{model} could not be served on one T4"); continue
        try:
            cli = Cli(model, port, conc=48 if kind == "write" else 32)
            if kind == "write":
                out[model] = write_ledger(model, cli)
                gates(out[model], model)
                D["ctx"][f"L{SHORT[model]}P"] = ledger_contexts(out[model], True, f"L{SHORT[model]}P")   # CPU, overlaps GPU work
            else:
                arm = f"Z_{short(model)}"
                t = time.time()
                read_arm(arm, D["ctx"]["A0"], cli, f"{short(model)}|v3b_{arm}")
                R["zoo"][arm] = {"model": model, "gpus": 1, "load_min": srv.load_min, "minutes": round((time.time() - t) / 60, 1)}
        except Exception as e:
            STATE["errors"][f"{kind}_{model}"] = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            srv.stop(drop_weights=True)


@stage("v3b_single_gpu")
def stage_single():
    """Small writers, then zoo readers; GPU 0 and GPU 1 in parallel."""
    led = {}
    t0 = [("write", WRITER_SMALL[0])] + [("read", a) for a, _ in ZOO_PAIRS]
    t1 = [("write", WRITER_SMALL[1])] + [("read", b) for _, b in ZOO_PAIRS]
    th = [threading.Thread(target=gpu_worker, args=("0", 8011, t0, led)), threading.Thread(target=gpu_worker, args=("1", 8012, t1, led))]
    [x.start() for x in th]; [x.join() for x in th]
    D["led"] = led


@stage("v3b_moe")
def stage_moe():
    """The 35B-A3B MoE on both GPUs: writer, then reader of A0 and of its own L35P packets."""
    if hours() > OPTIONAL_STOP_H:
        R["notes"].append("MoE skipped: time"); return
    srv = Srv(MOE, "0,1", 8011)
    if not srv.start():
        R["notes"].append("MoE could not be served on T4 x2"); return
    try:
        cli = Cli(MOE, 8011, conc=48)
        D["led"][MOE] = write_ledger(MOE, cli)
        gates(D["led"][MOE], MOE)
        D["ctx"]["L35P"] = ledger_contexts(D["led"][MOE], True, "L35P")
        cli = Cli(MOE, 8011, conc=32)
        read_arm("Z35_A0", D["ctx"]["A0"], cli, "qwen3.5-35b-a3b|v3b_Z35_A0")
        read_arm("Z35_L35P", D["ctx"]["L35P"], cli, "qwen3.5-35b-a3b|v3b_Z35_L35P")
    finally:
        srv.stop(drop_weights=True)


@stage("v3b_reader")
def stage_reader():
    """Qwen3.5-9B on both GPUs: A0 (cached), its own ledger, then every ledger arm."""
    srv = Srv(READER, "0,1", 8011)
    if not srv.start():
        raise RuntimeError("vLLM could not serve the reader")
    try:
        cli = Cli(READER, 8011, conc=48)
        read_arm("A0", D["ctx"]["A0"], cli, f"{LABEL}|v3a_A0")         # v3a's label: same cache keys, no new calls
        D["led"][READER] = write_ledger(READER, cli)
        gates(D["led"][READER], READER)
        D["ctx"]["L9"] = ledger_contexts(D["led"][READER], False, "L9")
        D["ctx"]["L9P"] = ledger_contexts(D["led"][READER], True, "L9P")
        cli = Cli(READER, 8011, conc=32)
        for arm in ("L9", "L9P", "L35P", "L2P", "L08P"):
            if arm in D["ctx"] and hours() < v0.TIME_LIMIT_H:
                read_arm(arm, D["ctx"][arm], cli, f"{LABEL}|v3b_{arm}")
    finally:
        srv.stop(drop_weights=True)


@stage("v3b_judge")
def stage_judge():
    j = Srv(JUDGE_ID, "0,1", 8011)
    if not j.start():
        raise RuntimeError("judge could not be served")
    try:
        jc = Cli(JUDGE_ID, 8011, conc=64)
        for arm, rs in ROWS.items():
            Bm.judge(jc, rs)
            V.write_rows(f"dev_{arm}", rs)
    finally:
        j.stop()


# ---------------------------------------------------------------------------------------------- statistics + decisions
def grp(r):
    return r.get("type") if r["bench"] == "lme" else L.CATS.get(r.get("cat"))


def acc(rs, bench=None, groups=None):
    xs = [r["qwen_correct"] for r in rs if (r["bench"] != "locomo" or r["cat"] != 5) and (bench is None or r["bench"] == bench)
          and (groups is None or grp(r) in groups[r["bench"]])]
    return float(np.mean(xs)) if xs else None


def summary(rs):
    a = Bm.dev_acc(rs)
    lme = [r for r in rs if r["bench"] == "lme"]
    by = defaultdict(list)
    for r in rs:
        if r["bench"] != "locomo" or r["cat"] != 5:
            by[f"{r['bench']}:{grp(r)}"].append(r["qwen_correct"])
    return {**a, "hard": acc(rs, groups=HARD), "safe": acc(rs, groups=SAFE), "by_group": {k: float(np.mean(v)) for k, v in by.items()},
            "group_n": {k: len(v) for k, v in by.items()},
            "lme_acc_if_all_evidence": float(np.mean([r["qwen_correct"] for r in lme if r.get("evidence_all")])) if lme else None,
            "with_block": sum(1 for r in rs if r.get("ledger")), "mean_tokens": float(np.mean([r["input_tokens"] for r in rs]))}


def paired(a, b):
    keep = lambda rs: [r for r in rs if r["bench"] != "locomo" or r["cat"] != 5]
    return L.paired(keep(a), keep(b), key="qwen_correct", cluster=None)


def items_delta(a, b, groups):
    """Net items gained (a vs b, paired by qid) within groups."""
    bb = {r["qid"]: r for r in b}
    return sum(int(r["qwen_correct"]) - int(bb[r["qid"]]["qwen_correct"]) for r in a
               if r["qid"] in bb and (r["bench"] != "locomo" or r["cat"] != 5) and grp(r) in groups[r["bench"]])


def decide():
    s, out = R["arms"], {}
    a0 = s.get("A0")
    if not a0 or "mean" not in a0:
        return
    for arm in ("L9", "L9P", "L35P", "L2P", "L08P"):
        x = s.get(arm)
        if not x or "mean" not in x:
            continue
        dm, dh = x["mean"] - a0["mean"], x["hard"] - a0["hard"]
        ku = x["by_group"].get("lme:knowledge-update")
        safe = items_delta(ROWS[arm], ROWS["A0"], SAFE)
        verdict = ("KEEP" if dm >= 0.010 and dh >= 0.030 and (ku or 0) >= 0.885 and safe >= -2 else
                   "KILL" if dm <= -0.005 or dh < 0.010 else "ITERATE")
        out[arm] = {"mean_gain": dm, "hard_gain": dh, "ku": ku, "safe_items_net": safe, "verdict": verdict}
    if "L9P" in s and "L9" in s and "mean" in s["L9P"] and "mean" in s["L9"]:
        pk = lambda arm: s[arm]["by_group"].get("lme:single-session-preference")
        n_pref = s["L9P"]["group_n"].get("lme:single-session-preference", 20)
        p9p = round((pk("L9P") or 0) * n_pref)
        out["PREF_route"] = {"pref_items_L9P": p9p, "pref_items_L9": round((pk("L9") or 0) * n_pref),
                             "verdict": "KEEP" if p9p >= 13 and s["L9P"]["mean"] >= a0["mean"] else
                             "KILL" if p9p <= 10 or s["L9P"]["mean"] < a0["mean"] - 0.005 else "ITERATE"}
    if "L9P" in s and "mean" in s["L9P"]:
        g9 = s["L9P"]["mean"] - a0["mean"]
        for arm in ("L2P", "L08P", "L35P"):
            if arm in s and "mean" in s[arm]:
                g = s[arm]["mean"] - a0["mean"]
                out[f"writer_{arm}"] = {"gain": g, "share_of_L9P_gain": (g / g9) if g9 > 0 else None,
                                        "viable": g9 > 0 and g >= 0.7 * g9}
    for arm, x in s.items():
        if arm.startswith("Z") and "mean" in x:
            out[arm] = {"mean_gain": x["mean"] - a0["mean"], "s_per_question": x.get("s_per_question"),
                        "candidate": x["mean"] - a0["mean"] >= 0.02 and (x.get("s_per_question") or 99) <= 2 * A1_S_PER_Q}
    R["decisions"] = out


def stage_stats():
    for arm, rs in ROWS.items():
        if rs and "qwen_correct" in rs[0]:
            R["arms"].setdefault(arm, {}).update(summary(rs))
    if "A0" in ROWS and "qwen_correct" in ROWS["A0"][0]:
        for arm, rs in ROWS.items():
            if arm != "A0" and rs and "qwen_correct" in rs[0]:
                R["paired"][f"{arm}-A0"] = paired(rs, ROWS["A0"])
    if "L9P" in ROWS and "L9" in ROWS and "qwen_correct" in ROWS["L9P"][0]:
        R["paired"]["L9P-L9"] = paired(ROWS["L9P"], ROWS["L9"])
    decide()
    save_r()


def write_report():
    f = lambda x: "n/a" if x is None else f"{100 * x:.1f}"
    out = [f"# DAMA paper v3b: event ledger + model zoo, dev ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{', SMOKE' if SMOKE else ''})", "",
           "Dev only (v2c's sample). Judge Qwen2.5-7B. A0 = v2c C3_m0 with Qwen3.5-9B. L* = A0 + LEDGER block (writer in the name: "
           "9 = Qwen3.5-9B, 35 = Qwen3.5-35B-A3B GPTQ, 2 = Qwen3.5-2B, 08 = Qwen3.5-0.8B; P = with the PREF route). Z_* = other readers on "
           "the A0 packets. Rules: README.md.", "",
           "| arm | mean | LoCoMo | LME | hard | safe | KU | pref | multi-session | blocks | tokens | s/question |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in R["arms"].items():
        if "mean" in v:
            g = v["by_group"]
            out.append(f"| {k} | {f(v['mean'])} | {f(v['locomo'])} | {f(v['lme'])} | {f(v['hard'])} | {f(v['safe'])} | "
                       f"{f(g.get('lme:knowledge-update'))} | {f(g.get('lme:single-session-preference'))} | {f(g.get('lme:multi-session'))} | "
                       f"{v['with_block']} | {v['mean_tokens']:.0f} | {v.get('s_per_question')} |")
    out += ["", "## Decisions", "```", json.dumps(R["decisions"], indent=1, default=str), "```", "",
            "## Ledgers (write side) and gates", "```", json.dumps({"ledgers": R["ledgers"], "gates": R["gates"], "contexts": R.get("contexts")}, indent=1, default=str), "```", "",
            "## Zoo", "```", json.dumps(R["zoo"], indent=1, default=str), "```", "",
            "## Paired vs A0 (pooled LoCoMo cat. 1-4 + LME; exact McNemar, item bootstrap)", "```", json.dumps(R["paired"], indent=1, default=str), "```", "",
            "By group: `" + json.dumps({k: v.get("by_group") for k, v in R["arms"].items()}, default=str) + "`", "",
            f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`", f"- notes: {R['notes']}"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_V3B.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


import atexit
atexit.register(lambda: [s.stop() for s in list(Bm.RUNNING)])

if __name__ == "__main__":
    log("environment", v0.environment())
    Cm.stage_setup()
    if "v2c_setup" not in STATE["errors"]:
        V.stage_select_prior()
        D["ce_name"], D["ce_n"] = None, 0
        R["settings"]["prior"] = V.V2["settings"].get("prior", {}).get("choice")
        stage_contexts()
        if "v3b_contexts" not in STATE["errors"] and Bm.install_vllm(VLLM):
            D["led"] = {}
            stage_single()
            stage_moe()
            stage_reader()
            stage_judge()
    try:
        stage_stats()
    except Exception as e:
        STATE["errors"]["v3b_stats"] = f"{type(e).__name__}: {str(e)[:150]}"
    save_r()
    write_report()
    log(f"v3b done; errors: {STATE['errors'] or 'none'}")

"""DAMA paper v2b (2026-10-29): v2's memory, a better reader. Private Kaggle notebook `s41r4j/dama-paper-v2b` (T4 x2).
Plan, pre-registered rules and cost: README.md next to this file.

v2 (round 3) showed the memory is no longer the bottleneck: on the LongMemEval holdout V2_fast's packet held all the
evidence for 92% of questions, yet Qwen2.5-7B answered 48.5%. On LongMemEval dev (v1 rows), with all evidence in the
packet, the reader still got only 33% of multi-session, 45% of temporal and 15% of preference questions. This round
changes the reader only; memory (V2_fast, DAMA_H, B3 contexts) is v2's code, unchanged.

- Engine: vLLM (OpenAI server in its own venv, tensor parallel over both T4s, continuous batching). Every call cached.
- Prompt P1 ("notes first"): short evidence notes, counting and date arithmetic, then one "ANSWER:" line. P0 = v2's
  prompts. The reader's answer line is what the judges see.
- Readers: Qwen2.5-7B (v2's), Qwen3-8B, Qwen3.5-9B, Qwen3-14B-AWQ (non-thinking). Chosen on dev only,
  with a GPU-time cap, then frozen (FROZEN_V2B.json) before any holdout or test row. The judge stays Qwen2.5-7B.
- MODE "judge": DeepSeek labels for every final row, run once from the website (secret), no GPU.
"""
from __future__ import annotations
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODE = "full"   # "full": GPU run. "judge": no GPU; DeepSeek-judge every final row of this notebook's previous output
SMOKE = False   # True: every stage and every candidate reader on a few questions, to test the run cheaply first
FAKE_ENGINE = bool(os.environ.get("DAMA_FAKE_ENGINE"))   # local dry run: a stub server instead of vLLM (no GPU)
HERE = Path(globals().get("__file__", "dama_paper_v2b.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent / Path(name).parent


if __name__ == "__main__" and "--stub-server" in sys.argv:      # local dry run only: answers every call with fixed text
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = body["messages"][-1]["content"]
            if "Answer yes or no" in text:
                out = "yes"
            elif '"label"' in text:
                out = '{"label": "CORRECT"}'
            elif '"declines"' in text:
                out = '{"declines": false}'
            elif "NOTES" in body["messages"][0]["content"]:
                out = "<think>\n\n</think>\n\nNOTES: [m:1] says something.\nANSWER: 7 May 2023"
            else:
                out = "7 May 2023"
            time.sleep(0.01)
            msg = json.dumps({"choices": [{"message": {"content": out}}], "usage": {"prompt_tokens": len(text) // 4}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(msg)

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[-1])), Stub).serve_forever()

sys.path.insert(0, str(_exp_dir("v2_2026-10-28/dama_paper_v2.py")))
import dama_paper_v2 as V       # v2's memory, contexts and statistics (imports v1, v0 and v2lib); its main block does not run
L, v0, v1 = V.L, V.v0, V.v1
import numpy as np

V.SMOKE = v0.SMOKE = v1.SMOKE = SMOKE
v0.TIME_LIMIT_H, v0.FINAL_LIMIT_H = (2.0, 2.5) if SMOKE else (9.5, 10.5)
v1.DS_CAP_USD = 0.05 if SMOKE else 2.00
log, stage, OUT, D, STATE, SEED = v0.log, v0.stage, v0.OUT, v0.D, v0.STATE, v0.SEED

# ---------------------------------------------------------------------------------------------- pre-registered settings
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"          # the Qwen judge of rounds 1-3, now served by vLLM; fixed for every arm
CANDIDATES = [                                  # cheapest first: ties within TIE_PT go to the earlier one
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
    ("Qwen/Qwen3-8B", "qwen3-8b"),
    ("Qwen/Qwen3.5-9B", "qwen3.5-9b"),
    ("Qwen/Qwen3-14B-AWQ", "qwen3-14b-awq"),
]   # smoke run (version 1, 12 dev questions, speed only): Qwen3-32B-AWQ 14.7 s/question (~10 h projected) and
    # Qwen3.5-27B-GPTQ-Int4 (13.9 GiB per GPU, out of memory) do not fit the budget; replaced before any dev accuracy
PROMPT_GAIN_PT = 0.01        # P1 replaces P0 if it adds >= 1 point of mean dev accuracy with the 7B reader
TIE_PT = 0.01                # readers within 1 point of the best dev accuracy count as tied -> the cheaper one
FINAL_READER_BUDGET_H = 4.5  # a reader whose projected final-run reading time exceeds this is not eligible (smoke: 7B 2.2 h, 14B 4.4 h)
DEV_LOCO, DEV_LOCO_ADV, DEV_LME_PER_TYPE = (10, 2, 2) if SMOKE else (100, 30, 20)
VLLM_VERSIONS = ("0.32.0", "0.20.2", "0.10.1.1")   # newest first; the first that serves the 7B reader on the T4s is kept
VENV, HF_HOME, PORT = Path("/tmp/vllm_env"), Path("/tmp/hf"), 8011
MAX_LEN, CONCURRENCY = 8192, 48
B = {"settings": {}, "dev": {}, "lme": {}, "locomo": {}, "stats": {}, "engine": {}, "notes": []}
H = V.H

LOCO_P1 = """You answer questions about a long conversation between two people, using only the memory provided.
Rules:
- Use only the MEMORY section. Do not use outside knowledge.
- First write NOTES: list the memory lines that matter for the question, each with its date. If the question needs several facts, counting or date arithmetic, work it out step by step. Keep the notes under 120 words.
- Then write one final line that starts with "ANSWER:" and gives a short answer (at most 15 words).
- For time questions, give absolute dates (for example "7 May 2023"), using the session dates shown and any resolved dates in [= ...].
- If the memory does not contain the answer, the final line is exactly: ANSWER: Not mentioned in the conversation."""
LME_P1 = """You answer questions about a user's past conversations with an AI assistant, using only the memory provided.
Rules:
- Use only the MEMORY section. Do not use outside knowledge.
- First write NOTES: list every memory line that matters for the question, with its session date. When the question asks how many, which ones or what changed, go through all the sessions and list each item before counting; if a fact was updated, the latest session wins. For time questions, work out the dates and the difference step by step from the session dates and the current date. Keep the notes under 150 words.
- If the question asks for a suggestion or recommendation, base it on the user's own preferences and details found in the memory.
- Then write one final line that starts with "ANSWER:" and gives the answer in at most 30 words.
- If the memory does not contain the answer, the final line is exactly: ANSWER: Not mentioned in the conversation."""
PROMPTS = {"P0": {"locomo": (v0.READER_SYSTEM, 40), "lme": (v1.LME_SYSTEM, 64)},
           "P1": {"locomo": (LOCO_P1, 400), "lme": (LME_P1, 400)}}


def save_b():
    (OUT / "results_v2b.json").write_text(json.dumps(B, indent=1, default=str))


def disk_gb():
    return round(shutil.disk_usage("/tmp").free / 2**30, 1)


# ---------------------------------------------------------------------------------------------- engine (vLLM server)
def _run(cmd, timeout):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def install_vllm(version):
    """vLLM in its own venv (Python 3.12), so the notebook's torch, sentence-transformers and onnxruntime stay intact."""
    if FAKE_ENGINE:
        return True
    shutil.rmtree(VENV, ignore_errors=True)
    t = time.time()
    _run([sys.executable, "-m", "pip", "install", "-q", "uv"], 600)
    rc, out = _run([sys.executable, "-m", "uv", "venv", "-q", "--python", "3.12", str(VENV)], 900)
    if rc == 0:
        rc, out = _run([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", str(VENV / "bin/python"), f"vllm=={version}"], 1500)
    B["engine"].setdefault("installs", []).append({"vllm": version, "ok": rc == 0, "minutes": round((time.time() - t) / 60, 1), "tail": out[-600:] if rc else ""})
    log(f"vLLM {version} install:", "ok" if rc == 0 else f"failed: {out[-600:]}", f"({(time.time() - t) / 60:.1f} min, /tmp free {disk_gb()} GB)")
    return rc == 0


RUNNING = []      # servers to stop if the run ends early (they run in their own process group)


class Server:
    """One model at a time on both T4s. `vllm serve` (OpenAI API) on localhost; stop() frees both GPUs."""
    def __init__(self, model):
        self.model, self.proc, self.log_path = model, None, OUT / f"vllm_{model.split('/')[-1]}.log"

    def start(self, timeout_min=30):
        if not FAKE_ENGINE and disk_gb() < 30:     # room for the next download: drop every other cached model first
            for d in (HF_HOME / "hub").glob("models--*"):
                if d.name != "models--" + self.model.replace("/", "--"):
                    shutil.rmtree(d, ignore_errors=True)
            log("dropped cached weights for disk space; /tmp free", disk_gb(), "GB")
        if FAKE_ENGINE:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--stub-server", str(PORT)]
        else:
            cmd = [str(VENV / "bin/vllm"), "serve", self.model, "--served-model-name", "m", "--port", str(PORT), "--dtype", "half",
                   "--tensor-parallel-size", "2", "--max-model-len", str(MAX_LEN), "--gpu-memory-utilization", "0.88",
                   "--max-num-seqs", "64", "--seed", str(SEED), "--disable-custom-all-reduce"]
        env = {**os.environ, "HF_HOME": str(HF_HOME), "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        t = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.log_path.open("w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
        RUNNING.append(self)
        while time.time() - t < timeout_min * 60:
            if self.proc.poll() is not None:
                break
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
                    self.load_min = round((time.time() - t) / 60, 1)
                    log(f"served {self.model} in {self.load_min} min; /tmp free {disk_gb()} GB")
                    return True
            except Exception:
                pass
            time.sleep(5)
        tail = self.log_path.read_text(errors="replace")[-3000:] if self.log_path.exists() else ""
        log(f"could not serve {self.model}:", tail)
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
        if self in RUNNING:
            RUNNING.remove(self)
        time.sleep(3 if FAKE_ENGINE else 10)
        if drop_weights:                       # keeps /tmp from filling up across four candidate downloads
            shutil.rmtree(HF_HOME / "hub" / ("models--" + self.model.replace("/", "--")), ignore_errors=True)


def clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class Client:
    """Same call signature as v0.LLM: (list of message lists, max_new) -> (outputs, prompt_tokens, ms). Greedy; cached in
    the shared SQLite cache under (model, messages, max_new, "vllm")."""
    def __init__(self, model):
        self.name = model
        self.failed = 0

    def _post(self, messages, max_new):
        body = {"model": "m", "messages": messages, "max_tokens": max_new, "temperature": 0, "seed": SEED,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            t = time.time()
            try:
                j = json.loads(urllib.request.urlopen(req, timeout=1800).read())
                return clean(j["choices"][0]["message"].get("content")), int(j.get("usage", {}).get("prompt_tokens", 0)), (time.time() - t) * 1000
            except urllib.error.HTTPError as e:      # 400: prompt longer than MAX_LEN; counted in `failed`, not retried
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
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            for i, (o, pt, ms) in zip(todo, ex.map(lambda i: self._post(convs[i], max_new), todo)):
                if o is None:
                    self.failed += 1; res[i] = ("", pt, ms)
                else:
                    v0.CACHE.put(keys[i], self.name, o, pt, ms); res[i] = (o, pt, ms)
        return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]


def answer_of(raw, prompt):
    """P1: the text after the last "ANSWER:" (first line). No such line (notes cut at the token cap): the last line."""
    if prompt == "P0":
        return raw, True
    m = list(re.finditer(r"ANSWER\s*:\s*", raw or "", flags=re.I))
    if m:
        return raw[m[-1].end():].strip().split("\n")[0].strip() or v0.NOT_FOUND, True
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    return (lines[-1] if lines else ""), False


def holdout_rows(path):
    """v2's LongMemEval rows carry split "holdout"; v1's (same file names, dama-paper-v0 output) are dev rows without it."""
    with open(path) as f:
        return json.loads(f.readline()).get("split") == "holdout"


# ---------------------------------------------------------------------------------------------- setup (v2's, embeddings batched)
@stage("v2b_setup")
def stage_setup():
    import torch
    from sentence_transformers import SentenceTransformer
    from dama_agent.fastmodels import OnnxEncoder
    v0.stage_data()
    convs, lq = L.load_locomo(OUT / "locomo10.json")
    old = {q["qid"]: q["evidence"] for q in D["questions"]}
    if {q["qid"]: q["evidence"] for q in lq if q["qid"] in old} != old:
        raise ValueError("v2lib.load_locomo disagrees with dama_paper_v0.load_data")
    lme_path = Path(os.environ.get("DAMA_LME_PATH", "/tmp/longmemeval_s_cleaned.json"))
    if not lme_path.exists():
        urllib.request.urlretrieve(L.LME_URL, lme_path)
    if L.sha256(lme_path) != L.LME_SHA256:
        raise ValueError("LongMemEval-S file differs from the pinned SHA-256")
    meta = L.lme_splits(lme_path)
    v1_rows = [p for p in sorted(glob.glob("/kaggle/input/**/paper/methods/LME_DAMA_H.jsonl", recursive=True)) if not holdout_rows(p)]
    v1_dev = [json.loads(l)["qid"] for l in open(v1_rows[0])] if v1_rows else None
    B["leakage"] = L.leakage_audit(meta, v1_dev)
    lme = L.load_lme(lme_path, limit=int(os.environ.get("DAMA_LME_LIMIT", 12)) if SMOKE else None, meta=meta)   # env: local dry runs
    if SMOKE:
        keep = []
        for ci in (0, 1, 4, 7):
            keep += [q for q in lq if q["ci"] == ci][:12]
        lq = keep
    D["convs"], D["lq"], D["lme"] = convs, lq, [it for it in lme if it["split"] in ("dev", "holdout")]
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
    allq = lq + D["lme"]
    D["qv"] = dict(zip([q["qid"] for q in allq], enc([q["question"] for q in allq])))
    t = time.time()
    hist = [(("locomo", c["ci"]), c["turns"]) for c in convs if any(q["ci"] == c["ci"] for q in lq)]
    hist += [(("lme", it["qid"]), it["turns"]) for it in D["lme"]]
    texts = [x for _, turns in hist for x in L.key_texts(turns)]
    vec = D["bge"].encode(texts, batch_size=128 if dev == "cuda" else 16, normalize_embeddings=True, convert_to_numpy=True)   # one call: sorted by length
    s = 0
    for key, turns in hist:
        H[key] = {"reader": L.build_reader(turns, vec[s:s + len(turns)], enc), "turns": turns}
        s += len(turns)
    B["settings"]["embedding_minutes"] = round((time.time() - t) / 60, 1)
    B["settings"]["data"] = {"locomo_questions": len(lq), "lme": dict(Counter(it["split"] for it in D["lme"]))}
    log("v2b setup:", B["settings"]["data"], f"embeddings {B['settings']['embedding_minutes']} min")
    save_b()


@stage("v2b_prior")
def stage_prior():
    """v2's prior selection, rerun exactly (deterministic); must match v2's frozen choice."""
    V.stage_select_prior()
    D["ce_name"], D["ce_n"] = None, 0           # v2: V2_acc not run (no cross-encoder beat the prior on dev)
    choice = V.V2["settings"].get("prior", {}).get("choice")
    B["settings"]["prior"] = choice
    if choice != "v2|gbm":
        B["notes"].append(f"prior selection gave {choice}, v2 froze v2|gbm")
    save_b()


# ---------------------------------------------------------------------------------------------- jobs: contexts first, GPU freed after
def b3_ctx_fn():
    ce = V.ce_model(*V.B3_CE)
    def b3(q):
        turns = V.reader_of(q)[1]
        z = ce.predict([(q["question"], L.pair_text(t)) for t in turns], batch_size=256, show_progress_bar=False)
        chosen, used = [], 0
        for j in np.argsort(-np.asarray(z)):
            n = L.ntok(L.render(turns[j])) + 1
            if used + n <= L.BUDGET:
                chosen.append(turns[j]); used += n
        return {"context": L.render_many(chosen) if chosen else v0.NOT_FOUND, "dia": [t["dia"] for t in chosen]}
    return b3, ce


def dev_sample():
    rng = random.Random(SEED)
    ld = [q for q in D["lq"] if q["split"] == "dev"]
    loco = rng.sample([q for q in ld if q["cat"] != 5], min(DEV_LOCO, sum(q["cat"] != 5 for q in ld)))
    loco += rng.sample([q for q in ld if q["cat"] == 5], min(DEV_LOCO_ADV, sum(q["cat"] == 5 for q in ld)))
    lme = []
    by = defaultdict(list)
    for it in D["lme"]:
        if it["split"] == "dev":
            by[it["type"]].append(it)
    for t in sorted(by):
        lme += rng.sample(by[t], min(DEV_LME_PER_TYPE, len(by[t])))
    return loco, lme


@stage("v2b_contexts")
def stage_contexts():
    """Every context of every arm is built now (CPU memory + GPU cross-encoder), so the GPUs are free for vLLM after."""
    import torch
    loco_dev, lme_dev = dev_sample()
    hold = [it for it in D["lme"] if it["split"] == "holdout"]
    sub = V.test_subset()
    if SMOKE:
        hold, sub = hold[:8], sub[:12]
    oracle = [q for q in D["lq"] if q["split"] == "test" and q["cat"] != 5 and q["evidence"]][: (12 if SMOKE else None)]
    b3, ce = b3_ctx_fn()
    J = {"dev": [("dev", q, V.v2_ctx(q, notes=True)) for q in loco_dev + lme_dev],
         "LME_V2b_fast": [("LME_V2b_fast", it, V.v2_ctx(it, notes=True)) for it in hold],
         "LME_V2b_DAMA_H": [("LME_V2b_DAMA_H", it, V.h_ctx_v1(it)) for it in hold],
         "LME_V2b_B3": [("LME_V2b_B3", it, b3(it)) for it in hold],
         "V2b_fast": [("V2b_fast", q, V.v2_ctx(q, notes=True)) for q in sub],
         "V2b_DAMA_H": [("V2b_DAMA_H", q, V.h_ctx_v1(q)) for q in sub],
         "V2b_B3": [("V2b_B3", q, b3(q)) for q in sub],
         "ORACLE_gold_notes_v2b": [("ORACLE_gold_notes_v2b", q, V.oracle_ctx(q, True)) for q in oracle]}
    D["jobs"] = J
    del ce, b3
    for k in ("bge",):
        D.pop(k, None)
    import gc
    gc.collect(); torch.cuda.empty_cache()
    B["settings"]["jobs"] = {k: len(v) for k, v in J.items()}
    log("v2b contexts:", B["settings"]["jobs"], "GPU memory in use (GiB):",
        [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else "no GPU")
    save_b()


# ---------------------------------------------------------------------------------------------- reading + judging
def messages(q, ctx, prompt):
    system, _ = PROMPTS[prompt][q["bench"]]
    if q["bench"] == "lme":
        user = f"MEMORY:\n{ctx}\n\nCURRENT DATE: {q['date']}\nQUESTION: {q['question']}\n" + ("ANSWER:" if prompt == "P0" else "NOTES:")
    else:
        user = f"MEMORY:\n{ctx}\n\nQUESTION: {q['question']}\n" + ("ANSWER:" if prompt == "P0" else "NOTES:")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read(client, jobs, prompt, label):
    """jobs: (arm, question, context dict). Returns rows (unjudged) and the wall-clock seconds per question."""
    t = time.time()
    msgs = [messages(q, c["context"], prompt) for _, q, c in jobs]
    rows = [None] * len(jobs)
    for bench in ("lme", "locomo"):          # one call per max_new value
        idx = [i for i, (_, q, _) in enumerate(jobs) if q["bench"] == bench]
        if not idx:
            continue
        outs, ptoks, ms = client([msgs[i] for i in idx], max_new=PROMPTS[prompt][bench][1])
        for i, o, pt, m in zip(idx, outs, ptoks, ms):
            arm, q, c = jobs[i]
            ans, ok = answer_of(o, prompt)
            ev = set(q["evidence"])
            r = {"method": arm, "reader": label, "prompt": prompt, "qid": q["qid"], "question": q["question"], "gold": q["gold"],
                 "answer": ans, "raw": o, "parsed": ok, "context": c["context"], "reader_prompt_tokens": pt, "reader_ms": m,
                 "input_tokens": v0.ntok(msgs[i][0]["content"]) + v0.ntok(msgs[i][1]["content"]), "read_ms": c.get("read_ms"),
                 "evidence_all": (ev <= set(c.get("dia") or [])) if ev and c.get("dia") is not None else None,
                 "budget_limited": c.get("budget_limited"), "flags": None, "bench": q["bench"]}
            if q["bench"] == "lme":
                r.update({"type": q["type"], "abs": q["abs"], "split": q["split"]})
            else:
                r.update({"ci": q["ci"], "cat": q["cat"]})
            rows[i] = r
    sec = (time.time() - t) / max(1, len(jobs))
    log(f"read {label}/{prompt}: {len(jobs)} questions, {sec:.2f} s/question wall, unparsed {sum(not r['parsed'] for r in rows)}, failed calls {client.failed}")
    return rows, sec


def judge(client, rows):
    """Qwen2.5-7B labels: LoCoMo with rounds 1-3's rules (v0.judge_rows), LongMemEval with the benchmark's prompts."""
    loco = [r for r in rows if r["bench"] == "locomo"]
    if loco:
        v0.judge_rows(loco, llm=client)
    lme = [r for r in rows if r["bench"] == "lme"]
    if lme:
        outs, _, _ = client([[{"role": "user", "content": v1.lme_judge_prompt(
            {"question_id": r["qid"], "question_type": r["type"], "question": r["question"], "answer": r["gold"]}, r["answer"])}] for r in lme], max_new=6)
        for r, o in zip(lme, outs):
            r["qwen_correct"] = "yes" in (o or "").lower()
    for r in loco:
        r["qwen_correct"] = r["correct"]
    return rows


def dev_acc(rows):
    """Mean over the two benchmarks of Qwen-judged accuracy (LoCoMo categories 1-4; LongMemEval all types)."""
    lo = [r["correct"] for r in rows if r["bench"] == "locomo" and r["cat"] != 5]
    lm = [r["qwen_correct"] for r in rows if r["bench"] == "lme"]
    adv = [r["correct"] for r in rows if r["bench"] == "locomo" and r["cat"] == 5]
    by = defaultdict(list)
    for r in rows:
        if r["bench"] == "lme":
            by[r["type"]].append(r["qwen_correct"])
    return {"mean": float(np.mean([np.mean(lo), np.mean(lm)])), "locomo": float(np.mean(lo)), "lme": float(np.mean(lm)),
            "locomo_abstention": float(np.mean(adv)) if adv else None, "lme_by_type": {t: float(np.mean(v)) for t, v in by.items()},
            "parsed": float(np.mean([r["parsed"] for r in rows])), "n": len(rows)}


def final_long_prompts():
    J = D["jobs"]
    n = sum(len(v) for k, v in J.items() if k not in ("dev", "ORACLE_gold_notes_v2b"))
    return n + 0.2 * len(J["ORACLE_gold_notes_v2b"]), (2319 + 0.2 * 954)   # this run, a full run (3 x 272 + 3 x 501 + oracle)


@stage("v2b_select")
def stage_select():
    """Dev only. (1) Prompt on the 7B reader: P1 if it adds >= 1 point. (2) Reader: highest mean dev accuracy among the
    candidates that load and fit the GPU-time cap; ties within 1 point -> the cheaper. The 7B judge labels everything."""
    jobs = D["jobs"]["dev"]
    gen, sec = {}, {}
    ok = False
    for ver in VLLM_VERSIONS:
        if install_vllm(ver):
            srv = Server(JUDGE_ID)
            if srv.start():
                B["engine"]["vllm"], ok = ver, True
                B["engine"]["load_min"] = {JUDGE_ID: srv.load_min}
                break
        B["notes"].append(f"vLLM {ver} could not serve {JUDGE_ID} on this GPU")
    if not ok:
        raise RuntimeError("no vLLM version could serve the 7B reader")
    judge_client = Client(JUDGE_ID)
    c7 = Client(CANDIDATES[0][0])
    for p in ("P0", "P1"):
        gen[(CANDIDATES[0][1], p)], sec[(CANDIDATES[0][1], p)] = read(c7, jobs, p, CANDIDATES[0][1])
        judge(judge_client, gen[(CANDIDATES[0][1], p)])
    a0, a1 = dev_acc(gen[(CANDIDATES[0][1], "P0")]), dev_acc(gen[(CANDIDATES[0][1], "P1")])
    prompt = "P1" if a1["mean"] - a0["mean"] >= PROMPT_GAIN_PT else "P0"
    B["dev"]["prompt"] = {"P0": a0, "P1": a1, "choice": prompt, "rule": f"P1 if mean dev accuracy gain >= {PROMPT_GAIN_PT:.2f} with the 7B reader"}
    log("v2b prompt choice:", prompt, {"P0": a0["mean"], "P1": a1["mean"]})
    srv.stop()
    for model, label in CANDIDATES[1:]:
        if (time.time() - v0.T_START) / 3600 > (1.5 if SMOKE else 4.0):
            B["notes"].append(f"{label} not tried: dev selection time cap"); continue
        s = Server(model)
        if s.start():
            B["engine"]["load_min"][model] = s.load_min
            gen[(label, prompt)], sec[(label, prompt)] = read(Client(model), jobs, prompt, label)
        else:
            B["notes"].append(f"{label} could not be served on T4 x2 with vLLM {B['engine']['vllm']}")
        s.stop(drop_weights=True)
    srv = Server(JUDGE_ID)
    if not srv.start():
        raise RuntimeError("judge could not be served again")
    D["judge_server"] = srv
    for (label, p), rows in gen.items():
        if "qwen_correct" not in rows[0]:
            judge(judge_client, rows)
    n_final_this, n_final_full = final_long_prompts()
    table = {}
    for (label, p), rows in gen.items():
        a = dev_acc(rows)
        table[f"{label}|{p}"] = {**a, "s_per_question": round(sec[(label, p)], 3), "projected_full_run_h": round(sec[(label, p)] * n_final_full / 3600, 2)}
    eligible = [(m, l) for m, l in CANDIDATES if f"{l}|{prompt}" in table and table[f"{l}|{prompt}"]["projected_full_run_h"] <= FINAL_READER_BUDGET_H]
    if not eligible:
        eligible = [CANDIDATES[0]]
        B["notes"].append("no candidate fits the GPU-time cap; the 7B reader is kept")
    best = max(table[f"{l}|{prompt}"]["mean"] for _, l in eligible)
    model, label = next((m, l) for m, l in eligible if table[f"{l}|{prompt}"]["mean"] >= best - TIE_PT)
    D["reader"], D["reader_label"], D["prompt"] = model, label, prompt
    B["dev"]["grid"] = table
    B["dev"]["choice"] = {"reader": model, "prompt": prompt, "rule": f"highest mean dev accuracy among readers whose projected full-run reading time <= {FINAL_READER_BUDGET_H} h; ties within {TIE_PT:.2f} -> cheaper"}
    for (label_, p), rows in gen.items():
        V.write_rows(f"dev_{label_}_{p}", rows)
    log("v2b reader choice:", label, prompt, json.dumps(table, default=str)[:3000])
    save_b()


@stage("v2b_freeze")
def stage_freeze():
    frozen = {"reader": D["reader"], "prompt": D["prompt"], "judge": JUDGE_ID, "vllm": B["engine"].get("vllm"), "prior": B["settings"].get("prior"),
              "pool": L.POOL, "B": L.BUDGET, "neighbours": L.NEIGHBOURS, "time_notes": True,
              "systems": {b: PROMPTS[D["prompt"]][b][0] for b in ("locomo", "lme")},
              "sha256": {"dama_paper_v2b.py": V.file_sha(globals().get("__file__", "")), "dama_paper_v2.py": V.file_sha(V.__file__), "v2lib.py": V.file_sha(L.__file__)},
              "frozen_at_min": round((time.time() - v0.T_START) / 60, 1)}
    (OUT / "FROZEN_V2B.json").write_text(json.dumps(frozen, indent=1))
    B["settings"]["frozen"] = frozen
    log("v2b frozen before any holdout or test row:", frozen)
    save_b()


FINAL_ARMS = ["LME_V2b_fast", "LME_V2b_DAMA_H", "LME_V2b_B3", "V2b_fast", "V2b_DAMA_H", "V2b_B3", "ORACLE_gold_notes_v2b"]


def v2_rows_to_rejudge():
    """v2's LongMemEval holdout rows (Qwen2.5-7B, P0, transformers): re-judged by this run's judge, for a same-judge comparison."""
    out = {}
    for name in ("LME_V2_fast", "LME_DAMA_H", "LME_B3"):
        hits = [p for p in sorted(glob.glob(f"/kaggle/input/**/paper/methods/{name}.jsonl", recursive=True)) if holdout_rows(p)]
        if hits:
            rows = [json.loads(l) for l in open(hits[0])]
            for r in rows:
                r.update({"bench": "lme", "qwen_correct_v2": r.get("qwen_correct"), "method": f"{name}_v2", "reader": "qwen2.5-7b-hf", "prompt": "P0"})
                r.pop("ds_correct", None)
            out[f"{name}_v2"] = rows[:8] if SMOKE else rows
    return out


@stage("v2b_final")
def stage_final():
    """Holdout first. Generation with the frozen reader, then every row judged by the 7B judge."""
    J, rows = D["jobs"], {}
    judge_srv = D.pop("judge_server", None)
    if D["reader"] != JUDGE_ID:
        judge_srv.stop()
        srv = Server(D["reader"])
        if not srv.start():
            raise RuntimeError(f"{D['reader']} could not be served for the final run")
    else:
        srv = judge_srv
    rc = Client(D["reader"])
    for arm in FINAL_ARMS:
        if (time.time() - v0.T_START) / 3600 > v0.TIME_LIMIT_H:
            STATE["errors"][arm] = "time guard"; continue
        rows[arm], s = read(rc, J[arm], D["prompt"], D["reader_label"])
        B.setdefault("final_s_per_question", {})[arm] = round(s, 3)
        V.write_rows(arm, rows[arm])                 # unjudged, in case the judge step fails
    if D["reader"] != JUDGE_ID:
        srv.stop(drop_weights=True)
        srv = Server(JUDGE_ID)
        if not srv.start():
            raise RuntimeError("judge could not be served for the final rows")
    jc = Client(JUDGE_ID)
    rows.update(v2_rows_to_rejudge())
    for arm, rs in rows.items():
        judge(jc, rs)
        V.write_rows(arm, rs)
        summarize(arm, rs)
    srv.stop()
    D["rows"] = rows
    save_b()


def summarize(arm, rs):
    if rs and rs[0]["bench"] == "lme":
        B["lme"][arm] = {**V.lme_summary(rs), "parsed": float(np.mean([r.get("parsed", True) for r in rs]))}
        log(f"LME {arm}:", {k: B["lme"][arm][k] for k in ("n", "acc_ds", "acc_qwen", "mean_tokens", "evidence_recall")})
    else:
        s = v0.summarize(rs)
        if any(r.get("ds_correct") is not None for r in rs):
            s["acc_ds"] = V._acc([r for r in rs if r["cat"] != 5], "ds_correct")
            s["by_category_ds"] = {L.CATS[c]: V._acc([r for r in rs if r["cat"] == c], "ds_correct") for c in L.CATS}
        B["locomo"][arm] = {k: s.get(k) for k in ("n", "acc", "acc_ci", "acc_ds", "by_category", "by_category_ds", "abstention", "mean_tokens", "evidence_recall")}
        log(f"LoCoMo {arm}:", {k: B["locomo"][arm][k] for k in ("n", "acc", "abstention", "mean_tokens", "evidence_recall")})


# ---------------------------------------------------------------------------------------------- statistics + report
def stage_stats():
    rows, st = D.get("rows", {}), {}
    for key in ("ds_correct", "qwen_correct"):
        for a, b in (("LME_V2b_fast", "LME_V2b_DAMA_H"), ("LME_V2b_fast", "LME_V2b_B3"), ("LME_V2b_fast", "LME_V2_fast_v2"),
                     ("LME_V2b_B3", "LME_B3_v2"), ("LME_V2_fast_v2", "LME_DAMA_H_v2"), ("LME_V2_fast_v2", "LME_B3_v2")):
            if a in rows and b in rows:
                st.setdefault("lme_holdout", {}).setdefault(key, {})[f"{a} - {b}"] = L.paired(rows[a], rows[b], key=key, cluster=None)
        for a, b in (("V2b_fast", "V2b_DAMA_H"), ("V2b_fast", "V2b_B3")):
            if a in rows and b in rows:
                st.setdefault("locomo", {}).setdefault(key, {})[f"{a} - {b}"] = L.paired([r for r in rows[a] if r["cat"] != 5], [r for r in rows[b] if r["cat"] != 5], key=key)
    for key in ("ds_correct", "qwen_correct"):          # Holm over the two pre-registered primary comparisons
        prim = st.get("lme_holdout", {}).get(key) or {}
        ps = sorted((v["mcnemar_p"], k) for k, v in prim.items() if k in ("LME_V2b_fast - LME_V2b_DAMA_H", "LME_V2b_fast - LME_V2b_B3") and v)
        st.setdefault("primary_holm", {})[key] = {k: {"p": p, "diff": prim[k]["diff"], "ci": prim[k]["ci"],
                                                      "supported": prim[k]["ci"][0] > 0 and all(pp <= 0.05 / (len(ps) - j) for j, (pp, _) in enumerate(ps[:i + 1]))}
                                                  for i, (p, k) in enumerate(ps)}
    B["stats"] = st
    save_b()


def ds_judge_all():
    rows = D.get("rows", {})
    ds = v1.ds_client()
    if ds is None:
        return
    for arm, rs in rows.items():
        if rs[0]["bench"] == "locomo":
            v1.ds_judge_rows(rs)
        else:
            jd = ds.many([[{"role": "user", "content": v1.lme_judge_prompt({"question_id": r["qid"], "question_type": r["type"], "question": r["question"], "answer": r["gold"]}, r["answer"])}]
                          for r in rs], max_tokens=6)
            for r, o in zip(rs, jd):
                r["ds_correct"] = None if o is None else ("yes" in o.lower())
        V.write_rows(arm, rs)
        summarize(arm, rs)
    B["ds_cost"] = ds.summary()


def fmt(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def write_report():
    s, out = B["settings"], []
    out += [f"# DAMA paper v2b results ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, mode {MODE}{', SMOKE' if SMOKE else ''})", "",
            "Plan and pre-registration: `experiments/v2b_2026-10-29/README.md`. Memory = v2 (unchanged); this round changes the reader. "
            "LongMemEval holdout: second use (v2 ran it once with the 7B reader). LoCoMo: regression only.", "",
            "## 1. Frozen settings", f"`{json.dumps(s.get('frozen'), default=str)}`", "", f"- engine: `{json.dumps(B['engine'], default=str)}`",
            f"- leakage: `{json.dumps(B.get('leakage'), default=str)}`", "",
            "## 2. Dev selection (Qwen2.5-7B judge)", f"- prompt: `{json.dumps(B['dev'].get('prompt'), default=str)}`", "",
            "| reader prompt | mean | LoCoMo | LME | LoCoMo abstention | parsed | s/question | projected full run h |", "|---|---|---|---|---|---|---|---|"]
    for k, v in B["dev"].get("grid", {}).items():
        out.append(f"| {k} | {fmt(v['mean'])} | {fmt(v['locomo'])} | {fmt(v['lme'])} | {fmt(v['locomo_abstention'])} | {fmt(v['parsed'])} | {v['s_per_question']} | {v['projected_full_run_h']} |")
    out += ["", f"- choice: `{json.dumps(B['dev'].get('choice'), default=str)}`", "",
            "## 3. LongMemEval-S holdout", "| arm | n | DeepSeek % [95% CI] | Qwen % | tokens | evidence % | parsed % |", "|---|---|---|---|---|---|---|"]
    for k, v in B["lme"].items():
        ci = v.get("ci_ds") or (None, None)
        out.append(f"| {k} | {v['n']} | {fmt(v.get('acc_ds'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(v.get('acc_qwen'))} | {v.get('mean_tokens') or 0:.0f} | {fmt(v.get('evidence_recall'))} | {fmt(v.get('parsed'))} |")
    out += ["", "By type: `" + json.dumps({k: v.get("by_type") for k, v in B["lme"].items()}, default=str) + "`", "",
            "Paired (item bootstrap, exact McNemar):", "```", json.dumps(B["stats"].get("lme_holdout"), indent=1, default=str), "```",
            f"Primary (Holm over V2b_fast vs DAMA_H and vs B3): `{json.dumps(B['stats'].get('primary_holm'), default=str)}`", "",
            "## 4. LoCoMo test subset (501; regression)", "| arm | n | Qwen % [95% CI] | DeepSeek % | abstention | tokens | evidence % |", "|---|---|---|---|---|---|---|"]
    for k, v in B["locomo"].items():
        ci = v.get("acc_ci") or (None, None)
        out.append(f"| {k} | {v['n']} | {fmt(v.get('acc'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(v.get('acc_ds'))} | {fmt(v.get('abstention'))} | {v.get('mean_tokens') or 0:.0f} | {fmt(v.get('evidence_recall'))} |")
    out += ["", "By category: `" + json.dumps({k: {"qwen": v.get("by_category"), "ds": v.get("by_category_ds")} for k, v in B["locomo"].items()}, default=str) + "`", "",
            "```", json.dumps(B["stats"].get("locomo"), indent=1, default=str), "```", "",
            "## 5. Errors, notes, cost", f"- errors: `{json.dumps(STATE['errors'])}`", f"- stage minutes: `{json.dumps(STATE['stages'])}`",
            f"- final s/question: `{json.dumps(B.get('final_s_per_question'))}`", f"- notes: {B['notes']}", f"- DeepSeek: `{json.dumps(B.get('ds_cost'), default=str)}`"]
    text = "\n".join(out) + "\n"
    (OUT / "RESULTS_V2B.md").write_text(text)
    print("=====BEGIN REPORT=====\n" + text + "=====END REPORT=====", flush=True)


@stage("v2b_carry")
def stage_carry():
    """Judge mode: this notebook's previous output (rows, results)."""
    src = sorted(glob.glob("/kaggle/input/**/paper/results_v2b.json", recursive=True))
    if not src:
        raise FileNotFoundError("no earlier results_v2b.json attached")
    root = Path(src[0]).parent
    B.update(json.loads((root / "results_v2b.json").read_text()))
    rows = {}
    for f in sorted((root / "methods").glob("*.jsonl")):
        if f.stem in FINAL_ARMS or f.stem.endswith("_v2"):
            rows[f.stem] = [json.loads(l) for l in open(f)]
    D["rows"] = rows
    for f in root.rglob("*"):
        if f.is_file() and f.name != "llm_cache.sqlite":
            dst = OUT / f.relative_to(root); dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy(f, dst)
    log("v2b carried forward:", sorted(rows))


import atexit
atexit.register(lambda: [srv.stop() for srv in list(RUNNING)])

if __name__ == "__main__" and MODE == "judge":
    stage_carry()
    if v1.ds_client() is None:
        log("no DeepSeek key in this run (versions pushed from the CLI cannot read secrets): start it from the website"); sys.exit(1)
    ds_judge_all()
    stage_stats()
    write_report()
    sys.exit(0)

if __name__ == "__main__":
    if v0.ON_KAGGLE and not SMOKE and not glob.glob("/kaggle/input/**/paper/methods/LME_V2_fast.jsonl", recursive=True):
        log("v2's rows (dama-paper-v2 output) are not attached; stopping before using GPU time"); sys.exit(1)
    log("environment", v0.environment(), "/tmp free GB", disk_gb())
    try:
        log("nvidia-smi", subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    for fn in (stage_setup, stage_prior, stage_contexts):
        fn()
        if any(k in STATE["errors"] for k in ("v2b_setup", "v2_select_prior", "v2b_contexts")):
            save_b(); write_report(); log("v2b stopped before the readers:", STATE["errors"]); sys.exit(1)
    stage_select()
    if "reader" in D:
        stage_freeze()
        stage_final()
    try:
        stage_stats()
    except Exception as e:
        STATE["errors"]["v2b_stats"] = f"{type(e).__name__}: {str(e)[:150]}"
    save_b()
    write_report()
    log(f"v2b done; errors: {STATE['errors'] or 'none'}")

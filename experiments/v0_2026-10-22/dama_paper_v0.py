"""DAMA v0 paper experiments (LoCoMo), run as a private Kaggle notebook (T4 x2). Nothing is trained here.

Plan: experiments/v0_2026-10-22/README.md. Rules kept: no number is invented (every number below comes from this run and
is logged); settings are chosen on dev (LoCoMo conversations 0-3) and test (4-9) is run once; every method uses the same
reader, prompt and greedy decoding; every LLM call is cached in SQLite; seed 13. Reads/imports dama_train (loaders,
evidence repair, BM25, the trained Tier-1 runtime) from the attached code dataset; modifies nothing there.

Stages run in priority order; each is wrapped so a failure is recorded and the run continues, and a time guard keeps
the total under Kaggle's 12 h limit. RESULTS_FOR_PAPER.md is always written at the end.
"""
from __future__ import annotations
import glob, hashlib, json, math, os, platform, random, re, shutil, sqlite3, subprocess, sys, time, traceback, zipfile
from collections import Counter, defaultdict
from pathlib import Path

T_START = time.time()
SEED = 13
SMOKE = False  # True: a few questions per conversation, trimmed Mem0/latency stages, to test every stage cheaply first
TIME_LIMIT_H = 6.5                                    # stop starting new reader stages after this (sized to the weekly GPU quota left, not Kaggle's 12 h limit)
FINAL_LIMIT_H = 7.0                                   # second judge, audit, latency and report may still start until this
ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = Path("/kaggle/working/paper") if ON_KAGGLE else Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "methods").mkdir(exist_ok=True)
DEV, TEST, V1_CONVS = [0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [7, 8, 9]
READER_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGE2_ID = "microsoft/Phi-3.5-mini-instruct"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_ID, CE_ID = "BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L6-v2"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
K_CHOICES, B_CHOICES, ALPHAS, MAIN_ALPHA = (32, 64, 128), (500, 1000, 1500, 3000), (0.1, 0.2, 0.3), 0.2
MMR_LAMBDA, BORDER_MS, BORDER_BAND, BORDER_MAX = 0.7, 40.0, 0.1, 8
MEM0_TIMEBOX_S = 2 * 3600
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
NOT_FOUND = "NOT_FOUND: no stored memory matches this question."

READER_SYSTEM = """You answer questions about a long conversation between two people, using only the memory provided.
Rules:
- Use only the MEMORY section. Do not use outside knowledge.
- If the memory does not contain the answer, reply exactly: Not mentioned in the conversation.
- Give a short answer (at most 15 words). No explanation.
- For time questions, give absolute dates (for example "7 May 2023") using the session dates shown."""
PULL_SYSTEM = """You answer questions about a long conversation between two people. You start with no memory of it.
Rules:
- First call recall(...) with a short search query to look up the conversation. You may call it again with a different query.
- Answer only from what recall returned. Do not use outside knowledge.
- If recall's results do not contain the answer, reply exactly: Not mentioned in the conversation.
- Give a short answer (at most 15 words). No explanation.
- For time questions, give absolute dates (for example "7 May 2023") using the session dates shown."""
SIDECAR_SYSTEM = READER_SYSTEM + "\n- If the MEMORY section is not enough, you may call recall(...) to look up more memory."
JUDGE_TMPL = """You grade answers to questions about a conversation.
Question: {q}
Gold answer: {gold}
Model answer: {ans}
Mark CORRECT if the model answer contains the same key information as the gold answer, even if worded differently or shorter. For dates, the same date in another format is CORRECT; an off-by-one-day error caused by relative wording ("yesterday") is CORRECT only if the gold answer is itself relative. Otherwise mark WRONG.
Reply with JSON only: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""
DECLINE_TMPL = """Question: {q}
Model answer: {ans}
Does the model answer decline to answer, i.e. say that the information is not available or not mentioned?
Reply with JSON only: {{"declines": true}} or {{"declines": false}}"""
RECALL_TOOL = {"type": "function", "function": {"name": "recall", "description": "Look up the conversation memory. Returns the most relevant stored turns with their dates.",
                                                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}}}

random.seed(SEED)
LOG = (OUT / "run.log").open("a")


def log(*a):
    msg = f"[{(time.time() - T_START) / 60:7.1f} min] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


STATE = {"stages": {}, "errors": {}, "deviations": [], "notes": []}


def save_state():
    (OUT / "state.json").write_text(json.dumps(STATE, indent=1, default=str))


def stage(name, limit_h=None):
    def deco(fn):
        def run(*a, **kw):
            lim = limit_h or TIME_LIMIT_H
            if (time.time() - T_START) / 3600 > lim:
                STATE["errors"][name] = f"not run: time guard ({lim} h) reached"
                log(f"SKIP {name}: time guard"); save_state(); return None
            t0 = time.time(); log(f"=== stage {name}")
            try:
                r = fn(*a, **kw)
                STATE["stages"][name] = round((time.time() - t0) / 60, 1); save_state(); return r
            except Exception as e:
                STATE["errors"][name] = f"{type(e).__name__}: {str(e)[:120]}"
                log(f"FAILED {name}: {traceback.format_exc()[-3000:]}"); save_state(); return None
        return run
    return deco


# ----------------------------------------------------------------------------------------------- setup + environment
def find_code():
    hits = glob.glob("/kaggle/input/**/dama_train/__init__.py", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(min(hits, key=len)))
    local = Path(__file__).resolve().parents[2] / "code"
    return str(local)


sys.path.insert(0, find_code())
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "onnxruntime", "mem0ai", "ollama"], check=False)
os.environ["MEM0_TELEMETRY"] = "False"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import tiktoken
try:
    ENC = tiktoken.get_encoding("o200k_base"); ENC_NAME = "o200k_base"
except Exception:
    ENC = tiktoken.get_encoding("cl100k_base"); ENC_NAME = "cl100k_base"


def ntok(s):
    return len(ENC.encode(s))


def environment():
    import importlib.metadata as md
    pk = {}
    for p in ("torch", "transformers", "sentence-transformers", "scikit-learn", "scipy", "statsmodels", "numpy", "tiktoken", "onnxruntime", "mem0ai", "ollama", "accelerate"):
        try:
            pk[p] = md.version(p)
        except Exception:
            pk[p] = "not installed"
    mem = next((int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemTotal")), None) if os.path.exists("/proc/meminfo") else None
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
    return {"os": platform.platform(), "cpu": cpu, "cores": os.cpu_count(), "ram_mb": mem, "python": sys.version.split()[0],
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "packages": pk,
            "reader": READER_ID, "judge": READER_ID, "judge2": JUDGE2_ID, "embedder": EMBED_ID, "cross_encoder": CE_ID,
            "tokenizer_for_counts": ENC_NAME, "seed": SEED, "torch_threads": torch.get_num_threads()}


# ----------------------------------------------------------------------------------------------- LLM cache + reader
class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, model TEXT, out TEXT, prompt_tokens INT, ms REAL)")
        for prev in glob.glob("/kaggle/input/**/llm_cache.sqlite", recursive=True):   # an earlier run's cache, if attached
            try:
                self.db.execute(f"ATTACH DATABASE '{prev}' AS prev"); self.db.execute("INSERT OR IGNORE INTO llm SELECT * FROM prev.llm")
                self.db.commit(); self.db.execute("DETACH DATABASE prev"); log("imported LLM cache from", prev)
            except Exception as e:
                log("could not import cache", prev, e)

    @staticmethod
    def key(model, payload):
        return hashlib.sha256(json.dumps([model, payload], sort_keys=True).encode()).hexdigest()

    def get(self, k):
        r = self.db.execute("SELECT out, prompt_tokens, ms FROM llm WHERE k=?", (k,)).fetchone()
        return r

    def put(self, k, model, out, ptok, ms):
        self.db.execute("INSERT OR REPLACE INTO llm VALUES (?,?,?,?,?)", (k, model, out, ptok, ms)); self.db.commit()


CACHE = Cache(OUT / "llm_cache.sqlite")


class LLM:
    """Greedy decoding with a HF chat model on the GPUs; batched for short prompts; prompts over LONG tokens go one by one
    with chunked prefill, and prompts sharing a long prefix (full-context B1: one conversation, many questions) reuse
    that prefix's KV cache. Every call is cached by (model, messages, tools, max_new_tokens)."""
    LONG = 2000   # version 6: batches of ~3k-token prompts ran out of memory on the second T4 (and were slower than this path)

    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto").eval()
        self.max_ctx = getattr(self.model.config, "max_position_embeddings", 32768)
        eos = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                eos.add(i)
        self.stop = {e for e in eos if e is not None}
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def text(self, messages, tools=None):
        return self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

    def __call__(self, convs, max_new=40, tools=None, batch_tokens=16000, max_batch=24):
        """convs: list of message lists. Returns (outputs, prompt_tokens, ms_per_item)."""
        keys = [Cache.key(self.name, [c, tools, max_new]) for c in convs]
        res = [None] * len(convs)
        todo = []
        for i, k in enumerate(keys):
            hit = CACHE.get(k)
            if hit:
                res[i] = hit
            else:
                todo.append(i)
        if todo:
            texts = {i: self.text(convs[i], tools) for i in todo}
            lens = {i: len(self.tok(texts[i])["input_ids"]) for i in todo}
            long = [i for i in todo if lens[i] > self.LONG]
            short = sorted([i for i in todo if lens[i] <= self.LONG], key=lambda i: lens[i])
            b = []
            for i in short + [None]:
                L = max(lens[j] for j in b + [i]) if i is not None else 0
                if i is not None and b and (len(b) >= max_batch or (len(b) + 1) * L > batch_tokens or (len(b) + 1) * L * L > 2.5e7):
                    self._batch(b, texts, lens, keys, res, max_new); b = []
                if i is not None:
                    b.append(i)
            if b:
                self._batch(b, texts, lens, keys, res, max_new)
            self._long_group(long, texts, lens, keys, res, max_new)
        return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]

    def _batch(self, idx, texts, lens, keys, res, max_new):
        t0 = time.time()
        enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.model.device)
        try:
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                        eos_token_id=list(self.stop))
        except torch.OutOfMemoryError:   # halve the batch; a single prompt falls back to chunked prefill
            del enc; torch.cuda.empty_cache()
            log(f"OOM on a batch of {len(idx)} (max {max(lens[i] for i in idx)} tokens); splitting")
            if len(idx) == 1:
                return self._long_group(idx, texts, lens, keys, res, max_new)
            h = len(idx) // 2
            self._batch(idx[:h], texts, lens, keys, res, max_new)
            return self._batch(idx[h:], texts, lens, keys, res, max_new)
        outs = self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        ms = (time.time() - t0) * 1000 / len(idx)
        for i, o in zip(idx, outs):
            o = self._clean(o)
            CACHE.put(keys[i], self.name, o, lens[i], ms); res[i] = (o, lens[i], ms)

    def _clean(self, o):
        for t in ("<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"):
            o = o.split(t)[0]
        return o.replace(self.tok.pad_token or "\x00", "").strip()

    def _long_group(self, idx, texts, lens, keys, res, max_new):
        """Prompts over 6k tokens: group by common token prefix, prefill the prefix once in 512-token chunks, then for
        each prompt feed its suffix, decode greedily, and crop the cache back to the prefix."""
        from transformers import DynamicCache
        ids = {i: self.tok(texts[i], return_tensors="pt")["input_ids"][0] for i in idx}
        groups = defaultdict(list)
        for i in idx:
            groups[texts[i][:4000]].append(i)
        for members in groups.values():
            base = ids[members[0]]
            plen = len(base)
            for i in members[1:]:
                n = min(plen, len(ids[i]))
                same = (base[:n] == ids[i][:n]).int()
                plen = int(same.cumprod(0).sum())
            plen = max(1, plen - 4)
            cache = DynamicCache()
            with torch.no_grad():
                for s in range(0, plen, 512):
                    out = self.model(input_ids=base[s:min(s + 512, plen)][None].to(self.model.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                for i in members:
                    t0 = time.time()
                    suffix = ids[i][plen:]
                    out = self.model(input_ids=suffix[None].to(self.model.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache, new = out.past_key_values, []
                    for _ in range(max_new):
                        nxt = int(out.logits[0, -1].argmax())
                        if nxt in self.stop:
                            break
                        new.append(nxt)
                        out = self.model(input_ids=torch.tensor([[nxt]], device=self.model.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
                        cache = out.past_key_values
                    cache.crop(plen)
                    o = self._clean(self.tok.decode(new, skip_special_tokens=False))
                    ms = (time.time() - t0) * 1000
                    CACHE.put(keys[i], self.name, o, lens[i], ms); res[i] = (o, lens[i], ms)
            del cache
            torch.cuda.empty_cache()

    def unload(self):
        import gc
        self.model = None
        gc.collect(); torch.cuda.empty_cache()
        log("unloaded", self.name, "GPU memory in use (GiB):", [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])


# ----------------------------------------------------------------------------------------------- data
MONTHS = "January February March April May June July August September October November December".split()


def fmt_date(raw):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", raw or "")
    return f"{int(m.group(1))} {m.group(2)} {m.group(3)}" if m else (raw or "")


def load_data():
    from dama_train.data.locomo import parse_evidence
    path = OUT / "locomo10.json"
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json", path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = json.loads(path.read_text())
    convs, questions, repair = [], [], Counter()
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
        convs.append({"ci": ci, "sid": s["sample_id"], "turns": turns, "speakers": (conv.get("speaker_a"), conv.get("speaker_b"))})
        for qi, qa in enumerate(s["qa"]):
            ev_raw = qa.get("evidence") or []
            strict = all(re.fullmatch(r"D\d+:\d+", str(e)) for e in ev_raw)
            parsed = parse_evidence(ev_raw)
            kept = [e for e in parsed if e in known]
            if not strict:
                repair["items_repaired"] += 1
            repair["ids_dropped"] += len(parsed) - len(kept)
            if ev_raw and not kept:
                repair["questions_left_without_evidence"] += 1
            cat = int(qa.get("category", 0))
            questions.append({"qid": f"{s['sample_id']}:q{qi}", "ci": ci, "split": "dev" if ci in DEV else "test", "cat": cat,
                              "question": qa["question"].strip(), "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                              "evidence": kept})
    return {"convs": convs, "questions": questions, "sha256": sha, "repair": dict(repair)}


def render(t):
    return f"[m:{t['dia']}] ({t['date']}) {t['speaker']}: {t['text']}"


def render_many(turns):
    return "\n".join(render(t) for t in sorted(turns, key=lambda t: t["i"]))


# ----------------------------------------------------------------------------------------------- Tier 0 candidates
ENT_STOP = set("What When Where Who Whom Which Why How Did Does Do Is Was Were Are Has Have Had Would Could Should Can Will "
               "The A An In On At Of For To And Or But If Then This That These Those It Its I My Me We Our You Your He She "
               "His Her They Their Them".split())
_TOK = re.compile(r"\b[A-Z][a-zA-Z'-]+\b|\b\d+\b")


def entities(text):
    return {w for w in _TOK.findall(text) if w not in ENT_STOP}


class ConvIndex:
    def __init__(self, conv, embedder):
        from dama_train.data.bm25 import BM25
        self.conv = conv
        self.turns = conv["turns"]
        self.bm25 = BM25([t["text"] for t in self.turns])
        self.emb = embedder.encode([t["text"] for t in self.turns], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        self.ents = [entities(t["text"]) for t in self.turns]
        self.loglen = np.log([max(1, len(t["text"])) for t in self.turns])
        self.ntok = [ntok(render(t)) + 1 for t in self.turns]


def rrf(lists, k=60):
    sc = defaultdict(float)
    for l in lists:
        for r, i in enumerate(l):
            sc[i] += 1.0 / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i]), sc


def candidates(ix, q, qvec, K, mode="hybrid", depth=200):
    """Returns pool (turn indices, best first), per-candidate features and the three single rankings."""
    bm = np.array(ix.bm25.scores(q))
    dense = ix.emb @ qvec
    qe = entities(q)
    ent = np.array([len(qe & e) for e in ix.ents])
    lb = [int(i) for i in np.argsort(-bm)[:depth]]
    ld = [int(i) for i in np.argsort(-dense)[:depth]]
    le = [int(i) for i in np.lexsort((-dense, -ent))[:depth] if ent[i] > 0]   # ties in overlap broken by dense similarity
    lists = {"hybrid": [lb, ld, le], "bm25": [lb], "dense": [ld]}[mode]
    order, sc = rrf(lists)
    pool = order[:K]
    bsel = bm[pool] if pool else np.zeros(0)
    lo, hi = (bsel.min(), bsel.max()) if len(bsel) else (0, 1)
    feats = np.array([[dense[i], (bm[i] - lo) / (hi - lo) if hi > lo else 0.0, sc[i], ent[i], ix.loglen[i]] for i in pool]) if pool else np.zeros((0, 5))
    return pool, feats, {"bm25": lb, "dense": ld, "hybrid": rrf([lb, ld, le])[0]}


# ----------------------------------------------------------------------------------------------- statistics
def f1_score(pred, gold):
    import string
    def norm(s):
        s = str(s).lower(); s = "".join(c for c in s if c not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s); return s.split()
    p, g = norm(pred), norm(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return float(p == g)
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def boot_ci(x, n=10000, seed=SEED):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return (None, None)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def mcnemar_p(a, b):
    from statsmodels.stats.contingency_tables import mcnemar
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    t = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    return float(mcnemar(t, exact=True).pvalue)


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


# ----------------------------------------------------------------------------------------------- DAMA v0 read path
class Dama:
    """Tier 0 pool -> relevance probability (LR head, or CE+Platt, or trained v1) -> optional borderline CE rescoring ->
    split-conformal admission -> MMR packing under B tokens -> chronological rendering (or NOT_FOUND)."""
    def __init__(self, embedder, head, ce, platt, K):
        self.embedder, self.head, self.ce, self.platt, self.K = embedder, head, ce, platt, K

    def qvec(self, q):
        return self.embedder.encode([BGE_PREFIX + q], normalize_embeddings=True, convert_to_numpy=True)[0]

    def ce_p(self, ix, q, idx, model=None):
        z = (model or self.ce).predict([(q, f"{ix.turns[i]['speaker']}: {ix.turns[i]['text']}") for i in idx], batch_size=64, show_progress_bar=False)
        return self.platt.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]

    def packet(self, ix, q, lam, B, mode="hybrid", topk=None, mmr=True, ce_all=False, prob_fn=None, rescore=True):
        t0 = time.perf_counter()
        pool, feats, _ = candidates(ix, q, self.qvec(q), self.K, mode)
        if not pool:
            return {"context": NOT_FOUND, "dia": [], "tokens": ntok(NOT_FOUND), "budget_limited": False, "admitted": 0,
                    "read_ms": (time.perf_counter() - t0) * 1000, "rescored": 0}
        if prob_fn is not None:
            p = np.asarray(prob_fn(ix, q, pool), dtype=float)
        elif ce_all:
            p = self.ce_p(ix, q, pool)
        else:
            p = self.head.predict_proba(feats)[:, 1]
        rescored = 0
        if rescore and not ce_all and topk is None and np.isfinite(lam) and (time.perf_counter() - t0) * 1000 < BORDER_MS:
            band = [j for j in np.argsort(np.abs(p - lam)) if abs(p[j] - lam) <= BORDER_BAND][:BORDER_MAX]
            if band:
                p = p.copy(); p[band] = self.ce_p(ix, q, [pool[j] for j in band]); rescored = len(band)
        if topk is not None:
            adm = [int(j) for j in np.argsort(-p)[:topk]]
        else:
            adm = [j for j in range(len(pool)) if p[j] >= lam]
        adm_tokens = sum(ix.ntok[pool[j]] for j in adm)
        order = sorted(adm, key=lambda j: -p[j])
        chosen = mmr_pack(p, ix.emb[pool], adm, [ix.ntok[pool[j]] for j in order], B, mmr)
        turns = [ix.turns[pool[j]] for j in chosen]
        ctx = render_many(turns) if turns else NOT_FOUND
        return {"context": ctx, "dia": [t["dia"] for t in turns], "tokens": ntok(ctx), "budget_limited": adm_tokens > B,
                "admitted": len(adm), "read_ms": (time.perf_counter() - t0) * 1000, "rescored": rescored,
                "p_admitted": [float(p[j]) for j in chosen]}


def mmr_pack(p, V, adm, ntok, B, mmr=True):
    """Greedy packing in admission order: each step takes the candidate with the best MMR score
    (MMR_LAMBDA * p - (1 - MMR_LAMBDA) * max similarity to those already chosen; plain p order before the first pick or
    without MMR), keeps it if it still fits the budget B, and drops it otherwise. Returns indices into p/V."""
    cand = sorted(adm, key=lambda j: -p[j])
    if not cand:
        return []
    pc = np.asarray([p[j] for j in cand], dtype=float)
    S = V[cand] @ V[cand].T if mmr else None
    best = np.full(len(cand), -np.inf)
    alive = np.ones(len(cand), dtype=bool)
    chosen, used = [], 0
    while alive.any():
        score = MMR_LAMBDA * pc - (1 - MMR_LAMBDA) * best if (mmr and chosen) else -np.arange(len(cand), dtype=float)
        score = np.where(alive, score, -np.inf)
        k = int(np.argmax(score)); alive[k] = False
        if used + ntok[k] <= B:
            chosen.append(cand[k]); used += ntok[k]
            if mmr:
                best = np.maximum(best, S[k])
    return chosen


def fit_lambda(scores, alpha):
    s = sorted(scores)
    n = len(s); k = math.floor((n + 1) * alpha)
    return -math.inf if k < 1 or not n else s[k - 1]


def conformal_scores(qs, pools, probs):
    """s_i = min p over the evidence turns if all are in the pool, else -inf (dev categories 1-4 with evidence)."""
    out = []
    for q in qs:
        pool, p = pools[q["qid"]][0], probs[q["qid"]]
        pos = {q_ev for q_ev in q["evidence"]}
        idx = {t: j for j, t in enumerate(pool)}
        dias = [D["turn_dia"][q["ci"]][t] for t in pool]
        hit = [j for j, d in enumerate(dias) if d in pos]
        out.append(float(min(p[j] for j in hit)) if len(hit) == len(pos) and pos else -math.inf)
    return out


# ----------------------------------------------------------------------------------------------- evaluation helpers
READER = None


def reader_messages(q, ctx, system=READER_SYSTEM):
    return [{"role": "system", "content": system}, {"role": "user", "content": f"MEMORY:\n{ctx}\n\nQUESTION: {q}\nANSWER:"}]


def judge_rows(rows, llm=None):
    llm = llm or READER
    main = [r for r in rows if r["cat"] != 5]
    outs, _, _ = llm([[{"role": "user", "content": JUDGE_TMPL.format(q=r["question"], gold=r["gold"], ans=r["answer"])}] for r in main], max_new=24)
    for r, o in zip(main, outs):
        m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', o) or re.search(r"\b(CORRECT|WRONG)\b", o)
        r["judge_raw"], r["correct"] = o, (m.group(1) == "CORRECT") if m else False
        r["judge_unparsed"] = m is None
        r["f1"] = f1_score(r["answer"], r["gold"])
    adv = [r for r in rows if r["cat"] == 5]
    need = [r for r in adv if "not mentioned" not in r["answer"].lower()]
    outs, _, _ = llm([[{"role": "user", "content": DECLINE_TMPL.format(q=r["question"], ans=r["answer"])}] for r in need], max_new=24)
    dec = {id(r): bool(re.search(r'"declines"\s*:\s*true', o)) for r, o in zip(need, outs)}
    for r in adv:
        r["correct"] = "not mentioned" in r["answer"].lower() or dec.get(id(r), False)
        r["f1"] = None
    return rows


def evaluate(name, qs, ctx_of, system=READER_SYSTEM, extra=None):
    """Read every question with its context, judge it, save per-question rows. ctx_of(q) -> dict(context, dia, read_ms...)."""
    t0 = time.time()
    ctxs = [ctx_of(q) for q in qs]
    msgs = [reader_messages(q["question"], c["context"], system) for q, c in zip(qs, ctxs)]
    outs, ptoks, ms = READER(msgs, max_new=40)
    rows = []
    for q, c, m, o, pt, t in zip(qs, ctxs, msgs, outs, ptoks, ms):
        ev = set(q["evidence"])
        rows.append({"method": name, "qid": q["qid"], "ci": q["ci"], "cat": q["cat"], "question": q["question"], "gold": q["gold"],
                     "answer": o, "context": c["context"], "input_tokens": ntok(m[0]["content"]) + ntok(m[1]["content"]),
                     "reader_prompt_tokens": pt, "reader_ms": t, "read_ms": c.get("read_ms"),
                     "evidence_all": (ev <= set(c.get("dia") or [])) if ev and c.get("dia") is not None else None,
                     "budget_limited": c.get("budget_limited"), "flags": c.get("flags")})
    judge_rows(rows)
    with (OUT / "methods" / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    RESULTS[name] = summarize(rows)
    RESULTS[name]["minutes"] = round((time.time() - t0) / 60, 1)
    if extra:
        RESULTS[name].update(extra)
    ROWS[name] = rows
    log(f"{name}: acc {RESULTS[name]['acc']:.3f} f1 {RESULTS[name]['f1']:.3f} abst {RESULTS[name]['abstention']} tokens {RESULTS[name]['mean_tokens']:.0f} ({RESULTS[name]['minutes']} min)")
    save_results()
    return rows


def summarize(rows):
    main = [r for r in rows if r["cat"] != 5]
    adv = [r for r in rows if r["cat"] == 5]
    acc = [float(r["correct"]) for r in main]
    ev = [r["evidence_all"] for r in main if r["evidence_all"] is not None]
    rd = [r["read_ms"] for r in rows if r["read_ms"] is not None]
    by = {CATS[c]: float(np.mean([r["correct"] for r in rows if r["cat"] == c])) for c in CATS if any(r["cat"] == c for r in rows)}
    return {"n": len(rows), "n_main": len(main), "acc": float(np.mean(acc)) if acc else None, "acc_ci": boot_ci(acc),
            "f1": float(np.mean([r["f1"] for r in main])) if main else None,
            "abstention": float(np.mean([r["correct"] for r in adv])) if adv else None,
            "mean_tokens": float(np.mean([r["input_tokens"] for r in rows])), "median_tokens": float(np.median([r["input_tokens"] for r in rows])),
            "reader_prompt_tokens_mean": float(np.mean([r["reader_prompt_tokens"] for r in rows])),
            "evidence_recall": float(np.mean(ev)) if ev else None,
            "read_p50": pct(rd, 50), "read_p95": pct(rd, 95), "read_mean": float(np.mean(rd)) if rd else None,
            "reader_p50_ms": pct([r["reader_ms"] for r in rows], 50), "by_category": by,
            "judge_unparsed": int(sum(r.get("judge_unparsed", False) for r in main)),
            "budget_limited_share": float(np.mean([bool(r["budget_limited"]) for r in rows if r["budget_limited"] is not None])) if any(r["budget_limited"] is not None for r in rows) else None,
            "cost_usd": 0.0}


RESULTS, ROWS = {}, {}
D = {}


def save_results():
    (OUT / "results.json").write_text(json.dumps({"results": RESULTS, "state": STATE, "settings": D.get("settings", {})}, indent=1, default=str))


# ----------------------------------------------------------------------------------------------- stages
@stage("data")
def stage_data():
    data = load_data()
    D.update(data)
    D["turn_dia"] = {c["ci"]: [t["dia"] for t in c["turns"]] for c in data["convs"]}
    qs = data["questions"]
    D["counts"] = {s: {CATS.get(c, c): sum(1 for q in qs if q["split"] == s and q["cat"] == c) for c in sorted({q["cat"] for q in qs})} for s in ("dev", "test")}
    # category mapping evidence: 5 examples per category + share of "When ..." questions and missing answers
    ex = {}
    for c in sorted({q["cat"] for q in qs}):
        cq = [q for q in qs if q["cat"] == c]
        ex[c] = {"examples": [q["question"] for q in random.Random(SEED).sample(cq, min(5, len(cq)))],
                 "share_when": round(float(np.mean([q["question"].lower().startswith("when") for q in cq])), 3),
                 "share_no_gold_answer": round(float(np.mean([not q["gold"] for q in cq])), 3), "n": len(cq)}
    D["category_check"] = ex
    for c, v in ex.items():
        log(f"category {c}: n={v['n']} when={v['share_when']} no-answer={v['share_no_gold_answer']} e.g. {v['examples'][:3]}")
    try:
        import urllib.request
        readme = urllib.request.urlopen("https://raw.githubusercontent.com/snap-research/locomo/main/README.md", timeout=30).read().decode()
        D["readme_category_lines"] = [l.strip() for l in readme.splitlines() if "categor" in l.lower()][:10]
    except Exception as e:
        D["readme_category_lines"] = [f"README not readable: {e}"]
    log("counts", D["counts"], "repair", D["repair"], "sha256", D["sha256"])
    if SMOKE:   # 8 questions from each of conversations 0, 1 (dev) and 4, 7 (test)
        keep = []
        for ci in (0, 1, 4, 7):
            keep += [q for q in qs if q["ci"] == ci][:8]
        D["questions"] = keep
        log("SMOKE: kept", len(keep), "questions")


@stage("index")
def stage_index():
    from sentence_transformers import SentenceTransformer
    D["embedder"] = SentenceTransformer(EMBED_ID, device="cpu")
    D["ix"] = {c["ci"]: ConvIndex(c, D["embedder"]) for c in D["convs"]}
    # all question vectors (CPU) and the rankings of every source, for the ceiling table
    qs = D["questions"]
    D["qvec"] = dict(zip([q["qid"] for q in qs], D["embedder"].encode([BGE_PREFIX + q["question"] for q in qs], batch_size=64,
                                                                          normalize_embeddings=True, convert_to_numpy=True)))
    ceil = {}
    for q in [q for q in qs if q["cat"] != 5 and q["evidence"]]:
        _, _, ranks = candidates(D["ix"][q["ci"]], q["question"], D["qvec"][q["qid"]], 128)
        dias = D["turn_dia"][q["ci"]]
        for src, r in ranks.items():
            for K in (16, 32, 64, 128):
                ceil.setdefault((q["split"], src, K), []).append(set(q["evidence"]) <= {dias[i] for i in r[:K]})
    D["ceiling"] = {f"{s}|{src}|{K}": float(np.mean(v)) for (s, src, K), v in ceil.items()}
    D["ceiling_all"] = {f"{src}|{K}": float(np.mean(ceil.get(("dev", src, K), []) + ceil.get(("test", src, K), []))) for src in ("bm25", "dense", "hybrid") for K in (16, 32, 64, 128)}
    dev_cov = {K: D["ceiling"][f"dev|hybrid|{K}"] for K in K_CHOICES}
    D["K"] = min(K for K in K_CHOICES if dev_cov[K] >= dev_cov[128] - 0.02)
    D.setdefault("settings", {})["K"] = {"value": D["K"], "dev_all_evidence_in_pool": dev_cov, "rule": "smallest K within 2 points of K=128 on dev"}
    log("ceiling (all 10 convs)", D["ceiling_all"], "K chosen", D["K"], dev_cov)


def pools_for(qs, mode="hybrid"):
    return {q["qid"]: candidates(D["ix"][q["ci"]], q["question"], D["qvec"][q["qid"]], D["K"], mode)[:2] for q in qs}


@stage("head")
def stage_head():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sentence_transformers import CrossEncoder
    qs = D["questions"]
    D["pools"] = pools_for(qs)
    def pairs(sel):
        X, y = [], []
        for q in sel:
            pool, feats = D["pools"][q["qid"]]
            dias = D["turn_dia"][q["ci"]]
            for j, t in enumerate(pool):
                X.append(feats[j]); y.append(int(dias[t] in set(q["evidence"])))
        return np.array(X), np.array(y)
    dev_m = [q for q in qs if q["split"] == "dev" and q["cat"] != 5 and q["evidence"]]
    test_m = [q for q in qs if q["split"] == "test" and q["cat"] != 5 and q["evidence"]]
    X, y = pairs(dev_m)
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(X, y)
    Xt, yt = pairs(test_m)
    D["head"] = head
    lr = head[-1]; sc = head[0]
    D["settings"]["lr_head"] = {"features": ["dense_cos", "bm25_minmax_in_pool", "rrf", "entity_overlap", "log_turn_chars"],
                                "coef_on_standardized": [round(float(c), 4) for c in lr.coef_[0]], "intercept": round(float(lr.intercept_[0]), 4),
                                "scaler_mean": [round(float(m), 4) for m in sc.mean_], "scaler_scale": [round(float(s), 4) for s in sc.scale_],
                                "dev_pairs": int(len(y)), "dev_positives": int(y.sum()),
                                "ece_test_pairs": ece(head.predict_proba(Xt)[:, 1], yt), "test_pairs": int(len(yt))}
    # cross-encoder + Platt (dev pool pairs), on CPU
    ce = CrossEncoder(CE_ID, device="cpu")                                   # read path (timed)
    D["ce_gpu"] = CrossEncoder(CE_ID, device="cuda") if torch.cuda.is_available() else ce   # offline dev scoring only
    zs, ys = [], []
    for q in dev_m:
        pool, _ = D["pools"][q["qid"]]
        ix = D["ix"][q["ci"]]
        zs += list(D["ce_gpu"].predict([(q["question"], f"{ix.turns[t]['speaker']}: {ix.turns[t]['text']}") for t in pool], batch_size=128, show_progress_bar=False))
        ys += [int(D["turn_dia"][q["ci"]][t] in set(q["evidence"])) for t in pool]
    platt = LogisticRegression(random_state=SEED).fit(np.array(zs).reshape(-1, 1), np.array(ys))
    D["settings"]["platt_ce"] = {"a": round(float(platt.coef_[0][0]), 4), "b": round(float(platt.intercept_[0]), 4), "dev_pairs": len(ys)}
    D["dama"] = Dama(D["embedder"], head, ce, platt, D["K"])
    # conformal lambdas (LR head, and CE-on-whole-pool for ablation g)
    probs = {q["qid"]: head.predict_proba(D["pools"][q["qid"]][1])[:, 1] for q in qs}
    D["probs_lr"] = probs
    s = conformal_scores(dev_m, D["pools"], probs)
    D["lam"] = {a: fit_lambda(s, a) for a in ALPHAS}
    D["settings"]["lambda"] = {str(a): D["lam"][a] for a in ALPHAS}
    D["settings"]["lambda_n_dev"] = len(s); D["settings"]["lambda_dev_minus_inf_share"] = float(np.mean([x == -math.inf for x in s]))
    pce = {}
    for q in dev_m:
        pce[q["qid"]] = D["dama"].ce_p(D["ix"][q["ci"]], q["question"], D["pools"][q["qid"]][0], model=D["ce_gpu"])
    D["lam_ce"] = {a: fit_lambda(conformal_scores(dev_m, D["pools"], pce), a) for a in ALPHAS}
    D["settings"]["lambda_ce_all"] = {str(a): D["lam_ce"][a] for a in ALPHAS}
    # test-set omission at each alpha (conformal admission only, before budget)
    st = conformal_scores(test_m, D["pools"], probs)
    D["test_omission"] = {str(a): float(np.mean([x < D["lam"][a] or x == -math.inf for x in st])) for a in ALPHAS}
    log("head", D["settings"]["lr_head"], "platt", D["settings"]["platt_ce"], "lambda", D["settings"]["lambda"], "test omission", D["test_omission"])


@stage("loco")
def stage_loco():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    qs = [q for q in D["questions"] if q["cat"] != 5 and q["evidence"]]
    out = {a: [] for a in ALPHAS}; inf_folds = {a: 0 for a in ALPHAS}
    for c in range(10):
        tr = [q for q in qs if q["ci"] != c]; te = [q for q in qs if q["ci"] == c]
        if not te:
            continue
        X, y = [], []
        for q in tr:
            pool, feats = D["pools"][q["qid"]]
            X += list(feats); y += [int(D["turn_dia"][q["ci"]][t] in set(q["evidence"])) for t in pool]
        h = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(np.array(X), np.array(y))
        ptr = {q["qid"]: h.predict_proba(D["pools"][q["qid"]][1])[:, 1] for q in tr + te}
        s_tr, s_te = conformal_scores(tr, D["pools"], ptr), conformal_scores(te, D["pools"], ptr)
        for a in ALPHAS:
            lam = fit_lambda(s_tr, a)
            inf_folds[a] += lam == -math.inf
            out[a].append(float(np.mean([x == -math.inf or x < lam for x in s_te])))
    D["loco"] = {str(a): {"target": a, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(min(v)), "max": float(max(v)),
                          "folds_lambda_minus_inf": int(inf_folds[a]), "per_fold": [round(x, 3) for x in v]} for a, v in out.items()}
    log("LOCO", D["loco"])


@stage("mem0")
def stage_mem0():
    """B4: the mem0ai library with the same model family through Ollama (qwen2.5:7b), local qdrant, bge-small embedder on
    CPU. One memory space per test conversation; sessions added in order with their date as metadata; search top 10."""
    t0 = time.time()
    prev = [p for p in glob.glob("/kaggle/input/**/paper/mem0_contexts.json", recursive=True)]
    if prev and not SMOKE:
        prev_dir = Path(prev[0]).parent
        ctx = json.loads(Path(prev[0]).read_text())
        info_f = prev_dir / "mem0_info.json"
        if info_f.exists():
            info = json.loads(info_f.read_text())
        else:   # version 5 logged it only: "mem0 info {...}"
            import ast
            line = [l for l in (prev_dir / "run.log").read_text().splitlines() if "mem0 info {" in l][-1]
            info = ast.literal_eval(line[line.index("{"):])
        test_ids = {q["qid"] for q in D["questions"] if q["split"] == "test"}
        if test_ids <= set(ctx):
            info["reused_from"] = f"{prev_dir} (earlier version of this notebook: same code, data and settings)"
            D["mem0"] = {"contexts": ctx, "info": info}
            (OUT / "mem0_contexts.json").write_text(json.dumps(ctx)); (OUT / "mem0_info.json").write_text(json.dumps(info))
            log("mem0: reused", info)
            return
    info = {"library": None, "ingest_llm_calls": 0, "ingest_prompt_tokens": 0, "ingest_output_tokens": 0}
    import importlib.metadata as md
    info["library"] = md.version("mem0ai")
    if subprocess.run("command -v ollama", shell=True).returncode != 0:
        subprocess.run("apt-get -qq update && apt-get -qq install -y zstd pciutils", shell=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, timeout=900, capture_output=True, text=True)
        (OUT / "ollama_install.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"ollama install failed (exit {r.returncode}): {(r.stdout + r.stderr)[-800:]}")
    server = subprocess.Popen(["ollama", "serve"], stdout=(OUT / "ollama_serve.log").open("w"), stderr=subprocess.STDOUT)
    try:
        import urllib.request
        for _ in range(60):
            try:
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2); break
            except Exception:
                time.sleep(2)
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import ollama
        orig = ollama.Client.chat
        def counted(self, *a, **kw):
            r = orig(self, *a, **kw)
            info["ingest_llm_calls"] += 1
            info["ingest_prompt_tokens"] += int(getattr(r, "prompt_eval_count", 0) or (r.get("prompt_eval_count", 0) if isinstance(r, dict) else 0) or 0)
            info["ingest_output_tokens"] += int(getattr(r, "eval_count", 0) or (r.get("eval_count", 0) if isinstance(r, dict) else 0) or 0)
            return r
        ollama.Client.chat = counted
        from mem0 import Memory
        contexts, search_ms, done = {}, [], []
        for ci in sorted({q["ci"] for q in D["questions"] if q["split"] == "test"}):
            conv = D["convs"][ci]; sid = conv["sid"]
            cfg = {"llm": {"provider": "ollama", "config": {"model": OLLAMA_MODEL, "temperature": 0, "max_tokens": 2000, "ollama_base_url": "http://localhost:11434"}},
                   "embedder": {"provider": "huggingface", "config": {"model": EMBED_ID, "embedding_dims": 384, "model_kwargs": {"device": "cpu"}}},
                   "vector_store": {"provider": "qdrant", "config": {"collection_name": f"locomo_{sid.replace('-', '_')}", "path": f"/tmp/qdrant_{sid}",
                                                                     "embedding_model_dims": 384, "on_disk": True}},
                   "history_db_path": f"/tmp/mem0_history_{sid}.db"}
            m = Memory.from_config(cfg)
            sessions = defaultdict(list)
            for t in conv["turns"]:
                sessions[t["session"]].append(t)
            for n in sorted(sessions)[:2] if SMOKE else sorted(sessions):
                if time.time() - t0 > MEM0_TIMEBOX_S:
                    raise TimeoutError(f"time-box of {MEM0_TIMEBOX_S / 3600:.0f} h reached during {sid} session {n}; finished: {done}")
                msgs = [{"role": "user", "content": f"{t['speaker']}: {t['text']}"} for t in sessions[n]]
                m.add(msgs, user_id=sid, metadata={"session_date": sessions[n][0]["date"], "session": n})
            for q in [q for q in D["questions"] if q["ci"] == ci]:
                ts = time.perf_counter()
                try:
                    r = m.search(q["question"], top_k=10, filters={"user_id": sid})
                except TypeError:
                    r = m.search(q["question"], user_id=sid, limit=10)
                ms = (time.perf_counter() - ts) * 1000
                items = r.get("results", r) if isinstance(r, dict) else r
                items = sorted(items, key=lambda x: (x.get("metadata") or {}).get("session", 0))
                lines = [f"[m:{k + 1}] ({(x.get('metadata') or {}).get('session_date', '')}) {x.get('memory', '')}" for k, x in enumerate(items)]
                contexts[q["qid"]] = {"context": "\n".join(lines) if lines else NOT_FOUND, "dia": None, "read_ms": ms}
                search_ms.append(ms)
            done.append(sid)
            log(f"mem0: {sid} ingested and searched ({(time.time() - t0) / 60:.1f} min, {info['ingest_llm_calls']} LLM calls)")
        info.update({"minutes": round((time.time() - t0) / 60, 1), "search_p50_ms": pct(search_ms, 50), "search_p95_ms": pct(search_ms, 95)})
        D["mem0"] = {"contexts": contexts, "info": info}
        (OUT / "mem0_contexts.json").write_text(json.dumps(contexts)); (OUT / "mem0_info.json").write_text(json.dumps(info))
        log("mem0 info", info)
    finally:
        D.setdefault("mem0_info", info)
        server.terminate()
        subprocess.run("pkill -f 'ollama' || true", shell=True)
        torch.cuda.empty_cache()


@stage("load_reader")
def stage_load_reader():
    global READER
    READER = LLM(READER_ID)
    log("reader loaded", READER.max_ctx, READER.model.hf_device_map if hasattr(READER.model, "hf_device_map") else "")


def test_subset(n=500):
    """Fixed stratified random subset of the test split (seed 13), for the topology runs and ablations."""
    test = [q for q in D["questions"] if q["split"] == "test"]
    if len(test) <= n:
        return test
    rng, out = random.Random(SEED), []
    for c in sorted({q["cat"] for q in test}):
        cq = [q for q in test if q["cat"] == c]
        out += rng.sample(cq, round(n * len(cq) / len(test)))
    return sorted(out, key=lambda q: q["qid"])


def dama_ctx(q, B, lam=None, **kw):
    ix = D["ix"][q["ci"]]
    return D["dama"].packet(ix, q["question"], D["lam"][MAIN_ALPHA] if lam is None else lam, B, **kw)


@stage("choose_B")
def stage_choose_b():
    dev = [q for q in D["questions"] if q["split"] == "dev" and q["cat"] != 5]
    sample = random.Random(SEED).sample(dev, min(100, len(dev)))
    acc = {}
    for B in B_CHOICES:
        rows = evaluate(f"dev_B{B}", sample, lambda q: dama_ctx(q, B))
        acc[B] = RESULTS[f"dev_B{B}"]["acc"]
    best = max(acc.values())
    D["B"] = min(B for B in B_CHOICES if acc[B] == best)
    D["settings"]["B"] = {"value": D["B"], "dev_sample_acc": acc, "rule": "highest dev accuracy on a 100-question dev sample (cat 1-4, seed 13); ties -> smaller B"}
    # B3's k: mean packet tokens of top-k cross-encoder packets closest to DAMA's mean packet tokens on all dev questions
    devall = [q for q in D["questions"] if q["split"] == "dev"]
    dt = [dama_ctx(q, D["B"])["tokens"] for q in devall]
    ce_rank = {}
    for q in devall:
        pool = D["pools"][q["qid"]][0]
        p = D["dama"].ce_p(D["ix"][q["ci"]], q["question"], pool, model=D["ce_gpu"])
        ce_rank[q["qid"]] = [pool[j] for j in np.argsort(-p)]
    D["ce_rank_dev"] = ce_rank
    target = float(np.mean(dt))
    tok_k = {k: float(np.mean([ntok(render_many([D["ix"][q["ci"]].turns[t] for t in ce_rank[q["qid"]][:k]])) for q in devall])) for k in range(1, D["K"] + 1)}
    D["k3"] = min(tok_k, key=lambda k: abs(tok_k[k] - target))
    D["settings"]["B3_k"] = {"value": D["k3"], "dama_dev_mean_packet_tokens": target, "b3_dev_mean_packet_tokens": tok_k[D["k3"]],
                             "rule": "k whose mean dev packet tokens is closest to DAMA's"}
    log("B", D["settings"]["B"], "B3 k", D["settings"]["B3_k"])
    save_results()


def b3_ctx(q):
    ix = D["ix"][q["ci"]]; dama = D["dama"]
    t0 = time.perf_counter()
    pool, _, _ = candidates(ix, q["question"], dama.qvec(q["question"]), D["K"])
    p = dama.ce_p(ix, q["question"], pool)
    turns = [ix.turns[pool[j]] for j in np.argsort(-p)[:D["k3"]]]
    ctx = render_many(turns) if turns else NOT_FOUND
    return {"context": ctx, "dia": [t["dia"] for t in turns], "read_ms": (time.perf_counter() - t0) * 1000}


def b2_ctx(q):
    turns, used = [], 0
    for t in reversed(D["ix"][q["ci"]].turns):
        n = D["ix"][q["ci"]].ntok[t["i"]]
        if used + n > D["B"]:
            break
        turns.append(t); used += n
    return {"context": render_many(turns), "dia": [t["dia"] for t in turns], "read_ms": None}


B1_CTX = {}


def b1_ctx(q):
    """Whole conversation; if it does not fit the reader's window, the newest turns that fit (flagged)."""
    if q["ci"] not in B1_CTX:
        all_turns = D["convs"][q["ci"]]["turns"]
        turns = list(all_turns)
        ctx = render_many(turns); flags = None
        limit = READER.max_ctx - 400
        while len(READER.tok(ctx)["input_ids"]) > limit:
            turns = turns[max(1, len(turns) // 20):]; ctx = render_many(turns); flags = f"conv {q['ci']}: truncated to the newest {len(turns)} of {len(all_turns)} turns"
        B1_CTX[q["ci"]] = {"context": ctx, "dia": [t["dia"] for t in turns], "read_ms": None, "flags": flags}
    return B1_CTX[q["ci"]]


def run_b1_child():
    """Child process: answer B1 conversation by conversation into the shared cache. Kept out of the main process
    because long-prompt decoding hit intermittent 'CUDA error: unspecified launch failure', which poisons the CUDA
    context of whatever process it happens in."""
    global READER
    stage_data()
    READER = LLM(READER_ID)
    for ci in TEST:
        qs = [q for q in D["questions"] if q["ci"] == ci]
        if not qs:
            continue
        READER([reader_messages(q["question"], b1_ctx(q)["context"]) for q in qs], max_new=40)
        log(f"B1 child: conversation {ci} answered ({len(qs)} questions); {b1_ctx(qs[0])['flags'] or 'not truncated'}")


@stage("main")
def stage_main():
    test = [q for q in D["questions"] if q["split"] == "test"]
    v0 = {q["qid"]: dama_ctx(q, D["B"]) for q in test}
    D["v0_packets"] = v0
    evaluate("DAMA_v0", test, lambda q: v0[q["qid"]], extra={"alpha": MAIN_ALPHA, "lambda": D["lam"][MAIN_ALPHA]})
    evaluate("B3", test, b3_ctx, extra={"k": D["k3"]})
    evaluate("B2", test, b2_ctx)
    if D.get("mem0"):
        evaluate("B4", test, lambda q: D["mem0"]["contexts"][q["qid"]], extra=D["mem0"]["info"])


@stage("b1")
def stage_b1():
    global READER
    READER.unload(); READER = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--b1-child"], capture_output=True, text=True, timeout=5 * 3600)
    (OUT / "b1_child.log").write_text(r.stdout[-200000:] + "\n----- stderr -----\n" + r.stderr[-50000:])
    READER = LLM(READER_ID)
    test = [q for q in D["questions"] if q["split"] == "test"]
    done = [q for q in test if CACHE.get(Cache.key(READER_ID, [reader_messages(q["question"], b1_ctx(q)["context"]), None, 40]))]
    D["b1_truncated"] = sorted({b1_ctx(q)["flags"] for q in test if b1_ctx(q)["flags"]})
    extra = {"truncated": D["b1_truncated"], "answered": len(done), "of": len(test), "child_exit": r.returncode}
    if len(done) < len(test):
        STATE["errors"]["b1"] = f"partial: {len(done)} of {len(test)} questions answered; child exit {r.returncode}: {r.stderr[-300:]}"
    if done:
        evaluate("B1", done, b1_ctx, extra=extra)


@stage("topology")
def stage_topology():
    test = test_subset()
    v0 = D["v0_packets"]
    for mode in ("pull", "sidecar"):
        t0 = time.time()
        sysm = PULL_SYSTEM if mode == "pull" else SIDECAR_SYSTEM
        st = {}
        for q in test:
            mem = v0[q["qid"]]["context"] if mode == "sidecar" else "(no memory given; use recall)"
            st[q["qid"]] = {"msgs": reader_messages(q["question"], mem, sysm), "calls": 0, "seen": set(v0[q["qid"]]["dia"]) if mode == "sidecar" else set(),
                            "pulled": set(), "ms": 0.0, "answer": None, "ptok": 0}
        for step in range(4):
            active = [q for q in test if st[q["qid"]]["answer"] is None]
            if not active:
                break
            tools = [RECALL_TOOL] if step < 3 else None
            outs, ptoks, ms = READER([st[q["qid"]]["msgs"] for q in active], max_new=96, tools=tools)
            for q, o, pt, t in zip(active, outs, ptoks, ms):
                s = st[q["qid"]]; s["ms"] += t; s["ptok"] += pt
                m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", o, re.S)
                call = None
                if m and tools:
                    try:
                        call = json.loads(m.group(1)); query = str((call.get("arguments") or {}).get("query") or q["question"])
                    except Exception:
                        call = None
                if call:
                    pk = dama_ctx({**q, "question": query}, D["B"])
                    s["calls"] += 1; s["seen"] |= set(pk["dia"]); s["pulled"] |= set(pk["dia"])
                    s["msgs"] = s["msgs"] + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "recall", "arguments": {"query": query}}}]},
                                             {"role": "tool", "name": "recall", "content": pk["context"]}]
                else:
                    s["answer"] = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", o, flags=re.S).strip()
        rows = []
        for q in test:
            s = st[q["qid"]]; ev = set(q["evidence"])
            rows.append({"method": f"topology_{mode}", "qid": q["qid"], "ci": q["ci"], "cat": q["cat"], "question": q["question"], "gold": q["gold"],
                         "answer": s["answer"] or "", "context": None, "input_tokens": None, "reader_prompt_tokens": s["ptok"], "reader_ms": s["ms"],
                         "read_ms": None, "evidence_all": (ev <= s["seen"]) if ev else None, "calls": s["calls"],
                         "pull_all": (ev <= s["pulled"]) if ev and s["calls"] else None, "budget_limited": None, "flags": None})
        judge_rows(rows)
        with (OUT / "methods" / f"topology_{mode}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        main = [r for r in rows if r["cat"] != 5]
        RESULTS[f"topology_{mode}"] = {"acc": float(np.mean([r["correct"] for r in main])), "acc_ci": boot_ci([float(r["correct"]) for r in main]),
                                       "coverage": float(np.mean([r["evidence_all"] for r in main if r["evidence_all"] is not None])),
                                       "mean_calls": float(np.mean([r["calls"] for r in rows])), "called_share": float(np.mean([r["calls"] > 0 for r in rows])),
                                       "pull_recall_when_called": float(np.mean([r["pull_all"] for r in main if r["pull_all"] is not None])) if any(r["pull_all"] is not None for r in main) else None,
                                       "reader_p50_s": pct([r["reader_ms"] for r in rows], 50) / 1000, "n": len(rows), "minutes": round((time.time() - t0) / 60, 1)}
        ROWS[f"topology_{mode}"] = rows
        log(f"topology {mode}", RESULTS[f"topology_{mode}"]); save_results()


@stage("ablations")
def stage_ablations():
    test = test_subset()
    B, lam = D["B"], D["lam"][MAIN_ALPHA]
    runs = {"abl_a_no_conformal_topk": lambda q: dama_ctx(q, B, topk=D["k3"]),       # in run order: answered in version 6 /
            "abl_b_no_mmr": lambda q: dama_ctx(q, B, mmr=False),                     # same packets as DAMA v0 / short first,
            "abl_e_alpha_0.1": lambda q: dama_ctx(q, B, lam=D["lam"][0.1]),          # so the time guard drops the costly ones
            "abl_f_B3000": lambda q: dama_ctx(q, 3000),
            "abl_f_B500": lambda q: dama_ctx(q, 500),
            "abl_c_bm25_only": lambda q: dama_ctx(q, B, mode="bm25"),
            "abl_d_dense_only": lambda q: dama_ctx(q, B, mode="dense"),
            "abl_g_ce_whole_pool": lambda q: dama_ctx(q, B, lam=D["lam_ce"][MAIN_ALPHA], ce_all=True),
            "abl_e_alpha_0.3": lambda q: dama_ctx(q, B, lam=D["lam"][0.3])}
    for name, fn in runs.items():
        if (time.time() - T_START) / 3600 > TIME_LIMIT_H:
            STATE["errors"][name] = "time guard"; continue
        try:
            evaluate(name, test, fn)
        except Exception as e:
            STATE["errors"][name] = f"{type(e).__name__}: {str(e)[:120]}"; log("ablation failed", name, traceback.format_exc()[-1500:])


@stage("v1")
def stage_v1():
    from dama_train.runtime import Tier1
    bundle = os.path.dirname(min(glob.glob("/kaggle/input/**/dama_tier1.json", recursive=True), key=len))
    t1 = Tier1(bundle, threads=2)
    D["v1_bundle"] = {"path": bundle, "checkpoint": t1.meta.get("checkpoint")}
    def prob(ix, q, pool):   # the trained rel head, on the turn format it was trained on
        return t1.relevance(q, [f"[{ix.turns[i]['date_raw']}] {ix.turns[i]['speaker']}: {ix.turns[i]['text'][:1000]}" for i in pool])
    dev_m = [q for q in D["questions"] if q["split"] == "dev" and q["cat"] != 5 and q["evidence"]]
    pv = {q["qid"]: prob(D["ix"][q["ci"]], q["question"], D["pools"][q["qid"]][0]) for q in dev_m}
    lam = fit_lambda(conformal_scores(dev_m, D["pools"], pv), MAIN_ALPHA)
    D["settings"]["v1_lambda"] = lam
    qs = [q for q in D["questions"] if q["ci"] in V1_CONVS]
    evaluate("DAMA_v1_conv7-9", qs, lambda q: dama_ctx(q, D["B"], lam=lam, prob_fn=prob), extra={"lambda": lam, "convs": V1_CONVS})
    save_results()


@stage("judge2", limit_h=FINAL_LIMIT_H)
def stage_judge2():
    global READER
    rows = [r for r in ROWS.get("DAMA_v0", []) if r["cat"] != 5]
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)
    sample = []
    for c, rs in by.items():
        sample += rng.sample(rs, min(len(rs), max(1, round(200 * len(rs) / len(rows)))))
    sample = sample[:200]
    first = [r["correct"] for r in sample]
    READER.unload(); READER = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    j2 = LLM(JUDGE2_ID)
    copies = [dict(r) for r in sample]
    judge_rows(copies, llm=j2)
    second = [r["correct"] for r in copies]
    from sklearn.metrics import cohen_kappa_score
    D["judge2"] = {"model": JUDGE2_ID, "n": len(sample), "agreement": float(np.mean([a == b for a, b in zip(first, second)])),
                   "kappa": float(cohen_kappa_score(first, second)), "unparsed": int(sum(r.get("judge_unparsed", False) for r in copies))}
    j2.unload()
    log("judge2", D["judge2"])


@stage("audit", limit_h=FINAL_LIMIT_H)
def stage_audit():
    import csv
    rng = random.Random(SEED)
    pool = [r for m in ("B1", "B2", "B3", "B4", "DAMA_v0") for r in ROWS.get(m, []) if r["cat"] != 5]
    by = defaultdict(list)
    for r in pool:
        by[r["cat"]].append(r)
    pick = []
    for c, rs in by.items():
        pick += rng.sample(rs, min(len(rs), round(100 * len(rs) / len(pool))))
    with (OUT / "judge_audit.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "category", "question", "gold", "answer", "label"])
        for r in pick[:100]:
            w.writerow([r["method"], CATS[r["cat"]], r["question"], r["gold"], r["answer"], "CORRECT" if r["correct"] else "WRONG"])


@stage("latency", limit_h=FINAL_LIMIT_H)
def stage_latency():
    """CPU, real weights: per-component timing on a synthetic store of 10^5 turns (LoCoMo turns tiled with ID suffixes)."""
    from dama_train.data.bm25 import BM25
    base = [t for c in D["convs"] for t in c["turns"]]
    embs = np.concatenate([D["ix"][c["ci"]].emb for c in D["convs"]])
    size = 10000 if SMOKE else 100000
    reps = math.ceil(size / len(base))
    turns = [{**t, "dia": f"{t['dia']}#{r}", "i": r * len(base) + k} for r in range(reps) for k, t in enumerate(base)][:size]
    emb = np.tile(embs, (reps, 1))[:size]
    t0 = time.time(); bm = BM25([t["text"] for t in turns]); build_s = time.time() - t0
    ents = [entities(t["text"]) for t in turns]
    loglen = np.log([max(1, len(t["text"])) for t in turns]); ntoks = [ntok(render(t)) + 1 for t in turns]
    test_q = [q for q in D["questions"] if q["split"] == "test"]
    qs = random.Random(SEED).sample(test_q, min(300, len(test_q)))
    dama, comp = D["dama"], defaultdict(list)
    for q in qs:
        T = {}
        t = time.perf_counter(); qv = dama.qvec(q["question"]); T["query_embedding"] = time.perf_counter() - t
        t = time.perf_counter(); s_b = np.array(bm.scores(q["question"])); lb = list(np.argsort(-s_b)[:200]); T["bm25"] = time.perf_counter() - t
        t = time.perf_counter(); s_d = emb @ qv; ld = list(np.argpartition(-s_d, 200)[:200]); ld.sort(key=lambda i: -s_d[i]); T["dense"] = time.perf_counter() - t
        t = time.perf_counter(); qe = entities(q["question"]); ent = np.array([len(qe & e) for e in ents]); le = [i for i in np.argsort(-ent, kind="stable")[:200] if ent[i] > 0]; T["entity"] = time.perf_counter() - t
        t = time.perf_counter(); order, sc = rrf([lb, ld, le]); pool = order[:D["K"]]; T["fusion"] = time.perf_counter() - t
        t = time.perf_counter()
        bsel = s_b[pool]; lo, hi = bsel.min(), bsel.max()
        feats = np.array([[s_d[i], (s_b[i] - lo) / (hi - lo) if hi > lo else 0, sc[i], ent[i], loglen[i]] for i in pool])
        p = dama.head.predict_proba(feats)[:, 1]; T["head"] = time.perf_counter() - t
        lam = D["lam"][MAIN_ALPHA]
        t = time.perf_counter()
        if np.isfinite(lam):
            band = [j for j in np.argsort(np.abs(p - lam)) if abs(p[j] - lam) <= BORDER_BAND][:BORDER_MAX]
            if band:
                z = dama.ce.predict([(q["question"], f"{turns[pool[j]]['speaker']}: {turns[pool[j]]['text']}") for j in band], show_progress_bar=False)
                p[band] = dama.platt.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]
        T["borderline_rescoring"] = time.perf_counter() - t
        t = time.perf_counter(); adm = [j for j in range(len(pool)) if p[j] >= lam]; T["threshold"] = time.perf_counter() - t
        t = time.perf_counter()
        order = sorted(adm, key=lambda j: -p[j])
        chosen = mmr_pack(p, emb[pool], adm, [ntoks[pool[j]] for j in order], D.get("B", 1500))
        render_many([turns[pool[j]] for j in chosen]); T["packing"] = time.perf_counter() - t
        for k, v in T.items():
            comp[k].append(v * 1000)
        comp["whole_read_path"].append(sum(T.values()) * 1000)
    D["latency_1e5"] = {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in comp.items()}
    D["latency_1e5"]["bm25_index_build_s"] = round(build_s, 1)
    D["latency_1e5"]["store_size"] = len(turns)
    log("latency 1e5", D["latency_1e5"])


# ----------------------------------------------------------------------------------------------- report
def f(x, nd=3, pct_=False):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "-inf" if x < 0 else str(x)
    return f"{100 * x:.1f}" if pct_ else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def acc_cell(r):
    if not r or r.get("acc") is None:
        return "n/a"
    lo, hi = r.get("acc_ci", (None, None))
    return f"{100 * r['acc']:.1f} [{f(lo, pct_=True)}, {f(hi, pct_=True)}]"


def paired(a, b):
    ra = {r["qid"]: r for r in ROWS.get(a, []) if r["cat"] != 5}
    rb = {r["qid"]: r for r in ROWS.get(b, []) if r["cat"] != 5}
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    x = np.array([float(ra[i]["correct"]) for i in ids]); y = np.array([float(rb[i]["correct"]) for i in ids])
    d = x - y
    return {"diff": float(d.mean()), "ci": boot_ci(d), "p": mcnemar_p(x.astype(bool), y.astype(bool)), "n": len(ids)}


def write_report():
    sub = {q["qid"] for q in test_subset()} if D.get("questions") else set()
    for name in ("DAMA_v0", "B1", "B2", "B3", "B4"):
        if name in ROWS:
            RESULTS[f"{name}_subset"] = summarize([r for r in ROWS[name] if r["qid"] in sub])
    for name in ("DAMA_v0", "B1", "B2", "B3", "B4"):   # the same conversations as v1, for a like-for-like comparison
        if name in ROWS and "DAMA_v1_conv7-9" in RESULTS:
            RESULTS[f"{name}_conv7-9"] = summarize([r for r in ROWS[name] if r["ci"] in V1_CONVS])
    run_id = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local") + "-" + time.strftime("%Y%m%d-%H%M%S")
    E, S, R = D.get("env", {}), D.get("settings", {}), RESULTS
    L = [f"# DAMA v0 results  (generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, run id {run_id})", ""]
    L += ["## 1. Environment",
          f"- OS {E.get('os')}; CPU {E.get('cpu')}, {E.get('cores')} cores; RAM {E.get('ram_mb')} MB; GPUs {E.get('gpus')}; Python {E.get('python')}.",
          f"- Packages: " + ", ".join(f"{k} {v}" for k, v in (E.get("packages") or {}).items()) + ".",
          f"- Reader and judge: `{READER_ID}` (fp16, greedy, max 40 new tokens); second judge `{JUDGE2_ID}`; embedder `{EMBED_ID}` (CPU); cross-encoder `{CE_ID}` (CPU); B4 ingestion via Ollama `{OLLAMA_MODEL}`.",
          f"- Token counts: tiktoken `{E.get('tokenizer_for_counts')}`; reader-reported prompt tokens are Qwen tokenizer counts.",
          f"- Dataset: locomo10.json SHA-256 `{D.get('sha256')}`. Seed {SEED}. Total API cost: USD 0.00 (all models run on Kaggle; GPU quota only).",
          f"- Wall time per stage (min): " + ", ".join(f"{k} {v}" for k, v in STATE["stages"].items()) + f"; total {round((time.time() - T_START) / 60, 1)} min.", ""]
    cc = D.get("category_check", {})
    L += ["## 2. Data",
          "- Category mapping used: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial. Evidence from the data: "
          + "; ".join(f"cat {c}: n={v['n']}, starts with 'When' {100 * v['share_when']:.0f}%, no gold answer {100 * v['share_no_gold_answer']:.0f}%" for c, v in cc.items()) + ".",
          "- Example questions per category (5 each) are in run.log. README lines mentioning categories: " + (" | ".join(D.get("readme_category_lines", [])[:3]) or "none found") + ".",
          f"- Evidence repair (dama_train.data.locomo.parse_evidence): {D.get('repair')}.",
          "", "| split | " + " | ".join(CATS.values()) + " | total |", "|---|" + "---|" * (len(CATS) + 1)]
    for s in ("dev", "test"):
        c = D.get("counts", {}).get(s, {})
        L.append(f"| {s} | " + " | ".join(str(c.get(n, 0)) for n in CATS.values()) + f" | {sum(c.values())} |")
    L.append("")
    lr = S.get("lr_head", {})
    L += ["## 3. Settings chosen on dev",
          f"- K = {S.get('K', {}).get('value')} ({S.get('K', {}).get('rule')}; dev all-evidence-in-pool {json.dumps({k: round(v, 3) for k, v in S.get('K', {}).get('dev_all_evidence_in_pool', {}).items()})}).",
          f"- B = {S.get('B', {}).get('value')} ({S.get('B', {}).get('rule')}; dev accuracy {json.dumps({k: round(v, 3) for k, v in S.get('B', {}).get('dev_sample_acc', {}).items()})}).",
          f"- B3 k = {S.get('B3_k', {}).get('value')} ({S.get('B3_k', {}).get('rule')}: DAMA {f(S.get('B3_k', {}).get('dama_dev_mean_packet_tokens'), 1)} vs B3 {f(S.get('B3_k', {}).get('b3_dev_mean_packet_tokens'), 1)} tokens).",
          f"- LR head on standardized features {lr.get('features')}: coefficients {lr.get('coef_on_standardized')}, intercept {lr.get('intercept')} ({lr.get('dev_pairs')} dev pairs, {lr.get('dev_positives')} positive).",
          f"- Cross-encoder Platt: a = {S.get('platt_ce', {}).get('a')}, b = {S.get('platt_ce', {}).get('b')} ({S.get('platt_ce', {}).get('dev_pairs')} dev pairs).",
          f"- lambda (LR head): " + ", ".join(f"alpha {a}: {f(v, 4)}" for a, v in S.get("lambda", {}).items()) + f" from {S.get('lambda_n_dev')} dev questions ({f(S.get('lambda_dev_minus_inf_share'), pct_=True)}% with evidence outside the pool, score -inf). lambda for CE-on-whole-pool (ablation g): " + ", ".join(f"{a}: {f(v, 4)}" for a, v in S.get("lambda_ce_all", {}).items()) + ".",
          f"- MMR lambda 0.7 on embedding similarity; borderline rescoring if < {BORDER_MS:.0f} ms elapsed, at most {BORDER_MAX} candidates within +-{BORDER_BAND} of lambda.", ""]
    L += ["## 4. Main results, LoCoMo test (conversations 4-9)",
          "| method | n | judge acc % [95% CI] | token F1 | abstention acc % (cat 5) | mean input tokens | evidence recall % | read p50 / p95 ms | cost USD |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, label in (("B1", "B1 full context"), ("B2", "B2 recent window"), ("B3", "B3 retrieval + CE top-k"), ("B4", "B4 Mem0"), ("DAMA_v0", "DAMA v0 (alpha 0.2)"), ("DAMA_v1_conv7-9", "DAMA v1 (conv 7-9 only)")):
        r = R.get(name)
        if not r:
            why = STATE["errors"].get({"B4": "mem0", "DAMA_v1_conv7-9": "v1"}.get(name, "main"), "not run")
            L.append(f"| {label} | failed or not run: {why} | | | | | | | |"); continue
        rd = f"{f(r.get('read_p50'), 1)} / {f(r.get('read_p95'), 1)}" if r.get("read_p50") is not None else "n/a"
        L.append(f"| {label} | {r['n']} | {acc_cell(r)} | {f(r.get('f1'))} | {f(r.get('abstention'), pct_=True)} | {f(r.get('mean_tokens'), 0)} | {f(r.get('evidence_recall'), pct_=True)} | {rd} | 0.00 |")
    if R.get("B4"):
        i = R["B4"]
        L.append(f"\nB4 ingestion: {i.get('ingest_llm_calls')} LLM calls, {i.get('ingest_prompt_tokens')} prompt + {i.get('ingest_output_tokens')} output tokens, {i.get('minutes')} min (incl. search); search p50/p95 {f(i.get('search_p50_ms'), 1)} / {f(i.get('search_p95_ms'), 1)} ms (mem0ai {i.get('library')}).")
    if D.get("b1_truncated"):
        L.append(f"\nB1 truncation (reader context {READER.max_ctx if READER else 32768} tokens): {D['b1_truncated']}.")
    if R.get("DAMA_v1_conv7-9"):
        L.append("\nSame conversations 7-9 for comparison with v1: " + "; ".join(f"{n} {acc_cell(R.get(n + '_conv7-9'))}" for n in ("DAMA_v0", "B1", "B2", "B3", "B4") if R.get(n + "_conv7-9")) + ".")
    L += ["", "## 5. Accuracy by category (judge acc %, test)", "| method | multi-hop | temporal | open-domain | single-hop | adversarial (abstention) |", "|---|---|---|---|---|---|"]
    for name in ("B1", "B2", "B3", "B4", "DAMA_v0", "DAMA_v1_conv7-9"):
        if R.get(name):
            b = R[name]["by_category"]
            L.append(f"| {name} | " + " | ".join(f(b.get(c), pct_=True) for c in CATS.values()) + " |")
    L += ["", "## 6. Paired comparisons (DAMA v0 vs each baseline)", "| comparison | diff (points) [95% CI] | McNemar p | non-inferior at -5? (vs B3) |", "|---|---|---|---|"]
    for b in ("B1", "B2", "B3", "B4"):
        pr = paired("DAMA_v0", b)
        if pr:
            ni = ("yes" if pr["ci"][0] > -0.05 else "no") if b == "B3" else ""
            L.append(f"| DAMA v0 - {b} (n={pr['n']}) | {100 * pr['diff']:+.1f} [{100 * pr['ci'][0]:+.1f}, {100 * pr['ci'][1]:+.1f}] | {pr['p']:.4f} | {ni} |")
    L += ["", "## 7. Topology (push vs pull vs sidecar)", f"On the fixed stratified {len(sub)}-question test subset (seed 13); gateway = DAMA v0 on the same subset.",
          "| mode | judge acc % | evidence coverage % | mean tool calls | reader time p50 s |", "|---|---|---|---|---|"]
    g = R.get("DAMA_v0_subset")
    if g:
        L.append(f"| gateway (DAMA v0 packet) | {acc_cell(g)} | {f(g.get('evidence_recall'), pct_=True)} | 0 | {f((g.get('reader_p50_ms') or 0) / 1000, 2)} |")
    for m in ("pull", "sidecar"):
        t = R.get(f"topology_{m}")
        L.append(f"| {m} | {acc_cell(t)} | {f(t.get('coverage'), pct_=True)} | {f(t.get('mean_calls'), 2)} | {f(t.get('reader_p50_s'), 2)} |" if t else f"| {m} | not run: {STATE['errors'].get('topology', 'n/a')} | | | |")
    tp, ts = R.get("topology_pull"), R.get("topology_sidecar")
    if tp and ts and g:
        cov = {"gateway": g["evidence_recall"], "pull": tp["coverage"], "sidecar": ts["coverage"]}
        L.append(f"\np_a (pull-only called the tool) = {f(tp['called_share'], 3)}; p_h (gateway packet held all evidence) = {f(g['evidence_recall'], 3)}; "
                 f"rho (all-evidence recall of pull results when called) = {f(tp.get('pull_recall_when_called'), 3)}. Proposition 1 (sidecar >= max(gateway, pull)): "
                 f"{'yes' if cov['sidecar'] >= max(cov['gateway'], cov['pull']) else 'no'} ({f(cov['sidecar'], 3)} vs {f(cov['gateway'], 3)}, {f(cov['pull'], 3)}). "
                 "Reader time is the amortized share of a batched forward pass, not single-request latency.")
    L += ["", "## 8. Conformal and calibration", "| source | K=16 | K=32 | K=64 | K=128 |", "|---|---|---|---|---|"]
    for src in ("bm25", "dense", "hybrid"):
        L.append(f"| {src} | " + " | ".join(f(D.get("ceiling_all", {}).get(f"{src}|{K}"), pct_=True) for K in (16, 32, 64, 128)) + " |")
    L += ["(share of category 1-4 questions with all evidence in the pool, all 10 conversations)", "",
          "| alpha | target omission | mean | SD | min | max | folds with lambda=-inf |", "|---|---|---|---|---|---|---|"]
    for a, v in D.get("loco", {}).items():
        L.append(f"| {a} | {a} | {f(v['mean'])} | {f(v['sd'])} | {f(v['min'])} | {f(v['max'])} | {v['folds_lambda_minus_inf']} |")
    L.append(f"\nTest-set omission (conformal admission only, before the token budget): " + ", ".join(f"alpha {a}: {f(v)}" for a, v in D.get("test_omission", {}).items())
             + f". ECE of the LR head on test pool pairs: {f(lr.get('ece_test_pairs'), 4)} ({lr.get('test_pairs')} pairs). Budget-limited packets (DAMA v0, test): {f((R.get('DAMA_v0') or {}).get('budget_limited_share'), pct_=True)}%.")
    L += ["", "## 9. Ablations (test, judge acc % and mean tokens)", f"On the fixed stratified {len(sub)}-question test subset (seed 13). Reference on the same subset: "
          + "; ".join(f"{n} {acc_cell(R.get(n + '_subset'))} ({f((R.get(n + '_subset') or {}).get('mean_tokens'), 0)} tokens)" for n in ("DAMA_v0", "B3", "B1") if R.get(n + "_subset")) + ".",
          "| ablation | judge acc % [95% CI] | mean input tokens | evidence recall % | read p50 ms |", "|---|---|---|---|---|"]
    for name in ("abl_a_no_conformal_topk", "abl_b_no_mmr", "abl_c_bm25_only", "abl_d_dense_only", "abl_e_alpha_0.1", "abl_e_alpha_0.3", "abl_f_B500", "abl_f_B3000", "abl_g_ce_whole_pool"):
        r = R.get(name)
        L.append(f"| {name} | {acc_cell(r)} | {f(r.get('mean_tokens'), 0)} | {f(r.get('evidence_recall'), pct_=True)} | {f(r.get('read_p50'), 1)} |" if r else f"| {name} | not run: {STATE['errors'].get(name, STATE['errors'].get('ablations', 'n/a'))} | | | |")
    L += ["", "## 10. Latency with real weights (CPU)"]
    lat = D.get("latency_1e5")
    if lat:
        L += ["| component (10^5-turn synthetic store, 300 queries) | p50 ms | p95 ms |", "|---|---|---|"]
        for k, v in lat.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {f(v['p50'], 2)} | {f(v['p95'], 2)} |")
        L.append(f"BM25 index build: {lat.get('bm25_index_build_s')} s (pure-Python BM25 from dama_train).")
    else:
        L.append(f"not run: {STATE['errors'].get('latency', 'n/a')}")
    if g:
        L.append(f"Whole read path on LoCoMo test (per-conversation store, same CPU): p50 {f(g.get('read_p50'), 1)} ms, p95 {f(g.get('read_p95'), 1)} ms, mean {f(g.get('read_mean'), 1)} ms. ONNX int8 export of bge-small: not run (optional; not needed for this CPU path).")
    j2 = D.get("judge2")
    L += ["", "## 11. Judge reliability",
          (f"Second judge `{j2['model']}` on {j2['n']} stratified DAMA v0 test answers (cat 1-4): agreement {100 * j2['agreement']:.1f}%, Cohen's kappa {j2['kappa']:.3f} ({j2['unparsed']} unparsed)." if j2 else f"not run: {STATE['errors'].get('judge2', 'n/a')}")
          + (" judge_audit.csv (100 stratified judged items from B1-B4 and DAMA v0) is in the output folder." if (OUT / "judge_audit.csv").exists() else f" Judge audit: not run: {STATE['errors'].get('audit', 'n/a')}.")]
    L += ["", "## 12. LongMemEval-S subset (Stage B)", "not run: Stage B is optional and was not started in this run (plan: only after Stage A, after asking)."]
    L += ["", "## 13. Example packets"]
    rows = ROWS.get("DAMA_v0", [])
    ex = [next((r for r in rows if r["cat"] != 5 and r["correct"]), None),     # packets are shown cut to 900 characters
          next((r for r in rows if r["cat"] != 5 and not r["correct"]), None),
          next((r for r in rows if r["context"].startswith("NOT_FOUND")), None)]
    for name, r in zip(("correct", "wrong", "NOT_FOUND"), ex):
        if r:
            L += [f"**{name}** ({CATS[r['cat']]}) Q: {r['question']} | gold: {r['gold']} | answer: {r['answer']}", "```", r["context"][:900] + (" ..." if len(r["context"]) > 900 else ""), "```"]
        else:
            L.append(f"**{name}**: no such packet in this run.")
    L += ["", "## 14. Problems, deviations and caveats"]
    for k, v in STATE["errors"].items():
        L.append(f"- Stage `{k}` failed or was skipped: {v}")
    L += [f"- {d}" for d in STATE["deviations"]]
    L += ["", "## 15. Files",
          "Kaggle notebook output (`s41r4j/dama-paper-v0`, this version): `paper/run.log` (full log), `paper/results.json` (all numbers), `paper/state.json`, "
          "`paper/methods/*.jsonl` (per-question contexts, answers, labels), `paper/llm_cache.sqlite` (every LLM call), `paper/mem0_contexts.json`, "
          "`paper/judge_audit.csv`, `paper/RESULTS_FOR_PAPER.md`, `dama_v0_results.zip` (everything except caches). Code: `experiments/v0_2026-10-22/dama_paper_v0.py` in the repo."]
    text = "\n".join(L) + "\n"
    if len(text) > 20000:
        text = text[:19900] + "\n\n(truncated to 20,000 characters; full numbers in results.json)\n"
    (OUT / "RESULTS_FOR_PAPER.md").write_text(text)
    return text


DEVIATIONS = [
    "Reader/judge is Qwen2.5-7B-Instruct on Kaggle (no API keys available; chosen by the user). Its context is 32k tokens, not >=128k: B1 is truncated to the newest turns where a conversation does not fit (listed in section 4).",
    "Second judge is microsoft/Phi-3.5-mini-instruct instead of a second provider (Llama 3.1 is gated on Hugging Face).",
    "B4 (Mem0) ingests with qwen2.5:7b through Ollama (4-bit GGUF), the same model family as the reader but a quantized build, because mem0ai cannot call an in-process transformers model.",
    "DAMA v1 (the trained L1.2 bundle) is reported on conversations 7-9 only: its training used conversations 0-4, validation 5 and calibration 6. Its lambda is refitted on dev (conversations 0-3), which were part of its training data, so v1's threshold is likely too optimistic.",
    "Borderline rescoring replaces p for up to 8 candidates after lambda was fitted on LR-head scores; the conformal guarantee strictly holds only for the LR scores.",
    "Entity matching ignores a fixed list of capitalized function words (What, When, The, ...), otherwise every question would match every sentence-initial word. Ties in entity-overlap count are broken by dense similarity (a smoke run showed that breaking them by turn order filled the fused pool with early turns).",
    "BM25 uses dama_train's tokenizer (lowercase, alphanumeric tokens, English stopwords removed) instead of rank_bm25.",
    "Ablation (a) keeps MMR packing and the budget, replacing only the conformal admission by the top-k (k = B3's k) LR-head candidates.",
    "Reader time in the topology runs is the amortized per-question share of batched generation, not single-request latency.",
    "Pool-level evidence checks only cover questions whose repaired evidence is non-empty; accuracy covers all questions.",
    "Topology runs and ablations use a fixed stratified random 500-question subset of the test split (seed 13), not all 1,229 test questions: at ~2 s per question (reader + judge) on 2x T4 the full plan exceeded Kaggle's 12 h limit. Main methods are also reported on that subset for paired comparison.",
    "Prompts over 2,000 tokens (all B=3000 packets) are decoded one at a time with chunked prefill instead of in padded batches (version 6 ran out of GPU memory in batches). LLM calls answered in version 6 are reused from its cache, so a few answers in DAMA v1 and ablations a-b come from batched decoding; both are greedy fp16 and differ only by numerical noise.",
    "The pull-mode system prompt was rewritten after version 6: it inherited the push rules ('use only the MEMORY section; otherwise reply Not mentioned'), so with an empty MEMORY the reader called recall for only 6% of questions (pull accuracy 3.8%). The new prompt tells it to call recall first. This change was made after seeing that test-subset result.",
    "B4 (Mem0) ingestion and search are reused from version 5 of this notebook (same code, data and settings; its run failed later in an unrelated stage), saving ~1 h of GPU time.",
    "Cross-encoder scores used only to fit settings on dev (Platt scaling, the CE-on-whole-pool lambda, the B3 k) are computed on the GPU; every read-path latency uses the CPU model.",
    "B1 is answered in a child process (one conversation at a time, shared LLM cache) because long-prompt decoding on the T4s hit intermittent 'CUDA error: unspecified launch failure' in earlier runs; if the child fails, B1 is reported on the questions it answered.",
]


def zip_outputs():
    z = Path("/kaggle/working/dama_v0_results.zip") if ON_KAGGLE else OUT.parent / "dama_v0_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file() and p.name not in ("llm_cache.sqlite", "locomo10.json"):
                zf.write(p, p.relative_to(OUT.parent))
        zf.write(Path(__file__), "paper/code/dama_paper_v0.py")
    return z


# ----------------------------------------------------------------------------------------------- main
if __name__ == "__main__" and "--b1-child" in sys.argv:
    OUT = OUT  # same folder and cache as the parent
    run_b1_child()
    sys.exit(0)

if __name__ == "__main__":
    if ON_KAGGLE and not SMOKE and not glob.glob("/kaggle/input/**/llm_cache.sqlite", recursive=True):
        log("no earlier LLM cache attached; stopping before using GPU time"); sys.exit(1)   # a resume run without its cache would redo ~10 h
    STATE["deviations"] = DEVIATIONS
    D["env"] = environment(); log("environment", D["env"])
    stage_data()
    stage_index()
    stage_head()
    stage_loco()
    stage_mem0()
    stage_load_reader()
    if READER is not None:
        stage_choose_b()
        if "B" in D:
            stage_main()
            stage_v1()
            stage_b1()
            if READER is not None:
                stage_topology()
                stage_ablations()
            stage_audit()
            if READER is not None:
                stage_judge2()
    stage_latency()
    save_results()
    text = write_report()
    z = zip_outputs()
    log(f"report written ({len(text)} chars); zip {z}; errors: {STATE['errors'] or 'none'}")

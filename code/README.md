# DAMA, Tier-1 model training (free Colab / Kaggle)

This folder trains the small, fast models that the DAMA Memory Agent runs on every request, calibrates them,
and exports them as int8 ONNX files that run on a CPU in milliseconds. Nothing here needs a paid GPU:
the whole pipeline is sized for one free Colab or Kaggle session (T4, 16 GB), and every stage can be
interrupted and re-run.

## What gets trained

| Model | Starts from | Params | int8 size | What it decides |
|---|---|---|---|---|
| **Decision** | `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | ~22 MB | `rel`: is this stored turn relevant to the request? · `sal`: is this new turn worth storing? · `need`: does the request need memory at all? · `route`: which stores to query (`episodic`, `facts`, `timeline`, `expand`) |
| **Op** | `cross-encoder/nli-deberta-v3-xsmall` | 70.8M | 81–273 MB | `ADD` / `UPDATE` / `DELETE` / `NOOP` for a pair (existing memory, new statement) |
| **Retriever** | `BAAI/bge-small-en-v1.5` | 33M | ~33 MB | first stage: embeddings that pick the top-k turns for `rel` to re-score (fine-tuned on the train split, kept only if it beats the base model) |

The op model ships int8 only if an int8 variant agrees with PyTorch on ≥ 98% of test labels, otherwise fp32;
the retriever needs ≥ 95% top-5 overlap. `export` records which variant was kept.

Both are cross-encoders (they read the two texts together), so they can see contradictions such as
"lives in Berlin" vs "moved to Pune", which separate embeddings averaged into one vector cannot.
Both start from models that already solve a close task: MS MARCO relevance for `rel`, and NLI
(entail / neutral / contradict) for the op head. The new 4-way op head is initialised from the NLI head
(NOOP ← entailment, ADD ← neutral, UPDATE/DELETE ← contradiction).

After training, `calibrate` turns the raw scores into calibrated probabilities (Platt scaling and temperature)
and fits a **conformal threshold λ** for each rerank depth k: on held-out conversations, a memory packet built from
the first-stage top-k candidates with p ≥ λ contains every gold evidence turn for at least 1 − α of requests
(α = 0.05 / 0.1 / 0.2, k = 4 / 8 / 16 / 32 / 64). The `rel` calibration is fitted on those same retrieved pools.

## Training data

| Source | Used for | Notes |
|---|---|---|
| LoCoMo (10 long conversations, ~2k QA) | rel, sal, need, route | evidence turn ids are in the dataset; BM25 hard negatives |
| LongMemEval (500 questions, `_s` haystacks) | rel, sal, need, route, real UPDATE pairs | evidence = turns with `has_answer`; knowledge-update questions give old→new value pairs |
| MNLI (40k sample) | op model | keeps the NLI skill: entail→NOOP, neutral→ADD, contradict→UPDATE |
| Dolly-15k (6k sample) | need = 0 | ordinary requests that need no memory |
| SQuAD + Quora (3k questions each) | need = 0 | questions that need no memory, many first-person ("How do I …?"), so `need` cannot just learn "question vs. instruction" |
| Memory-request templates (generated) | need = 1 | casual requests like "What did I tell you about my job?"; frames held out per split |
| Synthetic slot statements (generated) | op model | 14 slots, 11 kinds of change (explicit and implicit update, add, duplicate, hedge, hearsay, chatter, retraction …); frames and values are held out per split |

**Splits are by conversation**, never by row: LoCoMo conversations 0–4 train, 5 val, 6 calib, 7–9 test;
LongMemEval by a hash of the question id (60/10/10/20, an `_abs` question stays with its base question).
`build` refuses to write data if any conversation appears in two splits, and checks that no synthetic
template is shared between splits.

## Run it on Colab or Kaggle

Open `notebooks/DAMA_train.ipynb`.

* **Colab:** *Runtime → Change runtime type → T4 GPU*, then *Run all*. Allow the Google Drive prompt:
  data and checkpoints go to `MyDrive/DAMA` (about 3 GB), so a disconnect loses nothing.
* **Kaggle:** *Settings → Accelerator → GPU T4 x2* and *Settings → Internet → On* (needs a phone-verified
  account). Do not pick P100: current PyTorch builds no longer ship kernels for it (the code detects this
  and warns). For an unattended run use *Save Version → Save & Run All*; the bundle appears under *Output*.

The notebook finds the code in this order: an existing copy, `MyDrive/DAMA/code` on Drive, a Kaggle dataset
you attached (folder or zip), `REPO_URL` (git), or it asks you to upload `dama_code.zip` (Colab).

**If the session dies, reconnect and Run all again.** Download and build skip finished work, training resumes
from `runs/<model>/checkpoint.pt`, and `--max-minutes` (default 150 in the notebook) makes each training
stage checkpoint and stop cleanly before a free session would be cut off.

Expected time on a T4: the whole pipeline should fit well inside one session (both models together are
under 100M parameters and the data is tens of thousands of pairs). These are estimates, the first run
prints real rows/s and an ETA every 30 seconds.

## Command line

```bash
pip install -r requirements.txt          # plus torch, if not preinstalled
python -m dama_train smoke               # offline end-to-end check with tiny random models (~3 min on CPU)

python -m dama_train download            # LoCoMo, LongMemEval, MNLI, Dolly, SQuAD and Quora samples
python -m dama_train build               # training JSONL + manifest.json (hashes, stats)
python -m dama_train train-retriever     # fine-tune bge-small (first stage), ~3 min on a T4
python -m dama_train train-decision      # --max-minutes 150 --epochs 3 --batch-size 64 --fp32
python -m dama_train train-ops
python -m dama_train calibrate           # picks the first stage on calib; --no-dense for BM25 only
python -m dama_train export              # ONNX bundle + zip (quantization chosen by measurement)
python -m dama_train all                 # everything in order
```

`--work DIR` picks the work folder (default: Drive on Colab, `/kaggle/working/dama` on Kaggle,
`./dama_work` locally). Layout:

```
<work>/raw/                  downloaded datasets
<work>/data/                 decision/, ops/, pools/, dialogues/ JSONL + manifest.json
<work>/runs/retriever/       log.jsonl, best/ (fine-tuned bge-small, or the base model if tuning did not help)
<work>/runs/decision|ops/    checkpoint.pt, log.jsonl, best/
<work>/runs/calibration/     calibration.json, report.json (held-out test numbers)
<work>/export/dama_tier1/    retriever, decision and ops ONNX files, tokenizers, dama_tier1.json, runtime.py
<work>/CHECKPOINT.json       label (L1.0, L1.1, ...), notes, code hash, headline metrics   (notebook)
<work>/code_snapshot.zip     the exact code that produced this output                     (notebook)
```

**Checkpoints live on Kaggle.** Each run of the notebook is one checkpoint, labelled by `config.CHECKPOINT`.
A saved Kaggle notebook version keeps its own output, so every checkpoint's models, report and code stay
there without downloading anything. The `dama-code` dataset gets one version per checkpoint too.

## Using the exported models (CPU, no PyTorch)

```bash
pip install -r requirements-runtime.txt
```

```python
from runtime import Tier1                       # runtime.py is inside the bundle
t1 = Tier1("dama_tier1")
vecs = t1.embed(memories)                       # store with each memory when it is written
order = t1.first_stage(request, vecs)           # dense first stage, best first (see meta["calibration"]["first_stage"])
t1.need("Where do I live these days?")          # probability the request needs memory
t1.route("When did I start the new job?")       # {"episodic": p, "facts": p, "timeline": p, "expand": p}
t1.relevance(request, memories)                 # calibrated p for each candidate memory
t1.select(request, candidates, alpha=0.1, budget_tokens=1500)  # candidates ordered by your retriever, best first
# -> {"ids": [...], "p": [...], "complete": True/False, "tokens": n}; only the top k (default 8) are re-scored
t1.op("The user lives in Berlin.", "I moved to Pune last month")  # ADD / UPDATE / DELETE / NOOP probabilities
```

Measured CPU latency of the exported int8 models (same architectures with random weights, 2 vCPUs,
onnxruntime; real trained weights have the same cost):

| Model | batch 1 | batch 16 | batch 64 |
|---|---|---|---|
| Decision (MiniLM-L6), seq 128 / 128 / 96 | 17 ms | 264 ms | 789 ms |
| Op (DeBERTa-v3-xsmall), seq 64 | 25 ms | 358 ms | 1,468 ms |

So on a CPU the `rel` head re-scores only the first-stage top k (default 8, about 16 ms per pair on 2 vCPUs), never
the whole store. `export` measures latency on real test requests and their own top-k turns (k = 1 / 4 / 8 / 16, at
max_length 256 and 128), and the test report shows coverage for each k, so k is chosen from measurements. `complete` is False when the token budget dropped an admitted memory:
the coverage guarantee only holds for complete packets.

## What the test report shows

`calibrate` prints, on the test conversations only:

* **First-stage sources:** share of questions with every gold turn in the top-k for BM25, bge-small, the fine-tuned
  retriever and their RRF fusions, on calib (where the source is chosen) and test.
* **Retrieval:** gold turns inside the candidate pool, recall@5/10/20 and MRR before and after the `rel` head.
  Packet coverage can never exceed "questions with every gold turn in the pool"; if the report says the pool is
  too small for an α, raise `pool_size` or improve the retriever.
* **Conformal packets:** for each rerank depth k, target vs. achieved coverage and how many turns/tokens get admitted.
* **Heads:** AUC, best F1 and calibration error (ECE) before → after calibration; per-route F1.
* **Op model:** accuracy, macro-F1, ECE, accuracy per data source and the confusion matrix.
* **Probes:** hand-written behaviour checks (`probes.py`: 30 `need` requests, 24 op pairs such as Berlin → Pune),
  never trained on and never used for any choice. The notebook runs them again through the shipped CPU bundle.

## Known limitations

* `sal` labels are weak: a turn counts as salient if some benchmark question later uses it as evidence.
  Turns that matter but were never asked about count as negatives.
* `need` learned "benchmark question vs. instruction" in L1.0 (AUC 0.999, yet 0.049 on "Where do I live these days?").
  L1.1 adds question-form negatives and casual memory-request positives; the probes show whether that was enough.
  (`need` and `route` read the bare question: only LongMemEval questions carry a date, so a date prefix would be a shortcut.)
* `route` labels are mapped from each question's type, not annotated per question.
* Real UPDATE pairs from LongMemEval knowledge-update questions are few (tens). Most op training data is synthetic,
  so check `ops.by_source` in the report and add real examples from your own logs when you have them.
* Mapping MNLI contradiction → UPDATE keeps the NLI skill but is not a perfect match for memory updates;
  the synthetic and LongMemEval rows are what teach the real distinction.
* Check the licences of LoCoMo, LongMemEval, MNLI and Dolly before you publish trained weights.

## Level 2: the Memory Agent (`dama_agent`)

DAMA's Memory Agent runs the Tier-1 bundle as a sidecar: SQLite store + numpy, no server and no GPU.

```python
from dama_agent import MemoryAgent, load_tier1
agent = MemoryAgent(load_tier1("dama_tier1"))                 # any bundle; scopes keep users/projects apart
agent.observe("alice", "I moved to Pune last month", speaker="Alice", ts="2023-05-02")   # write path
agent.packet("alice", "Where do I live now?")                  # push: need gate, dense first stage, top 8 in budget
agent.recall("alice", "Alice's cities")                        # pull tool: more memories, no gate, no deadline
agent.timeline("alice", "where Alice lives")                   # facts incl. superseded ones, oldest first
```

* **Read path:** `need` gate → dense first stage (fine-tuned retriever) → the top 8 within the token budget → superseded
  turns bring their newer fact and are marked outdated → time-ordered lines with `[m:ID]`, or `NOT_FOUND`.
  Measured on held-out LoCoMo: 19 ms p50 / 26 ms p95 on CPU.
* **The `rel` rerank is off by default** (`use_rerank=True` turns it on). Over the fine-tuned first stage it changed
  coverage by 0.000 and cost ~220 ms on CPU: at k = 8 its conformal threshold admits every candidate, so it only
  reordered them. When on, it is deadline-aware (largest calibrated k that fits, else first-stage order).
* **Many agents, one memory:** identical requests to an unchanged scope are answered once (a cache keyed by the scope's
  store version, plus in-flight deduplication for simultaneous requests); the store is thread-safe.
* **Write path:** append the turn verbatim → `sal` gate → compare with the nearest current facts of the same speaker →
  op model → ADD / UPDATE (supersede) / DELETE (retract) / NOOP; an unsure UPDATE becomes a CONFLICT. Every decision is
  audited. `observe(..., defer=True)` + `commit()` runs it after the reply.
* **Tools** for the Primary LLM (`dama_agent/tools.py`, MCP-shaped): `recall`, `timeline`, `note`, `observe`, `commit`.
* Not built yet: Tier-2 fact extraction (a fact is its turn), BM25/entity search, cross-request batching, PostgreSQL,
  an MCP server.

```bash
python tests/test_agent.py && python tests/test_answer.py                       # no models needed
python -m dama_agent.eval_locomo --bundle dama_tier1 --locomo locomo10.json --out l2/   # Level 2 on LoCoMo 7-9
python -m dama_agent.answer --eval l2/ --model Qwen/Qwen2.5-7B-Instruct --out l3/      # Level 3 (GPU)
```

**Level 2** (`eval_locomo`) replays each held-out conversation through the write path, then asks every question through
the read path, and reports the 4 pillars: latency per stage, evidence coverage of the packets, packet tokens vs. the
history, and latency/throughput with 1–16 agents firing at once. It saves `packets.jsonl`, `histories.json` and
`writes.jsonl`. **Level 3** (`answer`) has one fixed LLM answer every question from the DAMA packet, from nothing, from
the gold evidence only and from the full history, and scores LoCoMo token F1 and an LLM-judge accuracy; it also checks
the write path's UPDATE / CONFLICT / DELETE decisions. On Kaggle both run in the notebook `dama-evaluation`, once with the
CPU bundle and once with the PyTorch weights on the GPU (`load_tier1_torch`, `dama_agent/gpu.py`); see `kaggle/README.md`.

## Files

```
dama_train/
  env.py            platform detection (Colab / Kaggle / local), GPU check, precision
  config.py         all settings (data sizes, learning rates, time budget)
  data/             download, LoCoMo + LongMemEval loaders, BM25, splits, data builders
  models/           decision model (4 heads) and op model (NLI re-headed to 4 ops)
  train/            resumable training loop with time budget, early stopping, fp16/bf16; retriever fine-tuning
  retrieval.py      BM25 + dense rankings (base and fine-tuned bge-small) and their RRF fusions
  probes.py         hand-written behaviour probes for need and op
  calibrate.py      Platt / temperature, conformal threshold, held-out test report
  export.py         ONNX export, int8 quantization, parity and latency checks
  runtime.py        CPU runtime class (copied into the bundle)
  smoke.py          offline end-to-end test
dama_agent/
  store.py          SQLite store: episodes, facts (current / superseded / deleted), audit; per-scope vector cache
  agent.py          MemoryAgent: read path (packet, recall, timeline) and write path (observe, commit, note)
  tools.py          tool schemas + dispatch for the Primary LLM
  fake.py           FakeTier1 for tests without models
  eval_locomo.py    Level 2: replay on held-out LoCoMo conversations, 4-pillar metrics, concurrency test
  answer.py         Level 3: answer accuracy (LoCoMo F1 + LLM judge) and write-path check
  gpu.py            TorchTier1: the same Tier-1 interface on GPU (PyTorch weights + the bundle's calibration)
notebooks/DAMA_train.ipynb   Colab / Kaggle notebook
kaggle/                      one folder per Kaggle item + push.sh / publish_code.sh (see kaggle/README.md)
tests/fixtures.py            small fake datasets for the smoke test
tests/test_agent.py          Memory Agent unit tests
tests/test_answer.py         Level 3 scorer unit tests
```

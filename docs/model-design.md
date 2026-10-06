# DAMA model design

DAMA is a standalone **learned decision model** for memory. It decides what to do with memory. It doesn't
store memories, run an agent or answer questions. A caller (the Primary Agent's memory layer) applies the decision.

## The flow

```text
ModelInput
  task (write | retrieve), text, source_role, recent turns, temporal_mode
  candidates: [id, text, status (current|superseded|archived), memory_type]
        │
batching.Collator  — items per example: [current text, recent turns, candidate 1..N]
        │            only real texts are tokenized; nothing is ever truncated (overflow → abstain)
        ▼
pretrained encoder (MiniLM or BGE; frozen, or LoRA on query/value) ── token states of the current text ──┐
        │ pooled vector per item                                                                         │
        ├── raw cosine(current text, candidate)  ← the similarity MiniLM/BGE were trained for            │
        ▼                                                                                                │
projection + learned embeddings for structured state                                                     │
  (item kind, task, role, temporal mode, candidate status, candidate type)                               │
        ▼                                                                                                │
2-layer transformer over the items (no position embeddings → candidate order cannot matter)              │
        │                                                                                                │
        ├─ event vector ──► operation (4) · memory type (5) · fallback (2)                               │
        ├─ event × candidate pairs [e, c, e−c, e·c, cosine] ──► relevance (per candidate) · update/archive target (+ null)
        └─ event vector + token states ──► evidence span start/end ◄─────────────────────────────────────┘
        ▼
inference.decode — typed Decision; abstains on: low confidence, non-user source (writes),
                   target that is not a current candidate, invalid span, non-finite scores, overflow
```

### Why it's built this way

- **Structured state is given as embeddings, not prose.** The earlier version wrote `status=current`, ISO timestamps
  and other metadata into the text that a frozen sentence encoder averaged. That drowned out the message and hid the
  most important signal for updates and historical queries: whether a memory is current or superseded.
- **The message is encoded on its own**, and recent turns are encoded as a separate item. Mean pooling over
  "metadata + history + message" blurred the message itself. Keeping recent turns lets the model handle replies like
  "Oh, Linode." after "Which cloud provider do we use?".
- **The raw cosine similarity is fed to the relevance and target heads.** MiniLM and BGE are trained so that this
  similarity ranks relevant text, which gives the heads a strong prior from step one.
- **The span head sees the contextualized event vector**, so the evidence it picks can depend on the candidates (for
  an UPDATE, the new value).
- **ARCHIVE is learned.** The old decoder only accepted the literal string `Archive memory <id>.`, so the ARCHIVE
  head did nothing.
- **Order equivariance and batch independence** were checked locally: reversing the candidates reverses the scores,
  and a row's outputs don't change when it's batched with other rows.

### Size

| Config | Total | Trainable |
|---|---|---|
| `minilm-frozen` (default) | 23.8M | 1.19M |
| `minilm-lora` | 23.8M | 1.27M |
| `bge-frozen` | 111.1M | 2.25M |
| `bge-lora` | 111.4M | 2.54M |

## Data

`dama.data.generate` writes about 10k synthetic examples (8000 train / 1200 dev / 1200 test). They cover 16 memory
slots across all five types:
- **key facts:** database, language, frontend, hosting, city, job
- **preferences:** theme, drink, answer style, meeting time
- **plans:** launch, trip
- **routines:** backups, standup, exercise
- **emotional:** mood

There are 14 behaviours.

| Write | Retrieve |
|---|---|
| add (fact inside chatty text → span is only the fact) | current value (with the outdated value as a hard negative) |
| add from context (answer to the assistant's question) | historical value (`temporal_mode=historical`) |
| duplicate → NOOP | missing memory → nothing relevant, fallback |
| update (target = the *current* memory, not a superseded one) | off-topic request → nothing relevant, no fallback |
| archive ("please forget my city") | all memories of a type ("What are my preferences?") |
| hedged / hearsay → NOOP + fallback | |
| chatter, prompt injection → NOOP | |
| assistant/tool text → NOOP | |

**Split by template and value.** Each message template and each slot value belongs to one split, so dev and test
only contain phrasings and entities the model never saw in training. `generate` fails if any template group leaks
across splits.

**Limits of synthetic data.** It's still templated English with clean labels. High scores here show the model can
learn the decision structure and generalise to new wording within these patterns. They don't show performance on
real conversations. The next step for research is real logged conversations with reviewed labels, which can be
saved in the same JSONL format.

## Training

`dama.training.train(config, data_dir, output)`:
- **Loss:** a multitask loss in which each head learns only from rows that carry its label (no relevance loss on
  writes, no span loss on NOOPs, and so on).
- **Optimizer:** AdamW with linear warmup and decay. LoRA adapters get their own learning rate.
- **Precision:** FP16 on a T4, BF16 where it's natively supported.
- **Selection:** dev is evaluated after every epoch, and the best epoch is exported to `output/export`. An export is
  self-contained (weights, configs, tokenizer, SHA-256 hashes, attention kernel) and loads offline.

## Evaluation

`dama.evaluate.evaluate` reports:
- **Writes:** operation accuracy and macro-F1 with a confusion matrix, type accuracy, target accuracy, span exact
  match and token F1, write precision, and the share of useful writes missed.
- **Retrieval:** precision, recall and F1.
- **Fallback:** fallback accuracy.
- **Calibration:** operation ECE.
- **Speed:** p50/p95 latency and input tokens.

The notebook also prints accuracy per behaviour. Thresholds (0.6 operation, 0.5 relevance and fallback) are
defaults. Tune them on dev only. Scores are uncalibrated probabilities.

## How this relates to the research

| Work | What it measures | Relation to DAMA |
|---|---|---|
| MemRouter (2026) | ADD/NOOP write routing, ~12M trainable on a **frozen 7B** backbone; LoCoMo F1 52.0 vs 45.6 for an LLM manager; 970 → 58 ms p50 | Same idea (a small learned head instead of LLM decoding). DAMA covers more decisions (update target, archive, retrieval, span, fallback) on a 23.8M model, so its forward pass should be much cheaper. Not measured yet. |
| SAGE (2026) | Novelty gate ADD/NOOP, escalates uncertain cases to an LLM | DAMA's fallback head plays the same role |
| Mem0 (2025) | End-to-end LoCoMo QA with an LLM extractor; 91% lower p95 latency, >90% token savings vs full context | End-to-end system numbers. DAMA would be one component of such a system. |
| Jev / System One (2026, vendor) | Typed, calibrated decisions; company-reported speed | Inspiration for the interface. DAMA scores are not calibrated yet. |

None of these numbers can be compared directly with DAMA's synthetic test scores. The research reports
**end-to-end QA** on LoCoMo or LongMemEval: a full memory system plus an answering LLM, judged on its answers. DAMA's
test set measures **decision accuracy** on synthetic data. A fair comparison needs DAMA inside a memory pipeline,
evaluated on LoCoMo or LongMemEval with the same answering model and official scoring, and never trained on their
QA answers.

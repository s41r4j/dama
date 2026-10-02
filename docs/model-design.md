# DAMA System One decision model

The deliverable is a **standalone learned decision model**, trained for memory tasks.
The user's clarification supersedes the application-oriented handoff: do not build
a Primary Agent, memory database, vector store, or conversational agent framework.

## What the research supports

The project concept calls for a fast specialized
model that emits decisions and eventually uses multiple heads. The summary and gap
PDFs are broader than the finalized report. The finalized comparison identifies
write gating, temporal updates and retrieval routing as separate capabilities to
measure, with no DAMA result yet. Earlier novelty language is not established.

[MemRouter](https://arxiv.org/html/2605.00356), section 3.2, performs BGE chunk
embedding -> trainable projection -> frozen Qwen2.5-7B body -> ADD/NOOP and type
heads. Its ~12M is the *trainable* count, not the deployed count. This implementation
adopts supervised forward-only decisions, not its complete backbone or reported
results. Its paper's limits and the project's finalized comparison motivate measuring
both admitted useful facts and rejected useful facts.

[Jev's official description](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
emphasizes structured state, parallel typed outputs and calibrated decisions. It
is conceptual inspiration. The public description does not disclose enough to
reproduce Jev's weights, training recipe or capability. Type-safe output alone does
not establish correctness or calibration. DAMA does not claim Jev equivalence.

## Architecture implemented

```text
observed text + recent context + task/time/role metadata
candidate memory texts + status/validity metadata
           |
shared pretrained MiniLM encoder (mean pooling; frozen initially)
           |
384 -> 192 learned projection
           |
2-layer bidirectional candidate contextualizer
(no candidate-position embeddings; order-equivariant)
           |
parallel heads:
  operation: ADD / NOOP / UPDATE / ARCHIVE
  memory type: key_fact / preference / plan / routine / emotional
  candidate relevance: independent logits
  update/archive target: candidate logits + null target
  fallback: supported / needs more evidence
  source span: start/end token logits on observed text
           |
typed decoding: IDs mapped from supplied candidates, exact source substring,
invalid targets/spans abstain; scores explicitly uncalibrated
```

No token-by-token decoder, generated rationale, answering model, database or agent
loop is involved. The encoder runs over event and candidate texts as one batched
invocation; a contextualizer lets write decisions see possible duplicates/updates.
Model artifacts record and restore the encoder attention implementation (eager or
SDPA) to avoid changing the inference kernel silently after loading a checkpoint.
Cost scales with candidate count and sequence length, not just head parameters.
A single forward invocation is not constant-time with respect to input size.

## Initial backbone and capacity

[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
is an Apache-2.0 pretrained English encoder with 384-dimensional features. The
exact revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Its model card gives the mean-pooling interface and 256-token default input limit.
We use 256 as a strict per-text limit, with at most 32 candidates. Oversized inputs
abstain explicitly; no text or evidence is silently truncated.

The instantiated architecture has 23,717,392 total parameters, of which 1,004,176
are trainable with a frozen encoder. FP32 weights alone are approximately 90.5 MiB;
activations, padding, runtime and CUDA allocations require additional memory.
Counts were verified from the official pinned encoder config using meta tensors,
without downloading weights.

This is a concrete size/latency-oriented starting hypothesis, not a claim that a
23.7M model handles every memory decision reliably. A 0.5B-1.5B decoder with decision
heads may be a later capacity ablation if held-out errors justify it. Qwen2.5-1.5B
JSON SFT would introduce autoregressive decoding and is no longer the default.

## Training

Supervised multitask losses: operation CE, type CE only when labeled, relevance
BCE only on retrieval rows and real candidates, target CE only on writes,
fallback CE, and source-span CE only for supported ADD/UPDATE rows. Masks prevent
unlabeled heads and padded candidates from contributing. NOOP/ADD train null target.
Start with the encoder frozen; train the projection, contextualizer and heads.
Optional encoder LoRA (query/value, rank 8) is implemented and tested for forward
and artifact loading. It must earn its extra cost on dev data. RL and foundation
pretraining are deferred. QLoRA is unnecessary for this small encoder and is not
claimed as a supported path.

A stronger candidate is [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
(MIT), revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
`accuracy-frozen.json` and `accuracy.json` compare frozen/LoRA adaptation. BGE uses
CLS pooling per its official card, a 256-wide contextualizer, and microbatch 1.
This is a capacity option, not a claimed measured accuracy improvement. Choose on
the same reviewed dev data and evaluate final selected models on untouched test.

The source-role and scope/time checks are interface constraints. The neural model
must learn relevance, admission, corrections and abstention. Type checks cannot
prove semantic correctness. An exact substring can still be an incorrect or
irrelevant user claim. Human review remains necessary for research labels.

## Evaluation boundary

Measure model-level operation confusion, write precision and useful facts missed,
relevance precision/recall, update-target accuracy, evidence-span exact match,
fallback accuracy and ECE, as well as total input tokens, candidate count,
forward-only latency, parameter count and resources. Keep whole source families,
conversations, users and projects disjoint. A repeated syntax pattern in tiny
software fixtures is not evidence of linguistic generalization.

Default thresholds (0.6 operation, 0.5 relevance/fallback) are provisional, not
tuned or calibrated. Tune only on development data and predeclare before final test.
ECE currently reports raw operation confidence against raw argmax correctness;
gated decision accuracy is reported separately. Empty evidence and low scores can
abstain. Calibration, Brier/NLL, risk-coverage curves, bootstrap intervals and
independent annotation agreement are required before deploying confidence as a
probability; those research experiments have not run.

The full-context/agent benchmarks and cost hypotheses in the old report apply to
later downstream integration, not this standalone-model milestone. No Mem0/Letta
installation is required. LoCoMo/LongMemEval QA answers must not become policy
training labels. Any later use needs official pinned sources, clean conversation
splits and separate end-to-end evaluation.

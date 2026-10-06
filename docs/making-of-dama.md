# Making of DAMA — notes

Short dated notes on how the model got to where it is. Newest at the bottom.
Design details live in [model-design.md](model-design.md).

---

## 29 Sep 2026 — the idea
- DAMA = **Dual Agent Memory Architecture**: a big Primary Agent answers; a small Memory Agent decides what goes in and out of memory.
- Goal: not a bigger context window, but making every token in it matter.
- Key choice: the Memory Agent **makes decisions, not paragraphs**. It classifies (ADD / NOOP / UPDATE / ARCHIVE, type, target, span, relevance, fallback) instead of generating text.
- "System One" = the fast, small model; uncertain cases fall back to a bigger "System Two" LLM.
- Proposal: `research/notes/proposal-2026-09-29.md`.

## Research — what we read
- **Reference set 1** (memory systems): MemGPT, Generative Agents, LongLLMLingua, SwiftSage, LongMemEval, Zep/Graphiti, A-MEM, Mem0, memory survey, MemRouter, SAGE, Gated Memory Routing.
- **Reference set 2** (agents and memory frameworks): agent surveys, Internet of Agents, DAWN, Memoria, memory fabric, travel-planning memory, NPC memory.
- Closest prior work:
  - **MemRouter**: a small learned head on a frozen 7B model routes ADD/NOOP; 970 → 58 ms.
  - **SAGE**: a novelty gate that escalates uncertain cases to an LLM.
  - **Mem0**: end-to-end memory system with LLM extraction.
- Takeaway: a small classifier can replace LLM decoding for memory writes. Nobody covers all the decisions (update target, archive, retrieval, span, fallback) in one small model.
- Reports and comparisons: `docs/reports/`. Slides: `docs/slides/`.

## 2–3 Oct 2026 — first implementation (v1)
- `src/dama/` package with CLI, `pyproject.toml`, bootstrap scripts, tests, an 80-row fixture dataset.
- Encoder: MiniLM or BGE, frozen or LoRA. Heads for each decision.
- Colab fixes along the way:
  - a Python environment without `ensurepip`
  - the attention backend being lost after saving and reloading
  - T4 GPUs only emulate BF16 (slowly) → use FP16
  - a three-step GPU smoke run added to the notebook
- Problems found on review:
  - metadata (`status=current`, timestamps) was written into the text, which drowned out the message
  - ARCHIVE fired only on the literal text `Archive memory <id>.`
  - the 80 rows were 20 sentences repeated with a trailing space, so test scores were meaningless
  - too much tooling (CLI, packaging, bootstrap) for a project whose only job right now is the model

## 6 Oct 2026 — cleanup
- Removed archive/, experiments/, stale artifacts, duplicate READMEs, fixtures, tests, scripts, `pyproject.toml`.
- All runnable code moved to `code/`. Papers stay in `research/` (git-ignored).

## 6 Oct 2026 — model-only rewrite (v2)
- **Dropped the CLI.** The notebook calls the package directly: `generate → train → evaluate → predict`.
- Kept 8 modules (~1,000 lines): `contracts`, `data`, `batching`, `model`, `inference`, `training`, `evaluate`, `artifacts`.
- Model changes:
  - metadata goes in as **learned embeddings**, not text
  - message and recent turns are **encoded separately**
  - **raw cosine** similarity is fed to the relevance and target heads
  - the span head sees the **event vector** (so it can pick out the new value on an UPDATE)
  - ARCHIVE is **learned**
  - the transformer over items has no position embeddings, so **candidate order can't matter**
- Training: AdamW, warmup + linear decay, FP16/BF16, dev score every epoch, **best epoch exported**.
- Data: generator for ~10k examples, 16 slots × 14 behaviours. **Templates and values are held out per split**, and generation fails if a template leaks across splits.
- Checked locally with a tiny random encoder (no training on the Mac):
  - losses are finite and gradients reach every weight
  - reordering candidates only reorders scores
  - a row gives the same output alone or in a batch
  - save and reload give identical outputs
  - abstention paths work
- Bug caught: inputs with zero candidates crashed (metadata tensors were float) → fixed.

## 6 Oct 2026 — first Colab run
- `minilm-frozen` on a T4: 8000 train / 1200 dev, 2000 steps, 23.9M parameters (1.19M trainable).
- **Crash at step 55: relevance loss = NaN.**
  - Cause: a batch with no retrieval rows. The empty-head loss was `logits.sum() * 0`; summing the −1e4 masked logits overflows to −inf in FP16, and −inf × 0 = NaN.
  - Fix: return a float32 zero instead (`7e57932`).
- Lesson: anything that sums masked logits must be checked under FP16, not only FP32 on the Mac.

## Next
- Finish the `minilm-frozen` run → first real test numbers and per-behaviour accuracy.
- Compare with `minilm-lora` and `bge-frozen` if needed.
- Real conversations with reviewed labels (same JSONL format).
- Fair comparison with the research: DAMA inside a memory pipeline on LoCoMo / LongMemEval, same answering LLM, official scoring.

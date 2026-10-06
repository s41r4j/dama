# Making of DAMA — milestones

Design details live in [model-design.md](model-design.md).

## 1. The idea
- DAMA = **Dual Agent Memory Architecture**: a big agent answers; a small memory model decides what goes in and out of memory.
- The memory model **makes decisions, not text**: ADD / NOOP / UPDATE / ARCHIVE, type, target, evidence span, relevance, fallback to a bigger LLM.

## 2. Research
- Read two paper sets: memory systems (MemGPT, Mem0, A-MEM, Zep, LongMemEval, MemRouter, SAGE…) and agent/memory frameworks.
- Closest work: **MemRouter** (small learned head routes writes) and **SAGE** (novelty gate with LLM fallback).
- Gap: no single small model covers all memory decisions.

## 3. First implementation
- Pretrained encoder (MiniLM or BGE) plus decision heads, trained on Colab.
- Lessons from review: metadata written as text drowned the message, ARCHIVE never really learned, the tiny dataset made scores meaningless, and the tooling outweighed the model.

## 4. Model-only rewrite
- Cut everything that isn't the model: the CLI, packaging, scripts and tests are gone. The notebook calls the code directly.
- Better model:
  - metadata goes in as learned embeddings
  - the message and recent context are encoded separately
  - a similarity feature is fed to retrieval
  - ARCHIVE is learned
  - candidate order can't change results
- Better data: ~10k generated examples covering 14 behaviours, with phrasings and values held out for test.
- Better training: learning-rate schedule and best-epoch export.

## 5. First GPU training
- `minilm-frozen` on a Colab T4.
- Found and fixed an FP16 NaN in the loss on batches without retrieval examples.

## Next
- First real test scores, then compare model sizes.
- Real conversation data with reviewed labels.
- Fair benchmark against the research on LoCoMo / LongMemEval.

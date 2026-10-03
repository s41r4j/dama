# DAMA decision model — V0.1 research implementation

DAMA is now being developed as a **small System One model for memory decisions**.
It reads text and candidate state and produces typed operation, relevance,
update-target, evidence-span and fallback outputs in a forward pass. It does not
run a conversational agent or persist memories.

**Implemented model code, not a trained checkpoint.** No optimizer steps, fine-tuning,
large labeling jobs or paid API calls have run on this Mac. Local checks use a tiny
random network and controlled logits. Their success is software correctness only.

The initial pretrained encoder is all-MiniLM-L6-v2 (Apache-2.0, exact revision in
configs). With the implemented context layer and heads: **23.72M total / 1.00M
trainable parameters**. Training defaults to a frozen encoder; optional encoder
LoRA is provided. A larger BGE-base + LoRA candidate is configured in
`configs/accuracy.json` for an accuracy/speed comparison; its advantage is unproven. Read [the model design](docs/model-design.md) for the research
rationale, scope, losses and limitations. MemRouter/Jev are inspirations, not
claimed reproductions or beaten baselines.

## Project layout

Run commands from this directory, which is the active Python package root.

| Location | Purpose |
|---|---|
| `src/dama/`, `tests/` | Active model implementation and software tests |
| `configs/`, `fixtures/` | Training configurations and small synthetic fixtures |
| `scripts/`, `notebooks/` | Colab setup, label tooling and notebook |
| [docs/](docs/README.md) | Model design, training instructions and validation |
| [artifacts/](artifacts/README.md) | Local generated runs, datasets and checkpoints |

This repository contains the DAMA model and its development/training tools.

## Local setup and verified commands

Python 3.12 is the tested local interpreter. The current `.venv` contains the pinned
model and development libraries. For a fresh environment:

```bash
cd DAMA
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,dev]'
export TOKENIZERS_PARALLELISM=false
```

Run:

```bash
dama resources
dama test
dama validate-config --config configs/heads.json
dama validate-config --config configs/lora.json
dama validate-config --config configs/accuracy.json
dama validate-config --config configs/no-context-ablation.json
dama validate-data --input fixtures/model-v1
dama model-smoke --output artifacts/runs/new-forward-smoke
```

Output directories must be empty/new to keep experiments reproducible. To regenerate
fixtures: `dama prepare-data --output artifacts/datasets/new-fixtures`.
There are 80 authored synthetic examples, grouped into 56 train / 12 dev / 12 test.
These are software fixtures; human research review is pending. Do not treat their
scores or their limited vocabulary as evidence of generalization.

The smoke network is ~46K random parameters, deliberately different from the
23.72M proposed pretrained model. Its latency is not DAMA's expected latency.
No base-model weights were downloaded locally. Resource estimates count model
weights only; inference should be bounded and explicitly measured later.

## Colab and inference after training

Open the [Colab notebook](https://colab.research.google.com/github/s41r4j/dama/blob/main/notebooks/DAMA_System_One_Colab.ipynb)
and select a GPU runtime. Run its first code cell to clone/update this repository,
install the isolated environment and validate the model code and fixtures. Rerun
the same cell after code fixes are pushed. No source ZIP or Drive mount is needed.
The cell is also available as `notebooks/colab_setup_cell.py` for an existing notebook.
See [the Colab training guide](docs/colab-training.md) for training and persistence.

An optional offline upload archive can still be created with:

```bash
python scripts/package_colab.py --output artifacts/bundles/dama-model-colab.zip
```

The notebook calls portable commands, inspects CUDA/BF16/VRAM, and runs the explicit
three-step cloud smoke test in section 2 without Drive permissions. Mount Drive
before longer runs to persist full resume state. Research training remains gated
behind a separate user action. macOS and CPU training are refused.

After downloading a trained export:

```bash
dama predict --artifact path/to/export --input fixtures/inputs/correction.json --device cpu
dama evaluate --artifact path/to/export --input fixtures/model-v1/test.jsonl --output artifacts/runs/trained-test --repeats 3
```

A trained artifact contains exact encoder revision, all encoder/head weights,
tokenizer, model config, metrics, dataset manifest and file hashes. Inference loads
it offline and rejects hash mismatches. Untrained exports require explicit opt-in.
Model outputs are decisions for a caller to apply; they are not executed here.

## Optional label tooling

`python scripts/teacher_requests.py --data fixtures/model-v1 --output artifacts/datasets/teacher-requests`
prepares train-only prompts **offline**, without sending them or consuming a budget.
Imported responses can be checked with `scripts/validate_teacher_labels.py`; it logs
rejections and writes a review sample. Human review and a new dataset manifest are
required before using accepted labels. Test questions/answers are never supplied.

## Status and remaining work

Local: active unit tests, frozen/LoRA forward paths, schema/mask/span checks,
checkpoint round trips, fixture manifests, notebook syntax, CLI and Mac training
refusal verified. [Validation report](docs/validation.md) records exact results.

Cloud: Python 3.13/T4 setup, CUDA availability and all 35 setup tests passed.
Actual pretrained-encoder inference, optimizer steps, resume after interruption
and trained quality/latency are **unverified**.
Expand/review labels, measure the frozen-head baseline, then compare context-layer
and LoRA ablations on a fixed held-out test. Confidence calibration and downstream
answer/cost hypotheses remain research work.

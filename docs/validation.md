# Local validation — 2 October 2026

The active deliverable is a forward-only System One **model implementation**, not
an agent application. The user authorized selecting the architecture for speed
and accuracy; that choice will be decided from measured held-out performance.

## Measured software checks

Python 3.12.14, macOS arm64, 8 reported CPU cores. Approximately 98 GiB disk free
at final check. Initial vm_stat showed substantial compression; no local base-model
weights, training, optimizer steps, distillation or teacher/API batches were run.

32 active tests passed in 2.18 seconds in the final test run before packaging.
They cover strict contracts, task loss masks, zero/padded candidate handling,
permutation equivariance, scope/future-data rejection, span offsets, bounded-input
abstention, safe target/archive outputs, non-finite score abstention, frozen and
LoRA forward paths, pooling/context ablations, dataset grouping/hashes, precision
selection, artifact hashes/round trips and the Mac training guard. Tests use a
tiny random encoder and controlled logits; no backward/optimizer step is invoked.

Package editable installation and wheel/sdist build passed. `pip check` reports
no broken requirements. All notebook code cells and active scripts parse. Dataset
validation passed: 80 synthetic fixtures, 56 train / 12 dev / 12 test. Five training
configs validate, including the stronger BGE frozen/LoRA variants. Teacher-request
preparation produced 56 train-only requests offline, with zero API calls.

The random-network smoke ran 8 fixture examples, three repetitions after one
excluded warmup. It recorded 24 predictions, 1,938 total input bytepiece/WordPiece
tokens as counted by its toy tokenizer and zero generated tokens. Operation
accuracy was 0 on this selected subset; random scores mostly abstained. Median
~0.70 ms, p95 ~0.76 ms, and process peak RSS ~280 MiB were observed for the **46K
random test network**. These are not latency/accuracy results for MiniLM or BGE,
and must not be used as evidence that DAMA is fast or accurate. Raw results are
in local `artifacts/runs/model-forward-final/` with untrained status; generated
results are excluded from Git.

Architecture counts were checked with meta tensors from pinned official configs:
MiniLM variant: 23,717,392 total / 1,004,176 trainable.
BGE frozen variant: 111,329,552 total / 1,847,312 trainable.
BGE LoRA variant: 111,624,464 total / 2,142,224 trainable.
Counts are static architecture measurements, not quality or performance results.

## Validation commands

The original results above were recorded before the folder reorganization.
Commands below use the current project-root paths.

From the project root using its project-local Python:

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m build --no-isolation --outdir artifacts/bundles/packages .
.venv/bin/python -m pip check
TOKENIZERS_PARALLELISM=false .venv/bin/dama test
.venv/bin/dama resources
.venv/bin/dama validate-config --config configs/heads.json
.venv/bin/dama validate-config --config configs/lora.json
.venv/bin/dama validate-config --config configs/no-context-ablation.json
.venv/bin/dama validate-config --config configs/accuracy.json
.venv/bin/dama validate-config --config configs/accuracy-frozen.json
.venv/bin/dama validate-data --input fixtures/model-v1
TOKENIZERS_PARALLELISM=false .venv/bin/dama model-smoke --output artifacts/runs/model-forward-final
.venv/bin/python scripts/teacher_requests.py --data fixtures/model-v1 --output /tmp/dama-model-teacher-requests
```

The smoke output directory now exists, so choose a new output when repeating.
`prepare-data` was also run to create `fixtures/model-v1`, using the same interpreter.
The notebook's cells were AST-checked; they were not executed on Colab.

Mac refusal check (expected exit 2, before model/optimizer initialization):

```bash
.venv/bin/dama train --config configs/heads.json \
  --data fixtures/model-v1 --output /tmp/dama-must-not-train
```

Result: `Training on this Mac is forbidden. Run this entry point on Colab CUDA.`
No training output directory was created by this command.

## Unverified and remaining research work

Colab setup repair: the user's runtime log confirms a T4 GPU and failure in
`venv`'s `ensurepip` subprocess. Setup now uses `--without-pip` and the notebook's
pip `--python` option, including for partial environments. The target environment
receives its own pinned pip through the requirements file. An offline integration
check installs a local probe wheel into a pip-less environment and verifies that
a retry preserves it. All 33 tests passed in 3.50s locally. Notebook subprocess
output is streamed through Python stdout for readable Colab diagnostics. Actual
CUDA setup still needs a successful user rerun.

Model-only repository preparation: the superseded orchestration prototype was
removed; reference collections and historical snapshots are excluded from Git.
The notebook now fetches GitHub updates and runs isolated setup, config/data
validation and tests in one cell. After these changes, 32 tests passed in 2.21s,
the dataset manifest validated, `pip check` passed, and all seven notebook code
cells and scripts parsed. The setup cell has not yet executed on a Colab GPU.

Colab runtime follow-up: the user's screenshot shows Python 3.13.15 and the old
notebook's 3.12 upper-bound rejection, before model code or CUDA inspection ran.
The notebook now accepts 3.13. Official PyPI Torch 2.6.0 and NumPy 2.2.6 release
files include CPython 3.13 Linux x86-64 wheels. This is package availability evidence,
not successful runtime execution. Setup now installs in a separate venv using the
kernel's current Python and records its dependency/import/CUDA check output.

Actual pretrained model inference, CUDA environment execution, GPU memory/latency,
optimizer updates, interruption/resume equivalence and trained quality remain
unverified. Colab commands and notebook are prepared, not advertised as cloud-tested.
The synthetic fixtures are insufficient for training a reliable general controller;
review and expand labels with hard negatives, corrections and linguistic diversity.
The validator relies on dataset authors marking paraphrase/source families correctly;
it cannot automatically discover unannotated semantic overlap.

Confidence outputs are uncalibrated. Default decision thresholds are provisional.
Calibration, selective-risk curves, uncertainty intervals, annotator agreement and
accuracy/speed across input lengths/candidate counts are future measured experiments.
No end-to-end answer accuracy, dollar saving, Jev-equivalent capability, MemRouter
reproduction, or novelty claim is established. Full-context/agent comparisons belong
to later integration after the model milestone, per the user's scope correction.

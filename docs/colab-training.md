# Train DAMA's System One model on Google Colab

No training is allowed on this Mac. `dama train` refuses macOS before constructing
an optimizer, and refuses non-CUDA runtimes. Remote access/SSH is not configured.

Run local packaging commands from the `DAMA/` project root.

## 1. Open the notebook and choose GPU

Open the [DAMA notebook in Colab](https://colab.research.google.com/github/s41r4j/dama/blob/main/notebooks/DAMA_System_One_Colab.ipynb),
select a GPU runtime, and run the first code cell. It clones the model repository
into `/content/dama` on first use, or pulls updates on subsequent runs. It then
installs dependencies, checks CUDA, validates config/data and runs software tests.
No source ZIP upload or Drive permission is needed for setup. Rerun this same
cell after a code fix is pushed. Local edits in the checkout cause an explicit
stop instead of being overwritten. For an existing notebook, paste the contents
of `notebooks/colab_setup_cell.py` into one cell.

This cell performs no training. Training outputs can later be persisted to Drive
using the separate mount cell. The Colab runtime itself is temporary.

## 2. Install and validate

The notebook runs `scripts/bootstrap_colab.py`, installing the pinned versions
and editable package into `/content/dama-env` using Colab's current Python version.
Python 3.10–3.13 is accepted; the notebook no longer rejects 3.13 merely because
local validation used 3.12. Torch 2.6.0 and NumPy 2.2.6 publish CPython 3.13 Linux
x86-64 wheels. Full import/dependency/CUDA checks still have to pass in Colab.
**CUDA execution has not been tested in this session.** No environment is
advertised as CUDA-verified. No kernel restart or Python downgrade is required.
Setup writes `/content/dama-setup.log` and starts no training.
The environment is created with `venv --without-pip`; the notebook interpreter's
pip manages it through `--python`, so Colab's missing/broken `ensurepip` does not
block setup. Re-running repairs a partial environment without clearing packages.

Notebook commands use `/content/dama-env/bin/python -m dama`. When using a Colab
terminal, prefix the CLI examples below with that path instead of bare `dama`.
Colab upload and Drive mount still run in the notebook kernel environment.

Validate configs and the dataset. Start with `configs/heads.json`: frozen MiniLM,
192-wide contextualizer, two context layers, max 256 tokens per text, max 32
candidates, microbatch 4, accumulation 4, head LR 1e-3, checkpoint every 50 steps.
No training data is inferred from evaluation QA. All fixtures are explicitly
synthetic, license CC0 for newly authored text; private data is excluded.

For real data, use the same JSONL Example schema with source/provenance/license,
observed-at snapshot, candidate IDs and target labels. Provide `train.jsonl`,
`dev.jsonl`, `test.jsonl`, plus a manifest with each file's SHA-256. Assign related
paraphrases to one family; keep all families/conversations/users/projects disjoint.
The validator enforces group/hash/schema/scope/time boundaries; it cannot infer
semantic lineage that a dataset author failed to annotate. Review labels before
training. The 80 shipped examples only test the mechanics.

## 3. Explicit cloud smoke test

Mount Drive and choose a new persistent output directory. Run:

```bash
dama train --config configs/heads.json --data fixtures/model-v1 \
  --output /content/drive/MyDrive/DAMA/model-runs/smoke-v1 --smoke
```

This performs at most three optimizer steps **on CUDA only**, then dev evaluation
and artifact export. Inspect `training.jsonl`, `run_manifest.json`, `metrics.json`,
peak VRAM and `export/manifest.json`. It is not a useful training run or quality
result. Sequence overflow is rejected rather than silently dropping evidence.
Conservative settings are a starting point, not a guarantee against OOM.

`precision=auto` selects BF16 when supported, otherwise FP16 with GradScaler.
FP32 is available explicitly. Unsupported requested BF16 fails. Memory can be
reduced with microbatch 1 or fewer candidates/shorter *prepared* inputs; changing
limits requires rebuilding/revalidating the dataset rather than silently truncating.

## 4. Real run — separate user action

Replace the fixtures with reviewed, sufficiently varied data and choose a fresh
run directory. Set `RUN_REAL_TRAINING=True` in the separate notebook cell only
when ready. The cell also refuses a synthetic dataset unless the user explicitly
sets `ALLOW_SYNTHETIC_EXPERIMENT=True`. Portable command:

```bash
dama train --config configs/heads.json --data /content/reviewed-data \
  --output /content/drive/MyDrive/DAMA/model-runs/heads-v1
```

Use `configs/no-context-ablation.json` for the smaller decision-head ablation.
Use `configs/accuracy-frozen.json` and `configs/accuracy.json` to compare a
stronger BGE-base encoder before/after encoder LoRA (MIT license, exact revision
in config, CLS pooling as documented by its official model card). These use
microbatch 1 with accumulation 16. No superiority is assumed; select the fastest
variant meeting the declared accuracy constraint on held-out decisions.
Use `configs/lora.json` for MiniLM query/value LoRA after the frozen baseline is
measured. LoRA rank/alpha, encoder/head learning rates, sequence bound, microbatch,
accumulation, epochs, seed and save frequency are validated/exposed. No causal-LM
QLoRA or RL path is claimed. Small encoder head fitting is supervised specialization,
not foundation-model pretraining. No time/quality guarantee is made.

## 5. Resume after interruption

Only checkpoint directories with a `COMPLETE` marker are valid. Select the latest
completed checkpoint, not a partially written one:

```bash
dama train --config configs/heads.json --data /content/reviewed-data \
  --output /content/drive/MyDrive/DAMA/model-runs/heads-v1 \
  --resume /content/drive/MyDrive/DAMA/model-runs/heads-v1/checkpoint-000050
```

Config and dataset-manifest fingerprints must match exactly. Checkpoints retain
model, optimizer, scaler, Python/NumPy/CPU/CUDA RNG states, optimizer-step number,
epoch and next microbatch index. Checkpoints occur at accumulation boundaries.
Their hashes and completion marker are checked before loading state. The final
checkpoint can be at a configured terminal step; it will refuse to resume beyond
a completed step limit. Increase total limits only as a deliberately new experiment
(the current strict resume contract refuses arbitrary config changes).

## 6. Evaluate and export

The trainer evaluates **dev only** and exports automatically. Test labels stay
held out. After freezing thresholds/model selection, evaluate once on test:

```bash
dama evaluate --artifact /content/drive/MyDrive/DAMA/model-runs/heads-v1/export \
  --input /content/reviewed-data/test.jsonl \
  --output /content/drive/MyDrive/DAMA/model-runs/heads-v1/test-eval --device cuda --repeats 3
```

A checkpoint can be exported independently without optimizer steps:

```bash
dama export --checkpoint /content/drive/MyDrive/DAMA/model-runs/heads-v1/checkpoint-000050 \
  --output /content/drive/MyDrive/DAMA/model-runs/heads-v1/export-from-checkpoint
```

Compare frozen-head, no-context and LoRA variants using identical splits and limits.
An unfitted decision head has random scores; it is not a pretrained memory manager.
Report it as an untrained reference if desired, never as Qwen/Jev capability.
Use stratified errors, write precision/recall, target/span metrics, latency, input
size, candidate count and ECE. Confidence remains uncalibrated; fit/validate a
calibration method on dev before claiming probabilistic reliability. Final report
hypotheses about answer accuracy/cost require a later controlled downstream study.

## 7. Load locally later

Download the full `export/` directory (or zip it in Colab); preserve its hashes.
It includes all frozen encoder weights, trained modules, exact base revision,
tokenizer, config, dataset manifest and metrics. The optional LoRA variant is saved
as a complete self-contained model state, not an incompatible decoder/GGUF adapter.
CPU loading is the conservative supported local path:

```bash
dama predict --artifact downloaded-export --input fixtures/inputs/correction.json --device cpu
```

Measure actual local memory and latency before selecting MPS or larger batches.
Keep model weights/checkpoints/private datasets out of source control. No secrets
are written to source, notebooks or logs. The built-in pipeline makes no paid calls.

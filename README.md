# DAMA

Research implementation of DAMA, a memory system for language model agents. The repository contains the memory agent, model training and export pipeline, benchmark adapters, and experiment scripts.

**Author:** [s41r4j](https://github.com/s41r4j)

**Code:** [github.com/s41r4j/dama](https://github.com/s41r4j/dama)

## Repository structure

| Path | Contents |
| --- | --- |
| `code/dama_agent/` | Memory storage, retrieval, update decisions, and agent tools |
| `code/dama_train/` | Data preparation, training, calibration, and ONNX export |
| `code/dmbm/` | Benchmark source adapters and memory system comparisons |
| `code/notebooks/` | Training notebook for Colab and Kaggle |
| `code/kaggle/` | Kaggle configuration and execution scripts |
| `code/tests/` | Unit tests and offline fixtures |
| `experiments/` | Versioned paper experiment scripts and analysis code |

## Installation

Use Python 3.12. Install a PyTorch build appropriate for your hardware before installing the training dependencies.

```bash
git clone https://github.com/s41r4j/dama.git
cd dama
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -r code/requirements.txt
export PYTHONPATH="$PWD/code"
```

For CPU inference with an exported ONNX bundle, install `code/requirements-runtime.txt` instead of the training dependencies.

## Training and evaluation

The [training notebook](code/notebooks/DAMA_train.ipynb) supports Colab and Kaggle. The command line pipeline can also be run in stages:

```bash
python -m dama_train download
python -m dama_train build
python -m dama_train train-retriever
python -m dama_train train-decision
python -m dama_train train-ops
python -m dama_train calibrate
python -m dama_train export
```

See [the implementation guide](code/README.md) for configuration, dataset splits, calibration, and inference examples. Experiment entry points are maintained under `experiments/`; their dependencies, model access, and input paths are specified in the scripts. Benchmark data, trained weights, and generated memory packets must be obtained or generated separately.

## Verification

```bash
python -m pip install pytest
PYTHONPATH=code python -m pytest code/tests
python -m dama_train smoke
```

## Research use

This repository provides source code accompanying the DAMA research project. Reported results depend on the experiment version, dataset split, reader model, and configuration. Dataset and pretrained model use remains subject to the respective providers' terms.

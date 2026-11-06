# Kaggle setup (all private, account `s41r4j`)

Every folder here is one Kaggle item; nothing is downloaded to this machine. Run commands from the repo root.

| Folder | Kaggle item | Role | Command |
|---|---|---|---|
| `dataset/` | Dataset **DAMA code** (`s41r4j/dama-code`) | the code, one version per change; its page is the project map | `code/kaggle/publish_code.sh "message"` |
| `training/` | Notebook **DAMA Tier1 Training** (`dama-tier1-training`) | Level 1: train, calibrate, export; registers each checkpoint in the Model (T4, ~35 min) | `code/kaggle/push.sh "L1.x: message"` (publishes code, then runs) |
| `model/` | Model **DAMA Tier-1 Models** (`dama-tier1`) | `other/tier1` = CPU bundle, `transformers/tier1-pytorch` = PyTorch weights, one version per checkpoint | `kaggle models update -p code/kaggle/model` (title/description only) |
| `evaluation/` | Notebook **DAMA Evaluation** (`dama-evaluation`) | Levels 2 + 3 in one run: the Memory Agent on held-out LoCoMo with the CPU bundle and with the PyTorch weights on GPU (4 pillars), then answer accuracy of both packets and the no-memory / gold-evidence baselines (T4, ~40 min) | `kaggle kernels push -p code/kaggle/evaluation --accelerator NvidiaTeslaT4` |
| `register/` | (not kept on Kaggle) | fallback: register a training output in the Model if its upload cell failed | `kaggle kernels push -p code/kaggle/register` |
| `../../experiments/v0_2026-10-22/` | Notebook **DAMA Paper v0** (`dama-paper-v0`) | the paper experiments (baselines, Mem0, topology, ablations, conformal checks) | see `dama_paper_v0.py` |

Order for a new checkpoint: `push.sh` → (training registers itself) → pin the new Model versions in
`evaluation/kernel-metadata.json` → push `evaluation`.
Read any result without downloading: `.venv/bin/kaggle kernels logs s41r4j/<notebook>`; a single small file:
`.venv/bin/kaggle kernels output s41r4j/<notebook> --file-pattern summary.json -p <scratch dir>`.

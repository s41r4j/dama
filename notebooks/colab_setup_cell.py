# Paste this entire file into one Colab code cell. Rerun after code updates.
# Setup and software checks only; no training or Drive mount.
from pathlib import Path
import json
import os
import platform
import subprocess
import sys

if platform.system() != "Linux":
    raise RuntimeError("Run this cell in the Colab Linux GPU runtime")
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise RuntimeError("Python 3.10–3.13 required; newer versions need validation")
print("Notebook Python:", sys.version, flush=True)
subprocess.run(["nvidia-smi"], check=True)
SOURCE = Path("/content/dama")
REPO = "https://github.com/s41r4j/dama.git"
if SOURCE.exists():
    origin = subprocess.check_output(
        ["git", "-C", str(SOURCE), "remote", "get-url", "origin"], text=True
    ).strip()
    if origin != REPO:
        raise RuntimeError("/content/dama belongs to another repository")
    changes = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    )
    if changes.strip():
        raise RuntimeError("Save local checkout edits before fetching updates:\n" + changes)
    subprocess.run(["git", "-C", str(SOURCE), "pull", "--ff-only", "origin", "main"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", REPO, str(SOURCE)], check=True)
subprocess.run(["git", "-C", str(SOURCE), "log", "-1", "--oneline"], check=True)
os.chdir(SOURCE)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, str(SOURCE / "scripts/bootstrap_colab.py"),
                "--source", str(SOURCE)], check=True)
DAMA_PYTHON = Path("/content/dama-env/bin/python")
def dama(*args):
    subprocess.run([str(DAMA_PYTHON), "-m", "dama", *map(str, args)], check=True)
CONFIG = SOURCE / "configs/heads.json"
DATA = SOURCE / "fixtures/model-v1"
dama("validate-config", "--config", CONFIG)
dama("validate-data", "--input", DATA)
dama("test")
print("DAMA READY. Setup and tests passed; training has not started.", flush=True)

# Paste this entire file into one Colab code cell. Rerun after code updates.
# Setup and software checks only; no training or Drive mount.
from pathlib import Path
import json
import os
import platform
import subprocess
import sys

def run(command):
    # Capturing subprocess output makes it visible through notebook Python stdout.
    with subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError("Command failed: " + " ".join(map(str, command)))

if platform.system() != "Linux":
    raise RuntimeError("Run this cell in the Colab Linux GPU runtime")
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise RuntimeError("Python 3.10–3.13 required; newer versions need validation")
print("Notebook Python:", sys.version, flush=True)
run(["nvidia-smi"])
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
    run(["git", "-C", str(SOURCE), "pull", "--ff-only", "origin", "main"])
else:
    run(["git", "clone", "--depth", "1", "--branch", "main", REPO, str(SOURCE)])
run(["git", "-C", str(SOURCE), "log", "-1", "--oneline"])
os.chdir(SOURCE)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
run([sys.executable, str(SOURCE / "scripts/bootstrap_colab.py"),
     "--source", str(SOURCE)])
DAMA_PYTHON = Path("/content/dama-env/bin/python")
def dama(*args):
    run([str(DAMA_PYTHON), "-m", "dama", *map(str, args)])
CONFIG = SOURCE / "configs/heads.json"
DATA = SOURCE / "fixtures/model-v1"
dama("validate-config", "--config", CONFIG)
dama("validate-data", "--input", DATA)
dama("test")
print("DAMA READY. Setup and tests passed; training has not started.", flush=True)

#!/usr/bin/env python3
"""Install DAMA using the current Python in isolation; never train in setup."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


def prepare_environment(run, interpreter, env_dir):
    # Colab's system Python can create a venv but may lack working ensurepip.
    # Re-running without --clear repairs a partial venv without deleting packages.
    run([interpreter, "-m", "venv", "--without-pip", env_dir])
    python = env_dir / "bin/python"
    pip = [interpreter, "-m", "pip", "--python", str(python)]
    return python, pip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--env", type=Path, default=Path("/content/dama-env"))
    parser.add_argument("--log", type=Path, default=Path("/content/dama-setup.log"))
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("This bootstrap is for the Linux Colab GPU machine only")
    if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
        parser.error("Python 3.10–3.13 required; newer versions need validation")
    source = args.source.resolve()
    if not (source / "pyproject.toml").is_file():
        parser.error("Fetch the DAMA repository or extract the source archive first")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with args.log.open("a", encoding="utf-8") as log:
        def run(command):
            label = "\nRUN: " + " ".join(map(str, command)) + "\n"
            print(label, flush=True)
            log.write(label)
            log.flush()
            with subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  env=environment) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"Command failed; full setup log: {args.log}")

        run(["nvidia-smi"])
        # Reuse Colab's Python version; isolate model libraries from the kernel.
        python, pip = prepare_environment(run, sys.executable, args.env)
        run([python, "-c", "import sys; print(sys.version); "
             f"assert sys.version_info[:2] == {sys.version_info[:2]!r}"])
        run([*pip, "install",
             "-r", source / "scripts/requirements-colab.txt"])
        run([*pip, "install", "-e", source])
        run([*pip, "check"])
        run([python, "-c", "import torch, transformers, peft; "
             "from dama.training import runtime; "
             "print({'torch': torch.__version__, 'transformers': transformers.__version__, "
             "'peft': peft.__version__}); print(runtime())"])
    print(f"SETUP COMPLETE. DAMA Python: {python}. Log: {args.log}", flush=True)


if __name__ == "__main__":
    main()

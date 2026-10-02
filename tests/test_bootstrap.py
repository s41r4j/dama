"""Offline integration check for Colab's partial venv/ensurepip failure."""
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile


def test_partial_environment_without_pip_can_install_and_retry(tmp_path):
    prepare = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                "scripts/bootstrap_colab.py"))["prepare_environment"]
    env = tmp_path / "partial-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(env)], check=True)
    python = env / "bin/python"
    subprocess.run([str(python), "-c",
                    "import importlib.util; assert importlib.util.find_spec('pip') is None"], check=True)

    def run(command):
        subprocess.run(list(map(str, command)), check=True, capture_output=True, text=True)

    python, pip = prepare(run, sys.executable, env)
    # A local wheel exercises real installation into a pip-less environment,
    # without downloading packages or touching the model/training runtime.
    wheel = tmp_path / "dama_bootstrap_probe-0.0.1-py3-none-any.whl"
    info = "dama_bootstrap_probe-0.0.1.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("dama_bootstrap_probe.py", "VALUE = 42\n")
        archive.writestr(f"{info}/METADATA",
                         "Metadata-Version: 2.1\nName: dama-bootstrap-probe\nVersion: 0.0.1\n")
        archive.writestr(f"{info}/WHEEL",
                         "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    run([*pip, "install", "--no-index", "--no-deps", wheel])
    # Retry must preserve already installed packages while repairing venv files.
    python, pip = prepare(run, sys.executable, env)
    run([python, "-c", "import dama_bootstrap_probe; assert dama_bootstrap_probe.VALUE == 42"])
    run([*pip, "check"])

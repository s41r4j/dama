"""Where are we running (Colab, Kaggle, local) and what can the hardware do?"""
from __future__ import annotations
import importlib.util
import json
import os
import platform
from pathlib import Path


def platform_name() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "local"


def default_work_dir() -> Path:
    """Persistent-ish place for data, checkpoints and exports.

    Kaggle keeps /kaggle/working when you save a version. On Colab, mount Google Drive first
    (the notebook does this) so checkpoints survive a disconnect; otherwise /content is used.
    """
    name = platform_name()
    if name == "kaggle":
        return Path("/kaggle/working/dama")
    if name == "colab":
        drive = Path("/content/drive/MyDrive")
        return (drive / "DAMA") if drive.is_dir() else Path("/content/dama_work")
    return Path(os.environ.get("DAMA_WORK_DIR", "dama_work")).resolve()


def _bf16_native(torch) -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch has no including_emulation argument
        return torch.cuda.get_device_capability(0)[0] >= 8


_GPU_CHECK = {}


def gpu_ok() -> bool:
    """True when a GPU exists AND this PyTorch build has kernels for it. New PyTorch wheels (CUDA 12.8+) dropped
    Pascal GPUs, so Kaggle's P100 can be visible but unusable; in that case we warn and fall back to CPU."""
    import torch
    if "ok" in _GPU_CHECK:
        return _GPU_CHECK["ok"]
    ok = torch.cuda.is_available()
    if ok:
        try:
            (torch.ones(8, device="cuda") * 2).sum().item()
        except Exception as e:  # "no kernel image is available for execution on the device"
            print(f"WARNING: {torch.cuda.get_device_name(0)} cannot run this PyTorch build ({str(e)[:120]}). "
                  "On Kaggle choose the 'GPU T4 x2' accelerator instead of P100. Falling back to CPU (very slow).")
            ok = False
    _GPU_CHECK["ok"] = ok
    return ok


def best_device() -> str:
    return "cuda" if gpu_ok() else "cpu"


def device_info() -> dict:
    import torch
    info = {"platform": platform_name(), "python": platform.python_version(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "cuda": torch.cuda.is_available()}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_count=torch.cuda.device_count(), gpu_memory_gb=round(p.total_memory / 2**30, 1),
                    compute_capability=f"{p.major}.{p.minor}", gpu_usable=gpu_ok(), bf16_native=_bf16_native(torch))
    return info


def amp_dtype(torch):
    """bf16 on Ampere+ (A100/L4), fp16 on T4/P100, None (fp32) on CPU."""
    if not gpu_ok():
        return None
    return torch.bfloat16 if _bf16_native(torch) else torch.float16


def print_env():
    print(json.dumps(device_info(), indent=1))

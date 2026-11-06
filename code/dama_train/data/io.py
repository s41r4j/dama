"""JSONL files with a hash manifest, so a run can prove which data it was trained on."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def write_jsonl(path, rows) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "sha256": sha256(path)}


def read_jsonl(path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(directory) -> dict:
    """Raise if any file listed in manifest.json changed since it was built."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, meta in manifest["files"].items():
        if sha256(root / name) != meta["sha256"]:
            raise ValueError(f"{name} does not match manifest.json; rebuild the data")
    return manifest

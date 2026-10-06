"""Self-contained model export: weights, configs, tokenizer and file hashes; loads offline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
from .contracts import ModelConfig
from .model import DAMADecisionModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(model, tokenizer, directory, metrics=None):
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(root / "model.safetensors"))
    (root / "model_config.json").write_text(model.config.model_dump_json(indent=2) + "\n")
    model.encoder.config.save_pretrained(root / "encoder_config")
    tokenizer.save_pretrained(root / "tokenizer")
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    # Attention kernel is recorded so a reload cannot silently switch eager <-> SDPA.
    manifest = {"attention": model.encoder.config._attn_implementation, "parameters": model.parameter_counts(), "metrics": metrics or {}, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_model(directory, device="cpu"):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    root = Path(directory); manifest = json.loads((root / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        if sha(root / relative) != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    config = ModelConfig.model_validate_json((root / "model_config.json").read_text())
    encoder_config = AutoConfig.from_pretrained(root / "encoder_config", local_files_only=True)
    encoder = AutoModel.from_config(encoder_config, attn_implementation=manifest["attention"])
    model = DAMADecisionModel(config, encoder=encoder)
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    return model.to(device).eval(), tokenizer, manifest

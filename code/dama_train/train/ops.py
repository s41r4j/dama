"""Stage 2: train the op model (ADD / UPDATE / DELETE / NOOP) on data/ops/*.jsonl."""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..config import OPS, TrainConfig
from ..data.io import read_jsonl
from ..metrics import macro_f1
from ..models.factory import build_ops
from .loop import train_loop


def make_collate(tokenizer, max_length):
    def collate(rows):
        enc = tokenizer([r["a"] for r in rows], [r["b"] for r in rows], truncation="longest_first", max_length=max_length,
                        padding=True, return_tensors="pt")
        inputs = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        return inputs, {"y": torch.tensor([int(r["y"]) for r in rows])}
    return collate


def op_loss(logits, labels, cfg):
    l = F.cross_entropy(logits, labels["y"])
    return l, {"op": round(float(l), 4)}


@torch.no_grad()
def predict_logits(model, rows, collate, batch_size=128):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(rows), batch_size):
        inputs, _ = collate(rows[i:i + batch_size])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(**{k: v.to(device) for k, v in inputs.items()}).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(OPS)))


def evaluate_ops(model, rows, collate):
    z = predict_logits(model, rows, collate)
    y = np.array([r["y"] for r in rows]); pred = z.argmax(-1)
    m = {"accuracy": float((pred == y).mean()) if len(y) else None, "macro_f1": macro_f1(y, pred, len(OPS))}
    by_src = defaultdict(list)
    for r, p in zip(rows, pred):
        by_src[r["src"]].append(int(p == r["y"]))
    for src, oks in by_src.items():
        m[f"acc_{src}"] = float(np.mean(oks))
    return m


def save_ops(wrapper, tokenizer, directory, meta):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    wrapper.model.save_pretrained(root / "model")
    tokenizer.save_pretrained(root / "tokenizer")
    (root / "meta.json").write_text(json.dumps({"kind": "ops", "ops": OPS, **meta}, indent=1) + "\n")


def train_ops(data_dir, run_dir, cfg: TrainConfig, max_val_rows=6000, log=print):
    data, run = Path(data_dir), Path(run_dir)
    train_rows = read_jsonl(data / "ops" / "train.jsonl")
    val_rows = read_jsonl(data / "ops" / "val.jsonl")[:max_val_rows]
    texts = [r["a"] for r in train_rows[:5000]] + [r["b"] for r in train_rows[:5000]]
    model, tokenizer, warm = build_ops(cfg, tiny_texts=texts, work_dir=run)
    log(f"backbone {cfg.backbone if not cfg.tiny else 'tiny-random'}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"op head warm-started from NLI head: {warm}")
    collate = make_collate(tokenizer, cfg.max_length)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n")

    def save_best(m, directory, metrics):
        save_ops(m, tokenizer, directory, {"backbone": cfg.backbone, "max_length": cfg.max_length, "val": metrics})

    return train_loop(model, train_rows, val_rows, cfg, run, collate=collate, loss_fn=op_loss,
                      evaluate=lambda m, rows: evaluate_ops(m, rows, collate), metric="macro_f1", save_best=save_best, log=log)

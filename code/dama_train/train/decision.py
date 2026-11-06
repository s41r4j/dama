"""Stage 1: train the decision model (rel / sal / need / route heads) on data/decision/*.jsonl."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..config import ROUTES, TrainConfig
from ..data.io import read_jsonl
from ..metrics import auc, best_f1, sigmoid
from ..models.decision import save_decision
from ..models.factory import build_decision
from .loop import train_loop

TASK = {"rel": 0, "sal": 1, "need": 2, "route": 3}


def make_collate(tokenizer, max_length):
    def collate(rows):
        enc = tokenizer([r["a"] for r in rows], [r["b"] for r in rows], truncation="longest_first", max_length=max_length,
                        padding=True, return_tensors="pt")
        inputs = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        task = torch.tensor([TASK[r["task"]] for r in rows])
        y_bin = torch.tensor([float(r["y"]) if r["task"] != "route" else -1.0 for r in rows])
        y_route = torch.tensor([r["y"] if r["task"] == "route" else [-1.0] * len(ROUTES) for r in rows], dtype=torch.float32)
        return inputs, {"task": task, "y_bin": y_bin, "y_route": y_route}
    return collate


def decision_loss(logits, labels, cfg):
    task, parts, total = labels["task"], {}, logits.new_zeros(())
    for name, t in (("rel", 0), ("sal", 1), ("need", 2)):
        m = task == t
        if m.any():
            l = F.binary_cross_entropy_with_logits(logits[m, t], labels["y_bin"][m])
            parts[name] = l.detach()
            total = total + cfg.loss_weights.get(name, 1.0) * l
    m = task == 3
    if m.any():
        l = F.binary_cross_entropy_with_logits(logits[m, 3:], labels["y_route"][m])
        parts["route"] = l.detach()
        total = total + cfg.loss_weights.get("route", 1.0) * l
    return total, parts


@torch.no_grad()
def predict_logits(model, rows, collate, batch_size=128):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(rows), batch_size):
        inputs, _ = collate(rows[i:i + batch_size])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(**{k: v.to(device) for k, v in inputs.items()}).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 3 + len(ROUTES)))


def evaluate_decision(model, rows, collate):
    z = predict_logits(model, rows, collate)
    m = {}
    for name, t in (("rel", 0), ("sal", 1), ("need", 2)):
        idx = [i for i, r in enumerate(rows) if r["task"] == name]
        if idx:
            y = [rows[i]["y"] for i in idx]
            m[f"{name}_auc"] = auc(y, z[idx, t])
            m[f"{name}_best_f1"] = best_f1(y, sigmoid(z[idx, t]))["f1"]
    idx = [i for i, r in enumerate(rows) if r["task"] == "route"]
    if idx:
        y = np.array([rows[i]["y"] for i in idx]); p = sigmoid(z[idx, 3:]) >= 0.5
        tp = (p & (y == 1)).sum(); m["route_micro_f1"] = float(2 * tp / max(1, p.sum() + (y == 1).sum()))
    keys = [k for k in ("rel_auc", "sal_auc", "need_auc", "route_micro_f1") if m.get(k) is not None]
    m["score"] = float(np.mean([m[k] for k in keys])) if keys else None
    return m


def train_decision(data_dir, run_dir, cfg: TrainConfig, max_val_rows=6000, log=print):
    data, run = Path(data_dir), Path(run_dir)
    train_rows = read_jsonl(data / "decision" / "train.jsonl")
    val_rows = read_jsonl(data / "decision" / "val.jsonl")[:max_val_rows]
    texts = [r["a"] for r in train_rows[:5000]] + [r["b"] for r in train_rows[:5000]]
    model, tokenizer = build_decision(cfg, tiny_texts=texts, work_dir=run)
    log(f"backbone {cfg.backbone if not cfg.tiny else 'tiny-random'}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"rel head warm-started from the cross-encoder: {model.warm_started}")
    collate = make_collate(tokenizer, cfg.max_length)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n")

    def save_best(m, directory, metrics):
        save_decision(m, tokenizer, directory, meta={"backbone": cfg.backbone, "max_length": cfg.max_length, "val": metrics})

    state = train_loop(model, train_rows, val_rows, cfg, run, collate=collate, loss_fn=decision_loss,
                       evaluate=lambda m, rows: evaluate_decision(m, rows, collate), metric="score", save_best=save_best, log=log)
    return state

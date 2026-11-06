"""One training loop for both models, built for free notebooks.

* mixed precision: fp16 + loss scaling on T4/P100, bf16 on Ampere, fp32 on CPU
* resumable: a checkpoint (model, optimizer, scheduler, scaler, data position) is written every `save_every`
  steps and when the time budget runs out; re-running the same cell continues where it stopped
* time budget: stops cleanly after `max_minutes` so a free session never dies mid-write
* best model on validation is saved to run_dir/best via `save_best`
"""
from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from ..env import amp_dtype, best_device


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fmt(parts: dict) -> dict:
    """Loss parts stay on the GPU during training; convert only when logging (avoids a sync every step)."""
    return {k: round(float(v), 4) for k, v in parts.items()}


def train_loop(model, train_rows, val_rows, cfg, run_dir, *, collate, loss_fn, evaluate, metric, save_best, log=print):
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    device = best_device()
    model.to(device)
    dtype = None if getattr(cfg, "precision", "auto") == "fp32" else amp_dtype(torch)
    torch.manual_seed(cfg.seed)

    head = model.head_parameters()
    head_ids = {id(p) for p in head}
    encoder = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": encoder, "lr": cfg.lr}, {"params": head, "lr": cfg.head_lr}], weight_decay=cfg.weight_decay)
    per_epoch = math.ceil(len(train_rows) / (cfg.batch_size * cfg.grad_accum))
    total = max(1, per_epoch * cfg.epochs)
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    state = {"step": 0, "epoch": 0, "pos": 0, "best": -math.inf, "bad_evals": 0, "history": [], "done": False}
    ckpt = run / "checkpoint.pt"
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"]); sched.load_state_dict(saved["sched"])
        scaler.load_state_dict(saved["scaler"]); state = saved["state"]
        log(f"resumed from step {state['step']} (epoch {state['epoch']}, row {state['pos']})")
        if state.get("done"):
            log("this run already finished; delete the run folder to train again")
            return state

    def checkpoint():
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "state": state}, ckpt)

    def run_eval():
        model.eval()
        with torch.no_grad():
            m = evaluate(model, val_rows)
        model.train()
        score = m[metric] if m.get(metric) is not None else -math.inf
        improved = score > state["best"]
        if improved:
            state["best"], state["bad_evals"] = score, 0
            save_best(model, run / "best", m)
        else:
            state["bad_evals"] += 1
        state["history"].append({"step": state["step"], "epoch": state["epoch"], **{k: v for k, v in m.items() if isinstance(v, (int, float)) or v is None}})
        with (run / "log.jsonl").open("a") as f:
            f.write(json.dumps(state["history"][-1]) + "\n")
        log(f"step {state['step']}/{total} | val {metric} {score:.4f}{' (best)' if improved else ''} | "
            + " ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float) and k != metric)[:300])

    log(f"training on {device} ({dtype or 'fp32'}), {len(train_rows)} rows, {per_epoch} steps/epoch, {total} steps total")
    started, seen, micro, last_log, bad_batches = time.time(), 0, 0, time.time(), 0
    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(state["epoch"], cfg.epochs):
        state["epoch"] = epoch
        order = np.random.RandomState(cfg.seed + epoch).permutation(len(train_rows))
        for start in range(state["pos"], len(order), cfg.batch_size):
            rows = [train_rows[i] for i in order[start:start + cfg.batch_size]]
            inputs, labels = collate(rows)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            with torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(**inputs)
            loss, parts = loss_fn(logits.float(), labels, cfg)
            if not torch.isfinite(loss):  # fp16 overflow in the forward pass: skip this batch, give up if it keeps happening
                bad_batches += 1
                opt.zero_grad(set_to_none=True)
                if bad_batches > 20:
                    raise RuntimeError(f"non-finite loss {bad_batches} times (last at step {state['step']}: {_fmt(parts)}). "
                                       "Re-run with precision='fp32' (CLI: --fp32).")
                continue
            scaler.scale(loss / cfg.grad_accum).backward()
            micro += 1
            seen += len(rows)
            state["pos"] = start + cfg.batch_size
            if micro % cfg.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            state["step"] += 1
            if time.time() - last_log > 30:
                rate = seen / (time.time() - started)
                eta = (total - state["step"]) * cfg.batch_size * cfg.grad_accum / max(rate, 1e-9) / 60
                log(f"step {state['step']}/{total} loss {loss.item():.4f} {_fmt(parts)} | {rate:.0f} rows/s | eta {eta:.1f} min")
                last_log = time.time()
            if state["step"] % cfg.eval_every == 0:
                run_eval()
                if state["bad_evals"] >= cfg.patience:
                    log("early stop: validation stopped improving")
                    state["done"] = True; checkpoint(); return state
            if state["step"] % cfg.save_every == 0:
                checkpoint()
            if (time.time() - started) / 60 > cfg.max_minutes:
                if not (run / "best").exists():
                    run_eval()  # so later stages always have a usable model, even from a partial run
                checkpoint()
                log(f"time budget of {cfg.max_minutes} min reached at step {state['step']}; run the same cell again to resume")
                return state
        state["pos"] = 0
        state["epoch"] = epoch + 1
        if not state["history"] or state["history"][-1]["step"] != state["step"]:
            run_eval()
        checkpoint()
    state["done"] = True
    checkpoint()
    log(f"finished: best val {metric} {state['best']:.4f} in {(time.time() - started) / 60:.1f} min")
    return state

"""Supervised multitask training on a CUDA GPU (Colab). Keeps the best dev epoch and exports it."""
from __future__ import annotations
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
from .contracts import TrainConfig


def require_cuda():
    import torch
    if platform.system() == "Darwin" or not torch.cuda.is_available():
        raise RuntimeError("Training runs on a CUDA GPU (e.g. Colab), not on this machine")
    return torch


def choose_precision(requested, torch):
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)  # T4 emulates BF16 slowly -> FP16
    if requested == "bf16" and not native_bf16:
        raise ValueError("BF16 is not natively supported on this GPU")
    name = ("bf16" if native_bf16 else "fp16") if requested == "auto" else requested
    return name, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def train(config, data_dir, output, *, smoke=False, log=print):
    """Train from a TrainConfig (or a path to its JSON) on data_dir/{train,dev}.jsonl; export the best dev epoch to output/export."""
    torch = require_cuda()
    from transformers import AutoTokenizer
    from .artifacts import save_model
    from .batching import Collator, to_device
    from .data import load_split
    from .evaluate import evaluate, score
    from .inference import Predictor
    from .model import DAMADecisionModel, decision_loss

    if not isinstance(config, TrainConfig):
        config = TrainConfig.model_validate_json(Path(config).read_text())
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root} is not empty; use a new run directory")
    root.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)

    train_rows, dev_rows = load_split(data_dir, "train"), load_split(data_dir, "dev")
    if smoke:
        train_rows, dev_rows = train_rows[:48], dev_rows[:32]
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder, revision=config.model.encoder_revision)
    collator = Collator(tokenizer, config.model)
    every = train_rows + dev_rows
    for i in range(0, len(every), 64):  # fail on overflow/span misalignment before spending GPU time
        collator(every[i:i + 64])
    model = DAMADecisionModel(config.model).cuda()

    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    groups = [{"params": heads, "lr": config.learning_rate}] + ([{"params": adapters, "lr": config.encoder_learning_rate}] if adapters else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = -(-len(train_rows) // config.batch_size)
    total = 3 if smoke else steps_per_epoch * config.epochs
    warmup = max(1, int(total * config.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))
    precision, dtype = choose_precision(config.precision, torch)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    log(f"{len(train_rows)} train / {len(dev_rows)} dev rows, {total} steps, {precision}, parameters {model.parameter_counts()}")

    best, best_score, history, step = None, -1.0, [], 0
    for epoch in range(1 if smoke else config.epochs):
        model.train(); started = time.time()
        order = np.random.RandomState(config.seed + epoch).permutation(len(train_rows))
        for i in range(0, len(order), config.batch_size):
            if step >= total:
                break
            batch = collator([train_rows[j] for j in order[i:i + config.batch_size]])
            labels = {k: v.cuda() for k, v in batch["labels"].items()}
            with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
                outputs = model(**to_device(batch, "cuda"))
            loss, parts = decision_loss(outputs, labels, batch["model_inputs"]["candidate_mask"].cuda())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {parts}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(heads + adapters, config.grad_clip)
            scaler.step(optimizer); scaler.update(); scheduler.step(); step += 1
            if step % 50 == 0 or smoke:
                log(f"step {step}/{total} loss {float(loss):.4f}")
        metrics = evaluate(Predictor(model, tokenizer, device="cuda"), dev_rows)
        current = score(metrics)
        history.append({"epoch": epoch, "step": step, "dev_score": current, "seconds": time.time() - started, **{k: v for k, v in metrics.items() if k != "confusion"}})
        log(f"epoch {epoch}: dev score {current:.4f} | " + " | ".join(f"{k} {metrics[k]:.3f}" for k in ["operation_macro_f1", "target_accuracy", "span_f1", "retrieval_f1", "fallback_accuracy"] if metrics[k] is not None))
        if current > best_score:  # keep only trainable weights in memory; the frozen encoder never changes
            best_score, best = current, {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    model.load_state_dict(best, strict=False)
    summary = {"config": config.model_dump(), "best_dev_score": best_score, "history": history, "steps": step, "smoke": smoke,
               "precision": precision, "gpu": torch.cuda.get_device_name(), "peak_vram_bytes": torch.cuda.max_memory_allocated()}
    save_model(model, tokenizer, root / "export", metrics=summary)
    (root / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model, tokenizer, summary

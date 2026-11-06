"""Stage 1b (L1.1): fine-tune the first-stage dense retriever (bge-small) on the train split.

Why: in L1.0 every gold turn was in the hybrid top-8 for only 55% of test questions, so the rel head and the
conformal threshold could not reach their targets at a k that fits the latency budget. The retriever, not the
reranker, was the ceiling.

How (standard dense-retrieval fine-tuning, as in DPR / bge / ANCE):
* pairs (question, gold turn) from the train dialogues (data/dialogues/train.jsonl)
* L1.2: hard negatives are mined with the base dense model itself (its top-30 non-gold turns of the same dialogue),
  which are the turns the first stage actually confuses; L1.1 used BM25 negatives
* L1.2: each turn is embedded as a key with the start of the previous turn (retrieval.memory_key)
* InfoNCE over the batch: each question against every positive and hard negative in the batch, temperature 0.05
* a batch never holds the same question twice, and turns that are gold for a question are masked out of its
  negatives (multi-evidence questions would otherwise push their own evidence away)
* model selection on the val split by first-stage coverage (all gold turns in the top-k). The untouched base model is
  evaluated first as "epoch 0", so fine-tuning is kept only if it beats bge-small as shipped.
"""
from __future__ import annotations
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..config import RetrieverConfig
from ..data.io import read_jsonl
from ..env import amp_dtype, best_device
from ..retrieval import BGE_QUERY_PREFIX, DenseEncoder, memory_keys

EVAL_KS = (8, 16, 32)


def build_pairs(dialogues, encoder, cfg, rng):
    """(question, gold key, [hard negative keys]) for every gold turn, plus each question's set of gold keys.
    Negatives: the base model's top `mine_depth` non-gold turns of the same dialogue, `hard_negatives` sampled."""
    pairs, gold = [], defaultdict(set)
    for d in dialogues:
        qs = [q for q in d["questions"] if q["gold"]]
        if not qs:
            continue
        keys = memory_keys(d["texts"], cfg.context_chars)
        order = np.argsort(-(encoder.encode([q["query"] for q in qs], query=True) @ encoder.encode(keys).T), axis=1)
        for j, q in enumerate(qs):
            g = set(q["gold"])
            hard = [int(i) for i in order[j, :cfg.mine_depth + len(g)] if i not in g][:cfg.mine_depth]
            gold[q["query"]].update(keys[i] for i in g)
            for i in g:
                negs = rng.sample(hard, min(cfg.hard_negatives, len(hard)))
                pairs.append((q["query"], keys[i], [keys[n] for n in negs]))
    return pairs, gold


def batches(pairs, size, rng):
    """Shuffle, then fill batches with distinct questions (a question's other gold turns are not negatives)."""
    pending = pairs[:]
    rng.shuffle(pending)
    while pending:
        batch, seen, rest = [], set(), []
        for p in pending:
            if len(batch) < size and p[0] not in seen:
                batch.append(p); seen.add(p[0])
            else:
                rest.append(p)
        yield batch
        pending = rest


def coverage(encoder, dialogues, ks=EVAL_KS, context_chars=0) -> dict:
    """Share of questions whose gold turns are all in the dense top-k (the number the conformal packet is capped by)."""
    hits = {k: [] for k in ks}
    for d in dialogues:
        qs = [q for q in d["questions"] if q["gold"]]
        if not qs:
            continue
        docs = encoder.encode(memory_keys(d["texts"], context_chars))
        qv = encoder.encode([q["query"] for q in qs], query=True)
        order = np.argsort(-(qv @ docs.T), axis=1)
        for j, q in enumerate(qs):
            for k in ks:
                hits[k].append(set(q["gold"]) <= set(order[j, :k].tolist()))
    m = {f"all_gold@{k}": float(np.mean(v)) if v else 0.0 for k, v in hits.items()}
    m["score"] = float(np.mean(list(m.values())))
    return m


def _encode(model, tok, texts, max_length, device):
    enc = tok(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
    return F.normalize(model(**enc).last_hidden_state[:, 0], dim=-1)


def build_retriever(cfg: RetrieverConfig, tiny_texts=None, work_dir=None):
    from transformers import AutoModel, AutoTokenizer
    if cfg.tiny:
        from transformers import BertModel
        from ..models.factory import _tiny_config, tiny_tokenizer
        tok = tiny_tokenizer(tiny_texts, Path(work_dir) / "tiny_vocab_retriever")
        return BertModel(_tiny_config(tok)), tok
    return AutoModel.from_pretrained(cfg.backbone), AutoTokenizer.from_pretrained(cfg.backbone)


def save_retriever(model, tok, directory, meta):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(root / "model")
    tok.save_pretrained(root / "model")
    (root / "meta.json").write_text(json.dumps({"kind": "retriever", "query_prefix": BGE_QUERY_PREFIX, **meta}, indent=1) + "\n")


def train_retriever(data_dir, run_dir, cfg: RetrieverConfig, log=print):
    data, run = Path(data_dir), Path(run_dir)
    if (run / "best" / "meta.json").exists():
        log(f"retriever already trained in {run / 'best'} (delete it to train again)")
        return json.loads((run / "best" / "meta.json").read_text())
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = best_device()
    train = read_jsonl(data / "dialogues" / "train.jsonl")
    val = read_jsonl(data / "dialogues" / "val.jsonl")
    sample = [t for d in train[:20] for t in d["texts"]][:5000] + [q["query"] for d in train[:50] for q in d["questions"]]
    model, tok = build_retriever(cfg, tiny_texts=sample, work_dir=run)
    model.to(device).eval()
    encoder = DenseEncoder(model=model, tokenizer=tok, device=device, max_length=cfg.max_length)

    def evaluate(context_chars=cfg.context_chars):
        model.eval()
        m = coverage(encoder, val, context_chars=context_chars)
        model.train()
        return m

    t_mine = time.time()
    pairs, gold = build_pairs(train, encoder, cfg, rng)
    log(f"retriever {cfg.backbone if not cfg.tiny else 'tiny-random'}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"{len(pairs)} (question, gold turn) pairs from {len(gold)} train questions, {cfg.hard_negatives} mined negatives each "
        f"({time.time() - t_mine:.0f}s), keys with {cfg.context_chars} previous-turn chars")
    plain = evaluate(0)
    log(f"epoch 0 (base model, plain turns) val {json.dumps(plain)}")
    best = evaluate()
    history = [{"epoch": 0, **best}]
    log(f"epoch 0 (base model, neighbour keys) val {json.dumps(best)}")
    meta = {"backbone": cfg.backbone, "max_length": cfg.max_length, "context_chars": cfg.context_chars, "val_base_plain": plain}
    save_retriever(model, tok, run / "best", {**meta, "epoch": 0, "val": best, "val_base": best})
    base = best

    dtype = amp_dtype(torch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    steps_per_epoch = sum(1 for _ in batches(pairs, cfg.batch_size, random.Random(0)))  # distinct-question batches can exceed len/size
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    started, step = time.time(), 0
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for batch in batches(pairs, cfg.batch_size, rng):
            queries = [BGE_QUERY_PREFIX + q for q, _, _ in batch]
            docs = [p for _, p, _ in batch] + [n for _, _, negs in batch for n in negs]
            with torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                qv = _encode(model, tok, queries, cfg.query_max_length, device)
                dv = _encode(model, tok, docs, cfg.max_length, device)
            logits = (qv.float() @ dv.float().T) / cfg.temperature
            mask = torch.tensor([[j != i and docs[j] in gold[q] for j in range(len(docs))] for i, (q, _, _) in enumerate(batch)], device=device)
            loss = F.cross_entropy(logits.masked_fill(mask, -1e4), torch.arange(len(batch), device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item()); step += 1
        m = evaluate()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **m})
        improved = m["score"] > best["score"]
        log(f"epoch {epoch} loss {np.mean(losses):.4f} | val {json.dumps(m)}{' (best)' if improved else ''} | {(time.time() - started) / 60:.1f} min")
        if improved:
            best = m
            save_retriever(model, tok, run / "best", {**meta, "epoch": epoch, "val": m, "val_base": base})
    with (run / "log.jsonl").open("w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    log(f"retriever: val coverage score {plain['score']:.3f} (base, plain turns) / {base['score']:.3f} (base, neighbour keys) "
        f"-> {best['score']:.3f} (kept)")
    return json.loads((run / "best" / "meta.json").read_text())

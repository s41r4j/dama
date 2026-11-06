"""Stage 4: export the models to ONNX, quantize, check they still agree with PyTorch, and time them on CPU.

Result: one folder (and zip) that the Memory Agent loads with onnxruntime only - no PyTorch, no GPU:
  retriever  first-stage embeddings (fine-tuned bge-small)      read path
  decision   rel / sal / need / route                           read path (rel, need, route), write path (sal)
  ops        ADD / UPDATE / DELETE / NOOP                       write path, off the critical path

Quantization is chosen per model by measurement (L1.1): int8 cost the op model 5 points of accuracy in L1.0, so each
model tries a few int8 variants and keeps the first that agrees with PyTorch, else ships fp32.
Latency is measured on real test pools (a request and its first-stage top-k turns), not on arbitrary rows.
"""
from __future__ import annotations
import inspect
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .config import OPS, ROUTES
from .data.io import read_jsonl


class _DecisionExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


class _DecisionExportNoTT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def _onnx_export(module, inputs: dict, path: Path, prefer_dynamo: bool = True):
    """Export with dynamic batch and sequence length. Prefers the torch.export-based exporter (exact parity);
    falls back to the classic TorchScript exporter on old torch versions. Returns the exporter used."""
    module.eval()
    names = list(inputs)
    args = tuple(inputs[n] for n in names)
    params = inspect.signature(torch.onnx.export).parameters
    if prefer_dynamo and "dynamo" in params and "dynamic_shapes" in params:
        try:
            from torch.export import Dim
            batch, seq = Dim("batch"), Dim("seq", max=512)
            torch.onnx.export(module, args, str(path), input_names=names, output_names=["logits"], opset_version=18,
                              dynamic_shapes={n: {0: batch, 1: seq} for n in names}, dynamo=True)
            return "dynamo"
        except Exception as e:  # missing onnxscript, or an op the new exporter cannot handle
            print(f"dynamo export failed ({type(e).__name__}: {str(e)[:200]}); trying the classic exporter")
    kwargs = dict(input_names=names, output_names=["logits"], opset_version=17,
                  dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names}, "logits": {0: "batch"}})
    if "dynamo" in params:
        kwargs["dynamo"] = False
    torch.onnx.export(module, args, str(path), **kwargs)
    return "torchscript"


class _EmbedExport(torch.nn.Module):
    """Retriever: L2-normalised CLS embedding, as bge is used."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h, dim=-1)


# int8 variants tried in order; the first whose agreement with PyTorch reaches the bar is kept
QUANT_VARIANTS = [("int8", {}), ("int8_per_channel", {"per_channel": True}), ("int8_matmul_only", {"op_types_to_quantize": ["MatMul"]}),
                  ("int8_matmul_per_channel", {"per_channel": True, "op_types_to_quantize": ["MatMul"]})]


def _pick_quantization(fp32: Path, name: str, agreement, threads, bar: float, log=print):
    """Quantize `fp32` with each variant, score it with agreement(session) in [0, 1], keep the first >= bar.
    Falls back to the fp32 file. Returns (file name, report)."""
    tried = {}
    for variant, kw in QUANT_VARIANTS:
        dst = fp32.with_name(f"{name}.{variant}.onnx")
        try:
            _quantize(fp32, dst, **kw)
            tried[variant] = round(float(agreement(_session(dst, threads))), 4)
        except Exception as e:  # an option this onnxruntime build does not support
            tried[variant] = f"failed: {type(e).__name__}: {str(e)[:120]}"
            _remove_model(dst)
            continue
        if tried[variant] >= bar:
            log(f"{name}: kept {variant} (agreement {tried[variant]} >= {bar}); tried {tried}")
            for v, _ in QUANT_VARIANTS:
                if v != variant:
                    _remove_model(fp32.with_name(f"{name}.{v}.onnx"))
            _remove_model(fp32)
            return dst.name, {"chosen": variant, "agreement_bar": bar, "tried": tried}
        _remove_model(dst)
    log(f"{name}: no int8 variant reached {bar}; shipping fp32. tried {tried}")
    return fp32.name, {"chosen": "fp32", "agreement_bar": bar, "tried": tried}


def _pool_latency(sess, tok, pools, names, max_length, ks=(1, 4, 8, 16), questions=12, runs=3):
    """Time rel re-scoring of a real request against its first-stage top-k turns (padding to the longest pair, as at
    runtime). Reports p50/p95 over questions and the mean padded sequence length."""
    out = {}
    for k in ks:
        ts, lens = [], []
        for q in pools[:questions]:
            cands = [c["text"] for c in q["cands"][:k]]
            if len(cands) < k:
                continue
            feeds = _feeds(tok, [{"a": q["query"], "b": c} for c in cands], max_length, names)
            sess.run(None, feeds)
            per = []
            for _ in range(runs):
                t = time.perf_counter(); sess.run(None, feeds); per.append((time.perf_counter() - t) * 1000)
            ts.append(sorted(per)[len(per) // 2]); lens.append(feeds["input_ids"].shape[1])
        if ts:
            ts.sort()
            out[f"k{k}"] = {"p50_ms": round(ts[len(ts) // 2], 1), "p95_ms": round(ts[max(0, int(len(ts) * 0.95) - 1)], 1),
                            "mean_padded_tokens": round(float(np.mean(lens)), 1), "questions": len(ts)}
    return out


def _fuse(module, inputs: dict, out: Path, name: str, num_heads: int, hidden: int):
    """L1.2: TorchScript export + onnxruntime's transformer optimizer, which fuses attention, layer norm and GELU into
    single CPU kernels (the torch.export graph does not match its patterns), then int8. Returns (path, fused op counts)."""
    from onnxruntime.transformers import optimizer
    ts, fused, q = out / f"{name}.ts.onnx", out / f"{name}.fused.onnx", out / f"{name}.fused.int8.onnx"
    _onnx_export(module, inputs, ts, prefer_dynamo=False)
    m = optimizer.optimize_model(str(ts), model_type="bert", num_heads=num_heads, hidden_size=hidden)
    m.save_model_to_file(str(fused))
    stats = {k: v for k, v in m.get_fused_operator_statistics().items() if v}
    _quantize(fused, q)
    _remove_model(ts); _remove_model(fused)
    return q, stats


def _quantize(src: Path, dst: Path, **kw):
    """Dynamic int8 quantization of the weights. Stale shape annotations from the exporter are dropped first,
    because onnxruntime re-runs shape inference and rejects conflicting hints."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    clean = src.with_name(src.stem + ".clean.onnx")
    onnx.save(model, str(clean))
    quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8, **kw)
    clean.unlink()


def _session(path, threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def _size_mb(path: Path) -> float:
    """Model size including any external-data file the exporter wrote next to it."""
    files = [path] + list(path.parent.glob(path.name + ".data")) + list(path.parent.glob(path.stem + ".onnx_data"))
    return round(sum(f.stat().st_size for f in files if f.exists()) / 2**20, 1)


def _remove_model(path: Path):
    for f in [path] + list(path.parent.glob(path.name + ".data")) + list(path.parent.glob(path.stem + ".onnx_data")):
        if f.exists():
            f.unlink()


def _latency(sess, feeds, runs=30):
    for _ in range(3):
        sess.run(None, feeds)
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); sess.run(None, feeds); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 2), "p95_ms": round(ts[int(len(ts) * 0.95) - 1], 2)}


def _feeds(tok, rows, max_length, names):
    enc = tok([r["a"] for r in rows], [r["b"] for r in rows], truncation="longest_first", max_length=max_length,
              padding=True, return_tensors="np")
    return {n: enc[n].astype(np.int64) for n in names}


def export_all(data_dir, decision_dir, ops_dir, calibration_dir, out_dir, *, threads=2, retriever_dir=None, log=print):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .models.decision import load_decision
    data, out = Path(data_dir), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report = {}

    # ---- decision model
    dec, dtok, dmeta = load_decision(decision_dir, device="cpu")
    dec.eval()
    dmax = dmeta.get("max_length", 256)
    rows = [r for r in read_jsonl(data / "decision" / "test.jsonl") if r["task"] == "rel"][:64]
    sample = dtok([r["a"] for r in rows[:2]], [r["b"] for r in rows[:2]], truncation="longest_first", max_length=dmax, padding=True, return_tensors="pt")
    has_tt = "token_type_ids" in sample
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if has_tt else [])
    wrapper = _DecisionExport(dec) if has_tt else _DecisionExportNoTT(dec)
    exporter = _onnx_export(wrapper, {n: sample[n] for n in names}, out / "decision.onnx")
    _quantize(out / "decision.onnx", out / "decision.int8.onnx")
    dtok.save_pretrained(out / "decision_tokenizer")
    feeds = _feeds(dtok, rows, dmax, names)
    with torch.no_grad():
        ref = dec(**{k: torch.from_numpy(v) for k, v in feeds.items()}).numpy()
    fp32 = _session(out / "decision.onnx", threads).run(None, feeds)[0]
    int8 = _session(out / "decision.int8.onnx", threads).run(None, feeds)[0]
    p_ref, p_int8 = 1 / (1 + np.exp(-ref[:, 0])), 1 / (1 + np.exp(-int8[:, 0]))
    report["decision"] = {"exporter": exporter, "onnx_fp32_max_abs_diff": float(np.abs(fp32 - ref).max()),
                          "int8_rel_prob_max_abs_diff": float(np.abs(p_int8 - p_ref).max()),
                          "int8_rel_rank_corr": float(np.corrcoef(p_ref, p_int8)[0, 1]) if len(rows) > 2 else None,
                          "size_mb_fp32": _size_mb(out / "decision.onnx"), "size_mb_int8": _size_mb(out / "decision.int8.onnx")}
    if report["decision"]["onnx_fp32_max_abs_diff"] > 1e-3:
        raise RuntimeError(f"ONNX decision model disagrees with PyTorch: {report['decision']}")
    sess = _session(out / "decision.int8.onnx", threads)
    # k = re-scoring the first-stage top-k turns of one real test request (the rerank depth at runtime)
    pools = read_jsonl(data / "pools" / "test.jsonl") if (data / "pools" / "test.jsonl").exists() else []
    report["decision"]["cpu_latency_int8_real_pools"] = {f"max_length_{m}": _pool_latency(sess, dtok, pools, names, m) for m in sorted({dmax, 128})}
    # L1.2: try a fused graph; ship it only if it matches PyTorch as well and is >= 10% faster at k = 8 on real pools
    try:
        cfg = dec.encoder.config
        fpath, stats = _fuse(wrapper, {n: sample[n] for n in names}, out, "decision", cfg.num_attention_heads, cfg.hidden_size)
        p_f = 1 / (1 + np.exp(-_session(fpath, threads).run(None, feeds)[0][:, 0]))
        lat_f = _pool_latency(_session(fpath, threads), dtok, pools, names, dmax)
        base_k8 = report["decision"]["cpu_latency_int8_real_pools"].get(f"max_length_{dmax}", {}).get("k8", {}).get("p50_ms")
        fused_k8 = lat_f.get("k8", {}).get("p50_ms")
        corr = float(np.corrcoef(p_ref, p_f)[0, 1]) if len(rows) > 2 else 1.0
        keep = (corr >= report["decision"]["int8_rel_rank_corr"] - 0.002 and float(np.abs(p_f - p_ref).max()) <= 0.1
                and base_k8 and fused_k8 and fused_k8 <= 0.9 * base_k8)
        report["decision"]["fused"] = {"fused_ops": stats, "int8_rel_rank_corr": corr, "int8_rel_prob_max_abs_diff": float(np.abs(p_f - p_ref).max()),
                                       "cpu_latency_real_pools": lat_f, "shipped": bool(keep)}
        if keep:
            _remove_model(out / "decision.int8.onnx")
            fpath.rename(out / "decision.int8.onnx")
        else:
            _remove_model(fpath)
    except Exception as e:  # optimizer missing or the graph does not fuse: keep the plain int8 model
        report["decision"]["fused"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
        for f in out.glob("decision.ts.onnx*"):
            f.unlink()
        for f in out.glob("decision.fused*"):
            f.unlink()
    log(f"decision fused graph: {json.dumps(report['decision']['fused'])[:600]}")
    _remove_model(out / "decision.onnx")  # keep only the int8 file in the bundle
    log(f"decision model exported: {report['decision']}")

    # ---- op model
    op = AutoModelForSequenceClassification.from_pretrained(Path(ops_dir) / "model").eval()
    otok = AutoTokenizer.from_pretrained(Path(ops_dir) / "tokenizer")
    ometa = json.loads((Path(ops_dir) / "meta.json").read_text())
    omax = ometa.get("max_length", 160)
    orows = read_jsonl(data / "ops" / "test.jsonl")[:256]
    from .models.ops import OpLogits
    owrap = OpLogits(op).eval()
    osample = otok([r["a"] for r in orows[:2]], [r["b"] for r in orows[:2]], truncation="longest_first", max_length=omax, padding=True, return_tensors="pt")
    onames = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in osample else [])
    oexporter = _onnx_export(owrap, {n: osample[n] for n in onames}, out / "ops.onnx")
    otok.save_pretrained(out / "ops_tokenizer")
    feeds = _feeds(otok, orows, omax, onames)
    with torch.no_grad():
        ref = owrap(**{k: torch.from_numpy(v) for k, v in feeds.items()}).numpy()
    ofp32 = _session(out / "ops.onnx", threads).run(None, feeds)[0]
    if float(np.abs(ofp32 - ref).max()) > 1e-3:
        raise RuntimeError(f"ONNX op model disagrees with PyTorch: max abs diff {float(np.abs(ofp32 - ref).max())}")
    ofile, oquant = _pick_quantization(out / "ops.onnx", "ops", lambda sess: (sess.run(None, feeds)[0].argmax(-1) == ref.argmax(-1)).mean(),
                                       threads, bar=0.98, log=log)
    sess = _session(out / ofile, threads)
    chosen = sess.run(None, feeds)[0]
    y = np.array([r["y"] for r in orows])
    report["ops"] = {"exporter": oexporter, "onnx_fp32_max_abs_diff": float(np.abs(ofp32 - ref).max()), "quantization": oquant,
                     "label_agreement": float((ref.argmax(-1) == chosen.argmax(-1)).mean()),
                     "torch_accuracy_on_sample": float((ref.argmax(-1) == y).mean()),
                     "shipped_accuracy_on_sample": float((chosen.argmax(-1) == y).mean()), "size_mb": _size_mb(out / ofile),
                     "cpu_latency": {f"batch{b}": _latency(sess, _feeds(otok, orows[:b], omax, onames)) for b in (1, 16) if len(orows) >= b}}
    log(f"op model exported: {report['ops']}")

    # ---- metadata the Memory Agent needs at runtime
    cal_dir = Path(calibration_dir)
    calibration = json.loads((cal_dir / "calibration.json").read_text()) if (cal_dir / "calibration.json").exists() else None
    test_report = json.loads((cal_dir / "report.json").read_text()) if (cal_dir / "report.json").exists() else None
    # ---- retriever (first stage)
    rmeta_out = None
    if retriever_dir and (Path(retriever_dir) / "meta.json").exists():
        from transformers import AutoModel
        rdir = Path(retriever_dir)
        rmeta = json.loads((rdir / "meta.json").read_text())
        rtok = AutoTokenizer.from_pretrained(rdir / "model")
        rmodel = _EmbedExport(AutoModel.from_pretrained(rdir / "model").eval()).eval()
        rmax = rmeta.get("max_length", 256)
        groups = [q for q in pools[:8] if q["cands"]]
        queries = [rmeta["query_prefix"] + q["query"] for q in groups]
        spans, texts = [], []
        for q in groups:
            spans.append(slice(len(texts), len(texts) + len(q["cands"][:32]))); texts += [c["text"] for c in q["cands"][:32]]
        texts = texts or [r["b"] for r in rows]
        rs = rtok(texts[:2], truncation=True, max_length=rmax, padding=True, return_tensors="pt")
        rnames = ["input_ids", "attention_mask", "token_type_ids"]
        rexporter = _onnx_export(rmodel, {n: rs[n] for n in rnames}, out / "retriever.onnx")
        rtok.save_pretrained(out / "retriever_tokenizer")

        def embed(fn, items, m):
            enc = rtok(items, truncation=True, max_length=m, padding=True, return_tensors="np")
            return fn({n: enc[n].astype(np.int64) for n in rnames})

        with torch.no_grad():
            ref_doc = embed(lambda f: rmodel(**{k: torch.from_numpy(v) for k, v in f.items()}).numpy(), texts, rmax)
            ref_q = embed(lambda f: rmodel(**{k: torch.from_numpy(v) for k, v in f.items()}).numpy(), queries, 64) if queries else None
        rfp32 = embed(lambda f: _session(out / "retriever.onnx", threads).run(None, f)[0], texts, rmax)
        if float(np.abs(rfp32 - ref_doc).max()) > 1e-3:
            raise RuntimeError(f"ONNX retriever disagrees with PyTorch: max abs diff {float(np.abs(rfp32 - ref_doc).max())}")

        def top5_overlap(sess):
            """Mean overlap of the top-5 turns per test request (its first 32 pool turns) between PyTorch and this file."""
            if not queries:
                return 1.0
            run = lambda f: sess.run(None, f)[0]
            d, q = embed(run, texts, rmax), embed(run, queries, 64)
            hits = []
            for j, block in enumerate(spans):
                n = min(5, block.stop - block.start)
                a = set(np.argsort(-(ref_doc[block] @ ref_q[j]))[:n]); b = set(np.argsort(-(d[block] @ q[j]))[:n])
                hits.append(len(a & b) / max(1, n))
            return float(np.mean(hits))

        rfile, rquant = _pick_quantization(out / "retriever.onnx", "retriever", top5_overlap, threads, bar=0.95, log=log)
        rsess = _session(out / rfile, threads)
        one = rtok([queries[0] if queries else "where do I live?"], truncation=True, max_length=64, return_tensors="np")
        report["retriever"] = {"exporter": rexporter, "fine_tuned_epoch": rmeta.get("epoch"), "quantization": rquant, "size_mb": _size_mb(out / rfile),
                               "cpu_latency_query": _latency(rsess, {n: one[n].astype(np.int64) for n in rnames})}
        rmeta_out = {"file": rfile, "tokenizer": "retriever_tokenizer", "inputs": rnames, "max_length": rmax, "query_max_length": 64,
                     "query_prefix": rmeta["query_prefix"], "fine_tuned_epoch": rmeta.get("epoch"), "val": rmeta.get("val"), "val_base": rmeta.get("val_base"),
                     # memory keys: previous-turn chars prepended, unless calibration chose the same model on bare turns
                     "context_chars": 0 if (calibration or {}).get("first_stage", "").endswith("tuned_plain") else rmeta.get("context_chars", 0)}
        log(f"retriever exported: {report['retriever']}")

    manifest = json.loads((data / "manifest.json").read_text())
    from .config import CHECKPOINT
    meta = {"format": "dama-tier1/1", "checkpoint": CHECKPOINT, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decision": {"file": "decision.int8.onnx", "tokenizer": "decision_tokenizer", "inputs": names, "max_length": dmax,
                         "columns": ["rel", "sal", "need"] + [f"route:{r}" for r in ROUTES], "backbone": dmeta.get("backbone")},
            "retriever": rmeta_out,
            "ops": {"file": ofile, "tokenizer": "ops_tokenizer", "inputs": onames, "max_length": omax, "labels": OPS,
                    "backbone": ometa.get("backbone")},
            "routes": ROUTES, "calibration": calibration, "export_checks": report, "test_report": test_report,
            "training_data": {k: v["sha256"] for k, v in manifest["files"].items()}}
    (out / "dama_tier1.json").write_text(json.dumps(meta, indent=1) + "\n")
    shutil.copy(Path(__file__).with_name("runtime.py"), out / "runtime.py")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    log(f"bundle written to {out} and {archive}")
    return meta

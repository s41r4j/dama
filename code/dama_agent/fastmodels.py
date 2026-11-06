"""CPU models for the hybrid read path: ONNX exports of an off-the-shelf bi-encoder (query side) and cross-encoder.

    enc = OnnxEncoder.export("BAAI/bge-small-en-v1.5", "out/bge", prefix="Represent this sentence for searching relevant passages: ")
    ce = OnnxCrossEncoder.export("cross-encoder/ms-marco-TinyBERT-L2-v2", "out/ce", platt=(a, b))
    reader = HybridReader(enc.encode, scorer=ce.probs, ...)

Exports reuse dama_train.export (dynamo exporter with a TorchScript fallback). `quantize=True` adds a dynamic-int8
copy, kept only if it agrees with fp32 (cosine >= 0.99 for embeddings, |dp| <= 0.03 for probabilities, on the sample
texts given); otherwise fp32 is used. Stored memory vectors may come from the PyTorch model on a GPU: fp32 ONNX matches
it to ~1e-6.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def _session(path, threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class OnnxEncoder:
    """L2-normalised CLS embeddings (bge-style)."""
    def __init__(self, directory, threads: int = 4, variant: str | None = None):
        from transformers import AutoTokenizer
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "encoder.json").read_text())
        self.variant = variant or self.meta["use"]
        self.sess = _session(self.dir / self.meta["files"][self.variant], threads)
        self.tok = AutoTokenizer.from_pretrained(self.dir / "tokenizer")
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def encode(self, texts, query: bool = True) -> np.ndarray:
        texts = [self.meta["prefix"] + t for t in texts] if query else list(texts)
        enc = self.tok(texts, truncation=True, max_length=self.meta["max_length"], padding=True, return_tensors="np")
        return self.sess.run(None, {n: enc[n].astype(np.int64) for n in self.inputs})[0]

    @classmethod
    def export(cls, model_id, out_dir, *, prefix="", max_length=128, quantize=True, sample=(), threads=4, log=print):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from dama_train.export import _EmbedExport, _onnx_export, _quantize
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_id); tok.save_pretrained(out / "tokenizer")
        model = AutoModel.from_pretrained(model_id).eval()
        enc = tok(["a short example query", "another one"], return_tensors="pt", padding=True)
        inputs = {n: enc[n] for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc}
        _onnx_export(_EmbedExport(model) if "token_type_ids" in inputs else _NoTT(model), inputs, out / "fp32.onnx")
        meta = {"model": model_id, "prefix": prefix, "max_length": max_length, "files": {"fp32": "fp32.onnx"}, "use": "fp32"}
        (out / "encoder.json").write_text(json.dumps(meta))
        if quantize:
            try:
                _quantize(out / "fp32.onnx", out / "int8.onnx")
                meta["files"]["int8"] = "int8.onnx"; (out / "encoder.json").write_text(json.dumps(meta))
                texts = list(sample) or ["When did Caroline go to the LGBTQ support group?", "I moved to Pune last month."]
                a, b = cls(out, threads, "fp32").encode(texts), cls(out, threads, "int8").encode(texts)
                agree = float(np.min(np.sum(a * b, axis=1)))
                meta["int8_min_cosine"] = agree
                if agree >= 0.99:
                    meta["use"] = "int8"
                log(f"encoder {model_id}: int8 min cosine to fp32 {agree:.4f} -> using {meta['use']}")
            except Exception as e:
                log(f"encoder {model_id}: int8 failed ({type(e).__name__}: {str(e)[:150]}); using fp32")
        (out / "encoder.json").write_text(json.dumps(meta))
        return cls(out, threads)


class OnnxCrossEncoder:
    """Pair classifier (ms-marco style, one logit) -> Platt-calibrated probability."""
    def __init__(self, directory, threads: int = 4, variant: str | None = None):
        from transformers import AutoTokenizer
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "cross_encoder.json").read_text())
        self.variant = variant or self.meta["use"]
        self.sess = _session(self.dir / self.meta["files"][self.variant], threads)
        self.tok = AutoTokenizer.from_pretrained(self.dir / "tokenizer")
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.platt = tuple(self.meta.get("platt") or (1.0, 0.0))

    def logits(self, queries, docs, batch_size: int = 64) -> np.ndarray:
        out = []
        for s in range(0, len(queries), batch_size):
            enc = self.tok(list(queries[s:s + batch_size]), list(docs[s:s + batch_size]), truncation="longest_first",
                           max_length=self.meta["max_length"], padding=True, return_tensors="np")
            out.append(self.sess.run(None, {n: enc[n].astype(np.int64) for n in self.inputs})[0][:, 0])
        return np.concatenate(out) if out else np.zeros(0)

    def probs(self, queries, docs) -> np.ndarray:
        a, b = self.platt
        return _sigmoid(a * self.logits(queries, docs) + b)

    def set_platt(self, a: float, b: float):
        self.platt = (float(a), float(b)); self.meta["platt"] = list(self.platt)
        (self.dir / "cross_encoder.json").write_text(json.dumps(self.meta))

    @classmethod
    def export(cls, model_id, out_dir, *, max_length=256, platt=None, quantize=True, sample=(), threads=4, log=print):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from dama_train.export import _onnx_export, _quantize
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_id); tok.save_pretrained(out / "tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        enc = tok(["a question", "another question"], ["a passage", "another passage here"], return_tensors="pt", padding=True)
        inputs = {n: enc[n] for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc}
        _onnx_export(_CE(model) if "token_type_ids" in inputs else _CENoTT(model), inputs, out / "fp32.onnx")
        meta = {"model": model_id, "max_length": max_length, "files": {"fp32": "fp32.onnx"}, "use": "fp32", "platt": list(platt) if platt else None}
        (out / "cross_encoder.json").write_text(json.dumps(meta))
        if quantize:
            try:
                _quantize(out / "fp32.onnx", out / "int8.onnx")
                meta["files"]["int8"] = "int8.onnx"; (out / "cross_encoder.json").write_text(json.dumps(meta))
                pairs = list(sample) or [("When did Caroline go to the support group?", "Caroline: I went to a LGBTQ support group yesterday.")] * 2
                q, d = [p[0] for p in pairs], [p[1] for p in pairs]
                a, b = cls(out, threads, "fp32").probs(q, d), cls(out, threads, "int8").probs(q, d)
                gap = float(np.max(np.abs(a - b)))
                meta["int8_max_prob_gap"] = gap
                if gap <= 0.03:
                    meta["use"] = "int8"
                log(f"cross-encoder {model_id}: int8 max |dp| {gap:.4f} -> using {meta['use']}")
            except Exception as e:
                log(f"cross-encoder {model_id}: int8 failed ({type(e).__name__}: {str(e)[:150]}); using fp32")
        (out / "cross_encoder.json").write_text(json.dumps(meta))
        return cls(out, threads)


try:   # wrappers need torch only when exporting
    import torch

    class _NoTT(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model

        def forward(self, input_ids, attention_mask):
            h = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(h, dim=-1)

    class _CE(torch.nn.Module):
        """The logits tensor itself (named inputs, so the dynamo exporter can mark them dynamic)."""
        def __init__(self, model):
            super().__init__(); self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    class _CENoTT(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
except ImportError:   # CPU-only deployments load exported files without torch
    pass

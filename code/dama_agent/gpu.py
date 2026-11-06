"""Tier-1 on a GPU: the same interface and calibration as the CPU bundle (dama_train.runtime.Tier1), but running the
trained PyTorch weights (Model variation transformers/tier1-pytorch) on CUDA with fp16 autocast. The calibration
(Platt, temperature, conformal thresholds) and the input specs come from the CPU bundle of the same checkpoint, so the
agent and the evaluation cannot tell the two runtimes apart; only speed and float precision differ."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from dama_train.runtime import Tier1


class TorchTier1(Tier1):
    def __init__(self, bundle_dir, pytorch_dir, device=None):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        from dama_train.models.decision import load_decision
        from dama_train.models.ops import OpLogits
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_meta(bundle_dir)
        py = Path(pytorch_dir)
        self.dec, self.dtok, _ = load_decision(py / "decision", device=self.device)
        self.ops = OpLogits(AutoModelForSequenceClassification.from_pretrained(py / "ops" / "model")).to(self.device).eval()
        self.otok = AutoTokenizer.from_pretrained(py / "ops" / "tokenizer")
        self.ret = AutoModel.from_pretrained(py / "retriever" / "model").to(self.device).eval()
        self.rtok = AutoTokenizer.from_pretrained(py / "retriever" / "model")

    def _autocast(self):
        return self.torch.autocast(device_type="cuda", dtype=self.torch.float16, enabled=self.device.startswith("cuda"))

    def _run(self, model, tok, spec, a, b):
        enc = tok(a, b, truncation="longest_first", max_length=spec["max_length"], padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode(), self._autocast():
            return model(**{n: enc[n] for n in spec["inputs"] if n in enc}).float().cpu().numpy()

    def embed(self, texts, query=False) -> np.ndarray:
        r = self.meta["retriever"]
        texts = [r["query_prefix"] + t for t in texts] if query else list(texts)
        enc = self.rtok(texts, truncation=True, max_length=r["query_max_length"] if query else r["max_length"], padding=True,
                        return_tensors="pt").to(self.device)
        with self.torch.inference_mode(), self._autocast():
            h = self.ret(**enc).last_hidden_state[:, 0].float()
        return self.torch.nn.functional.normalize(h, dim=-1).cpu().numpy()

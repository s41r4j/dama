"""Stage functions with fixed folder layout under one work directory:

    <work>/raw/                 downloaded benchmarks
    <work>/data/                built JSONL + manifest.json
    <work>/runs/retriever/      log.jsonl, best/ (fine-tuned bge-small first stage)
    <work>/runs/decision/       checkpoint.pt, log.jsonl, best/
    <work>/runs/ops/            checkpoint.pt, log.jsonl, best/
    <work>/runs/calibration/    calibration.json, report.json
    <work>/export/dama_tier1/   int8 ONNX models + tokenizers + dama_tier1.json (+ dama_tier1.zip)

Every stage skips work that is already done, so on a free notebook you can simply re-run after a disconnect.
"""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
from .config import DataConfig, RetrieverConfig, TrainConfig, decision_defaults, ops_defaults
from .env import default_work_dir


def paths(work=None) -> dict:
    root = Path(work) if work else default_work_dir()
    p = {"work": root, "raw": root / "raw", "data": root / "data", "retriever": root / "runs" / "retriever", "decision": root / "runs" / "decision",
         "ops": root / "runs" / "ops", "calibration": root / "runs" / "calibration", "export": root / "export" / "dama_tier1"}
    root.mkdir(parents=True, exist_ok=True)
    return p


def download(work=None, cfg: DataConfig | None = None):
    from .data.download import fetch_all
    p = paths(work)
    return fetch_all(cfg or DataConfig(), p["raw"])


def build(work=None, cfg: DataConfig | None = None, max_lme_questions=None, force=False):
    from .data.build import build_all
    p = paths(work)
    if (p["data"] / "manifest.json").exists() and not force:
        print(f"data already built in {p['data']} (force=True to rebuild)")
        return json.loads((p["data"] / "manifest.json").read_text())
    sources = json.loads((p["raw"] / "sources.json").read_text())
    return build_all(cfg or DataConfig(), sources, p["data"], max_lme_questions=max_lme_questions)


def train_retriever(work=None, cfg: RetrieverConfig | None = None, **overrides):
    from .data.io import verify
    from .train.retriever import train_retriever as _train
    p = paths(work)
    verify(p["data"])
    return _train(p["data"], p["retriever"], replace(cfg or RetrieverConfig(), **overrides))


def train_decision(work=None, cfg: TrainConfig | None = None, **overrides):
    from .data.io import verify
    from .train.decision import train_decision as _train
    p = paths(work)
    verify(p["data"])
    cfg = replace(cfg or decision_defaults(), **overrides)
    return _train(p["data"], p["decision"], cfg)


def train_ops(work=None, cfg: TrainConfig | None = None, **overrides):
    from .data.io import verify
    from .train.ops import train_ops as _train
    p = paths(work)
    verify(p["data"])
    cfg = replace(cfg or ops_defaults(), **overrides)
    return _train(p["data"], p["ops"], cfg)


def calibrate(work=None, use_dense=True, max_pool_questions=None):
    from .calibrate import print_report, run_calibration
    p = paths(work)
    cal, report = run_calibration(p["data"], p["decision"] / "best", p["ops"] / "best", p["calibration"],
                                  use_dense=use_dense, max_pool_questions=max_pool_questions, retriever_dir=p["retriever"] / "best")
    print_report(report)
    return cal, report


def export(work=None, threads=2):
    from .export import export_all
    p = paths(work)
    return export_all(p["data"], p["decision"] / "best", p["ops"] / "best", p["calibration"], p["export"], threads=threads,
                      retriever_dir=p["retriever"] / "best")

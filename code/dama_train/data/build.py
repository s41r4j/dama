"""Turn downloaded benchmarks into the JSONL files the trainers read, with a hash manifest and statistics."""
from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path
from ..config import SPLITS, DataConfig
from .build_decision import build_decision_rows
from .build_ops import build_op_rows
from .io import read_jsonl, write_jsonl
from .locomo import load_locomo
from .longmemeval import load_longmemeval


def build_all(cfg: DataConfig, sources: dict, out_dir, max_lme_questions: int | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    dialogues, questions = load_locomo(sources["locomo"])
    lme_path = sources.get(cfg.lme_s) if cfg.use_lme_s and sources.get(cfg.lme_s) else sources.get(cfg.lme_oracle)
    if lme_path:
        d2, q2 = load_longmemeval(lme_path, max_questions=max_lme_questions)
        dialogues += d2
        questions += q2
    dolly = read_jsonl(sources["dolly"]) if sources.get("dolly") else []
    mnli = read_jsonl(sources["mnli"]) if sources.get("mnli") else []
    general_questions = read_jsonl(sources["questions"]) if sources.get("questions") else []
    print(f"loaded {len(dialogues)} dialogues, {sum(len(d.turns) for d in dialogues)} turns, {len(questions)} questions "
          f"({sum(q.evidence != [] for q in questions)} with evidence) in {time.time() - t0:.0f}s")

    dec_rows, pools, eval_dialogues, dec_stats = build_decision_rows(dialogues, questions, dolly, cfg, general_questions)
    op_rows, op_stats = build_op_rows(dialogues, questions, mnli, cfg)

    files = {}
    for s in SPLITS:
        files[f"decision/{s}.jsonl"] = write_jsonl(out / "decision" / f"{s}.jsonl", dec_rows[s])
        files[f"ops/{s}.jsonl"] = write_jsonl(out / "ops" / f"{s}.jsonl", op_rows[s])
        if s in pools:
            files[f"pools/{s}.jsonl"] = write_jsonl(out / "pools" / f"{s}.jsonl", pools[s])
        files[f"dialogues/{s}.jsonl"] = write_jsonl(out / "dialogues" / f"{s}.jsonl", eval_dialogues[s])
    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "config": asdict(cfg), "sources": {k: Path(v).name for k, v in sources.items()},
                "lme_file": Path(lme_path).name if lme_path else None, "files": files,
                "stats": {"decision": dec_stats, "ops": op_stats}, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest

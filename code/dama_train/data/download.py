"""Fetch the public datasets once into the work directory (re-runs reuse the cached files)."""
from __future__ import annotations
import json
import random
import time
import urllib.request
from pathlib import Path
from ..config import DataConfig


def _get(url: str, dest: Path, tries: int = 3):
    for attempt in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:  # network hiccups on free notebooks are common
            if attempt == tries - 1:
                raise RuntimeError(f"could not download {url}: {e}") from e
            time.sleep(3 * (attempt + 1))


def fetch_all(cfg: DataConfig, raw_dir) -> dict:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    paths = {}

    locomo = raw / "locomo10.json"
    if not locomo.exists():
        print("downloading LoCoMo ...")
        _get(cfg.locomo_url, locomo)
    paths["locomo"] = str(locomo)

    from huggingface_hub import hf_hub_download
    names = [cfg.lme_oracle] + ([cfg.lme_s] if cfg.use_lme_s else [])
    for name in names:
        target = raw / name
        if not target.exists():
            print(f"downloading LongMemEval {name} ...")
            hf_hub_download(cfg.lme_repo, name, repo_type="dataset", local_dir=str(raw))
        paths[name] = str(target)

    mnli = raw / "mnli_sample.jsonl"
    if not mnli.exists():
        print("downloading MNLI sample ...")
        from datasets import load_dataset
        ds = load_dataset(cfg.mnli_repo, split="train")
        idx = random.Random(cfg.seed).sample(range(len(ds)), min(cfg.mnli_samples, len(ds)))
        with mnli.open("w") as f:
            for r in ds.select(idx):
                if r["label"] in (0, 1, 2):
                    f.write(json.dumps({"premise": r["premise"], "hypothesis": r["hypothesis"], "label": r["label"]}) + "\n")
    paths["mnli"] = str(mnli)

    dolly = raw / "dolly_sample.jsonl"
    if not dolly.exists():
        print("downloading Dolly-15k instructions (non-memory requests) ...")
        from datasets import load_dataset
        ds = load_dataset(cfg.dolly_repo, split="train")
        idx = random.Random(cfg.seed).sample(range(len(ds)), min(cfg.dolly_samples, len(ds)))
        with dolly.open("w") as f:
            for r in ds.select(idx):
                f.write(json.dumps({"instruction": r["instruction"], "category": r["category"]}) + "\n")
    paths["dolly"] = str(dolly)

    questions = raw / "general_questions.jsonl"
    if not questions.exists():
        print("downloading SQuAD + Quora questions (need-head negatives) ...")
        from datasets import load_dataset
        rng = random.Random(cfg.seed)
        rows = []
        squad = load_dataset(cfg.squad_repo, split="train")
        texts = sorted(set(squad["question"]))
        rows += [{"text": t, "src": "squad"} for t in rng.sample(texts, min(cfg.question_samples, len(texts)))]
        quora = load_dataset(cfg.quora_repo, "pair", split="train")
        texts = sorted(set(quora["anchor"]))
        rows += [{"text": t, "src": "quora"} for t in rng.sample(texts, min(cfg.question_samples, len(texts)))]
        with questions.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    paths["questions"] = str(questions)

    (raw / "sources.json").write_text(json.dumps(paths, indent=1) + "\n")
    return paths

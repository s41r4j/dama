"""Command line: python -m dama_train <stage> [options]

Stages (run in this order; each resumes or skips finished work):
  env             print platform / GPU info
  download        fetch LoCoMo, LongMemEval, MNLI sample, Dolly sample, SQuAD + Quora questions
  build           build training JSONL files
  train-retriever fine-tune the bge-small first-stage retriever
  train-decision  train the rel/sal/need/route model
  train-ops       train the ADD/UPDATE/DELETE/NOOP model
  calibrate       fit calibration + conformal threshold, print held-out test report
  export          int8 ONNX bundle for CPU
  all             everything above
  smoke           offline end-to-end check with tiny random models (no GPU, no downloads)
"""
from __future__ import annotations
import argparse
import sys
from . import pipeline
from .env import print_env


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dama_train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["env", "download", "build", "train-retriever", "train-decision", "train-ops", "calibrate", "export", "all", "smoke"])
    ap.add_argument("--work", default=None, help="work directory (default: Drive on Colab, /kaggle/working on Kaggle)")
    ap.add_argument("--no-lme-s", action="store_true", help="skip the 277 MB LongMemEval_S file (fewer hard negatives)")
    ap.add_argument("--max-lme", type=int, default=None, help="use only the first N LongMemEval questions")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop and checkpoint after this many minutes")
    ap.add_argument("--fp32", action="store_true", help="train in fp32 (slower; only if fp16 gives non-finite losses)")
    ap.add_argument("--no-dense", action="store_true", help="BM25-only candidate pools during calibration")
    ap.add_argument("--force", action="store_true", help="rebuild data even if it exists")
    a = ap.parse_args(argv)

    if a.stage == "smoke":
        from .smoke import run_smoke
        return run_smoke(a.work)
    overrides = {k: v for k, v in (("epochs", a.epochs), ("batch_size", a.batch_size), ("max_minutes", a.max_minutes)) if v is not None}
    if a.fp32:
        overrides["precision"] = "fp32"
    from .config import DataConfig
    dcfg = DataConfig(use_lme_s=not a.no_lme_s)
    if a.stage in ("env", "all"):
        print_env()
    if a.stage in ("download", "all"):
        pipeline.download(a.work, dcfg)
    if a.stage in ("build", "all"):
        pipeline.build(a.work, dcfg, max_lme_questions=a.max_lme, force=a.force)
    if a.stage in ("train-retriever", "all"):
        pipeline.train_retriever(a.work)
    if a.stage in ("train-decision", "all"):
        pipeline.train_decision(a.work, **overrides)
    if a.stage in ("train-ops", "all"):
        pipeline.train_ops(a.work, **overrides)
    if a.stage in ("calibrate", "all"):
        pipeline.calibrate(a.work, use_dense=not a.no_dense)
    if a.stage in ("export", "all"):
        pipeline.export(a.work)
    return 0


if __name__ == "__main__":
    sys.exit(main())

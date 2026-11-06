"""Training code for DAMA's Tier-1 Memory Agent models.

Two small models are trained, both cheap enough for a free Colab/Kaggle GPU and fast on CPU after export:

* decision model - one cross-encoder (MiniLM-L6, 22.7M) with four heads:
    rel   : does this stored turn answer this request?              (read path)
    sal   : does this new turn carry durable information?           (write path)
    need  : does this request need long-term memory at all?         (push gate)
    route : which stores to search: episodic / facts / timeline / expand
* op model - an NLI cross-encoder (DeBERTa-v3-xsmall, 70.8M) that compares an existing memory with a new
  statement and outputs ADD / UPDATE / DELETE / NOOP.

Both are trained on real benchmark annotations (LoCoMo, LongMemEval) plus MNLI and a held-out synthetic
slot set, calibrated, and exported to int8 ONNX for CPU inference.
"""
__version__ = "0.2.0"

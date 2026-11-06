"""Offline end-to-end check: fake benchmark files (or real LoCoMo if it can be downloaded), tiny random models,
a few training steps, calibration, int8 export and the CPU runtime. Proves the plumbing, not the accuracy.
Takes about 2-4 minutes on a laptop CPU."""
from __future__ import annotations
import json
import shutil
import sys
import tempfile
from pathlib import Path


def run_smoke(work=None) -> int:
    root = Path(work) if work else Path(tempfile.mkdtemp(prefix="dama_smoke_"))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from fixtures import make_sources
    from .calibrate import print_report, run_calibration
    from .config import DataConfig, RetrieverConfig, TrainConfig, ops_defaults
    from .data.build import build_all
    from .export import export_all
    from .train.decision import train_decision
    from .train.ops import train_ops
    from .train.retriever import train_retriever

    locomo = None
    try:  # real LoCoMo is small and on GitHub; fall back to a fake file offline
        from .data.download import _get
        locomo = root / "locomo10.json"
        _get(DataConfig().locomo_url, locomo)
    except Exception as e:
        print(f"LoCoMo download unavailable ({e}); using a fake LoCoMo file")
        locomo = None
    sources = make_sources(root / "raw", locomo_path=locomo)
    cfg = DataConfig(synthetic_ops={"train": 3000, "val": 400, "calib": 400, "test": 400}, negatives_per_positive=5)
    manifest = build_all(cfg, sources, root / "data")
    print(json.dumps({s: sum(v.values()) for s, v in manifest["stats"]["decision"].items() if s != "pools"}))

    train_retriever(root / "data", root / "runs" / "retriever", RetrieverConfig(tiny=True, epochs=2, batch_size=16, lr=1e-3))
    dcfg = TrainConfig(tiny=True, batch_size=32, epochs=1, eval_every=100, save_every=50, max_minutes=2, max_length=128, lr=1e-3, head_lr=1e-3)
    train_decision(root / "data", root / "runs" / "decision", dcfg)
    ocfg = ops_defaults()
    ocfg.tiny, ocfg.batch_size, ocfg.epochs, ocfg.eval_every, ocfg.save_every = True, 32, 1, 50, 50
    ocfg.max_minutes, ocfg.max_length, ocfg.lr, ocfg.head_lr = 2, 96, 1e-3, 1e-3
    train_ops(root / "data", root / "runs" / "ops", ocfg)
    # resume check: running again must detect the finished run instead of retraining
    train_ops(root / "data", root / "runs" / "ops", ocfg)

    _, report = run_calibration(root / "data", root / "runs" / "decision" / "best", root / "runs" / "ops" / "best",
                                root / "runs" / "calibration", use_dense=False, max_pool_questions=10,
                                retriever_dir=root / "runs" / "retriever" / "best")
    print_report(report)
    meta = export_all(root / "data", root / "runs" / "decision" / "best", root / "runs" / "ops" / "best",
                      root / "runs" / "calibration", root / "export" / "dama_tier1",
                      retriever_dir=root / "runs" / "retriever" / "best")
    sys.path.insert(0, str(root / "export" / "dama_tier1"))
    from runtime import Tier1
    t1 = Tier1(root / "export" / "dama_tier1")
    vecs = t1.embed(["user: I moved to Pune.", "user: I like tea."])
    probe = {"first_stage": t1.first_stage("Where do I live?", vecs),
             "relevance": t1.relevance("Where do I live?", ["[2023/05/01] user: I moved to Pune.", "user: I like tea."]).round(3).tolist(),
             "need": round(t1.need("What did I tell you about my landlord?"), 3),
             "route": {k: round(v, 3) for k, v in t1.route("When did I start the new job?").items()},
             "op": {k: round(v, 3) for k, v in t1.op("The user lives in Berlin.", "I moved to Pune last month").items()},
             "select": t1.select("Where do I live?", ["[2023/05/01] user: I moved to Pune.", "user: I like tea."], alpha=0.1)}
    print("runtime probe:", json.dumps(probe))
    from .probes import run_runtime
    print("runtime probes (accuracy only; tiny models are random):", {k: v["accuracy"] for k, v in run_runtime(t1).items()})
    print(f"\nSMOKE TEST PASSED - work dir {root}")
    return 0

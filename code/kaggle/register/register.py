"""Fallback: register the latest dama-tier1-training output (one checkpoint) as new versions of the private Kaggle Model,
without retraining. The training notebook does this itself; use this only when its upload cell failed or a variation
was added later. CPU only. Set VARIATIONS to what is missing.
  other/tier1                  CPU bundle + CHECKPOINT.json + report + calibration + code snapshot
  transformers/tier1-pytorch   trained PyTorch models (retriever/, decision/, ops/) for further training"""
import glob, json, os, shutil, subprocess, sys

OWNER_MODEL = "s41r4j/dama-tier1"
VARIATIONS = ["transformers/tier1-pytorch"]   # L1.2's bundle is already Model version 2 of other/tier1

hits = glob.glob("/kaggle/input/**/CHECKPOINT.json", recursive=True)
if not hits:
    raise SystemExit("no CHECKPOINT.json under /kaggle/input: attach the dama-tier1-training output")
work = os.path.dirname(min(hits, key=len))
ckpt = json.load(open(os.path.join(work, "CHECKPOINT.json")))
print("checkpoint:", ckpt["label"], "| created", ckpt["created"], "| code", ckpt["code_sha256"][:12], "| from", work)
items = {
    "other/tier1": [(os.path.join(work, "export", "dama_tier1"), ".")] + [(os.path.join(work, n), os.path.basename(n)) for n in
                   ("CHECKPOINT.json", "code_snapshot.zip", "runs/calibration/report.json", "runs/calibration/calibration.json")],
    "transformers/tier1-pytorch": [(os.path.join(work, "runs", m, "best"), m) for m in ("retriever", "decision", "ops")]
                                  + [(os.path.join(work, "CHECKPOINT.json"), "CHECKPOINT.json")],
}
try:
    import kagglehub
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
    import kagglehub
stage = "/kaggle/working/model_upload"
for variation in VARIATIONS:
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for src, dst in items[variation]:
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(stage, dst), dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "checkpoint.pt"))
        elif os.path.exists(src):
            shutil.copy(src, os.path.join(stage, dst))
    print(variation, "files:", sorted(os.listdir(stage)))
    kagglehub.model_upload(f"{OWNER_MODEL}/{variation}", stage, version_notes=f"{ckpt['label']}: {ckpt['notes']}")
    print(f"REGISTERED {ckpt['label']} as a new version of {OWNER_MODEL}/{variation}")
shutil.rmtree(stage, ignore_errors=True)

from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .contracts import ModelConfig
from .model import DAMADecisionModel


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_model(model,tokenizer,directory,*,trained,metrics=None,dataset_manifest=None):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    # clone removes shared storage safely; these are model artifacts, not pickled code.
    state={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
    save_file(state,str(root/"model.safetensors"))
    (root/"model_config.json").write_text(model.config.model_dump_json(indent=2)+"\n")
    model.encoder.config.save_pretrained(root/"encoder_config")
    tokenizer.save_pretrained(root/"tokenizer")
    manifest={"schema_version":1,"training_status":"trained" if trained else "untrained",
              "encoder":model.config.encoder,"encoder_revision":model.config.encoder_revision,
              "encoder_mode":model.config.encoder_mode,"parameter_counts":model.parameter_counts(),
              "python":platform.python_version(),"torch":torch.__version__,"metrics":metrics or {},
              "dataset_manifest":dataset_manifest,"scores_calibrated":False,
              "architecture":"shared encoder + candidate contextualizer + parallel supervised heads",
              "files":{}}
    for path in root.rglob("*"):
        if path.is_file() and path.name!="manifest.json": manifest["files"][str(path.relative_to(root))]=sha(path)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def load_model(directory,*,allow_untrained=False,device="cpu"):
    root=Path(directory); manifest=json.loads((root/"manifest.json").read_text())
    if manifest["training_status"]!="trained" and not allow_untrained: raise ValueError("untrained artifacts require --allow-untrained; they have no learned capability")
    for relative,digest in manifest["files"].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError("artifact path escapes directory")
        if sha(path)!=digest: raise ValueError("artifact hash mismatch: "+relative)
    config=ModelConfig.model_validate_json((root/"model_config.json").read_text())
    if manifest["encoder_revision"]!=config.encoder_revision: raise ValueError("encoder revision mismatch")
    encoder_config=AutoConfig.from_pretrained(root/"encoder_config",local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_config(encoder_config,trust_remote_code=False)
    model=DAMADecisionModel(config,encoder=encoder,load_pretrained=False)
    model.load_state_dict(load_file(str(root/"model.safetensors")),strict=True)
    tokenizer=AutoTokenizer.from_pretrained(root/"tokenizer",local_files_only=True,trust_remote_code=False)
    return model.to(device).eval(),tokenizer,manifest


def export_checkpoint(checkpoint,directory):
    root=Path(checkpoint); metadata=json.loads((root/"resume_manifest.json").read_text())
    if not (root/"COMPLETE").exists() or sha(root/"state.pt")!=metadata["state_sha256"]:
        raise ValueError("incomplete or corrupt checkpoint")
    config=ModelConfig.model_validate(metadata["fingerprint"]["config"]["model"])
    encoder_config=AutoConfig.from_pretrained(root/"encoder_config",local_files_only=True,trust_remote_code=False)
    model=DAMADecisionModel(config,encoder=AutoModel.from_config(encoder_config,trust_remote_code=False),load_pretrained=False)
    state=torch.load(root/"state.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state["model"],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(root/"tokenizer",local_files_only=True,trust_remote_code=False)
    return export_model(model,tokenizer,directory,trained=state["step"]>0,metrics={"training_steps":state["step"]},dataset_manifest=metadata["dataset_manifest"])

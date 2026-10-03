"""Colab/CUDA-only supervised decision-head training, with optional encoder LoRA."""
from __future__ import annotations
import json
import platform
from pathlib import Path
from .contracts import TrainConfig


def require_cuda():
    if platform.system()=="Darwin": raise RuntimeError("Training on this Mac is forbidden. Run this entry point on Colab CUDA.")
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU training fallback is disabled")
    return torch


def choose_precision(requested,torch):
    bf16=torch.cuda.is_bf16_supported(including_emulation=False)
    if requested=="bf16" and not bf16: raise ValueError("BF16 is unsupported on this GPU")
    selected=("bf16" if bf16 else "fp16") if requested=="auto" else requested
    return selected,{"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[selected]


def runtime():
    torch=require_cuda()
    free,total=torch.cuda.mem_get_info()
    return {"gpu":torch.cuda.get_device_name(),"vram_free_bytes":free,"vram_total_bytes":total,
            "bf16_supported":torch.cuda.is_bf16_supported(including_emulation=False),"torch":torch.__version__,"cuda":torch.version.cuda}


def train(config_path,dataset_dir,output,*,resume=None,smoke=False,allow_synthetic=False):
    # Enforce before loading a base model, creating an optimizer, or touching an existing run.
    torch=require_cuda()
    import random
    import numpy as np
    from transformers import AutoTokenizer
    from .artifacts import export_model,sha
    from .batching import Collator,device_batch
    from .data import read_examples,validate
    from .evaluate import evaluate
    from .inference import Predictor
    from .model import DAMADecisionModel,decision_loss
    config=TrainConfig.model_validate_json(Path(config_path).read_text())
    valid=validate(dataset_dir); data_root=Path(dataset_dir)
    dataset_manifest=json.loads((data_root/"manifest.json").read_text())
    if valid["synthetic"] and not (allow_synthetic or smoke): raise ValueError("synthetic fixtures are only a software smoke test; pass --allow-synthetic explicitly to experiment")
    root=Path(output)
    if root.exists() and any(root.iterdir()) and not resume: raise ValueError("existing output requires --resume or a new run directory")
    root.mkdir(parents=True,exist_ok=True)
    fingerprint={"config":config.model_dump(),"dataset_manifest_sha256":sha(data_root/"manifest.json")}
    if resume:
        metadata=json.loads((Path(resume)/"resume_manifest.json").read_text())
        if metadata["fingerprint"]!=fingerprint: raise ValueError("resume config/dataset differ from original run")
        if not (Path(resume)/"COMPLETE").exists() or sha(Path(resume)/"state.pt")!=metadata["state_sha256"]: raise ValueError("incomplete or corrupt checkpoint")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    tokenizer=AutoTokenizer.from_pretrained(config.model.encoder,revision=config.model.encoder_revision,trust_remote_code=False)
    model=DAMADecisionModel(config.model).to("cuda")
    collator=Collator(tokenizer,config.model)
    train_rows=read_examples(data_root/"train.jsonl"); dev_rows=read_examples(data_root/"dev.jsonl")
    # Reject overflow and unsupported label mappings before constructing an optimizer.
    for row in train_rows+dev_rows: collator([row])
    selected,dtype=choose_precision(config.precision,torch)
    scaler=torch.amp.GradScaler("cuda",enabled=selected=="fp16")
    groups=[{"params":[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith("encoder.")],"lr":config.learning_rate}]
    encoder_params=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith("encoder.")]
    if encoder_params: groups.append({"params":encoder_params,"lr":config.encoder_learning_rate})
    optimizer=torch.optim.AdamW(groups,weight_decay=config.weight_decay)
    step=0; start_epoch=0; start_batch=0
    if resume:
        state=torch.load(Path(resume)/"state.pt",map_location="cuda",weights_only=True)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        step=state["step"]; start_epoch=state["epoch"]; start_batch=state["next_batch"]
        if start_epoch>=config.epochs: raise ValueError("checkpoint is beyond configured epochs")
        total_batches=(len(train_rows)+config.microbatch-1)//config.microbatch
        if start_epoch==config.epochs-1 and start_batch>=total_batches:
            raise ValueError("checkpoint already completed all configured epochs")
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]]); random.setstate(state["python_rng"])
        rng=state["numpy_rng"]; np.random.set_state((rng[0],np.array(rng[1],dtype=np.uint32),rng[2],rng[3],rng[4]))
    run_manifest={"fingerprint":fingerprint,"runtime":runtime(),"precision":selected,"parameter_counts":model.parameter_counts(),
                  "synthetic":valid["synthetic"],"smoke":smoke,"resumed_from":str(resume) if resume else None}
    (root/"run_manifest.json").write_text(json.dumps(run_manifest,indent=2)+"\n")
    max_steps=min(config.max_steps or 3,3) if smoke else config.max_steps
    if max_steps is not None and step>=max_steps: raise ValueError("checkpoint already reached configured step limit")
    torch.cuda.reset_peak_memory_stats()
    def checkpoint(epoch,next_batch):
        path=root/f"checkpoint-{step:06d}"; path.mkdir(exist_ok=True)
        (path/"COMPLETE").unlink(missing_ok=True)
        rng=np.random.get_state()
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"step":step,"epoch":epoch,"next_batch":next_batch,
               "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate(),
               "numpy_rng":(rng[0],rng[1].tolist(),rng[2],rng[3],rng[4])}
        # Write data then metadata then completion marker; interrupted checkpoints are invalid.
        temporary=path/"state.tmp"; torch.save(state,temporary); temporary.replace(path/"state.pt")
        model.encoder.config.save_pretrained(path/"encoder_config")
        tokenizer.save_pretrained(path/"tokenizer")
        (path/"resume_manifest.json").write_text(json.dumps({"fingerprint":fingerprint,"state_sha256":sha(path/"state.pt"),"dataset_manifest":dataset_manifest,
            "encoder_attention_implementation":model.encoder.config._attn_implementation},indent=2))
        (path/"COMPLETE").write_text("1\n")
        return path
    stopped=False
    for epoch in range(start_epoch,config.epochs):
        order=np.random.RandomState(config.seed+epoch).permutation(len(train_rows)).tolist()
        batches=[order[i:i+config.microbatch] for i in range(0,len(order),config.microbatch)]
        cursor=start_batch if epoch==start_epoch else 0
        while cursor<len(batches):
            model.train(); optimizer.zero_grad(set_to_none=True)
            group=batches[cursor:cursor+config.gradient_accumulation]; losses=[]
            for indexes in group:
                batch=collator([train_rows[i] for i in indexes]); labels={k:v.to("cuda") for k,v in batch["labels"].items()}
                with torch.amp.autocast("cuda",dtype=dtype,enabled=selected!="fp32"):
                    outputs=model(**device_batch(batch,"cuda")); loss,parts=decision_loss(outputs,labels,batch["model_inputs"]["candidate_mask"].to("cuda"))
                if not torch.isfinite(loss): raise RuntimeError("non-finite loss; retain last completed checkpoint")
                scaler.scale(loss/len(group)).backward(); losses.append(float(loss.detach()))
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config.grad_clip)
            scaler.step(optimizer); scaler.update(); step+=1; cursor+=len(group)
            with (root/"training.jsonl").open("a") as f: f.write(json.dumps({"step":step,"epoch":epoch,"loss":float(np.mean(losses)),"peak_vram_bytes":torch.cuda.max_memory_allocated()})+"\n")
            if step%config.save_steps==0: checkpoint(epoch,cursor)
            if max_steps is not None and step>=max_steps: stopped=True; break
        if stopped: break
    last=checkpoint(epoch,cursor)
    predictor=Predictor(model,tokenizer,version=f"dama-step-{step}",device="cuda")
    metrics=evaluate(predictor,dev_rows,repeats=1,warmup=1)
    metrics.update(training_steps=step,peak_vram_bytes=torch.cuda.max_memory_allocated(),smoke=smoke)
    export_model(model,tokenizer,root/"export",trained=True,metrics=metrics,dataset_manifest=dataset_manifest)
    (root/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    return {"export":str(root/"export"),"resume_checkpoint":str(last),"metrics":metrics,"note":"synthetic smoke quality is not a research result" if valid["synthetic"] else "dev-only evaluation; test remains held out"}

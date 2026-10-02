from __future__ import annotations
import time
import torch
from .batching import Collator, InputOverflow, device_batch
from .contracts import Decision, ModelInput, OPERATIONS, TYPES, Span


def abstain(x,version,reason):
    return Decision(task=x.task,operation="NOOP" if x.task=="write" else None,needs_fallback=True,fallback_score=1.0,reason=reason,model_version=version)


def decode(x,output,offsets,prefix,config,version):
    if any(not torch.isfinite(v).all() for v in output.values()):
        return abstain(x,version,"invalid_scores")
    op_scores=torch.softmax(output["operation_logits"].float(),dim=-1).tolist()
    fallback=float(torch.softmax(output["fallback_logits"].float(),dim=-1)[1])
    scores={c.id:float(torch.sigmoid(output["relevance_logits"][i]).item()) for i,c in enumerate(x.candidates)}
    base={"task":x.task,"operation_scores":dict(zip(OPERATIONS,op_scores)),"relevance_scores":scores,
          "fallback_score":fallback,"needs_fallback":fallback>=config.fallback_threshold,"reason":"model_decision","model_version":version}
    if x.task=="retrieve":
        if not x.candidates: return abstain(x,version,"no_candidates")
        allowed=lambda c: c.status!="archived" and (x.temporal_mode=="all" or c.status==("current" if x.temporal_mode=="current" else "superseded"))
        selected=sorted((c.id for c in x.candidates if allowed(c) and scores[c.id]>=config.relevance_threshold),key=lambda mid:(-scores[mid],mid))
        return Decision(**base,selected_ids=selected)
    if x.source_role!="user": return abstain(x,version,"unsupported_source")
    index=max(range(4),key=op_scores.__getitem__); operation=OPERATIONS[index]
    if op_scores[index]<config.decision_threshold or base["needs_fallback"]:
        return Decision(**{**base,"operation":"NOOP","needs_fallback":True,"reason":"low_score"})
    target=None
    if operation in {"UPDATE","ARCHIVE"}:
        target_idx=int(output["target_logits"].argmax())
        if target_idx>=len(x.candidates) or x.candidates[target_idx].status!="current": return abstain(x,version,"invalid_target")
        target=x.candidates[target_idx]
    if operation=="ARCHIVE" and x.text!=f"Archive memory {target.id}.": return abstain(x,version,"explicit_archive_required")
    span=None; evidence=None; kind=None
    if operation in {"ADD","UPDATE"}:
        # Joint constrained argmax: independent argmax can produce end < start.
        start=output["start_logits"].float(); end=output["end_logits"].float()
        valid=torch.tensor([b>a and a>=prefix and b<=prefix+len(x.text) for a,b in offsets.tolist()],device=start.device)
        allowed=torch.triu(torch.ones((len(start),len(start)),device=start.device,dtype=torch.bool)) & valid[:,None] & valid[None,:]
        if not allowed.any(): return abstain(x,version,"invalid_span")
        joint=(start[:,None]+end[None,:]).masked_fill(~allowed,-float("inf"))
        flat=int(joint.argmax()); i,j=divmod(flat,len(start))
        a,b=int(offsets[i,0])-prefix,int(offsets[j,1])-prefix
        if not 0<=a<b<=len(x.text): return abstain(x,version,"invalid_span")
        span=Span(start=a,end=b); evidence=x.text[a:b]; kind=TYPES[int(output["type_logits"].argmax())]
    return Decision(**base,operation=operation,memory_type=kind,target_id=target.id if target else None,
                    expected_version=target.version if target else None,evidence_span=span,evidence_text=evidence,
                    source_id=x.source_id if span else None)


class Predictor:
    def __init__(self,model,tokenizer,*,version="untrained",device="cpu"):
        self.model=model.to(device).eval(); self.collator=Collator(tokenizer,model.config); self.version=version; self.device=device
    def predict(self,x: ModelInput):
        start=time.perf_counter()
        try: batch=self.collator([x])
        except InputOverflow: return abstain(x,self.version,"input_overflow"),{"seconds":time.perf_counter()-start,"input_tokens":None,"forward_passes":0}
        with torch.inference_mode():
            outputs=self.model(**device_batch(batch,self.device))
            decision=decode(x,{k:v[0].cpu() for k,v in outputs.items()},batch["offsets"][0],batch["prefixes"][0],self.model.config,self.version)
        if self.device.startswith("cuda"): torch.cuda.synchronize()
        return decision,{"seconds":time.perf_counter()-start,"input_tokens":int(batch["model_inputs"]["attention_mask"].sum()),
                         "output_generated_tokens":0,"forward_passes":1,"candidates":len(x.candidates)}

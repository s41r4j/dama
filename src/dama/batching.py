from __future__ import annotations
import torch
from .contracts import Example, ModelInput, OPERATIONS, TYPES


class InputOverflow(ValueError): pass


def render_event(x: ModelInput):
    recent="\n".join(f"{t.at} {t.role}: {t.text}" for t in x.recent)
    prefix=f"task={x.task}\ntime={x.at}\nrole={x.source_role}\ntemporal_mode={x.temporal_mode}\nbudget_tokens={x.budget_tokens}\nrecent:\n{recent}\ncurrent:\n"
    return prefix+x.text,len(prefix)


def render_candidate(c):
    return f"status={c.status}\nvalid_from={c.valid_from}\nvalid_to={c.valid_to}\ntype={c.memory_type}\ncontent:\n{c.text}"


class Collator:
    def __init__(self, tokenizer, config):
        self.tokenizer,self.config=tokenizer,config
        if not tokenizer.is_fast: raise ValueError("fast tokenizer with offset mappings required")
    def __call__(self, rows: list[Example | ModelInput]):
        examples=rows if rows and isinstance(rows[0],Example) else None
        inputs=[r.input if isinstance(r,Example) else r for r in rows]
        if not inputs: raise ValueError("empty batch")
        count=max(len(x.candidates) for x in inputs)
        if count>self.config.max_candidates: raise InputOverflow("candidate limit exceeded; caller must supply a bounded set")
        texts=[]; prefixes=[]
        for x in inputs:
            event,prefix=render_event(x); prefixes.append(prefix)
            texts.extend([event]+[render_candidate(c) for c in x.candidates]+[""]*(count-len(x.candidates)))
        encoded=self.tokenizer(texts,padding=True,truncation=False,return_offsets_mapping=True,return_tensors="pt")
        length=encoded["input_ids"].shape[1]
        if length>self.config.max_length: raise InputOverflow(f"{length} tokens exceeds {self.config.max_length}; no silent truncation")
        batch=len(inputs); items=count+1
        offsets=encoded.pop("offset_mapping").reshape(batch,items,length,2)[:,0]
        model_inputs={k:v.reshape(batch,items,length) for k,v in encoded.items()}
        candidate_mask=torch.tensor([[i<len(x.candidates) for i in range(count)] for x in inputs],dtype=torch.bool).reshape(batch,count)
        source_mask=torch.zeros((batch,length),dtype=torch.bool)
        for b,x in enumerate(inputs):
            for i,(start,end) in enumerate(offsets[b].tolist()):
                source_mask[b,i]=end>start and start>=prefixes[b] and end<=prefixes[b]+len(x.text)
        model_inputs.update(candidate_mask=candidate_mask,source_mask=source_mask)
        labels=None
        if examples:
            labels={k:torch.full((batch,),-100,dtype=torch.long) for k in ["operation","memory_type","target","start","end","fallback"]}
            labels["relevance"]=torch.full((batch,count),-100.0)
            for b,row in enumerate(examples):
                y=row.label; labels["fallback"][b]=int(y.needs_fallback)
                if y.operation is not None:
                    labels["operation"][b]=OPERATIONS.index(y.operation)
                    labels["target"][b]=next((i for i,c in enumerate(row.input.candidates) if c.id==y.target_id),count)
                if y.memory_type is not None: labels["memory_type"][b]=TYPES.index(y.memory_type)
                if y.evidence_span:
                    span=y.evidence_span; starts=[]; ends=[]
                    for i,(start,end) in enumerate(offsets[b].tolist()):
                        if source_mask[b,i] and start<=prefixes[b]+span.start<end: starts.append(i)
                        if source_mask[b,i] and start<prefixes[b]+span.end<=end: ends.append(i)
                    if not starts or not ends: raise ValueError("evidence span cannot be mapped to observed tokens")
                    labels["start"][b]=starts[0]; labels["end"][b]=ends[-1]
                if y.relevant_ids is not None:
                    for i,c in enumerate(row.input.candidates): labels["relevance"][b,i]=float(c.id in y.relevant_ids)
        return {"model_inputs":model_inputs,"labels":labels,"inputs":inputs,"offsets":offsets,"prefixes":prefixes}


def device_batch(batch,device):
    return {k:v.to(device) for k,v in batch["model_inputs"].items()}

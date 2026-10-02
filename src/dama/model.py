"""Shared pretrained encoder + small contextualizer + parallel decision heads."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .contracts import ModelConfig


class DAMADecisionModel(nn.Module):
    def __init__(self, config: ModelConfig, encoder=None, *, load_pretrained=True):
        super().__init__(); self.config=config
        if encoder is None:
            if not load_pretrained: raise ValueError("supply a tiny test encoder explicitly")
            encoder=AutoModel.from_pretrained(config.encoder,revision=config.encoder_revision,trust_remote_code=False)
        if config.encoder_mode=="lora":
            from peft import LoraConfig, TaskType, get_peft_model
            encoder=get_peft_model(encoder,LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=config.lora_rank,
                                   lora_alpha=config.lora_alpha,lora_dropout=config.dropout,target_modules=["query","value"],bias="none"))
        else:
            for parameter in encoder.parameters(): parameter.requires_grad_(False)
        self.encoder=encoder
        width=encoder.config.hidden_size; dim=config.hidden_dim
        self.projection=nn.Sequential(nn.Linear(width,dim),nn.LayerNorm(dim),nn.GELU())
        self.context=None
        if config.context_layers:
            layer=nn.TransformerEncoderLayer(dim,config.attention_heads,dim*2,dropout=config.dropout,batch_first=True,activation="gelu")
            self.context=nn.TransformerEncoder(layer,config.context_layers,enable_nested_tensor=False)
        self.operation=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(dim,4))
        self.memory_type=nn.Linear(dim,5)
        self.fallback=nn.Linear(dim,2)
        pair_width=dim*4
        self.relevance=nn.Sequential(nn.Linear(pair_width,dim),nn.GELU(),nn.Linear(dim,1))
        self.target=nn.Sequential(nn.Linear(pair_width,dim),nn.GELU(),nn.Linear(dim,1))
        self.null_target=nn.Linear(dim,1)
        self.span=nn.Linear(width,2)

    def train(self, mode=True):
        super().train(mode)
        # Frozen feature extraction stays deterministic while trainable heads use dropout.
        if self.config.encoder_mode=="frozen": self.encoder.eval()
        return self

    def forward(self,input_ids,attention_mask,candidate_mask,source_mask,token_type_ids=None):
        batch,items,length=input_ids.shape
        args={"input_ids":input_ids.reshape(-1,length),"attention_mask":attention_mask.reshape(-1,length)}
        if token_type_ids is not None: args["token_type_ids"]=token_type_ids.reshape(-1,length)
        if self.config.encoder_mode=="frozen":
            self.encoder.eval()
            with torch.no_grad(): hidden=self.encoder(**args).last_hidden_state
        else: hidden=self.encoder(**args).last_hidden_state
        # Preserve gradients through the frozen features into trainable projections.
        expanded=args["attention_mask"].unsqueeze(-1)
        pooled=hidden[:,0] if self.config.pooling=="cls" else (hidden*expanded).sum(1)/expanded.sum(1).clamp(min=1)
        contextual=self.projection(pooled).reshape(batch,items,-1)
        valid=torch.cat([torch.ones((batch,1),device=candidate_mask.device,dtype=torch.bool),candidate_mask.bool()],dim=1)
        if self.context: contextual=self.context(contextual,src_key_padding_mask=~valid)
        event=contextual[:,0]; candidates=contextual[:,1:]
        expanded_event=event[:,None,:].expand_as(candidates)
        pair=torch.cat([expanded_event,candidates,expanded_event-candidates,expanded_event*candidates],dim=-1)
        relevance=self.relevance(pair).squeeze(-1).masked_fill(~candidate_mask.bool(),-1e4)
        targets=self.target(pair).squeeze(-1).masked_fill(~candidate_mask.bool(),-1e4)
        target=torch.cat([targets,self.null_target(event)],dim=1)
        source_hidden=hidden.reshape(batch,items,length,-1)[:,0]
        spans=self.span(source_hidden).masked_fill(~source_mask.bool().unsqueeze(-1),-1e4)
        return {"operation_logits":self.operation(event),"type_logits":self.memory_type(event),
                "fallback_logits":self.fallback(event),"relevance_logits":relevance,"target_logits":target,
                "start_logits":spans[:,:,0],"end_logits":spans[:,:,1]}

    def parameter_counts(self):
        return {"total":sum(p.numel() for p in self.parameters()),"trainable":sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder":sum(p.numel() for p in self.encoder.parameters())}


def masked_ce(logits,labels):
    active=labels>=0
    return F.cross_entropy(logits[active].float(),labels[active]) if active.any() else logits.sum()*0


def decision_loss(outputs,labels,candidate_mask):
    losses={"operation":masked_ce(outputs["operation_logits"],labels["operation"]),
            "memory_type":masked_ce(outputs["type_logits"],labels["memory_type"]),
            "target":masked_ce(outputs["target_logits"],labels["target"]),
            "fallback":masked_ce(outputs["fallback_logits"],labels["fallback"]),
            "span_start":masked_ce(outputs["start_logits"],labels["start"]),
            "span_end":masked_ce(outputs["end_logits"],labels["end"])}
    active=(labels["relevance"]>=0)&candidate_mask.bool()
    losses["relevance"]=F.binary_cross_entropy_with_logits(outputs["relevance_logits"][active].float(),labels["relevance"][active].float()) if active.any() else outputs["relevance_logits"].sum()*0
    return sum(losses.values()),losses

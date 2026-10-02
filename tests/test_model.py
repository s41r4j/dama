import json
import platform
import torch
import pytest
from dama.artifacts import export_checkpoint,export_model,load_model
from dama.batching import Collator,InputOverflow
from dama.contracts import Candidate,Decision,Example,Label,ModelConfig,ModelInput,Scope,Span,TrainConfig
from dama.data import fixtures,prepare,sha,validate
from dama.inference import Predictor,decode
from dama.model import DAMADecisionModel,decision_loss
from dama.testing import tiny_model
from dama.training import choose_precision,require_cuda


@pytest.fixture
def network(tmp_path):
    torch.manual_seed(42); torch.set_num_threads(1)
    return tiny_model(tmp_path/"tiny")

@pytest.mark.parametrize("payload",[
    {"encoder_revision":"main"},{"hidden_dim":31},{"max_candidates":0},
    {"context_layers":8},{"unexpected":1},{"max_length":8},
])
def test_invalid_config(payload):
    with pytest.raises(ValueError): ModelConfig(**payload)

@pytest.mark.parametrize("payload",[
    '{"operation":"DELETE"}', '{"operation":3}', '{"operation":"NOOP","unexpected":1}',
    '{"evidence_span":{"start":4,"end":1}}',
])
def test_invalid_labels(payload):
    with pytest.raises(ValueError): Label.model_validate_json(payload)

def test_forward_shapes_no_grad(network):
    model,tokenizer=network; rows=[fixtures()[0],fixtures()[2],fixtures()[1]]
    batch=Collator(tokenizer,model.config)(rows)
    with torch.inference_mode(): result=model(**batch["model_inputs"])
    assert result["operation_logits"].shape==(3,4)
    assert result["type_logits"].shape==(3,5)
    assert result["relevance_logits"].shape==(3,2)
    assert result["target_logits"].shape==(3,3)
    assert result["start_logits"].shape==batch["model_inputs"]["source_mask"].shape
    assert all(p.grad is None for p in model.parameters())

def test_zero_candidates(network):
    model,tokenizer=network; batch=Collator(tokenizer,model.config)([fixtures()[0]])
    with torch.inference_mode(): result=model(**batch["model_inputs"])
    assert result["relevance_logits"].shape==(1,0) and result["target_logits"].shape==(1,1)
    loss,parts=decision_loss(result,batch["labels"],batch["model_inputs"]["candidate_mask"])
    assert torch.isfinite(loss) and not loss.requires_grad

def test_loss_masks_retrieval_write_labels(network):
    model,tokenizer=network; batch=Collator(tokenizer,model.config)([fixtures()[1]])
    with torch.inference_mode(): result=model(**batch["model_inputs"]); loss,parts=decision_loss(result,batch["labels"],batch["model_inputs"]["candidate_mask"])
    assert parts["operation"]==0 and parts["span_start"]==0 and parts["target"]==0
    assert torch.isfinite(loss) and parts["relevance"]>0

def test_padding_mask_and_permutation_equivariance(network):
    model,tokenizer=network; collator=Collator(tokenizer,model.config)
    x=fixtures()[1].input; flipped=x.model_copy(update={"candidates":list(reversed(x.candidates))})
    b=collator([x,flipped])
    with torch.inference_mode(): outputs=model(**b["model_inputs"])
    assert torch.allclose(outputs["operation_logits"][0],outputs["operation_logits"][1],atol=1e-5)
    assert torch.allclose(outputs["relevance_logits"][0],outputs["relevance_logits"][1].flip(0),atol=1e-5)
    padded=collator([fixtures()[0].input,x])
    with torch.inference_mode(): p=model(**padded["model_inputs"])
    assert (p["relevance_logits"][0]<-1000).all()

def test_frozen_encoder_stays_eval(network):
    model,_=network; model.train()
    assert model.training and not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())

def test_overflow_is_explicit(network):
    model,tokenizer=network; x=fixtures()[0].input.model_copy(update={"text":"hello "*1000})
    with pytest.raises(InputOverflow): Collator(tokenizer,model.config)([x])
    d,timing=Predictor(model,tokenizer).predict(x)
    assert d.reason=="input_overflow" and timing["forward_passes"]==0

def test_candidate_overflow(network):
    model,tokenizer=network; x=fixtures()[1].input
    many=[x.candidates[0].model_copy(update={"id":str(i)}) for i in range(33)]
    with pytest.raises(InputOverflow): Collator(tokenizer,model.config)([x.model_copy(update={"candidates":many})])

def test_span_mapping_and_no_truncation(network):
    model,tokenizer=network; rows=[fixtures()[0]]; b=Collator(tokenizer,model.config)(rows)
    start=int(b["labels"]["start"][0]); end=int(b["labels"]["end"][0]); prefix=b["prefixes"][0]
    assert int(b["offsets"][0,start,0])-prefix==0
    assert int(b["offsets"][0,end,1])-prefix==len(rows[0].input.text)

def forced_output(batch,operation="ADD"):
    length=batch["model_inputs"]["source_mask"].shape[1]; count=batch["model_inputs"]["candidate_mask"].shape[1]
    op=torch.full((4,),-20.0); op[["ADD","NOOP","UPDATE","ARCHIVE"].index(operation)]=20
    start=torch.full((length,),-20.0); end=start.clone(); valid=batch["model_inputs"]["source_mask"][0].nonzero().flatten()
    start[valid[0]]=20; end[valid[-1]]=20
    return {"operation_logits":op,"type_logits":torch.tensor([20.,-20,-20,-20,-20]),"fallback_logits":torch.tensor([20.,-20]),
            "relevance_logits":torch.zeros(count),"target_logits":torch.zeros(count+1),"start_logits":start,"end_logits":end}

def test_decoder_extracts_only_source_span(network):
    model,tokenizer=network; x=fixtures()[0].input; batch=Collator(tokenizer,model.config)([x]); out=forced_output(batch)
    d=decode(x,out,batch["offsets"][0],batch["prefixes"][0],model.config,"controlled-logit-test")
    assert d.operation=="ADD" and d.evidence_text==x.text and d.evidence_span==Span(start=0,end=len(x.text))
    assert not d.scores_calibrated

def test_decoder_rejects_assistant_mutation(network):
    model,tokenizer=network; x=fixtures()[0].input.model_copy(update={"source_role":"assistant"}); batch=Collator(tokenizer,model.config)([x])
    d=decode(x,forced_output(batch),batch["offsets"][0],batch["prefixes"][0],model.config,"test")
    assert d.operation=="NOOP" and d.reason=="unsupported_source"

def test_decoder_cannot_invent_target(network):
    model,tokenizer=network; x=fixtures()[0].input; batch=Collator(tokenizer,model.config)([x]); out=forced_output(batch,"UPDATE")
    d=decode(x,out,batch["offsets"][0],batch["prefixes"][0],model.config,"test")
    assert d.operation=="NOOP" and d.reason=="invalid_target"

def test_decoder_archive_requires_exact_user_request(network):
    model,tokenizer=network; x=fixtures()[1].input.model_copy(update={"task":"write","source_id":"u1","text":"Archive everything."})
    batch=Collator(tokenizer,model.config)([x]); out=forced_output(batch,"ARCHIVE"); out["target_logits"][0]=20
    d=decode(x,out,batch["offsets"][0],batch["prefixes"][0],model.config,"test")
    assert d.reason=="explicit_archive_required"

def test_scope_and_future_leakage():
    x=fixtures()[1].input.model_dump(); x["candidates"][0]["scope"]["user"]="different"
    with pytest.raises(ValueError): ModelInput.model_validate(x)
    x=fixtures()[1].input.model_dump(); x["candidates"][0]["valid_from"]="2027-01-01T00:00:00+00:00"
    with pytest.raises(ValueError): ModelInput.model_validate(x)

def test_dataset_manifests_and_group_leakage(tmp_path):
    root=tmp_path/"dataset"; result=prepare(root); assert result["counts"]=={"train":56,"dev":12,"test":12}
    train=json.loads((root/"train.jsonl").read_text().splitlines()[0]); dev=(root/"dev.jsonl").read_text().splitlines(); row=json.loads(dev[0]); row["family"]=train["family"]; dev[0]=json.dumps(row)
    (root/"dev.jsonl").write_text("\n".join(dev)+"\n"); manifest=json.loads((root/"manifest.json").read_text()); manifest["files"]["dev.jsonl"]["sha256"]=sha(root/"dev.jsonl")
    (root/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="leakage"): validate(root)

def test_checkpoint_roundtrip_requires_untrained_optin(network,tmp_path):
    model,tokenizer=network; root=tmp_path/"artifact"; manifest=export_model(model,tokenizer,root,trained=False)
    with pytest.raises(ValueError,match="untrained"): load_model(root)
    restored,tok,m=load_model(root,allow_untrained=True)
    assert m["encoder_attention_implementation"]==model.encoder.config._attn_implementation
    assert restored.encoder.config._attn_implementation==model.encoder.config._attn_implementation
    x=fixtures()[1].input
    first,_=Predictor(model,tokenizer,version="test").predict(x); second,_=Predictor(restored,tok,version="test").predict(x)
    assert first.selected_ids==second.selected_ids and first.operation==second.operation
    assert first.operation_scores==pytest.approx(second.operation_scores,abs=1e-6)
    assert first.relevance_scores==pytest.approx(second.relevance_scores,abs=1e-6)
    assert first.fallback_score==pytest.approx(second.fallback_score,abs=1e-6)
    assert manifest["training_status"]=="untrained"
    (root/"model_config.json").write_text("{}")
    with pytest.raises(ValueError,match="hash"): load_model(root,allow_untrained=True)

def test_training_guard_runs_before_torch_import(monkeypatch):
    monkeypatch.setattr(platform,"system",lambda:"Darwin")
    with pytest.raises(RuntimeError,match="forbidden"): require_cuda()

def test_precision_detection():
    class GPU:
        def is_bf16_supported(self): return False
    class Fake:
        cuda=GPU(); bfloat16="bf16"; float16="fp16"; float32="fp32"
    assert choose_precision("auto",Fake())[0]=="fp16"
    with pytest.raises(ValueError): choose_precision("bf16",Fake())

def test_lora_forward_and_artifact_without_training(network,tmp_path):
    base,tokenizer=network
    config=base.config.model_copy(update={"encoder_mode":"lora","lora_rank":2,"lora_alpha":4})
    model=DAMADecisionModel(config,encoder=base.encoder,load_pretrained=False).eval()
    batch=Collator(tokenizer,config)([fixtures()[1]])
    with torch.inference_mode(): outputs=model(**batch["model_inputs"])
    assert outputs["operation_logits"].shape==(1,4)
    assert any(p.requires_grad for p in model.encoder.parameters())
    root=tmp_path/"lora-artifact"; export_model(model,tokenizer,root,trained=False)
    restored,_,_=load_model(root,allow_untrained=True)
    assert restored.encoder.config._attn_implementation==model.encoder.config._attn_implementation
    for name,weight in model.state_dict().items():
        assert torch.equal(weight,restored.state_dict()[name]),name
    with torch.inference_mode(): second=restored(**batch["model_inputs"])
    for head in outputs:
        torch.testing.assert_close(outputs[head],second[head],rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize("attention",["eager","sdpa"])
def test_checkpoint_export_preserves_attention_without_training(network,tmp_path,attention):
    from transformers import BertModel
    base,tokenizer=network
    base.encoder.config._attn_implementation=attention
    model=DAMADecisionModel(base.config,encoder=BertModel(base.encoder.config),load_pretrained=False).eval()
    checkpoint=tmp_path/"checkpoint"
    checkpoint.mkdir()
    model.encoder.config.save_pretrained(checkpoint/"encoder_config")
    tokenizer.save_pretrained(checkpoint/"tokenizer")
    # A forward-only fixture checkpoint: no optimizer or backward pass.
    torch.save({"model":model.state_dict(),"step":0},checkpoint/"state.pt")
    (checkpoint/"resume_manifest.json").write_text(json.dumps({
        "fingerprint":{"config":{"model":model.config.model_dump()}},
        "state_sha256":sha(checkpoint/"state.pt"),"dataset_manifest":{},
        "encoder_attention_implementation":attention}))
    (checkpoint/"COMPLETE").write_text("1\n")
    export=tmp_path/"export"
    export_checkpoint(checkpoint,export)
    restored,_,manifest=load_model(export,allow_untrained=True)
    assert manifest["encoder_attention_implementation"]==attention
    assert restored.encoder.config._attn_implementation==attention
    batch=Collator(tokenizer,model.config)([fixtures()[1]])
    with torch.inference_mode():
        first=model(**batch["model_inputs"])
        second=restored(**batch["model_inputs"])
    for head in first:
        torch.testing.assert_close(first[head],second[head],rtol=1e-5,atol=1e-6)

def test_nonfinite_logits_abstain(network):
    model,tokenizer=network; x=fixtures()[0].input; batch=Collator(tokenizer,model.config)([x]); outputs=forced_output(batch)
    outputs['operation_logits'][0]=float('nan')
    decision=decode(x,outputs,batch['offsets'][0],batch['prefixes'][0],model.config,'test')
    assert decision.reason=='invalid_scores' and decision.needs_fallback

def test_decision_contract_rejects_mutating_retrieval():
    with pytest.raises(ValueError):
        Decision(task='retrieve',operation='ADD',fallback_score=0.0,needs_fallback=False,reason='model_decision',model_version='test')

@pytest.mark.parametrize('pooling',['mean','cls'])
def test_pooling_and_no_context_ablation(network,pooling):
    base,tokenizer=network; config=base.config.model_copy(update={'pooling':pooling,'context_layers':0})
    model=DAMADecisionModel(config,encoder=base.encoder,load_pretrained=False).eval()
    batch=Collator(tokenizer,config)([fixtures()[0],fixtures()[1]])
    with torch.inference_mode(): output=model(**batch['model_inputs'])
    assert output['operation_logits'].shape==(2,4) and torch.isfinite(output['operation_logits']).all()

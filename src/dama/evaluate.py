"""Model-level metrics; no Primary Agent, memory store or conversational application."""
from __future__ import annotations
import json
import math
import platform
import resource
import statistics
import time
from pathlib import Path
import numpy as np
from .inference import Predictor


def percentile(xs,q): return float(np.quantile(xs,q)) if xs else None


def expected_calibration_error(confidences,correct,bins=10):
    if not confidences: return None
    result=0.0
    for i in range(bins):
        selected=[j for j,c in enumerate(confidences) if i/bins<=c<(i+1)/bins or (i==bins-1 and c==1)]
        if selected:
            result+=len(selected)/len(confidences)*abs(statistics.mean(confidences[j] for j in selected)-statistics.mean(correct[j] for j in selected))
    return result


def evaluate(predictor,examples,output=None,*,repeats=3,warmup=1,artifact=None):
    if repeats<1 or warmup<0: raise ValueError("invalid repetition count")
    for _ in range(warmup): predictor.predict(examples[0].input)
    rows=[]; op=[]; raw_correct=[]; type_correct=[]; target=[]; spans=[]; precision=[]; recall=[]; fallback=[]; confidences=[]; latencies=[]; confusion={}
    for repeat in range(repeats):
        for row in examples:
            decision,timing=predictor.predict(row.input); y=row.label; d=decision.model_dump()
            result={"example_id":row.id,"family":row.family,"task":row.input.task,"repeat":repeat,"decision":d,"label":y.model_dump(),"timing":timing}
            latencies.append(timing["seconds"])
            # Quality counts each example once; repeats measure runtime variation.
            if repeat==0:
                fallback.append(int(decision.needs_fallback==y.needs_fallback))
                if y.operation is not None:
                    correct=int(decision.operation==y.operation); op.append(correct)
                    confusion.setdefault(y.operation,{}); confusion[y.operation][decision.operation]=confusion[y.operation].get(decision.operation,0)+1
                    confidences.append(max(decision.operation_scores.values(),default=0))
                    raw_correct.append(int(max(decision.operation_scores,key=decision.operation_scores.get,default=None)==y.operation))
                    if y.memory_type is not None: type_correct.append(int(decision.memory_type==y.memory_type))
                    if y.target_id is not None: target.append(int(decision.target_id==y.target_id))
                    if y.evidence_span is not None: spans.append(int(decision.evidence_span==y.evidence_span))
                else:
                    gold=set(y.relevant_ids); pred=set(decision.selected_ids); shared=len(gold&pred)
                    precision.append(shared/len(pred) if pred else float(not gold))
                    recall.append(shared/len(gold) if gold else float(not pred))
            rows.append(result)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    def mean(xs): return statistics.mean(xs) if xs else None
    mutated=[r for r in rows if r["repeat"]==0 and r["task"]=="write" and r["decision"]["operation"] in {"ADD","UPDATE","ARCHIVE"}]
    useful=[r for r in rows if r["repeat"]==0 and r["task"]=="write" and r["label"]["operation"] in {"ADD","UPDATE","ARCHIVE"}]
    summary={"model_version":predictor.version,"research_result":False,"trained_artifact":bool(artifact and artifact.get("training_status")=="trained"),
             "evaluation_scope":"model-level metrics; no downstream answer/cost claim",
             "examples":len(examples),"repeats":repeats,"warmup_excluded":warmup,"operation_accuracy":mean(op),"type_accuracy_on_supported":mean(type_correct),
             "target_accuracy":mean(target),"span_exact_match":mean(spans),"retrieval_precision":mean(precision),"retrieval_recall":mean(recall),
             "fallback_accuracy":mean(fallback),"raw_operation_accuracy":mean(raw_correct),"operation_ece":expected_calibration_error(confidences,raw_correct),"scores_calibrated":False,
             "write_precision":mean([int(r["decision"]["operation"]==r["label"]["operation"]) for r in mutated]),
             "useful_facts_missed":sum(r["decision"]["operation"]=="NOOP" for r in useful),"confusion":confusion,
             "p50_seconds":percentile(latencies,.5),"p95_seconds":percentile(latencies,.95),
             "input_tokens_total":sum(r["timing"].get("input_tokens") or 0 for r in rows),"generated_output_tokens":0,
             "input_overflows":sum(r["decision"]["reason"]=="input_overflow" for r in rows),
             "peak_process_ram_bytes":rss if platform.system()=="Darwin" else rss*1024,
             "local_compute_cost_usd":None,"pricing_note":"local runtime/resources measured separately; no API calls",
             "parameters":predictor.model.parameter_counts(),"artifact":artifact,"platform":platform.platform()}
    if output:
        root=Path(output)
        if root.exists() and any(root.iterdir()): raise ValueError("evaluation output must be new or empty")
        root.mkdir(parents=True,exist_ok=True)
        (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        (root/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return summary

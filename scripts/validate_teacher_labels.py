#!/usr/bin/env python3
"""Validate imported teacher responses against observed evidence and ID contracts."""
import argparse
import json
from pathlib import Path
from dama.contracts import Example,Label
from dama.data import read_examples,validate
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--responses',required=True); p.add_argument('--output',required=True); p.add_argument('--teacher-revision',required=True); args=p.parse_args()
validate(args.data); examples={r.id:r for r in read_examples(Path(args.data)/'train.jsonl')}; accepted=[]; rejected=[]; seen=set()
for line in Path(args.responses).read_text().splitlines():
 try:
  row=json.loads(line); identifier=row['id']
  if identifier not in examples or identifier in seen: raise ValueError('unknown, held-out or duplicate ID')
  seen.add(identifier); original=examples[identifier]
  label=Label.model_validate(row['label']); new=original.model_dump(); new['label']=label.model_dump()
  new['provenance'].update(label_method='teacher:'+args.teacher_revision,review_status='deterministically validated; human review pending')
  accepted.append(Example.model_validate(new))
 except (ValueError,KeyError,TypeError) as error: rejected.append({'response':line,'error':str(error)})
root=Path(args.output)
if root.exists() and any(root.iterdir()): raise ValueError('output must be new or empty')
root.mkdir(parents=True,exist_ok=True)
(root/'accepted.jsonl').write_text(''.join(r.model_dump_json()+'\n' for r in accepted))
(root/'rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
(root/'review-sample.jsonl').write_text(''.join(r.model_dump_json()+'\n' for r in accepted[:20]))
print(json.dumps({'accepted':len(accepted),'rejected':len(rejected),'missing':len(set(examples)-seen),'ready_for_training':False,'next':'human review and rebuild dataset manifest before replacing train labels'}))

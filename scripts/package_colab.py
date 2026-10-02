#!/usr/bin/env python3
"""Package active source only; exclude virtualenv, weights, runs and archived stack."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
root=Path(__file__).resolve().parents[1]; out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
allow=['src','tests','configs','scripts','notebooks','docs','fixtures','pyproject.toml','README.md','.env.example']
files=[]
for name in allow:
 path=root/name
 candidates=[path] if path.is_file() else list(path.rglob('*'))
 files.extend(f for f in candidates if f.is_file() and f.name not in {'.DS_Store','reorganization.md'} and '__pycache__' not in f.parts and not any(part.endswith('.egg-info') for part in f.parts) and f.suffix not in {'.pyc','.safetensors','.gguf','.pt'})
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
 for path in sorted(files): archive.write(path,'dama-model/'+str(path.relative_to(root)))
print(json.dumps({'archive':str(out),'files':len(files),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))

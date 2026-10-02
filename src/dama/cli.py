from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from .contracts import ModelInput,TrainConfig


def main(argv=None):
    p=argparse.ArgumentParser(prog="dama",description="DAMA forward-only System One decision model")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("resources"); sub.add_parser("test")
    c=sub.add_parser("prepare-data"); c.add_argument("--output",required=True)
    c=sub.add_parser("validate-data"); c.add_argument("--input",required=True)
    c=sub.add_parser("validate-config"); c.add_argument("--config",required=True)
    c=sub.add_parser("model-smoke"); c.add_argument("--output")
    c=sub.add_parser("export"); c.add_argument("--checkpoint",required=True); c.add_argument("--output",required=True)
    c=sub.add_parser("train"); c.add_argument("--config",required=True); c.add_argument("--data",required=True); c.add_argument("--output",required=True); c.add_argument("--resume"); c.add_argument("--smoke",action="store_true"); c.add_argument("--allow-synthetic",action="store_true")
    for command in ["predict","evaluate"]:
        c=sub.add_parser(command); c.add_argument("--artifact",required=True); c.add_argument("--input",required=True); c.add_argument("--allow-untrained",action="store_true"); c.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
        if command=="evaluate": c.add_argument("--output",required=True); c.add_argument("--repeats",type=int,default=3)
    args=p.parse_args(argv)
    try:
        if args.command=="resources":
            result={"platform":platform.platform(),"python":platform.python_version(),"cpu_count":os.cpu_count(),"disk_free_bytes":shutil.disk_usage(Path.cwd()).free,"training":"CUDA only; macOS forbidden"}
        elif args.command=="test": return subprocess.call([sys.executable,"-m","pytest",str(Path(__file__).resolve().parents[2]/"tests"),"-q"])
        elif args.command=="prepare-data":
            from .data import prepare
            result=prepare(args.output)
        elif args.command=="validate-data":
            from .data import validate
            result=validate(args.input)
        elif args.command=="validate-config": result={"valid":True,"config":TrainConfig.model_validate_json(Path(args.config).read_text()).model_dump()}
        elif args.command=="train":
            from .training import train
            result=train(args.config,args.data,args.output,resume=args.resume,smoke=args.smoke,allow_synthetic=args.allow_synthetic)
        elif args.command=="model-smoke":
            import torch
            from .testing import tiny_model
            from .data import fixtures
            from .inference import Predictor
            from .evaluate import evaluate
            with tempfile.TemporaryDirectory() as temp:
                torch.manual_seed(42); torch.set_num_threads(1)
                model,tokenizer=tiny_model(temp)
                result=evaluate(Predictor(model,tokenizer,version="untrained-random-test-network"),fixtures()[:8],args.output,repeats=3,warmup=1)
                result["status"]="forward-only software smoke; random weights; not learned capability"
        elif args.command=="export":
            from .artifacts import export_checkpoint
            result=export_checkpoint(args.checkpoint,args.output)
        else:
            from .artifacts import load_model
            from .inference import Predictor
            model,tokenizer,manifest=load_model(args.artifact,allow_untrained=args.allow_untrained,device=args.device)
            predictor=Predictor(model,tokenizer,version=manifest["files"]["model.safetensors"],device=args.device)
            if args.command=="predict":
                decision,timing=predictor.predict(ModelInput.model_validate_json(Path(args.input).read_text())); result={"decision":decision.model_dump(),"timing":timing}
            else:
                from .data import read_examples
                from .evaluate import evaluate
                result=evaluate(predictor,read_examples(args.input),args.output,repeats=args.repeats,artifact=manifest)
        print(json.dumps(result,indent=2)); return 0
    except (ValueError,RuntimeError,OSError) as e:
        print(f"{type(e).__name__}: {e}",file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())

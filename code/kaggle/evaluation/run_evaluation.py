"""DAMA evaluation, Levels 2 and 3, in one run on a Kaggle T4 machine. Nothing is trained.

  1. Level 2 on CPU   the shipped bundle (int8/fp32 ONNX, 2 threads) inside the Memory Agent, held-out LoCoMo 7-9
  2. Level 2 on GPU   the same checkpoint's PyTorch weights on CUDA (fp16 autocast), same agent, same calibration
  3. parity           how often the two runtimes build the same packet
  4. Level 3          Qwen2.5-7B-Instruct answers every question from: the CPU packet, the GPU packet, no memory, the
                      gold evidence only, and the full history (FULL_MAX questions); LoCoMo F1 + the same LLM as judge;
                      it also judges the write path's UPDATE / CONFLICT / DELETE decisions
The checkpoint is pinned in kernel-metadata.json (model_sources: bundle + PyTorch variation of the same checkpoint).
Note: the CPU here is a GPU machine's CPU; the CPU-only reference latency is in docs/making-of-dama.md (L2.0).
Output (kept with this notebook version): summary.json, l2/{cpu,gpu}/, l3/{cpu,gpu}/."""
import glob, json, os, platform, shutil, subprocess, sys, time, urllib.request, zipfile

FULL_MAX = 0    # full-history baseline questions (~65 s each on 2x T4); 0 skips it. Off: on these T4s long-prompt decoding
                # hit intermittent "CUDA error: unspecified launch failure" (versions 1-2); measured in dama-benchmark v2 and
                # re-measured as B1 in the paper experiments
WORK = "/kaggle/working"
os.environ.setdefault("HF_HOME", "/tmp/hf")   # LLM weights stay out of the saved output
t_start = time.time()

code = os.path.join(WORK, "code")
hits = glob.glob("/kaggle/input/**/dama_agent/__init__.py", recursive=True)
if hits:   # Kaggle unpacks dataset zips when it mounts them
    shutil.copytree(os.path.dirname(os.path.dirname(min(hits, key=len))), code)
else:
    with zipfile.ZipFile(glob.glob("/kaggle/input/**/dama_code.zip", recursive=True)[0]) as z:
        z.extractall(code)
sys.path.insert(0, code)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)

import torch
bundle = os.path.dirname(min(glob.glob("/kaggle/input/**/dama_tier1.json", recursive=True), key=len))
pytorch = os.path.dirname(os.path.dirname(min(glob.glob("/kaggle/input/**/decision/heads.safetensors", recursive=True), key=len)))
meta = json.load(open(os.path.join(bundle, "dama_tier1.json")))
cpu_name = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
machine = {"cpu": cpu_name, "vcpus": os.cpu_count(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "checkpoint": meta.get("checkpoint"), "bundle": bundle, "pytorch": pytorch}
print("machine:", json.dumps(machine))
locomo = os.path.join(WORK, "locomo10.json")
urllib.request.urlretrieve("https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json", locomo)

from dama_agent import AgentConfig, load_tier1, load_tier1_torch
from dama_agent import answer
from dama_agent.eval_locomo import print_report, run

summary = {"machine": machine, "level2": {}, "level3": {}}
for name, make in (("cpu", lambda: load_tier1(bundle, threads=2)), ("gpu", lambda: load_tier1_torch(bundle, pytorch, device="cuda"))):
    tier1 = make()
    report = run(tier1, locomo, (7, 8, 9), os.path.join(WORK, "l2", name), config=AgentConfig())
    print(f"\n===== Level 2 on {name.upper()} =====")
    print_report(report)
    summary["level2"][name] = {k: report[k] for k in ("overall", "context", "latency_ms", "concurrency", "write_ops", "by_category")}
    del tier1
    torch.cuda.empty_cache()

packets = {n: {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(WORK, "l2", n, "packets.jsonl"))} for n in ("cpu", "gpu")}
same = [packets["cpu"][q]["packet"] == packets["gpu"][q]["packet"] for q in packets["cpu"] if q in packets["gpu"]]
summary["parity_same_packet"] = sum(same) / max(1, len(same))
print(f"\nCPU vs GPU runtime: identical packets for {100 * summary['parity_same_packet']:.1f}% of questions")

generate = answer.HFGenerator("Qwen/Qwen2.5-7B-Instruct")
# each stage is independent and saved on its own, so one failure (v1 hit a CUDA launch failure in the full-history
# stage) cannot lose the others; the slow full-history baseline runs last
STAGES = [("dama_cpu", "cpu", ("dama",), True), ("dama_gpu", "gpu", ("dama",), False),
          ("baselines", "cpu", ("none", "oracle"), False), ("full", "cpu", ("full",), False)]
summary["level3"], summary["errors"] = {}, {}
for stage, l2, contexts, check in STAGES:
    if stage == "full" and not FULL_MAX:
        continue
    try:
        r = answer.run(os.path.join(WORK, "l2", l2), generate, os.path.join(WORK, "l3", stage), full_max=FULL_MAX, contexts=contexts,
                       check_writes=check)
        print(f"\n===== Level 3: {stage} =====")
        answer.print_report(r)
        summary["level3"][stage] = r
    except Exception as e:
        summary["errors"][stage] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"stage {stage} failed: {summary['errors'][stage]}")
        torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(WORK, "summary.json"), "w"), indent=1)
summary["minutes"] = round((time.time() - t_start) / 60, 1)
json.dump(summary, open(os.path.join(WORK, "summary.json"), "w"), indent=1)

l2c, l2g = summary["level2"]["cpu"], summary["level2"]["gpu"]
ctx = lambda stage, c: summary["level3"].get(stage, {}).get("contexts", {}).get(c)
fmt = lambda x: f"{x['f1_excl_adversarial']:.3f} / {x['judge_excl_adversarial']:.3f}" if x else "failed"
print("\n===== SUMMARY (held-out LoCoMo 7-9) =====")
print(f"{'':28s}{'CPU (ONNX)':>18s}{'GPU (PyTorch)':>18s}")
print(f"{'read p50 / p95 ms':28s}{l2c['latency_ms']['read_p50']:>10} / {l2c['latency_ms']['read_p95']:<6}{l2g['latency_ms']['read_p50']:>10} / {l2g['latency_ms']['read_p95']:<6}")
print(f"{'write p50 / p95 ms':28s}{l2c['latency_ms']['write_p50']:>10} / {l2c['latency_ms']['write_p95']:<6}{l2g['latency_ms']['write_p50']:>10} / {l2g['latency_ms']['write_p95']:<6}")
for scen in ("distinct", "team"):
    print(f"{'16 agents ' + scen + ' p95 ms':28s}{l2c['concurrency'][scen]['16']['p95_ms']:>18}{l2g['concurrency'][scen]['16']['p95_ms']:>18}")
print(f"{'all gold in packet':28s}{l2c['overall']['all_gold_in_packet']:>18.3f}{l2g['overall']['all_gold_in_packet']:>18.3f}")
print(f"{'answer F1 / J':28s}{fmt(ctx('dama_cpu', 'dama')):>18s}{fmt(ctx('dama_gpu', 'dama')):>18s}")
print(f"{'packet tokens':28s}{l2c['overall']['mean_packet_tokens']:>18.0f}{l2g['overall']['mean_packet_tokens']:>18.0f}")
for stage, c in (("baselines", "none"), ("baselines", "oracle"), ("full", "full")):
    print(f"baseline {c:8s} F1 / J {fmt(ctx(stage, c))}")
print(f"minutes: {summary['minutes']}  errors: {summary['errors'] or 'none'}")
shutil.rmtree(code, ignore_errors=True)
os.remove(locomo)

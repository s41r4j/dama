"""DAMA v3b, part 2 (2026-11-01): judge v3b's rows, and retry the MoE top-line reader. Private Kaggle notebook
`s41r4j/dama-v3b-judge` (T4 x2). v3b version 1 read every arm, then lost its judge to "Read-only file system:
/tmp/...lock" after 8 h; its Qwen3.5-35B-A3B server also had no memory left for the KV cache.

1. Qwen3.5-35B-A3B-GPTQ-Int4 on both T4s with tighter settings (memory 0.95, max length 4,608, eager mode, no image
   inputs) reads the A0 and L9P packets: Z35_A0, Z35_L9P (the L35P arm needed the MoE as writer; dropped).
2. Qwen2.5-7B judges every arm (identical answers hit the cache attached from v3b).
3. v3b's statistics and pre-registered decisions (README.md), unchanged code.
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

HERE = Path(globals().get("__file__", "dama_v3b_judge.py")).resolve().parent


def _exp_dir(name):
    hits = glob.glob(f"/kaggle/input/**/experiments/{name}", recursive=True)
    return Path(min(hits, key=len)).parent if hits else HERE.parent.parent / Path(name).parent


sys.path.insert(0, str(_exp_dir("v3b_2026-11-01/dama_paper_v3b.py")))
import dama_paper_v3b as B3
Bm, V, v0 = B3.Bm, B3.V, B3.v0
log, STATE = v0.log, v0.STATE
MOE_ARGS = ["--enforce-eager", "--limit-mm-per-prompt", '{"image": 0, "video": 0}']


def load_rows():
    hits = glob.glob("/kaggle/input/**/methods/dev_*.jsonl", recursive=True)
    if not hits:
        raise SystemExit("v3b's rows are not attached (kernel source s41r4j/dama-paper-v3b)")
    for f in sorted(hits, key=len):
        arm = Path(f).stem[4:]
        if arm not in B3.ROWS:
            B3.ROWS[arm] = [json.loads(l) for l in open(f)]
    log("rows:", {k: len(v) for k, v in B3.ROWS.items()})
    prev = sorted(glob.glob("/kaggle/input/**/paper/results_v3b.json", recursive=True), key=len)
    if prev:                                           # v3b's speeds, ledger statistics, gates and context counts
        B3.R.update({k: v for k, v in json.load(open(prev[0])).items() if k in ("settings", "arms", "ledgers", "gates", "contexts", "zoo", "notes")})


def ctx_of(arm):
    """The arm's packets in A0's question order (read_arm pairs them with D["qs"])."""
    by = {r["qid"]: r for r in B3.ROWS[arm]}
    return [{"context": by[r["qid"]]["context"], "dia": None, "read_ms": by[r["qid"]].get("read_ms"),
             "ledger": by[r["qid"]].get("ledger")} for r in B3.ROWS["A0"]]


@v0.stage("v3b_moe_read")
def stage_moe():
    srv = B3.Srv(B3.MOE, "0,1", 8011, max_len=4608, util=0.95, max_seqs=32, extra=MOE_ARGS)
    if not srv.start():
        B3.R["notes"].append("MoE could not be served on T4 x2 (second try)"); return
    try:
        cli = B3.Cli(B3.MOE, 8011, conc=32)
        B3.D["qs"] = [q_of(r) for r in B3.ROWS["A0"]]
        for src, arm in (("A0", "Z35_A0"), ("L9P", "Z35_L9P")):
            if src in B3.ROWS:
                B3.read_arm(arm, ctx_of(src), cli, f"qwen3.5-35b-a3b|v3b_{arm}")
                B3.R["zoo"][arm] = {"model": B3.MOE, "gpus": 2, "load_min": srv.load_min}
    finally:
        srv.stop(drop_weights=True)


def q_of(r):
    """The question fields Bm.read needs, from a row."""
    q = {"qid": r["qid"], "question": r["question"], "gold": r["gold"], "evidence": [], "bench": r["bench"]}
    q.update({"type": r["type"], "abs": r.get("abs"), "split": r.get("split"), "date": LME_DATES.get(r["qid"], "")} if r["bench"] == "lme"
             else {"ci": r["ci"], "cat": r["cat"]})
    return q


LME_DATES = {}


def lme_dates():
    path = Path("/tmp/longmemeval_s_cleaned.json")
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(B3.L.LME_URL, path)
    import re                                          # streamed regex, not json.load: the dates only (json.load needs ~2 GB)
    rx_id, rx_date = re.compile(r'"question_id":\s*"([^"]+)"'), re.compile(r'"question_date":\s*"([^"]*)"')
    cur, tail = None, ""
    with open(path) as f:
        while chunk := f.read(1 << 23):
            buf, keep = tail + chunk, 200                  # keep the last 200 characters: a key may span two chunks
            events = sorted([(m.start(), "id", m.group(1)) for m in rx_id.finditer(buf[:-keep])] +
                            [(m.start(), "date", m.group(1)) for m in rx_date.finditer(buf[:-keep])])
            for _, kind, val in events:
                if kind == "id":
                    cur = val
                elif cur is not None:
                    LME_DATES.setdefault(cur, val)
            tail = buf[-keep:]
    log("LME question dates:", len(LME_DATES))


if __name__ == "__main__":
    log("environment", v0.environment())
    load_rows()
    lme_dates()
    if Bm.install_vllm(B3.VLLM):
        stage_moe()
        B3.stage_judge()
    try:
        B3.stage_stats()
    except Exception as e:
        STATE["errors"]["v3b_stats"] = f"{type(e).__name__}: {str(e)[:150]}"
    B3.save_r()
    B3.write_report()
    log(f"v3b judge done; errors: {STATE['errors'] or 'none'}")

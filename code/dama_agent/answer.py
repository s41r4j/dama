"""Level 3: answer accuracy. A fixed answering LLM reads each question with a given memory context and answers in a
short phrase; answers are scored against LoCoMo's gold answers.

Contexts (same LLM, same prompt, so only the memory differs):
  dama    the Memory Agent's packet (eval_locomo packets.jsonl)
  none    no memory at all (lower bound)
  oracle  only the gold evidence turns (upper bound for a perfect retriever)
  full    the whole conversation history (the usual "full context" baseline; slow, so a fixed subset)
Scores:
  f1      LoCoMo token F1 after normalisation (the metric MemRouter reports); adversarial questions score 1 when the
          answer says the information is not available (LoCoMo's rule)
  judge   the same LLM judges CORRECT / WRONG against the gold answer (Mem0's "J" protocol, but with this open model
          as judge, not GPT-4o-mini, so J is not directly comparable with published numbers)
Also checks the write path: for every UPDATE / CONFLICT / DELETE in writes.jsonl the LLM says whether the new turn
really replaces (or retracts) the old fact, giving the precision of those decisions.

    python -m dama_agent.answer --eval results/ --model Qwen/Qwen2.5-7B-Instruct --out results/answers/
"""
from __future__ import annotations
import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ANSWER_SYSTEM = ("You answer questions about a long conversation between two people, using only the memories given. "
                 "Reply with a short phrase of a few words, not a sentence. Use the dates in the memories for questions about "
                 "time. If the memories do not contain the answer, reply exactly: Not mentioned.")
JUDGE_SYSTEM = ("You grade answers. Given a question, a gold answer and a generated answer, reply CORRECT if the generated "
                "answer means the same as the gold answer (it may be phrased differently or be slightly more detailed; dates "
                "in another format count as the same), otherwise reply WRONG. Reply with one word.")
WRITE_SYSTEM = ("You check a memory system. A stored memory and a newer statement are shown. Reply YES if the newer statement "
                "changes, corrects or retracts the information in the stored memory (the same attribute of the same person now "
                "has a different value, or is withdrawn). Reply NO if it only adds something, repeats it, or is unrelated. "
                "Reply with one word.")
NOT_AVAILABLE = ("not mentioned", "no information", "not available", "unknown", "not stated", "cannot be determined")


def normalize(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return float(p == g)
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def score(pred: str, gold: str, category: str) -> float:
    if category == "adversarial":
        return float(any(x in pred.lower() for x in NOT_AVAILABLE))
    return f1(pred, gold)


class HFGenerator:
    """Batched greedy generation with a chat model from Hugging Face transformers (GPU if available)."""
    def __init__(self, name, max_input_tokens=32000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left", truncation_side="left")   # never cut the question
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                          device_map="auto" if torch.cuda.is_available() else None).eval()
        self.max_input_tokens = max_input_tokens
        self.name = name

    def __call__(self, chats: list[list[dict]], max_new_tokens=32, batch_size=8) -> list[str]:
        out = []
        if batch_size == 1:   # long prompts (full history): one at a time, chunked prefill above 4k tokens
            for c in chats:
                ids = self.tok(self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True), return_tensors="pt",
                               truncation=True, max_length=self.max_input_tokens)["input_ids"]
                out.append(self._long(ids, max_new_tokens) if ids.shape[1] > 4096 else
                           self.__call__([c], max_new_tokens=max_new_tokens, batch_size=2)[0])
            return out
        for i in range(0, len(chats), batch_size):
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats[i:i + batch_size]]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
            out += [t.strip() for t in self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out


    def _long(self, ids, max_new_tokens, chunk=512) -> str:
        """Greedy decoding with the prompt fed in chunks through the KV cache. One 26k-token prefill made SDPA allocate the
        full attention matrix (107 GiB) on a T4; per chunk it is chunk x length, under 1 GiB."""
        torch = self.torch
        from transformers import DynamicCache
        cache, out = DynamicCache(), None
        ids = ids.to(self.model.device)
        stop = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        with torch.no_grad():
            for i in range(0, ids.shape[1], chunk):
                out = self.model(input_ids=ids[:, i:i + chunk], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            new = []
            for _ in range(max_new_tokens):
                nxt = int(out.logits[0, -1].argmax())
                if nxt in stop:
                    break
                new.append(nxt)
                out = self.model(input_ids=torch.tensor([[nxt]], device=self.model.device), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
        del cache
        torch.cuda.empty_cache()
        return self.tok.decode(new, skip_special_tokens=True).strip()


def answer_chat(question, context):
    return [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Memories:\n{context or '(none)'}\n\nQuestion: {question}\nShort answer:"}]


def judge_chat(question, gold, pred):
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\nGenerated answer: {pred}\nGrade:"}]


def run(eval_dir, generate, out_dir=None, full_max=90, contexts=("dama", "none", "oracle", "full"), check_writes=True, log=print) -> dict:
    ev = Path(eval_dir)
    rows = [json.loads(l) for l in (ev / "packets.jsonl").open()]
    histories = json.loads((ev / "histories.json").read_text()) if (ev / "histories.json").exists() else {}
    full_ids = {r["qid"] for r in rows[::max(1, len(rows) // full_max)][:full_max]} if full_max else set()   # fixed, evenly spread subset
    results = {}
    for ctx in contexts:
        sel = [r for r in rows if ctx != "full" or r["qid"] in full_ids]
        if ctx == "full" and (not histories or not full_max):
            continue
        t0 = time.time()
        context_of = {"dama": lambda r: "" if r["packet"] == "NOT_FOUND" else r["packet"], "none": lambda r: "",
                      "oracle": lambda r: r["gold_context"], "full": lambda r: histories[r["scope"]]}[ctx]
        preds = generate([answer_chat(r["question"], context_of(r)) for r in sel], batch_size=1 if ctx == "full" else 8)
        verdicts = generate([judge_chat(r["question"], r["answer"], p) for r, p in zip(sel, preds)], max_new_tokens=4)
        results[ctx] = [{"qid": r["qid"], "category": r["category"], "question": r["question"], "gold": r["answer"], "pred": p,
                         "f1": score(p, r["answer"], r["category"]), "judge": float(v.strip().upper().startswith("CORRECT")),
                         "context_tokens": len(context_of(r)) / 4} for r, p, v in zip(sel, preds, verdicts)]
        log(f"{ctx}: {len(sel)} questions in {time.time() - t0:.0f}s, F1 {np.mean([x['f1'] for x in results[ctx]]):.3f}, "
            f"judge {np.mean([x['judge'] for x in results[ctx]]):.3f} (all categories)")
        if out_dir:   # saved at once: a crash in a later context must not lose this one
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"answers_{ctx}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in results[ctx]))

    def summarize(items):
        main = [x for x in items if x["category"] != "adversarial"]
        cats = defaultdict(list)
        for x in items:
            cats[x["category"]].append(x)
        return {"n": len(items), "f1_excl_adversarial": float(np.mean([x["f1"] for x in main])) if main else None,
                "judge_excl_adversarial": float(np.mean([x["judge"] for x in main])) if main else None,
                "f1_all": float(np.mean([x["f1"] for x in items])), "mean_context_tokens": float(np.mean([x["context_tokens"] for x in items])),
                "by_category": {c: {"n": len(v), "f1": float(np.mean([x["f1"] for x in v])), "judge": float(np.mean([x["judge"] for x in v]))}
                                for c, v in sorted(cats.items())}}

    report = {"answer_model": getattr(generate, "name", "custom"), "contexts": {c: summarize(v) for c, v in results.items()}}
    if "full" in results:   # DAMA on exactly the full-context subset, for a paired comparison
        report["paired_with_full"] = {c: summarize([x for x in results[c] if x["qid"] in full_ids]) for c in results}

    writes_path = ev / "writes.jsonl"
    if check_writes and writes_path.exists():
        ws = [json.loads(l) for l in writes_path.open()]
        ws = [w for w in ws if w["action"] in ("UPDATE", "CONFLICT", "DELETE") and w["old"] and w["new"]]
        if ws:
            vs = generate([[{"role": "system", "content": WRITE_SYSTEM},
                            {"role": "user", "content": f"Stored memory ({w['old_speaker']}): {w['old']}\nNewer statement ({w['new_speaker']}): {w['new']}\nAnswer:"}]
                           for w in ws], max_new_tokens=3)
            by = defaultdict(list)
            for w, v in zip(ws, vs):
                w["llm_says_replaces"] = v.strip().upper().startswith("YES")
                by[w["action"]].append(w["llm_says_replaces"])
            report["write_check"] = {a: {"n": len(v), "llm_agrees": float(np.mean(v))} for a, v in by.items()}
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "write_check.jsonl").write_text("".join(json.dumps(w) + "\n" for w in ws))
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for c, v in results.items():
            (out / f"answers_{c}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in v))
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def print_report(r):
    print(f"\nLEVEL 3 answer accuracy, answering model {r['answer_model']} (F1 = LoCoMo token F1; J = same model as judge)")
    for c, s in r["contexts"].items():
        print(f"  {c:7s} n={s['n']:<4} F1 {s['f1_excl_adversarial']:.3f}  J {s['judge_excl_adversarial']:.3f}  (excl. adversarial)  "
              f"F1 all {s['f1_all']:.3f}  context {s['mean_context_tokens']:.0f} tokens")
        print("          " + "  ".join(f"{k} {v['f1']:.2f}/{v['judge']:.2f}" for k, v in s["by_category"].items()))
    if "paired_with_full" in r:
        print("  on the full-context subset: " + "  ".join(f"{c} F1 {s['f1_excl_adversarial']:.3f} J {s['judge_excl_adversarial']:.3f}"
                                                         for c, s in r["paired_with_full"].items()))
    if "write_check" in r:
        print("  write path, share of decisions the LLM agrees with: " + ", ".join(f"{a} {v['llm_agrees']:.2f} (n={v['n']})" for a, v in r["write_check"].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="folder with packets.jsonl, histories.json, writes.jsonl from eval_locomo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--full-max", type=int, default=90)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    print_report(run(a.eval, HFGenerator(a.model), a.out, full_max=a.full_max))


if __name__ == "__main__":
    main()

"""ENGRAM (Patel & Patel, ICLR 2026, arXiv 2511.12960), re-implemented from the paper: its code link (anonymous.4open.science
/r/engram-68FF) had expired on 2026-11-01. Prompts D.1-D.4 are the paper's Appendix D text.

Per message: the router (D.1) picks any of episodic / semantic / procedural; each picked type has its own extraction
prompt (D.2 episodic: title, summary, inferred event timestamp; D.3 semantic: one fact; D.4 procedural: title, content).
Records are embedded and stored per type. Retrieval: cosine top-20 per store, union, de-duplicated, best K = 25
(the paper's chosen budget), serialised as "date: text" per speaker.

Deviations (reported in DMBM): embedder bge-small-en-v1.5 on CPU instead of text-embedding-3-small; the router is asked to
end with a JSON object (the paper leaves its output format open); records are kept in memory (numpy), not SQLite, which
changes nothing about retrieval; de-duplication by exact text.
"""
from __future__ import annotations
import json
import re

import numpy as np

from .base import System, count_llm, lines

ROUTER = """Given this message, determine which storage types are most relevant:
Message: {message}
Storage types:
- episodic: Event and experiences with temporal context (think of this as a personal diary of events that is timelined)
- semantic: Facts, observations, preferences (only route to this if the message reveals a non-event fact or preference about a person, place, or thing)
- procedural: How-to information, processes
Determine which types are relevant for this message.
Return only a JSON object: {{"episodic": true or false, "semantic": true or false, "procedural": true or false}}"""

EPISODIC = """You are an intelligent memory assistant tasked with converting a single conversation message into an episodic memory JSON object.
Context
You receive a raw message string and a conversation timestamp. The conversation timestamp reflects when the message was sent, not necessarily when the event occurred. You must infer the actual event time from the message content and conversation context.
Instructions
1. Preserve specific nouns exactly as written: use exact place names, object/activity names, proper nouns, numbers, and titles verbatim.
2. Reason carefully about time. Do not blindly copy the input timestamp; infer the event timestamp from the message and context.
3. Write in third person.
4. Do not use temporal words in the summary (e.g., "yesterday", "last week").
5. Return only a single JSON object with keys title, summary, and timestamp.
Approach (think step by step)
1. Read the message and identify specific entities (names, places, objects) to preserve verbatim.
2. Extract temporal cues (e.g., "yesterday", "last month", holiday names) and map them to absolute dates based on the conversation timestamp.
3. If the message implies a date range (e.g., a month without a day), output the most specific resolvable form (e.g., "June 2024").
4. Form a concise title that includes specific details; then write a one-sentence summary in third person with exact nouns preserved.
5. Compute the timestamp as the actual event date/time derived from the message (not merely the conversation time).
Timestamp reasoning examples
- Input timestamp: "10:55 am on 22 July, 2024"; message: "I went to Cheesequake park yesterday with my friends" => event timestamp: "July 21 2024".
- Input timestamp: "10:55 am on 22 July, 2024"; message: "I went camping in Banff last month" => event timestamp: "June 2024".
Inputs
Message: {message}
Conversation Timestamp: {timestamp}
Output (return only this JSON)
{{"title": "...", "summary": "...", "timestamp": "..."}}"""

SEMANTIC = """Extract ONE key fact or observation from this message.
CRITICAL: Preserve specific nouns exactly as mentioned. Do NOT generalize or paraphrase:
- Use exact place names, not generic terms
- Use exact object/activity names, not categories
- Use exact relationship terms, not synonyms
- Keep proper nouns, numbers, and specific terms verbatim
RULES:
- One sentence with specific details preserved exactly as written
- Include exact names, places, objects, activities as mentioned
- Proper nouns must be preserved verbatim
- Be specific and detailed, avoid generic terms
Message: {message}
Return only a JSON object:
{{"fact": "detailed fact with specific nouns preserved exactly"}}"""

PROCEDURAL = """Convert this message into procedural memory format:
Inputs
Message: {message}
Extract
- title: A title for this procedure/instruction
- content: The procedural content/steps (can be a string or list of steps)
Return only a JSON object: {{"title": "...", "content": "..."}}"""


def _json(text):
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class Engram(System):
    name, needs_llm, native = "engram", True, "re-implemented"
    K_STORE, K_FINAL = 20, 25
    PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, encoder=None, **kw):
        super().__init__(**kw)
        count_llm()
        from openai import OpenAI
        self.client = OpenAI(base_url=self.llm_url, api_key="local")
        self._enc, self.mem = encoder, {}

    def enc(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        return self._enc

    def _ask(self, prompt):
        r = self.client.chat.completions.create(model=self.llm_model, messages=[{"role": "user", "content": prompt}], max_tokens=300)
        return _json(r.choices[0].message.content)

    def ingest(self, hid, sessions):
        recs = []                                    # (store, speaker, date, text)
        for s in sessions:
            if self.out_of_time():
                break
            for t in s.turns:
                msg = f"{t.speaker}: {t.text}"
                route = self._ask(ROUTER.format(message=msg))
                if route.get("episodic"):
                    e = self._ask(EPISODIC.format(message=msg, timestamp=s.date))
                    if e.get("summary") or e.get("title"):
                        recs.append(("episodic", t.speaker, str(e.get("timestamp") or s.date), f"{e.get('title', '')}: {e.get('summary', '')}".strip(": ")))
                if route.get("semantic"):
                    f = self._ask(SEMANTIC.format(message=msg))
                    if f.get("fact"):
                        recs.append(("semantic", t.speaker, s.date, str(f["fact"])))
                if route.get("procedural"):
                    p = self._ask(PROCEDURAL.format(message=msg))
                    if p.get("content") or p.get("title"):
                        c = p.get("content")
                        c = "; ".join(map(str, c)) if isinstance(c, list) else str(c or "")
                        recs.append(("procedural", t.speaker, s.date, f"{p.get('title', '')}: {c}".strip(": ")))
        vec = self.enc().encode([r[3] for r in recs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True) if recs else np.zeros((0, 384))
        self.mem[hid] = (recs, vec)

    def context(self, hid, question, q_date):
        recs, vec = self.mem[hid]
        if not recs:
            return lines([])
        q = self.enc().encode([self.PREFIX + question], normalize_embeddings=True, convert_to_numpy=True)[0]
        sc = vec @ q
        pool = []
        for store in ("episodic", "semantic", "procedural"):
            idx = [i for i, r in enumerate(recs) if r[0] == store]
            pool += sorted(idx, key=lambda i: -sc[i])[:self.K_STORE]
        seen, keep = set(), []
        for i in sorted(set(pool), key=lambda i: -sc[i]):
            if recs[i][3] not in seen:
                seen.add(recs[i][3]); keep.append(i)
        keep = keep[:self.K_FINAL]
        by_spk = {}
        for i in keep:                                # two banks, one per speaker, as in the paper's answer template
            by_spk.setdefault(recs[i][1], []).append(i)
        out = []
        for spk, idx in by_spk.items():
            out += [(recs[i][2], f"[{spk}] {recs[i][3]}") for i in idx]
        return lines(out)

    def close(self, hid):
        self.mem.pop(hid, None)

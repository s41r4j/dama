"""Two more memory systems from official code.

LightMem (zjunlp, ICLR 2026; MIT): sensory pre-compression (LLMLingua-2), topic segmentation, LLM metadata extraction and
summaries, offline update; settings copied from its own experiments (experiments/longmemeval/run_lightmem_qwen.py and
experiments/locomo/add_locomo.py at the pinned commit): LongMemEval as user/assistant pairs, LoCoMo as each turn = a
user message with the speaker's name plus an empty assistant message; messages_use "user_only"; retrieve limit 20.
Its exact version pins (torch==2.8.0, ...) clash with Kaggle's image, so the package source is fetched at the pinned
commit and used with the environment's libraries. CPU for LLMLingua-2 and the embedder (the GPUs hold the vLLM writer).

Graphiti (Zep's open-source temporal knowledge graph; Apache-2.0; PyPI graphiti-core): one episode per session with
its reference time; entity and fact extraction, deduplication and bi-temporal edges by the writer LLM through
OpenAIGenericClient (json_object mode, recommended in its README for local models); embedded FalkorDB (falkordblite);
embedder = bge-small through Graphiti's EmbedderClient interface (no embeddings endpoint is served); reranker =
Graphiti's BGE reranker client. Packet: the facts of the top 20 edges with their valid-from / invalid-at dates.
"""
from __future__ import annotations
import asyncio
import json
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from ..timeparse_shim import parse_any_date
from .base import System, count_llm, lines

LIGHTMEM = ("zjunlp/LightMem", "28cade249f847f7cfe7c208945be492f1bbda901")


def vendor_lightmem(root):
    d = Path(root) / "vendor"
    if not (d / "lightmem" / "memory" / "lightmem.py").exists():
        repo, sha = LIGHTMEM
        tree = json.loads(urllib.request.urlopen(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", timeout=60).read())
        for t in tree["tree"]:
            p = t["path"]
            if t["type"] == "blob" and p.startswith("src/lightmem/") and p.endswith(".py") and "/memory_toolkits/" not in p:
                dst = d / p[len("src/"):]
                dst.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{sha}/{p}", dst)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


class LightMem(System):
    name, needs_llm = "lightmem", True
    LIMIT = 20

    def __init__(self, **kw):
        super().__init__(**kw)
        count_llm()
        vendor_lightmem(self.workdir)
        self.mem = {}

    def _config(self, tag):
        path = Path(self.workdir) / "lightmem" / tag
        shutil.rmtree(path, ignore_errors=True)
        return {
            "pre_compress": True,
            "pre_compressor": {"model_name": "llmlingua-2", "configs": {"llmlingua_config": {
                "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", "device_map": "cpu", "use_llmlingua2": True}}},
            "topic_segment": True, "precomp_topic_shared": True, "topic_segmenter": {"model_name": "llmlingua-2"},
            "messages_use": "user_only", "metadata_generate": True, "text_summary": True,
            "memory_manager": {"model_name": "openai", "configs": {"model": self.llm_model, "api_key": "local", "max_tokens": 4000,
                                                                     "openai_base_url": self.llm_url}},
            "extract_threshold": 0.1, "index_strategy": "embedding",
            "text_embedder": {"model_name": "huggingface", "configs": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384,
                                                                       "model_kwargs": {"device": "cpu"}}},
            "retrieve_strategy": "embedding",
            "embedding_retriever": {"model_name": "qdrant", "configs": {"collection_name": tag[:60], "embedding_model_dims": 384, "path": str(path)}},
            "update": "offline",
        }

    def ingest(self, hid, sessions):
        from lightmem.memory.lightmem import LightMemory
        tag = "".join(c if c.isalnum() else "_" for c in hid)
        lm = LightMemory.from_config(self._config(tag))
        seg = getattr(getattr(lm, "segmenter", None), "model", None)
        if seg is not None:                                  # its topic segmenter reads BERT attention maps; transformers 5
            try:                                             # defaults to SDPA, which returns none ("tuple index out of range")
                seg.set_attn_implementation("eager")
            except Exception:
                seg.config._attn_implementation = "eager"
        user_assistant = {t.speaker for s in sessions for t in s.turns} <= {"user", "assistant"}
        for si, s in enumerate(sessions):
            if user_assistant:                               # LongMemEval: real user/assistant pairs
                ts = list(s.turns)
                while ts and ts[0].speaker != "user":
                    ts.pop(0)
                pairs = [ts[i:i + 2] for i in range(0, len(ts) - 1, 2)]
                pairs = [[{"role": "user", "content": a.text}, {"role": "assistant", "content": b.text}] for a, b in pairs
                         if a.speaker == "user" and b.speaker == "assistant"]
            else:                                            # LoCoMo: each turn + an empty assistant message
                pairs = [[{"role": "user", "content": t.text, "speaker_name": t.speaker},
                          {"role": "assistant", "content": "", "speaker_name": t.speaker}] for t in s.turns]
            when = parse_any_date(s.date)                    # LightMem parses only '2023/05/20 (Sat) 00:44' (its LoCoMo script converts too)
            stamp = when.strftime("%Y/%m/%d (%a) %H:%M") if when else s.date
            stop = False
            for pi, msgs in enumerate(pairs):
                for m in msgs:
                    m["time_stamp"] = stamp
                stop = self.out_of_time()
                last = stop or (si == len(sessions) - 1 and pi == len(pairs) - 1)
                lm.add_memory(messages=msgs, force_segment=last, force_extract=last)   # forced: nothing left in its buffer
                if stop:
                    break
            if stop:
                break
        try:
            lm.construct_update_queue_all_entries()
            lm.offline_update_all_entries(score_threshold=0.8)
        except Exception as e:                               # the update step is optional in its LongMemEval script
            print(f"lightmem offline update skipped: {type(e).__name__}: {e}", flush=True)
        self.mem[hid] = lm

    def context(self, hid, question, q_date):
        got = self.mem[hid].retrieve(question, limit=self.LIMIT)
        got = got if isinstance(got, list) else [got]
        return lines([("", str(x)) for x in got])

    def close(self, hid):
        self.mem.pop(hid, None)


class _BgeEmbedder:
    """Graphiti EmbedderClient over sentence-transformers bge-small (built lazily to keep the import light)."""
    def __new__(cls):
        from graphiti_core.embedder.client import EmbedderClient
        from sentence_transformers import SentenceTransformer

        class E(EmbedderClient):
            def __init__(self):
                self.m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

            async def create(self, input_data):
                x = input_data if isinstance(input_data, str) else " ".join(map(str, input_data))
                return self.m.encode([x], normalize_embeddings=True)[0].tolist()

            async def create_batch(self, input_data_list):
                return [v.tolist() for v in self.m.encode(list(input_data_list), normalize_embeddings=True, batch_size=64)]

        return E()


class Graphiti(System):
    name, needs_llm = "graphiti_zep", True
    NUM = 20

    def __init__(self, **kw):
        super().__init__(**kw)
        count_llm()
        self.mem, self.loop, self._server, self._port = {}, asyncio.new_event_loop(), None, None

    def _make(self, tag):
        """Graph store: embedded FalkorDB (falkordblite: a bundled Redis + FalkorDB server, one per worker process on its
        own TCP port), reached through Graphiti's standard FalkorDriver. The Kuzu backend is deprecated in graphiti-core
        0.31.2 and its full-text index fails (upstream Kuzu is unmaintained)."""
        import os
        from graphiti_core import Graphiti as G
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        if self._server is None:
            from redislite.falkordb_client import FalkorDB
            d = Path(self.workdir) / "graphiti"
            d.mkdir(parents=True, exist_ok=True)
            self._port = 20000 + os.getpid() % 20000
            self._server = FalkorDB(str(d / f"falkor_{os.getpid()}.db"), serverconfig={"port": str(self._port)})
        cfg = LLMConfig(api_key="local", model=self.llm_model, small_model=self.llm_model, base_url=self.llm_url)
        try:
            llm = OpenAIGenericClient(config=cfg, structured_output_mode="json_object")
        except TypeError:                                    # older signature
            llm = OpenAIGenericClient(config=cfg)
        drv = FalkorDriver(host="127.0.0.1", port=self._port, database=tag[:60])
        g = G(graph_driver=drv, llm_client=llm, embedder=_BgeEmbedder(), cross_encoder=BGERerankerClient())
        self.loop.run_until_complete(g.build_indices_and_constraints())
        return g

    def ingest(self, hid, sessions):
        from graphiti_core.nodes import EpisodeType
        tag = "".join(c if c.isalnum() else "_" for c in hid)
        g = self._make(tag)
        for s in sessions:
            if self.out_of_time():
                break
            body = "\n".join(f"{t.speaker}: {t.text}" for t in s.turns)
            when = parse_any_date(s.date) or datetime(2023, 1, 1)
            self.loop.run_until_complete(g.add_episode(name=f"session {s.session_id}", episode_body=body, source=EpisodeType.message,
                                                       source_description="chat session", reference_time=when, group_id=tag))
        self.mem[hid] = (g, tag)

    def context(self, hid, question, q_date):
        g, tag = self.mem[hid]
        edges = self.loop.run_until_complete(g.search(question, group_ids=[tag], num_results=self.NUM))
        out = []
        for e in edges:
            va = getattr(e, "valid_at", None); ia = getattr(e, "invalid_at", None)
            d = va.date().isoformat() if va else ""
            out.append((d, e.fact + (f" (no longer true from {ia.date().isoformat()})" if ia else "")))
        return lines(out)

    def close(self, hid):
        x = self.mem.pop(hid, None)
        if x:
            try:
                self.loop.run_until_complete(x[0].close())
            except Exception:
                pass

"""DMBM memory systems. Registry: name -> class (constructed with llm_url, llm_model, workdir)."""
from .engram import Engram
from .graph_light import Graphiti, LightMem
from .native import BM25, Dense, NoMemory, Recent
from .rag import RagChunk, RagHybrid, SummaryMemory, VectorMemory
from .thirdparty import AMem, Mem0, MemoryOS

REGISTRY = {
    # floors and retrieval (no LLM at write time)
    "no_memory": NoMemory, "recent": Recent, "bm25": BM25, "dense_bge_small": Dense,
    "rag_chunk": RagChunk, "rag_hybrid_rerank": RagHybrid, "vector_memory": VectorMemory,
    # LLM-written memories
    "summary_memory": SummaryMemory, "mem0": Mem0, "a_mem": AMem, "memoryos": MemoryOS, "engram": Engram,
    "lightmem": LightMem, "graphiti_zep": Graphiti,
}

"""DAMA v0 Memory Agent (Level 2): store + read path (push packet) + write path + tools, on top of the Tier-1 CPU bundle.

    from dama_agent import MemoryAgent, load_tier1
    agent = MemoryAgent(load_tier1("path/to/dama_tier1"))
    agent.observe("alice", "I moved to Pune last month", speaker="Alice", ts="2023-05-02")
    agent.packet("alice", "Where do I live now?")["text"]
"""
from .agent import AgentConfig, MemoryAgent, render
from .store import Store


def load_tier1(bundle_dir, threads: int = 2):
    """The CPU runtime from dama_train (newer than the copy inside older bundles), loaded on a bundle folder."""
    from dama_train.runtime import Tier1
    return Tier1(bundle_dir, threads=threads)


def load_tier1_torch(bundle_dir, pytorch_dir, device=None):
    """Tier-1 on a GPU: PyTorch weights (Model variation transformers/tier1-pytorch) + the bundle's calibration."""
    from .gpu import TorchTier1
    return TorchTier1(bundle_dir, pytorch_dir, device=device)


__all__ = ["AgentConfig", "MemoryAgent", "Store", "load_tier1", "load_tier1_torch", "render"]

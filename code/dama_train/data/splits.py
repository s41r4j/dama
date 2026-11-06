"""Deterministic splits at the dialogue level: no conversation (or LongMemEval question) appears in two splits."""
from __future__ import annotations
import hashlib

# LoCoMo has only 10 conversations; MemRouter/Memory-R1 train on 1 and test on 8. We use more for training
# because our heads need real negatives, but keep 2 conversations fully unseen for testing.
LOCOMO_SPLIT = {0: "train", 1: "train", 2: "train", 3: "train", 4: "train", 5: "val", 6: "calib", 7: "test", 8: "test", 9: "test"}
FRACTIONS = [("train", 0.60), ("val", 0.10), ("calib", 0.10), ("test", 0.20)]


def hashed_split(key: str, salt: str = "dama") -> str:
    x = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, frac in FRACTIONS:
        edge += frac
        if x < edge:
            return name
    return FRACTIONS[-1][0]


def split_of_dialogue(did: str, locomo_order: dict[str, int]) -> str:
    if did.startswith("locomo:"):
        return LOCOMO_SPLIT[locomo_order[did]]
    # LongMemEval: abstention variants share a haystack with their base question; keep them together.
    return hashed_split(did.removesuffix("_abs"))

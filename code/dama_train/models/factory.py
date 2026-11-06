"""Load pretrained backbones, or build tiny random ones (no downloads) for the offline smoke test."""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from ..config import TrainConfig
from .decision import DecisionModel
from .ops import OpLogits, rehead_for_ops


def tiny_tokenizer(texts, directory):
    from transformers import BertTokenizerFast
    words = Counter(w for t in texts for w in re.findall(r"[a-z0-9]+|[^\sa-z0-9]", t.lower()))
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    chars = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789"]
    vocab = specials + chars + [f"##{c}" for c in chars] + [w for w, _ in words.most_common(3000) if w not in chars]
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / "vocab.txt"
    path.write_text("\n".join(dict.fromkeys(vocab)) + "\n")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def _tiny_config(tokenizer, **kw):
    from transformers import BertConfig
    return BertConfig(vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=128, max_position_embeddings=512, **kw)


def build_decision(cfg: TrainConfig, tiny_texts=None, work_dir=None):
    """Returns (DecisionModel, tokenizer)."""
    if cfg.tiny:
        from transformers import BertForSequenceClassification
        tok = tiny_tokenizer(tiny_texts, Path(work_dir) / "tiny_vocab_decision")
        src = BertForSequenceClassification(_tiny_config(tok, num_labels=1))
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.backbone)
        src = AutoModelForSequenceClassification.from_pretrained(cfg.backbone)
    model = DecisionModel(src.base_model, warm_classifier=getattr(src, "classifier", None))
    return model, tok


def build_ops(cfg: TrainConfig, tiny_texts=None, work_dir=None):
    """Returns (OpLogits wrapper around a 4-way sequence classifier, tokenizer, warm_started)."""
    if cfg.tiny:
        from transformers import BertForSequenceClassification
        tok = tiny_tokenizer(tiny_texts, Path(work_dir) / "tiny_vocab_ops")
        src = BertForSequenceClassification(_tiny_config(tok, num_labels=3, id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
                                                         label2id={"contradiction": 0, "entailment": 1, "neutral": 2}))
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.backbone)
        src = AutoModelForSequenceClassification.from_pretrained(cfg.backbone)
    model, warm = rehead_for_ops(src)
    return OpLogits(model), tok, warm

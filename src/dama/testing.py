"""Tiny random network for forward-only software tests. No optimizer or training."""
from pathlib import Path
from transformers import BertConfig, BertModel, BertTokenizerFast
from .contracts import ModelConfig
from .model import DAMADecisionModel


def tiny_model(directory):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    vocabulary=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","the","project","uses","postgresql","mongodb","i","prefer","dark","mode","we","switched","to","what","database","now","before",".","?",":","=","memory"]
    (root/"vocab.txt").write_text("\n".join(vocabulary)+"\n")
    tokenizer=BertTokenizerFast(vocab_file=str(root/"vocab.txt"),do_lower_case=True)
    encoder=BertModel(BertConfig(vocab_size=len(vocabulary),hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,max_position_embeddings=512))
    config=ModelConfig(encoder="dama/tiny-random-bert-test",encoder_revision="0"*40,hidden_dim=32,context_layers=1,max_length=256,dropout=0.0)
    model=DAMADecisionModel(config,encoder=encoder,load_pretrained=False).eval()
    return model,tokenizer

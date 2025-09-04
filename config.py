from dataclasses import dataclass

@dataclass
class Config():
    vocab_size = 163842 # prebuild tokenizer vocab size
    embed_dim = 768
    dropout = 0.1
    n_head = 12
    n_layer = 12
    context_size = 1024


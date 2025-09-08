from dataclasses import dataclass

@dataclass
class Config():
    vocab_size = 163842 # prebuilt tokenizer vocab size
    dim = 768 # Embed Dim
    dropout = 0.1
    n_heads = 4
    n_layer = 4
    context_size = 1024
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    max_batch_size: int = 16
    max_seq_length:int = 1024



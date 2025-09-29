from dataclasses import dataclass
from typing import Literal
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
    n_activated_experts:int = 4
    n_expert_groups:int = 6
    n_limited_groups:int = 2
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    n_routed_experts = 4
    n_shared_experts = 2
    moe_inter_dim = 512 
    inter_dim = 512
    n_dense_layers = 4
    n_layers = 2



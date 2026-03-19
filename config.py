from dataclasses import dataclass
from typing import Literal
from transformers import AutoTokenizer


def get_vocab_size(tokenizer_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return len(tokenizer)


@dataclass
class Config:
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    vocab_size: int = None

    dim: int = 896
    dropout: float = 0.1
    n_heads: int = 8
    n_layers: int = 6
    max_seq_length: int = 1024

    qk_nope_head_dim: int = 56
    qk_rope_head_dim: int = 56
    v_head_dim: int = 112
    kv_lora_rank: int = 256
    max_batch_size: int = 16

    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_shared_experts: int = 2

    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0

    moe_inter_dim: int = 1024
    inter_dim: int = 1024

    n_dense_layers: int = 1

    aux_loss_coeff: float = 0.01

    def __post_init__(self):
        if self.vocab_size is None:
            self.vocab_size = get_vocab_size(self.tokenizer_name)

from dataclasses import dataclass
from typing import Literal
from transformers import AutoTokenizer
def get_vocab_size(tokenizer_name: str) -> int:
    """
    Loads a tokenizer from Hugging Face and returns its vocabulary size.

    Args:
        tokenizer_name (str): The name of the tokenizer on the Hugging Face Hub.

    Returns:
        int: The size of the tokenizer's vocabulary.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return len(tokenizer)

@dataclass
class Config:
    tokenizer_name:str = "HuggingFaceTB/SmolLM-135M"
    vocab_size: int = get_vocab_size(tokenizer_name) if get_vocab_size(tokenizer_name) is not None else 49152 # Original Kimi K2- 163,842, 
    dim: int = 768
    dropout: float = 0.1
    n_heads: int = 4
    n_layers: int = 4  # Total layers
    max_seq_length: int = 1024
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    max_batch_size: int = 16
    n_routed_experts: int = 4
    n_activated_experts: int = 2  # Activate 2 out of 4
    n_expert_groups: int = 2  # 4 experts / 2 groups = 2 experts per group
    n_shared_experts: int = 2
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    moe_inter_dim: int = 512
    inter_dim: int = 512
    n_dense_layers: int = 2  # First 2 layers dense, last 2 MoE
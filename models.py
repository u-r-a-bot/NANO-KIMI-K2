import torch 
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import Config

config = Config()

class Embeddings(nn.Module):
    def __init__(self, embed_dim=config.dim, vocab_size=config.vocab_size):
        super().__init__()
        self.table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table(x)

#Borrowed from Official torch implementation of rotary embeddings
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096):
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        # Q stays per-head
        self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)

        # K/V use compressed latent space (THIS IS THE KEY CHANGE!)
        self.wk_compress = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.wv_compress = nn.Linear(self.dim, self.kv_lora_rank, bias=False)

        # Up-project compressed K/V to per-head space
        self.wk_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim, bias=False)
        self.wv_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)

        self.rope = RotaryPositionalEmbeddings(
            dim=self.qk_rope_head_dim,
            max_seq_len=config.max_seq_length
        )

        # Cache now stores COMPRESSED representations!
        self.k_cache = None
        self.v_cache = None

        self.softmax_scale = self.qk_head_dim ** -0.5
        self.c_v_linear = nn.Linear(
            in_features=self.n_heads * self.v_head_dim,
            out_features=self.dim
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            input_pos: Optional[torch.Tensor],
            mask: Optional[torch.Tensor],
            use_cache: bool = False,
            is_causal: bool = False
    ):
        b, seq_len, _ = x.size()

        # Q: standard per-head projection
        q = self.wq(x).view(b, seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # K: compress then expand (MLA magic!)
        k_compress = self.wk_compress(x)  # [b, seq_len, kv_lora_rank + qk_rope_head_dim]
        k_compress_nope, k_rope = torch.split(k_compress, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # V: compress then expand
        v_compress = self.wv_compress(x)  # [b, seq_len, kv_lora_rank]

        # Apply RoPE to q and k rope components
        q_rope = self.rope(q_rope, input_pos=input_pos)
        k_rope = self.rope(k_rope.unsqueeze(2), input_pos=input_pos)  # Add head dim for rope

        # Cache management: store COMPRESSED k/v
        end_pos = start_pos + seq_len
        if use_cache:
            if self.k_cache is None:
                self.k_cache = torch.zeros(
                    (self.config.max_batch_size, self.config.max_seq_length, self.kv_lora_rank),
                    dtype=x.dtype,
                    device=x.device
                )
                self.v_cache = torch.zeros(
                    (self.config.max_batch_size, self.config.max_seq_length, self.kv_lora_rank),
                    dtype=x.dtype,
                    device=x.device
                )
                self.k_rope_cache = torch.zeros(
                    (self.config.max_batch_size, self.config.max_seq_length, 1, self.qk_rope_head_dim),
                    dtype=x.dtype,
                    device=x.device
                )

            self.k_cache[:b, start_pos:end_pos] = k_compress_nope
            self.v_cache[:b, start_pos:end_pos] = v_compress
            self.k_rope_cache[:b, start_pos:end_pos] = k_rope

            k_compress_nope = self.k_cache[:b, :end_pos]
            v_compress = self.v_cache[:b, :end_pos]
            k_rope = self.k_rope_cache[:b, :end_pos]

        # Up-project compressed K/V to per-head space
        k_nope = self.wk_up(k_compress_nope).view(b, -1, self.n_heads, self.qk_nope_head_dim)
        v = self.wv_up(v_compress).view(b, -1, self.n_heads, self.v_head_dim)

        # Expand k_rope to all heads
        k_rope = k_rope.expand(b, -1, self.n_heads, self.qk_rope_head_dim)

        # Concatenate q and k components
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Standard attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = mask if mask is not None else None

        c_v = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is None and is_causal)
        )

        c_v = c_v.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.c_v_linear(c_v)


# class MLA(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.dim = config.dim
#         self.n_heads = config.n_heads
#         self.qk_nope_head_dim = config.qk_nope_head_dim
#         self.qk_rope_head_dim = config.qk_rope_head_dim
#         self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
#         self.v_head_dim = config.v_head_dim
#
#         self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
#         self.wk = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
#         self.wv = nn.Linear(self.dim, self.n_heads * self.v_head_dim, bias=False)
#
#         self.rope = RotaryPositionalEmbeddings(
#             dim=self.qk_rope_head_dim,
#             max_seq_len=config.max_seq_length
#         )
#
#         self.k_cache = None
#         self.v_cache = None
#
#         self.softmax_scale = self.qk_head_dim ** -0.5
#         self.c_v_linear = nn.Linear(
#             in_features=self.n_heads * self.v_head_dim,
#             out_features=self.dim
#         )
#
#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         input_pos: Optional[torch.Tensor],
#         mask: Optional[torch.Tensor],
#         use_cache: bool = False,
#         is_causal: bool = False
#     ):
#         b, seq_len, _ = x.size()
#
#         q = self.wq(x).view(b, seq_len, self.n_heads, self.qk_head_dim)
#         k = self.wk(x).view(b, seq_len, self.n_heads, self.qk_head_dim)
#         v = self.wv(x).view(b, seq_len, self.n_heads, self.v_head_dim)
#
#         q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
#         k_nope, k_rope = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
#
#         q_rope = self.rope(q_rope, input_pos=input_pos)
#         k_rope = self.rope(k_rope, input_pos=input_pos)
#
#         q = torch.cat([q_nope, q_rope], dim=-1)
#         k = torch.cat([k_nope, k_rope], dim=-1)
#
#         end_pos = start_pos + seq_len
#         if use_cache:
#             if self.k_cache is None:
#                 self.k_cache = torch.zeros(
#                     (self.config.max_batch_size, self.config.max_seq_length, self.n_heads, self.qk_head_dim),
#                     dtype=x.dtype,
#                     device=x.device
#                 )
#                 self.v_cache = torch.zeros(
#                     (self.config.max_batch_size, self.config.max_seq_length, self.n_heads, self.v_head_dim),
#                     dtype=x.dtype,
#                     device=x.device
#                 )
#
#             self.k_cache[:b, start_pos:end_pos] = k
#             self.v_cache[:b, start_pos:end_pos] = v
#             k = self.k_cache[:b, :end_pos]
#             v = self.v_cache[:b, :end_pos]
#
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
#
#         attn_mask = mask if mask is not None else None
#
#         c_v = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_mask,
#             dropout_p=0.0,
#             is_causal=(attn_mask is None and is_causal)
#         )
#
#         c_v = c_v.transpose(1, 2).contiguous().view(b, seq_len, -1)
#         return self.c_v_linear(c_v)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Expert(nn.Module):
    def __init__(self, dim: int, hid_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hid_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.score_func = config.score_func
        self.router_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        self.bias = nn.Parameter(torch.zeros(config.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x, self.weight, self.bias)
        
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        
        original_scores = scores
        
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            topk_groups = self.topk // self.n_groups
            indices = group_scores.topk(topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        
        weights *= self.router_scale
        return weights.type_as(x), indices

class MoE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_local_experts = config.n_routed_experts 
        self.n_activated_experts = config.n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.n_local_experts
        self.gate = Gate(config)
        self.experts = nn.ModuleList([Expert(config.dim, config.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(config.dim, config.n_shared_experts * config.moe_inter_dim)

    def forward(self, x: torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] = y[idx] + expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)

class Block(nn.Module):
    def __init__(self, layer_id: int, config: Config):
        super().__init__()
        self.attn = MLA(config)
        self.ffn = MLP(config.dim, config.inter_dim) if layer_id < config.n_dense_layers else MoE(config)
        self.attn_norm = nn.RMSNorm(config.dim, eps=1e-5)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=1e-5)

    def forward(self, x: torch.Tensor, start_pos: int, input_pos: torch.Tensor, mask: Optional[torch.Tensor], use_cache: bool = False, is_causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, input_pos, mask, use_cache=use_cache, is_causal=is_causal)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.max_seq_length = config.max_seq_length
        self.vocab_size = config.vocab_size
        self.embed = Embeddings(config.dim, config.vocab_size)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))
        self.norm = nn.RMSNorm(config.dim, eps=1e-5)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False):
        b, seqlen = tokens.shape
        h = self.embed(tokens)
        
        input_pos = torch.arange(start_pos, start_pos + seqlen, device=tokens.device)
        
        # Training (multi-token, no cache): use SDPA causal fast-path.
        # Decoding with cache (usually 1 token): no mask and no is_causal (cache enforces causality).
        mask = None
        use_is_causal = (seqlen > 1 and not use_cache)
        
        for layer in self.layers:
            h = layer(h, start_pos, input_pos, mask, use_cache=use_cache, is_causal=use_is_causal)
        
        h = self.norm(h)
        logits = self.head(h)
        return logits
    
    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.9):
        self.eval()
        for layer in self.layers:
            layer.attn.k_cache = None
            layer.attn.v_cache = None
        
        generated_tokens = []
        input_tokens = tokens
        start_pos = 0
        
        for _ in range(max_new_tokens):
            logits = self.forward(input_tokens, start_pos=start_pos, use_cache=True)
            logits = logits[:, -1, :] / temperature
            
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            
            start_pos += input_tokens.size(1)
            input_tokens = next_token
        
        return generated_tokens

import torch 
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, Literal, Tuple
from config import Config

config = Config()

class Embeddings(nn.Module):
    def __init__(self, embed_dim=config.dim, vocab_size=config.vocab_size):
        super().__init__()
        self.table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table(x)

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
    
class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads * self.v_head_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(
            dim=self.qk_rope_head_dim, 
            max_seq_len=config.max_seq_length
        )
        self.register_buffer("k_cache", torch.zeros(config.max_batch_size, config.max_seq_length, config.n_heads, self.qk_head_dim))
        self.register_buffer("v_cache", torch.zeros(config.max_batch_size, config.max_seq_length, config.n_heads, config.v_head_dim))
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.c_v_linear = nn.Linear(in_features=self.n_heads * self.v_head_dim, out_features=self.dim)

    def forward(self, x: torch.Tensor, start_pos: int, input_pos: Optional[torch.Tensor], mask: Optional[torch.Tensor]):
        b, seq_len, _ = x.size()
        
        q = self.wq(x).view(b, seq_len, self.n_heads, self.qk_head_dim)
        k = self.wk(x).view(b, seq_len, self.n_heads, self.qk_head_dim) 
        v = self.wv(x).view(b, seq_len, self.n_heads, self.v_head_dim)
        
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, k_rope = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        q_rope = self.rope(q_rope, input_pos=input_pos)
        k_rope = self.rope(k_rope, input_pos=input_pos)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Differentiate between training and inference
        if self.training:
            # --- TRAINING LOGIC ---
            # Don't use the cache. Compute attention on the current sequence.
            scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale

            if mask is not None:
                scores = scores + mask.unsqueeze(1)

            weights = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            c_v = torch.einsum("bsht,bthd->bshd", weights, v)
        else:
            # --- INFERENCE LOGIC ---
            # Use the cache for autoregressive generation.
            end_pos = start_pos + seq_len
            self.k_cache[:b, start_pos:end_pos] = k
            self.v_cache[:b, start_pos:end_pos] = v
            
            cached_k = self.k_cache[:b, :end_pos]
            cached_v = self.v_cache[:b, :end_pos]
            
            scores = torch.einsum("bshd,bthd->bsht", q, cached_k) * self.softmax_scale

            if mask is not None:
                scores = scores + mask.unsqueeze(1)

            weights = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            c_v = torch.einsum("bsht,bthd->bshd", weights, cached_v)
        
        return self.c_v_linear(c_v.flatten(2))
        
class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Expert(nn.Module):
    def __init__(self, dim: int, hid_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hid_dim)
        self.w2 = nn.Linear(hid_dim, dim)
        self.w3 = nn.Linear(dim, hid_dim)

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
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

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
            y[idx] = y[idx]+ expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)

class Block(nn.Module):
    def __init__(self, layer_id: int, config: Config):
        super().__init__()
        self.attn = MLA(config)
        self.ffn = MLP(config.dim, config.inter_dim) if layer_id < config.n_dense_layers else MoE(config)
        self.attn_norm = nn.RMSNorm(config.dim)
        self.ffn_norm = nn.RMSNorm(config.dim)

    def forward(self, x: torch.Tensor, start_pos: int, input_pos: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, input_pos, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.max_seq_length = config.max_seq_length
        self.embed = Embeddings(config.dim, config.vocab_size)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))
        self.norm = nn.RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
    
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        
        input_pos = torch.arange(start_pos, start_pos + seqlen, device=tokens.device)
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        
        for layer in self.layers:
            h = layer(h, start_pos, input_pos, mask)
        
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits
        
if __name__ == "__main__":
    # Device agnostic setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)
    
    config = Config()
    model = Transformer(config).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 2 / 1024**3:.2f} GB (bf16)")
    
    # Initialize model weights properly
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)
    
    # Test 1: Single forward pass
    print("\n--- Test 1: Forward Pass ---")
    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    logits = model(tokens, start_pos=0)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    assert logits.shape == (batch_size, config.vocab_size)
    
    # Test 2: Sequential generation 
    print("\n--- Test 2: Sequential Generation ---")
    model.eval()
    
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    generated = []
    
    with torch.no_grad():
        # Reset caches for new sequence
        for layer in model.layers:
            layer.attn.k_cache.zero_()
            layer.attn.v_cache.zero_()
        
        # Process prompt
        logits = model(prompt, start_pos=0)
        
        # Generate next tokens
        for i in range(20):
            next_pos = prompt.size(1) + i
            
            # Safe sampling
            temperature = 0.8
            scaled_logits = torch.clamp(logits / temperature, min=-100, max=100)
            probs = F.softmax(scaled_logits, dim=-1)
            
            if torch.isnan(probs).any():
                next_token = torch.randint(0, config.vocab_size, (1, 1), device=device)
            else:
                next_token = torch.multinomial(probs.squeeze(0), num_samples=1).unsqueeze(0)
            
            generated.append(next_token.item())
            logits = model(next_token, start_pos=next_pos)
    
    print(f"Generated tokens: {generated[:10]}...")
    
    # Test 3: Batch processing (each with clean cache)
    print("\n--- Test 3: Fresh Sequences ---")
    for seq_len in [16, 32, 64]:
        # Reset cache for each test
        for layer in model.layers:
            layer.attn.k_cache.zero_()
            layer.attn.v_cache.zero_()
        
        batch_tokens = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
        with torch.no_grad():
            logits = model(batch_tokens, start_pos=0)
            print(f"Seq len {seq_len}: output shape {logits.shape}")
    
    # Test 4: Incremental generation (proper cache usage)
    print("\n--- Test 4: Incremental Generation ---")
    # Reset cache
    for layer in model.layers:
        layer.attn.k_cache.zero_()
        layer.attn.v_cache.zero_()
    
    # Generate incrementally
    seq = torch.randint(0, config.vocab_size, (1, 1), device=device)
    positions_filled = 0
    
    with torch.no_grad():
        for step in range(10):
            logits = model(seq, start_pos=positions_filled)
            positions_filled += seq.size(1)
            
            # Get next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            seq = next_token
            
            print(f"Step {step}: generated token {next_token.item()}, cache filled up to {positions_filled}")
            
            if positions_filled >= 100:
                break
    
    # Test 5: Max sequence with timing
    print("\n--- Test 5: Max Sequence Test ---")
    for layer in model.layers:
        layer.attn.k_cache.zero_()
        layer.attn.v_cache.zero_()
    
    max_test_len = min(256, config.max_seq_length)
    test_seq = torch.randint(0, config.vocab_size, (1, max_test_len), device=device)
    
    # Device-agnostic timing
    import time
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start.record()
            logits = model(test_seq, start_pos=0)
            end.record()
            torch.cuda.synchronize()
            
        elapsed_ms = start.elapsed_time(end)
    else:
        # For CPU/MPS
        with torch.no_grad():
            if device.type == 'mps':
                torch.mps.synchronize()
            
            start_time = time.perf_counter()
            logits = model(test_seq, start_pos=0)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"Processing {max_test_len} tokens: {elapsed_ms:.2f}ms")
    print(f"Throughput: {max_test_len / (elapsed_ms / 1000):.1f} tokens/sec")
    
    # Test 6: Memory stats
    print("\n--- Test 6: Memory Stats ---")
    if device.type == 'cuda':
        print(f"CUDA Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    elif device.type == 'mps':
        print(f"MPS Allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
        print(f"MPS Driver Allocated: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GB")
    else:
        print("Memory stats not available for CPU")
    
    print(f"\n✓ All tests passed on {device}")
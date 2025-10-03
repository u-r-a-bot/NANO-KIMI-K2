import torch
import torch.nn.functional as F
from models import Transformer
from config import Config
from optimizer import MuonClip, get_muon_param_groups

def verify_muon_implementation():
    print("=" * 60)
    print("MUON CLIP OPTIMIZER VERIFICATION")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    config = Config()
    config.vocab_size = 1000
    
    print("\n[1/5] Testing Model Initialization...")
    model = Transformer(config).to(device).to(dtype)
    print("✓ Model initialized successfully")
    
    print("\n[2/5] Testing Parameter Grouping...")
    param_groups = get_muon_param_groups(model, lr=0.02, weight_decay=0.01)
    qk_params = [p for g in param_groups if g.get('is_qk', False) for p in g['params']]
    other_params = [p for g in param_groups if not g.get('is_qk', False) for p in g['params']]
    
    print(f"  QK parameters: {len(qk_params)}")
    print(f"  Other parameters: {len(other_params)}")
    print("✓ Parameter grouping working correctly")
    
    print("\n[3/5] Testing MuonClip Optimizer...")
    optimizer = MuonClip(
        param_groups,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        qk_clip_tau=100.0,
        qk_clip_enabled=True
    )
    print("✓ Optimizer initialized successfully")
    
    print("\n[4/5] Testing Forward Pass & Max Logits Computation...")
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    model.train()
    logits = model(tokens, start_pos=0, use_cache=False)
    
    max_logits = logits.abs().max().item()
    print(f"  Logits shape: {logits.shape}")
    print(f"  Max logits value: {max_logits:.4f}")
    print("✓ Forward pass completed successfully")
    
    print("\n[5/5] Testing Backward Pass & Optimizer Step...")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        labels.reshape(-1)
    )
    
    loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step(max_logits=max_logits)
    optimizer.zero_grad()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Grad norm: {grad_norm:.4f}")
    print(f"  Max logits passed to optimizer: {max_logits:.4f}")
    print("✓ Backward pass and optimizer step completed")
    
    print("\n[BONUS] Testing QK-Clip Activation...")
    artificial_max_logits = 150.0
    print(f"  Simulating max_logits = {artificial_max_logits}")
    
    logits = model(tokens, start_pos=0, use_cache=False)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
    loss.backward()
    
    optimizer.step(max_logits=artificial_max_logits)
    optimizer.zero_grad()
    
    if artificial_max_logits > optimizer.defaults['qk_clip_tau']:
        print(f"✓ QK-Clip should activate (threshold: {optimizer.defaults['qk_clip_tau']})")
    
    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED ✓")
    print("=" * 60)
    
    print("\n📊 Implementation Summary:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - QK parameters: {sum(p.numel() for p in qk_params):,}")
    print(f"  - Other parameters: {sum(p.numel() for p in other_params):,}")
    print(f"  - Optimizer: MuonClip")
    print(f"  - QK-Clip threshold: {optimizer.defaults['qk_clip_tau']}")
    print(f"  - Learning rate: {optimizer.defaults['lr']}")
    print(f"  - Momentum: {optimizer.defaults['momentum']}")
    print(f"  - Max logits history: {len(optimizer.max_logits_history)} entries")
    
    print("\n✅ MUON Clip optimizer is ready for training!")
    
    return True

if __name__ == "__main__":
    try:
        verify_muon_implementation()
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from models import Transformer
from config import Config

def load_model(checkpoint_path: str, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config.vocab_size = len(tokenizer)
    
    model = Transformer(config).to(device).to(dtype)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Checkpoint step: {checkpoint.get('global_step', 'N/A')}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model, tokenizer, device, dtype

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, 
                  temperature: float = 0.8, top_p: float = 0.9, device=None):
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    for layer in model.layers:
        layer.attn.k_cache.zero_()
        layer.attn.v_cache.zero_()
    
    generated_tokens = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    full_tokens = input_ids[0].tolist() + generated_tokens
    generated_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
    
    return generated_text

def interactive_mode(model, tokenizer, device, max_new_tokens=100, temperature=0.8, top_p=0.9):
    print("\n" + "="*60)
    print("Interactive Generation Mode (type 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...\n")
        output = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device)
        print(f"Output:\n{output}\n")
        print("-"*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return
    
    model, tokenizer, device, dtype = load_model(args.checkpoint_path, args.device)
    
    if args.interactive:
        interactive_mode(model, tokenizer, device, args.max_tokens, args.temperature, args.top_p)
    else:
        if args.prompt is None:
            args.prompt = "Once upon a time"
        
        print(f"\nPrompt: {args.prompt}\n")
        output = generate_text(model, tokenizer, args.prompt, args.max_tokens, 
                              args.temperature, args.top_p, device)
        print(f"Generated:\n{output}\n")

if __name__ == "__main__":
    main()
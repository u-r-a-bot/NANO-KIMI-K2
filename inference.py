import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from models import Transformer
from config import Config
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.align import Align
import time

# Initialize rich console
console = Console()

def display_ascii_art():
    ascii_art = r"""
[bold cyan]
 _   _   ___   _   _ _____   _   __________  ________ _   __ _____ 
| \ | | / _ \ | \ | |  _  | | | / /_   _|  \/  |_   _| | / // __  \
|  \| |/ /_\ \|  \| | | | | | |/ /  | | | .  . | | | | |/ / `' / /'
| . ` ||  _  || . ` | | | | |    \  | | | |\/| | | | |    \   / /  
| |\  || | | || |\  \ \_/ / | |\  \_| |_| |  | |_| |_| |\  \./ /___
\_| \_/\_| |_/\_| \_/\___/  \_| \_/\___/\_|  |_/\___/\_| \_/\_____/
[/bold cyan]
"""
    console.print(ascii_art)

def load_model(checkpoint_path: str, device: str = None):
    display_ascii_art()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        
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
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = checkpoint["model_state_dict"]

        is_compiled = any(k.startswith("_orig_mod.") for k in state_dict)

        if is_compiled:
            console.print("[yellow]Detected torch.compile-style checkpoint — fixing keys...[/yellow]")
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_key = k.replace("_orig_mod.", "", 1)
                else:
                    new_key = k
                clean_state_dict[new_key] = v
            state_dict = clean_state_dict
        else:
            console.print("[green]Checkpoint is already clean — no fix needed[/green]")

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        
        progress.update(task, description="Model loaded successfully!")
        time.sleep(0.5)  # Brief pause for visual effect
    
    # Display model information in a nice table
    table = Table(title="Model Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="green")
    
    table.add_row("Checkpoint Path", checkpoint_path)
    table.add_row("Device", str(device))
    table.add_row("Data Type", str(dtype))
    table.add_row("Checkpoint Step", str(checkpoint.get('global_step', 'N/A')))
    table.add_row("Checkpoint Epoch", str(checkpoint.get('epoch', 'N/A')))
    table.add_row("Vocabulary Size", str(config.vocab_size))
    
    console.print(table)
    
    return model, tokenizer, device, dtype

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, 
                  temperature: float = 0.8, top_p: float = 0.9, device=None):
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # for layer in model.layers:
    #     layer.attn.k_cache.zero_()
    #     layer.attn.v_cache.zero_()
    
    start_time = time.time()
    generated_tokens = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    end_time = time.time()
    
    full_tokens = input_ids[0].tolist() + generated_tokens
    generated_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
    
    # Calculate statistics
    generation_time = end_time - start_time
    tokens_generated = len(generated_tokens)
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    stats = {
        'tokens_generated': tokens_generated,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second
    }
    
    return generated_text, stats

@torch.inference_mode()
def stream_generate(model, tokenizer, prompt: str, max_new_tokens: int = 100,
                    temperature: float = 0.8, top_p: float = 0.9, device=None):
    """
    Yields decoded text chunks as they are generated, without modifying core model logic.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Reset KV caches by running a dummy call identical to model.generate's setup
    for layer in model.layers:
        layer.attn.k_cache = None
        layer.attn.v_cache = None

    input_tokens = input_ids
    start_pos = 0
    tokens_generated = 0
    start_time = time.time()

    for _ in range(max_new_tokens):
        logits = model.forward(input_tokens, start_pos=start_pos, use_cache=True)
        logits = logits[:, -1, :] / temperature

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Yield decoded delta for just the new token
        text_delta = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
        if text_delta:
            yield text_delta
            tokens_generated += 1

        start_pos += input_tokens.size(1)
        input_tokens = next_token
    
    # Calculate statistics
    end_time = time.time()
    generation_time = end_time - start_time
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    stats = {
        'tokens_generated': tokens_generated,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second
    }
    
    # Return stats as the last item
    yield stats

def display_generation_stats(stats):
    stats_table = Table(title="Generation Statistics", show_header=True, header_style="bold blue")
    stats_table.add_column("Metric", style="cyan", width=20)
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Tokens Generated", str(stats['tokens_generated']))
    stats_table.add_row("Generation Time", f"{stats['generation_time']:.2f} seconds")
    stats_table.add_row("Tokens/Second", f"{stats['tokens_per_second']:.2f}")
    
    console.print(stats_table)

def interactive_mode(model, tokenizer, device, max_new_tokens=100, temperature=0.8, top_p=0.9):
    console.print(Panel.fit(
        "[bold cyan]Interactive Generation Mode[/bold cyan]\n[dim]Type 'quit' to exit[/dim]",
        border_style="cyan"
    ))
    
    while True:
        prompt = Prompt.ask("\n[bold]Enter prompt[/bold]", default="").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            console.print("[bold red]Exiting interactive mode...[/bold red]")
            break
        
        if not prompt:
            continue
        
        with console.status("[bold green]Generating..."):
            output, stats = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device)
        
        console.print(Rule(style="blue"))
        console.print(Panel(
            Syntax(output, "markdown", theme="monokai", line_numbers=False),
            title="[bold green]Generated Text[/bold green]",
            border_style="green"
        ))
        console.print(Rule(style="blue"))
        
        # Display generation statistics
        display_generation_stats(stats)

def main():
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--stream', action='store_true', help='Stream tokens as they generate')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint_path).exists():
        console.print(f"[bold red]Error: Checkpoint not found at {args.checkpoint_path}[/bold red]")
        return
    
    model, tokenizer, device, dtype = load_model(args.checkpoint_path, args.device)
    
    if args.interactive:
        if args.stream:
            console.print(Panel.fit(
                "[bold cyan]Interactive Streaming Mode[/bold cyan]\n[dim]Type 'quit' to exit[/dim]",
                border_style="cyan"
            ))
            while True:
                prompt = Prompt.ask("\n[bold]Enter prompt[/bold]", default="").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    console.print("[bold red]Exiting interactive mode...[/bold red]")
                    break
                if not prompt:
                    continue
                
                console.print(Rule(style="blue"))
                console.print("[bold green]Generating (streaming)...[/bold green]")
                
                # Direct streaming without Live display for better performance
                stats = None
                for delta in stream_generate(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_p, device):
                    if isinstance(delta, dict):
                        stats = delta
                    else:
                        print(delta, end='', flush=True)
                
                print()  # Add a newline after generation
                console.print(Rule(style="blue"))
                
                # Display generation statistics
                if stats:
                    display_generation_stats(stats)
        else:
            interactive_mode(model, tokenizer, device, args.max_tokens, args.temperature, args.top_p)
    else:
        if args.prompt is None:
            args.prompt = "Once upon a time"
        
        console.print(Panel(
            f"[bold cyan]Prompt:[/bold cyan] {args.prompt}",
            border_style="cyan"
        ))
        
        if args.stream:
            console.print("[bold green]Generating (streaming)...[/bold green]")
            
            # Direct streaming without Live display for better performance
            stats = None
            for delta in stream_generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p, device):
                if isinstance(delta, dict):
                    stats = delta
                else:
                    print(delta, end='', flush=True)
            
            print()  # Add a newline after generation
            
            # Display generation statistics
            if stats:
                display_generation_stats(stats)
        else:
            with console.status("[bold green]Generating..."):
                output, stats = generate_text(model, tokenizer, args.prompt, args.max_tokens, 
                                      args.temperature, args.top_p, device)
            
            console.print(Panel(
                Syntax(output, "markdown", theme="monokai", line_numbers=False),
                title="[bold green]Generated Text[/bold green]",
                border_style="green"
            ))
            
            # Display generation statistics
            display_generation_stats(stats)

if __name__ == "__main__":
    main()

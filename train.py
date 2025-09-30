import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Optional, Dict, List
import argparse

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box

from models import Transformer
from config import Config
from finewebdataset import FineWebStreamingDataset, FineWebDataset
from dataset import TextDataset, DataCollator

console = Console()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        eval_interval: int = 100,
        save_interval: int = 500,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        self.mixed_precision = mixed_precision and device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.mixed_precision,device=device ) if self.mixed_precision else None
        
        self.global_step = 0
        self.best_loss = float('inf')
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'steps': []
        }
        
        self.start_time = time.time()
        
    def get_lr(self):
        if self.global_step < self.warmup_steps:
            return self.learning_rate * (self.global_step + 1) / self.warmup_steps
        else:
            progress = (self.global_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return self.learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))
    
    def update_lr(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    @torch.no_grad()
    def evaluate(self, num_batches: int = 10):
        self.model.eval()
        losses = []
        
        eval_loader = self.val_loader or self.train_loader
        for i, batch in enumerate(eval_loader):
            if i >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16 if self.mixed_precision else torch.float32):
                logits = self.model(input_ids, start_pos=0, use_cache=False)
                
                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.vocab_size),
                    labels.reshape(-1),
                    ignore_index=0,
                    reduction='none'
                )
                masked_loss = loss * attention_mask.reshape(-1)
                loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)
            
            losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses) if losses else float('inf')
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'metrics': self.metrics,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            console.print(f"[green]✓ Saved best checkpoint (loss: {self.best_loss:.4f})[/green]")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        
        if not Path(checkpoint_path).exists():
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.metrics = checkpoint['metrics']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        console.print(f"[cyan]Resumed from step {self.global_step}[/cyan]")
        return True
    
    def plot_metrics(self):
        if len(self.metrics['steps']) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Metrics - Step {self.global_step}', fontsize=14)
        
        ax1.plot(self.metrics['steps'], self.metrics['train_loss'], 'b-', label='Train Loss')
        if self.metrics['val_loss']:
            val_steps = self.metrics['steps'][::self.eval_interval] if self.eval_interval > 0 else self.metrics['steps']
            val_steps = val_steps[:len(self.metrics['val_loss'])]
            ax1.plot(val_steps, self.metrics['val_loss'], 'r-', label='Val Loss', marker='o', markersize=3)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.metrics['steps'], self.metrics['learning_rate'], 'g-')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.metrics['steps'], self.metrics['gradient_norm'], 'orange')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Gradient Norm')
        ax3.grid(True, alpha=0.3)
        
        elapsed_per_step = (time.time() - self.start_time) / max(1, self.global_step)
        throughput = [1.0 / elapsed_per_step for _ in self.metrics['steps']]
        ax4.plot(self.metrics['steps'], throughput, 'purple')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Steps/sec')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def train(self):
        console.print(Panel.fit(
            f"[bold cyan]Starting Training[/bold cyan]\n"
            f"Model: {self.model.__class__.__name__}\n"
            f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            f"Device: {self.device}\n"
            f"Max Steps: {self.max_steps:,}\n"
            f"Mixed Precision: {self.mixed_precision}\n"
            f"Gradient Accumulation: {self.gradient_accumulation_steps}",
            title="Nano KIMI K2 Training",
            border_style="cyan"
        ))
        
        self.model.train()
        accumulation_counter = 0
        accumulated_loss = 0
        step_start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2
        ) as progress:
            
            train_task = progress.add_task("[cyan]Training...", total=self.max_steps)
            progress.update(train_task, completed=self.global_step)
            
            data_iter = iter(self.train_loader)
            
            while self.global_step < self.max_steps:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16 if self.mixed_precision else torch.float32):
                    logits = self.model(input_ids, start_pos=0, use_cache=False)
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, self.model.vocab_size),
                        labels.reshape(-1),
                        ignore_index=0,
                        reduction='none'
                    )
                    masked_loss = loss * attention_mask.reshape(-1)
                    loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)
                    loss = loss / self.gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_counter += 1
                
                if accumulation_counter >= self.gradient_accumulation_steps:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    lr = self.update_lr()
                    
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                    train_loss = accumulated_loss * self.gradient_accumulation_steps
                    
                    self.metrics['train_loss'].append(train_loss)
                    self.metrics['learning_rate'].append(lr)
                    self.metrics['gradient_norm'].append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                    self.metrics['steps'].append(self.global_step)
                    
                    throughput = 1.0 / (time.time() - step_start_time)
                    step_start_time = time.time()
                    
                    progress.update(train_task, advance=1)
                    
                    if self.global_step % 10 == 0:
                        progress.update(
                            train_task,
                            description=f"[cyan]Step {self.global_step}: Loss={train_loss:.4f} LR={lr:.6f} Grad={grad_norm:.3f}"
                        )
                    
                    if self.global_step % self.eval_interval == 0:
                        val_loss = self.evaluate()
                        self.metrics['val_loss'].append(val_loss)
                        
                        is_best = val_loss < self.best_loss
                        if is_best:
                            self.best_loss = val_loss
                        
                        console.print(f"\n[yellow]Step {self.global_step}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={lr:.6f}, Best={self.best_loss:.4f}[/yellow]")
                        
                        if is_best:
                            self.save_checkpoint(is_best=True)
                    
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint()
                        self.plot_metrics()
                        console.print(f"[blue]Checkpoint saved at step {self.global_step}[/blue]")
                    
                    accumulated_loss = 0
                    accumulation_counter = 0
        
        final_val_loss = self.evaluate()
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n"
            f"Total Steps: {self.global_step:,}\n"
            f"Final Val Loss: {final_val_loss:.4f}\n"
            f"Best Loss: {self.best_loss:.4f}\n"
            f"Time: {(time.time() - self.start_time) / 3600:.1f} hours",
            title="Training Summary",
            border_style="green"
        ))
        
        self.save_checkpoint()
        self.plot_metrics()
        
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train Nano KIMI K2 Model')
    parser.add_argument('--dataset', type=str, default='local', choices=['fineweb', 'streaming', 'local'])
    parser.add_argument('--data_path', type=str, default='data/train.txt')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if args.device == 'cuda' and not args.mixed_precision:
        args.mixed_precision = True
        console.print("[yellow]Enabled mixed precision for CUDA[/yellow]")
    elif args.device != 'cuda':
        args.mixed_precision = False
    
    config = Config()
    
    console.print("[yellow]Loading tokenizer...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config.vocab_size = len(tokenizer)
    config.max_batch_size = args.batch_size
    
    console.print("[yellow]Preparing dataset...[/yellow]")
    
    try:
        if args.dataset == 'streaming':
            dataset = FineWebStreamingDataset(
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
                num_samples=args.num_samples,
                subset="sample-10BT"
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
        elif args.dataset == 'fineweb':
            dataset = FineWebDataset(
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
                num_samples=args.num_samples,
                cache_dir="./fineweb_cache",
                streaming=True,
                min_length=100,
                rebuild_cache=False
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4 if args.device == 'cuda' else 0,
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
        else:
            if not Path(args.data_path).exists():
                console.print(f"[yellow]Data file not found, creating sample data...[/yellow]")
                Path("data").mkdir(exist_ok=True)
                sample_text = """
                The quick brown fox jumps over the lazy dog. This is a sample training text for our model.
                Machine learning is transforming the world in unprecedented ways.
                Natural language processing enables computers to understand human language.
                Deep learning models have revolutionized artificial intelligence.
                Transformers architecture has become the foundation of modern NLP.
                Attention mechanisms allow models to focus on relevant information.
                Training neural networks requires careful optimization and tuning.
                """ * 500
                Path(args.data_path).write_text(sample_text)
            
            dataset = TextDataset(
                file_path=args.data_path,
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
                stride=512,
                min_length=32,
                combine_short=True,
                cache_tokens=True
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2 if args.device == 'cuda' else 0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=DataCollator(tokenizer.pad_token_id),
                drop_last=True
            )
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        console.print("[yellow]Creating fallback dataset...[/yellow]")
        Path("data").mkdir(exist_ok=True)
        sample_path = Path("data/fallback.txt")
        sample_text = "This is sample training data. " * 5000
        sample_path.write_text(sample_text)
        
        dataset = TextDataset(
            file_path=sample_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            stride=512,
            min_length=32,
            combine_short=True,
            cache_tokens=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=DataCollator(tokenizer.pad_token_id),
            drop_last=True
        )
    
    dataset_size = len(dataset) if hasattr(dataset, '__len__') else 'streaming'
    console.print(f"[green]Dataset ready with {dataset_size} samples[/green]")
    
    console.print("[yellow]Initializing model...[/yellow]")
    
    model = Transformer(config)
    
    if args.device == 'cuda':
        model = model.to(dtype=torch.bfloat16)
        console.print("[green]Using bfloat16 precision on CUDA[/green]")
    elif args.device == 'mps':
        model = model.to(dtype=torch.float32)
        console.print("[green]Using float32 precision on MPS[/green]")
    else:
        model = model.to(dtype=torch.float32)
        console.print("[green]Using float32 precision on CPU[/green]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters: {total_params:,}[/cyan]")
    console.print(f"[cyan]Trainable parameters: {trainable_params:,}[/cyan]")
    console.print(f"[cyan]Model size: {total_params * 4 / 1024**3:.2f} GB (float32)[/cyan]")
    
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision
    )
    
    if args.resume:
        loaded = trainer.load_checkpoint(args.checkpoint_path)
        if not loaded:
            console.print("[yellow]No checkpoint found, starting from scratch[/yellow]")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        trainer.save_checkpoint()
        console.print("[green]Checkpoint saved[/green]")
    except Exception as e:
        console.print(f"\n[red]Training error: {e}[/red]")
        trainer.save_checkpoint()
        console.print("[green]Emergency checkpoint saved[/green]")
        raise e
    finally:
        console.print("[cyan]Exiting training script[/cyan]")
        sys.exit(0)

if __name__ == "__main__":
    main()
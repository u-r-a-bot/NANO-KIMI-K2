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

def forward_for_training(model, tokens, start_pos=0):
    """Custom forward pass for training that returns all logits"""
    batch_size, seqlen = tokens.shape
    h = model.embed(tokens)
    
    input_pos = torch.arange(start_pos, start_pos + seqlen, device=tokens.device)
    
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu(1)
    
    for layer in model.layers:
        h = layer(h, start_pos, input_pos, mask)
    
    h = model.norm(h)
    logits = model.head(h)
    return logits

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
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
        
        use_scaler = mixed_precision and torch.get_default_dtype() is not torch.bfloat16
        self.scaler = torch.amp.GradScaler(enabled=use_scaler) if device == "cuda" else None
        
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
            return self.learning_rate * self.global_step / self.warmup_steps
        else:
            progress = (self.global_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return self.learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))
    
    def update_lr(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def compute_logits_for_training(self, tokens, start_pos=0):
        """Forward pass for training that returns logits for all positions"""
        batch_size, seqlen = tokens.shape
        h = self.model.embed(tokens)
        
        input_pos = torch.arange(start_pos, start_pos + seqlen, device=tokens.device)
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu(1)
        
        for layer in self.model.layers:
            h = layer(h, start_pos, input_pos, mask)
        
        h = self.model.norm(h)
        logits = self.model.head(h)
        return logits
    
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
            attention_mask = batch['attention_mask'].to(self.device)
            
            for layer in self.model.layers:
                layer.attn.k_cache.zero_()
                layer.attn.v_cache.zero_()
            
            logits = self.compute_logits_for_training(input_ids, start_pos=0)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0
            )
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
            'config': self.model.__class__.__name__,
        }
        
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
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.metrics = checkpoint['metrics']
        
        console.print(f"[cyan]Resumed from step {self.global_step}[/cyan]")
        return True
    
    def plot_metrics(self):
        if len(self.metrics['steps']) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Metrics - Step {self.global_step}', fontsize=14)
        
        ax1.plot(self.metrics['steps'], self.metrics['train_loss'], 'b-', label='Train Loss')
        if self.metrics['val_loss']:
            ax1.plot(self.metrics['steps'][:len(self.metrics['val_loss'])], self.metrics['val_loss'], 'r-', label='Val Loss')
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
        
        throughput = [s / (t + 1e-6) for s, t in zip(
            range(len(self.metrics['steps'])),
            [(time.time() - self.start_time) / max(1, step) * step for step in self.metrics['steps']]
        )]
        if len(throughput) > 0:
            ax4.plot(self.metrics['steps'], throughput, 'purple')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Steps/sec')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def get_stats_table(self, train_loss, val_loss, lr, grad_norm, throughput):
        table = Table(title=f"Step {self.global_step}/{self.max_steps}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Train Loss", f"{train_loss:.4f}")
        if val_loss is not None:
            table.add_row("Val Loss", f"{val_loss:.4f}")
            table.add_row("Best Loss", f"{self.best_loss:.4f}")
        table.add_row("Learning Rate", f"{lr:.6f}")
        table.add_row("Grad Norm", f"{grad_norm:.3f}")
        table.add_row("Throughput", f"{throughput:.2f} steps/s")
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(1, self.global_step)) * (self.max_steps - self.global_step)
        table.add_row("Time Elapsed", f"{elapsed/3600:.1f}h")
        table.add_row("ETA", f"{eta/3600:.1f}h")
        
        return table
    
    def train(self):
        console.print(Panel.fit(
            f"[bold cyan]Starting Training[/bold cyan]\n"
            f"Model: {self.model.__class__.__name__}\n"
            f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            f"Device: {self.device}\n"
            f"Max Steps: {self.max_steps:,}",
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
            
            while self.global_step < self.max_steps:
                for batch in self.train_loader:
                    if self.global_step >= self.max_steps:
                        break
                    
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    if self.global_step % 100 == 0:
                        for layer in self.model.layers:
                            layer.attn.k_cache.zero_()
                            layer.attn.v_cache.zero_()
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            logits = forward_for_training(self.model, input_ids, start_pos=0)
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                labels.reshape(-1),
                                ignore_index=0
                            )
                        loss = loss / self.gradient_accumulation_steps
                        self.scaler.scale(loss).backward()
                    else:
                        logits = forward_for_training(self.model, input_ids, start_pos=0)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=0
                        )
                        loss = loss / self.gradient_accumulation_steps
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
                        
                        self.optimizer.zero_grad()
                        
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
                                description=f"[cyan]Training... Loss: {train_loss:.4f} LR: {lr:.6f}"
                            )
                        
                        if self.global_step % self.eval_interval == 0:
                            val_loss = self.evaluate()
                            self.metrics['val_loss'].append(val_loss)
                            
                            is_best = val_loss < self.best_loss
                            if is_best:
                                self.best_loss = val_loss
                            
                            console.print(self.get_stats_table(
                                train_loss, val_loss, lr, grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, throughput
                            ))
                            
                            if is_best:
                                self.save_checkpoint(is_best=True)
                        
                        if self.global_step % self.save_interval == 0:
                            self.save_checkpoint()
                            self.plot_metrics()
                            console.print(f"[blue]Checkpoint saved at step {self.global_step}[/blue]")
                        
                        accumulated_loss = 0
                        accumulation_counter = 0
        
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n"
            f"Total Steps: {self.global_step:,}\n"
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
    parser.add_argument('--dataset', type=str, default='streaming', choices=['fineweb', 'streaming', 'local'],
                      help='Dataset type to use')
    parser.add_argument('--data_path', type=str, default='data/train.txt',
                      help='Path to local text file (for local dataset)')
    parser.add_argument('--num_samples', type=int, default=10000,
                      help='Number of samples for FineWeb dataset')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum training steps')
    parser.add_argument('--eval_interval', type=int, default=50,
                      help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100,
                      help='Checkpoint save interval')
    parser.add_argument('--warmup_steps', type=int, default=100,
                      help='Warmup steps')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                      help='Gradient accumulation steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Use mixed precision training')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Specific checkpoint to resume from')
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float32)
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
                console.print(f"[red]Error: Data file {args.data_path} not found![/red]")
                console.print("[yellow]Creating sample data file...[/yellow]")
                Path("data").mkdir(exist_ok=True)
                sample_text = "This is a sample training text. " * 1000
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
        console.print("[yellow]Falling back to local dataset...[/yellow]")
        Path("data").mkdir(exist_ok=True)
        sample_path = Path("data/sample.txt")
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
    
    console.print(f"[green]Dataset ready with {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples[/green]")
    
    console.print("[yellow]Initializing model...[/yellow]")
    
    model = Transformer(config)
    
    if args.device != 'cuda':
        model = model.float()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters: {total_params:,}[/cyan]")
    console.print(f"[cyan]Trainable parameters: {trainable_params:,}[/cyan]")
    
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p, gain=0.5)
    
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=1.0,
        mixed_precision=args.mixed_precision
    )
    
    if args.resume:
        trainer.load_checkpoint(args.checkpoint_path)
    
    trainer.train()

if __name__ == "__main__":
    main()
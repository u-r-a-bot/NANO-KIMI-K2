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
import gc

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
from optimizer import MuonClip, get_muon_param_groups

console = Console()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.02,
        momentum: float = 0.95,
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
        dtype: torch.dtype = torch.float32,
        qk_clip_tau: float = 100.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.to(self.device)
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.qk_clip_tau = qk_clip_tau
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        param_groups = get_muon_param_groups(
            self.model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.optimizer = MuonClip(
            param_groups,
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=True,
            weight_decay=self.weight_decay,
            qk_clip_tau=self.qk_clip_tau,
            qk_clip_enabled=True
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if (mixed_precision and device == 'cuda' and dtype == torch.float16) else None
        self.mixed_precision = mixed_precision and device == 'cuda' and dtype == torch.float16
        self.use_amp = self.mixed_precision
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_steps': [],
            'learning_rate': [],
            'grad_norm': [],
            'max_logits': [],
            'steps': []
        }
    
    def update_lr(self):
        if self.global_step < self.warmup_steps:
            lr = self.learning_rate * (self.global_step + 1) / self.warmup_steps
        else:
            progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def compute_max_logits(self, logits):
        with torch.no_grad():
            max_logit = logits.abs().max().item()
        return max_logit
    
    def save_checkpoint(self, filename: Optional[str] = None):
        if filename is None:
            filename = f'checkpoint_step_{self.global_step}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'metrics': self.metrics
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        console.print(f"[green]Checkpoint saved: {checkpoint_path}[/green]")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_step_*.pt'))
            if not checkpoints:
                return False
            checkpoint_path = checkpoints[-1]
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.metrics = checkpoint['metrics']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        console.print(f"[green]Checkpoint loaded: {checkpoint_path}[/green]")
        return True
    
    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(input_ids, start_pos=0, use_cache=False)
            loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), labels.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:
                break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def plot_metrics(self):
        if not self.metrics['steps']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.metrics['steps'], self.metrics['train_loss'])
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Train Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True)
        
        if self.metrics['val_loss'] and self.metrics['val_steps']:
            axes[0, 1].plot(self.metrics['val_steps'], self.metrics['val_loss'])
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Validation Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.metrics['steps'], self.metrics['learning_rate'])
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.metrics['steps'], self.metrics['grad_norm'])
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png')
        plt.close(fig)
        plt.close('all')
    
    def cleanup_resources(self):
        """Clean up resources after training"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        plt.close('all')
        
        del self.optimizer
        del self.scaler
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def train(self):
        self.model.train()
        
        panel = Panel.fit(
            f"[bold green]Starting Training[/bold green]\n"
            f"Max Steps: {self.max_steps:,}\n"
            f"Batch Size: {self.train_loader.batch_size}\n"
            f"Learning Rate: {self.learning_rate}\n"
            f"Momentum: {self.momentum}\n"
            f"Weight Decay: {self.weight_decay}\n"
            f"QK-Clip Tau: {self.qk_clip_tau}\n"
            f"Gradient Accumulation: {self.gradient_accumulation_steps}\n"
            f"Device: {self.device}\n"
            f"Mixed Precision: {self.mixed_precision}",
            title="Training Configuration",
            border_style="green"
        )
        console.print(panel)
        
        accumulated_loss = 0
        accumulation_counter = 0
        training_completed = False
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            train_task = progress.add_task("[cyan]Training...", total=self.max_steps)
            
            data_iter = iter(self.train_loader)
            
            while self.global_step < self.max_steps:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                    self.epoch += 1
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(input_ids, start_pos=0, use_cache=False)
                        loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), labels.reshape(-1))
                else:
                    logits = self.model(input_ids, start_pos=0, use_cache=False)
                    loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), labels.reshape(-1))
                
                loss = loss / self.gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_counter += 1
                
                if accumulation_counter >= self.gradient_accumulation_steps:
                    max_logits = self.compute_max_logits(logits)
                    
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    lr = self.update_lr()
                    
                    if self.scaler:
                        self.scaler.step(self.optimizer, max_logits=max_logits)
                        self.scaler.update()
                    else:
                        self.optimizer.step(max_logits=max_logits)
                    
                    self.optimizer.zero_grad()
                    
                    self.metrics['train_loss'].append(accumulated_loss)
                    self.metrics['learning_rate'].append(lr)
                    self.metrics['grad_norm'].append(grad_norm.item())
                    self.metrics['max_logits'].append(max_logits)
                    self.metrics['steps'].append(self.global_step)
                    
                    progress.update(train_task, advance=1)
                    
                    if self.global_step % 10 == 0:
                        progress.console.print(
                            f"[cyan]Step {self.global_step}/{self.max_steps}[/cyan] | "
                            f"Loss: {accumulated_loss:.4f} | "
                            f"LR: {lr:.6f} | "
                            f"Grad: {grad_norm:.4f} | "
                            f"MaxLogits: {max_logits:.2f}"
                        )
                    
                    accumulated_loss = 0
                    accumulation_counter = 0
                    self.global_step += 1
                    
                    if self.global_step % self.eval_interval == 0 and self.val_loader:
                        val_loss = self.evaluate()
                        self.metrics['val_loss'].append(val_loss)
                        self.metrics['val_steps'].append(self.global_step)
                        console.print(f"[yellow]Validation Loss: {val_loss:.4f}[/yellow]")
                        
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint('best_model.pt')
                    
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint()
                    
                    if self.global_step >= self.max_steps:
                        training_completed = True
                        break
            
            del data_iter
            gc.collect()
        
        final_val_loss = self.evaluate() if self.val_loader else 0.0
        
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n"
            f"Total Steps: {self.global_step:,}\n"
            f"Final Val Loss: {final_val_loss:.4f}\n"
            f"Best Loss: {self.best_loss:.4f}\n"
            f"Time: {(time.time() - self.start_time) / 3600:.1f} hours",
            title="Training Summary",
            border_style="green"
        ))
        
        self.save_checkpoint('final_model.pt')
        self.plot_metrics()
        
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.cleanup_resources()
        
        return training_completed

def main():
    parser = argparse.ArgumentParser(description='Train Nano KIMI K2 Model with MuonClip')
    parser.add_argument('--dataset', type=str, default='local', choices=['fineweb', 'streaming', 'local'])
    parser.add_argument('--data_path', type=str, default='data/train.txt')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--qk_clip_tau', type=float, default=100.0)
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
    
    console.print("[yellow]Loading dataset...[/yellow]")
    if args.dataset == 'fineweb':
        dataset = FineWebDataset(
            num_samples=args.num_samples,
            max_length=config.max_seq_length,
            tokenizer=tokenizer,
            cache_dir="./cache",
            streaming=False
        )
    elif args.dataset == 'streaming':
        dataset = FineWebStreamingDataset(
            num_samples=args.num_samples,
            max_length=config.max_seq_length,
            tokenizer=tokenizer,
            batch_size=args.batch_size
        )
    else:
        dataset = TextDataset(
            file_path=args.data_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            stride=256
        )
    
    collator = DataCollator(tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if args.dataset != 'streaming' else False,
        num_workers=0 if args.dataset == 'streaming' else 2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=True
    )
    if args.dataset != 'streaming':
        console.print(f"[green]Dataset loaded: {len(dataset)} samples[/green]")
    
    console.print("[yellow]Initializing model...[/yellow]")
    model = Transformer(config)
    
    dtype = torch.float32
    if args.mixed_precision and args.device == 'cuda':
        dtype = torch.float16
        model = model
        console.print("[green]Using float16 precision on CUDA[/green]")
    elif args.device == 'mps':
        model = model
        console.print("[green]Using float32 precision on MPS[/green]")
    else:
        model = model
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
        momentum=args.momentum,
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
        mixed_precision=args.mixed_precision,
        dtype=dtype,
        qk_clip_tau=args.qk_clip_tau
    )
    
    if args.resume:
        loaded = trainer.load_checkpoint(args.checkpoint_path)
        if not loaded:
            console.print("[yellow]No checkpoint found, starting from scratch[/yellow]")
    
    training_success = False
    exception_occurred = False
    
    try:
        training_success = trainer.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        trainer.save_checkpoint('interrupted_checkpoint.pt')
        console.print("[green]Checkpoint saved[/green]")
        exception_occurred = True
    except Exception as e:
        console.print(f"\n[red]Training error: {e}[/red]")
        trainer.save_checkpoint('error_checkpoint.pt')
        console.print("[green]Emergency checkpoint saved[/green]")
        exception_occurred = True
        raise e
    finally:
        if training_success and not exception_occurred:
            console.print("[bold green]Training completed successfully![/bold green]")
        elif not exception_occurred:
            console.print("[yellow]Training ended before completion[/yellow]")
        
        del trainer
        del model
        del dataloader
        del dataset
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        console.print("[cyan]All resources cleaned up[/cyan]")
    
    if training_success:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
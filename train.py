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
import atexit
import os
import signal

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
from dataset import TextDataset, DataCollator, FileStreamingIterableDataset,DirectStreamingDataset
from optimizer import MuonClip, get_muon_param_groups

os.environ["TOKENIZERS_PARALLELISM"] = "false"


console = Console()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        val_samples: Optional[List[Dict[str, torch.Tensor]]] = None,
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
        qk_clip_tau: float = 100.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_samples = val_samples
        self.device = torch.device(device)
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
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        param_groups = get_muon_param_groups(self.model, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = MuonClip(
            param_groups,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            qk_clip_tau=self.qk_clip_tau,
            qk_clip_enabled=True
        )
        
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision else None
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
        
        self.metrics = {
            'steps': [],
            'train_loss': [],
            'val_loss': [],
            'val_steps': [],
            'learning_rate': [],
            'grad_norm': []
        }
        
        self.data_iter = None
        self.progress = None
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        return self.learning_rate
    
    def save_checkpoint(self, name: str = 'latest'):
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'metrics': self.metrics,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None
        }, checkpoint_path)
        console.print(f"[green]Checkpoint saved: {checkpoint_path}[/green]")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict= False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        if self.scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        console.print(f"[green]Checkpoint loaded: {checkpoint_path}[/green]")
        console.print(f"[cyan]Resuming from step {self.global_step}, epoch {self.epoch}[/cyan]")
        return True
    
    def evaluate(self):
        if self.val_samples is None and self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            iterable = self.val_samples if self.val_samples is not None else self.val_loader
            for batch in iterable:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, start_pos=0, use_cache=False)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(shift_logits.reshape(-1, self.model.vocab_size), shift_labels.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else None
    
    def plot_metrics(self):
        if not self.metrics['steps']:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            axes[0, 0].plot(self.metrics['steps'], self.metrics['train_loss'], label='Train Loss', alpha=0.7)
            if self.metrics['val_loss']:
                axes[0, 0].plot(self.metrics['val_steps'], self.metrics['val_loss'], label='Val Loss', alpha=0.7)
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(self.metrics['steps'], self.metrics['learning_rate'])
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True)
            
            if len(self.metrics['train_loss']) > 50:
                window = 50
                smoothed_loss = np.convolve(self.metrics['train_loss'], np.ones(window)/window, mode='valid')
                axes[1, 0].plot(self.metrics['steps'][:len(smoothed_loss)], smoothed_loss)
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Smoothed Loss')
                axes[1, 0].set_title('Smoothed Training Loss')
                axes[1, 0].grid(True)
            
            axes[1, 1].plot(self.metrics['steps'], self.metrics['grad_norm'])
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_metrics.png', dpi=100)
            console.print(f"[green]Metrics plot saved to {self.log_dir / 'training_metrics.png'}[/green]")
        except Exception as e:
            console.print(f"[red]Error plotting metrics: {e}[/red]")
        finally:
            plt.close(fig)
            plt.close('all')
    
    def cleanup_resources(self):
        console.print("[yellow]Cleaning up trainer resources...[/yellow]")
        
        if self.data_iter is not None:
            try:
                del self.data_iter
            except:
                pass
            self.data_iter = None
        
        plt.close('all')
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            del self.optimizer
        except:
            pass
        
        if self.scaler is not None:
            try:
                del self.scaler
            except:
                pass
            self.scaler = None
        
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        console.print("[green]Trainer resources cleaned up[/green]")
    
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
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        )
        
        try:
            with self.progress:

                train_task = self.progress.add_task("[cyan]Training...", total=self.max_steps,completed=self.global_step)
                
                self.data_iter = iter(self.train_loader)
                
                while self.global_step < self.max_steps:
                    try:
                        batch = next(self.data_iter)
                    except StopIteration:
                        if self.data_iter is not None:
                            del self.data_iter
                        self.data_iter = iter(self.train_loader)
                        batch = next(self.data_iter)
                        self.epoch += 1
                    
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if self.mixed_precision:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            logits = self.model(input_ids, start_pos=0, use_cache=False)
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            loss = F.cross_entropy(shift_logits.reshape(-1, self.model.vocab_size), shift_labels.reshape(-1))
                        loss = loss / self.gradient_accumulation_steps
                        self.scaler.scale(loss).backward()
                    else:
                        logits = self.model(input_ids, start_pos=0, use_cache=False)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = F.cross_entropy(shift_logits.reshape(-1, self.model.vocab_size), shift_labels.reshape(-1))
                        loss = loss / self.gradient_accumulation_steps
                        loss.backward()
                    
                    max_logits = logits.abs().max().item()
                    accumulated_loss += loss.item()
                    accumulation_counter += 1
                    
                    if accumulation_counter >= self.gradient_accumulation_steps:
                        if self.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        if self.mixed_precision:
                            self.scaler.step(self.optimizer, max_logits=max_logits)
                            self.scaler.update()
                        else:
                            self.optimizer.step(max_logits=max_logits)
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        lr = self.get_lr(self.global_step)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                        
                        self.metrics['steps'].append(self.global_step)
                        self.metrics['train_loss'].append(accumulated_loss)
                        self.metrics['learning_rate'].append(lr)
                        self.metrics['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                        
                        self.progress.update(
                            train_task,
                            advance=1,
                            description=f"[cyan]Step {self.global_step}/{self.max_steps} | Loss: {accumulated_loss:.4f} | LR: {lr:.6f}"
                        )
                        
                        accumulated_loss = 0
                        accumulation_counter = 0
                        self.global_step += 1
                        
                        if self.global_step % self.eval_interval == 0:
                            val_loss = self.evaluate()
                            if val_loss is not None:
                                self.metrics['val_loss'].append(val_loss)
                                self.metrics['val_steps'].append(self.global_step)
                                console.print(f"[blue]Step {self.global_step} | Val Loss: {val_loss:.4f}[/blue]")
                                
                                if val_loss < self.best_loss:
                                    self.best_loss = val_loss
                                    self.save_checkpoint('best')
                        
                        if self.global_step % self.save_interval == 0:
                            self.save_checkpoint('latest')
                        
                        if self.global_step >= self.max_steps:
                            training_completed = True
                            break
        finally:
            if self.data_iter is not None:
                del self.data_iter
                self.data_iter = None
        
        final_val_loss = self.evaluate()
        
        console.print(Panel.fit(
            f"[bold green]Training Complete[/bold green]\n"
            f"Total Steps: {self.global_step:,}\n"
            f"Final Val Loss: {final_val_loss if final_val_loss is not None else 'N/A'}\n"
            f"Best Loss: {self.best_loss:.4f}\n"
            f"Time: {(time.time() - self.start_time) / 3600:.1f} hours",
            title="Training Summary",
            border_style="green"
        ))
        
        self.save_checkpoint('final')
        self.plot_metrics()
        
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.cleanup_resources()
        
        return training_completed

def cleanup_handler(signum=None, frame=None):
    console.print("\n[yellow]Cleaning up before exit...[/yellow]")
    plt.close('all')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    console.print("[green]Cleanup complete[/green]")


def main():
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_handler)
    
    parser = argparse.ArgumentParser(description='Train Nano KIMI K2 Model with MuonClip')
    parser.add_argument('--dataset', type=str, default='local', choices=['fineweb', 'streaming', 'local','localstreaming','hfstreaming'])
    parser.add_argument('--hf_repo_name',type=str,default=None)
    parser.add_argument('--data_path', type=str, default='data/train.txt')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--qk_clip_tau', type=float, default=100.0)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--val_from_train', action='store_true', default=False, help='Cache a few training batches on CPU and reuse as validation')
    parser.add_argument('--val_batches', type=int, default=8, help='Number of training batches to cache for validation')
    
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        console.print(f"[red]Error loading tokenizer: {e}[/red]")
        sys.exit(1)
    
    config.vocab_size = len(tokenizer)
    
    console.print("[yellow]Loading dataset...[/yellow]")
    dataset = None
    try:
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
        elif args.dataset == 'hfstreaming':
            if args.hf_repo_name == None:
                raise ValueError("Dataset name not provided use \'--hf_repo_name\' argument")
            dataset = DirectStreamingDataset(
                dataset_name = args.hf_repo_name,
                tokenizer=tokenizer,
                max_length=1024,
                stride=512,
                hf_shuffle=True,
                seed=42
            )
        elif args.dataset == 'localstreaming':
            dataset = FileStreamingIterableDataset(
                file_path= args.data_path,
                tokenizer=tokenizer,
                max_length=512,
                stride=256,
                shuffle_buffer_size=1000,
                hf_shuffle=True,
                seed=42  
            )
        else:
            dataset = TextDataset(
                file_path=args.data_path,
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
                stride=256
            )
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        sys.exit(1)
    
    collator = DataCollator(tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if 'streaming' not in args.dataset else False,
        num_workers=0 if 'streaming' not in args.dataset else 2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=True,
        persistent_workers=False
    )
    
    if hasattr(dataset, '__len__'):
        console.print(f"[green]Dataset loaded: {len(dataset)} samples[/green]")
    else:
        console.print(f"[green]Streaming dataset loaded (num_samples: {args.num_samples if args.num_samples else 'unlimited'})[/green]")
    
    console.print("[yellow]Initializing model...[/yellow]")
    model = Transformer(config)
    
    if args.mixed_precision:
        console.print("[green]Mixed precision enabled - using autocast[/green]")
    else:
        console.print("[green]Using full precision (float32)[/green]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters: {total_params:,}[/cyan]")
    console.print(f"[cyan]Trainable parameters: {trainable_params:,}[/cyan]")
    console.print(f"[cyan]Model size: {total_params * 4 / 1024**3:.2f} GB (float32)[/cyan]")
    
    # Optionally cache a few batches from the same training DataLoader for validation
    cached_val_samples = None
    if args.val_from_train and args.eval_interval > 0:
        console.print(f"[yellow]Caching {args.val_batches} training batches for validation (CPU)...[/yellow]")
        cached_val_samples = []
        dl_iter = iter(dataloader)
        for _ in range(max(0, args.val_batches)):
            try:
                batch = next(dl_iter)
            except StopIteration:
                break
            cached_val_samples.append({
                'input_ids': batch['input_ids'].cpu(),
                'labels': batch['labels'].cpu(),
            })
        del dl_iter
        console.print(f"[green]Cached {len(cached_val_samples)} validation batches from training stream[/green]")

    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        val_samples=cached_val_samples,
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
        trainer.save_checkpoint('latest')
        console.print("[green]Checkpoint saved[/green]")
        exception_occurred = True
    except Exception as e:
        console.print(f"\n[red]Training error: {e}[/red]")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint('latest')
        console.print("[green]Emergency checkpoint saved[/green]")
        exception_occurred = True
    finally:
        if training_success and not exception_occurred:
            console.print("[bold green]Training completed successfully![/bold green]")
        elif not exception_occurred:
            console.print("[yellow]Training ended before completion[/yellow]")
        
        console.print("[yellow]Final cleanup...[/yellow]")
        
        try:
            if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
                del dataloader._iterator
        except:
            pass
        
        try:
            trainer.cleanup_resources()
        except:
            pass
        
        try:
            del trainer
        except:
            pass
        
        try:
            del model
        except:
            pass
        
        try:
            del dataloader
        except:
            pass
        
        try:
            del dataset
        except:
            pass
        
        try:
            del tokenizer
        except:
            pass
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        plt.close('all')
        
        console.print("[cyan]All resources cleaned up[/cyan]")
    
    exit_code = 0 if training_success else 1
    console.print(f"[cyan]Exiting with code {exit_code}[/cyan]")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

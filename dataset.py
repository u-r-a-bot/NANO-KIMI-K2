import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import random
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(
        self,
        file_path: Union[str, Path, List[str]],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
        min_length: int = 32,
        combine_short: bool = True,
        cache_tokens: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        self.combine_short = combine_short
        
        if isinstance(file_path, (str, Path)):
            file_paths = [Path(file_path)]
        else:
            file_paths = [Path(p) for p in file_path]
        
        self.samples = []
        
        for path in file_paths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            try:
                if cache_tokens and Path(f"{path}.tokens.npy").exists():
                    print(f"Loading cached tokens from {path}.tokens.npy")
                    try:
                        tokens = np.load(f"{path}.tokens.npy")
                        
                        if len(tokens) == 0:
                            print(f"Warning: Empty token cache for {path}, re-tokenizing...")
                            raise ValueError("Empty cache")
                        
                        self._create_samples(tokens.tolist())
                    except (ValueError, IOError) as e:
                        print(f"Warning: Failed to load cache ({e}), re-tokenizing...")
                        text = path.read_text(encoding='utf-8', errors='ignore')
                        tokens = self.tokenizer.encode(text)
                        
                        if cache_tokens and len(tokens) > 0:
                            np.save(f"{path}.tokens.npy", np.array(tokens))
                        
                        self._create_samples(tokens)
                else:
                    print(f"Processing {path}")
                    text = path.read_text(encoding='utf-8', errors='ignore')
                    
                    if not text.strip():
                        print(f"Warning: File {path} is empty, skipping...")
                        continue
                    
                    tokens = self.tokenizer.encode(text)
                    
                    if cache_tokens and len(tokens) > 0:
                        try:
                            np.save(f"{path}.tokens.npy", np.array(tokens))
                        except Exception as e:
                            print(f"Warning: Failed to cache tokens: {e}")
                    
                    self._create_samples(tokens)
            
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                raise
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples created from input files")
        
        print(f"Created {len(self.samples)} samples")
    
    def _create_samples(self, tokens: List[int]):
        if len(tokens) < self.min_length:
            return
        
        s = 0
        while s < len(tokens):
            end = min(s + self.max_length, len(tokens))
            chunk = tokens[s:end]
            
            if len(chunk) >= self.min_length:
                input_ids = chunk[:-1] if len(chunk) > 1 else chunk
                labels = chunk[1:] if len(chunk) > 1 else chunk
                
                if len(input_ids) < self.max_length:
                    pad_len = self.max_length - len(input_ids)
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                    attention_mask = [1] * len(chunk[:-1]) + [0] * pad_len
                else:
                    attention_mask = [1] * len(input_ids)
                
                self.samples.append({
                    'input_ids': input_ids[:self.max_length],
                    'labels': labels[:self.max_length],
                    'attention_mask': attention_mask[:self.max_length]
                })
            
            s += self.stride
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long)
        }

class DataCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        if not batch:
            raise ValueError("Empty batch")
        
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

class StreamingTextDataset(Dataset):
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 1024,
        chunk_size: int = 1024 * 1024,
    ):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        
        self.file_size = self.file_path.stat().st_size
        
        if self.file_size == 0:
            raise ValueError(f"File {self.file_path} is empty")
        
        self.n_chunks = (self.file_size // chunk_size) + 1
    
    def __len__(self):
        return self.n_chunks
    
    def __getitem__(self, idx):
        if idx >= self.n_chunks:
            raise IndexError(f"Index {idx} out of range for {self.n_chunks} chunks")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(idx * self.chunk_size)
                text = f.read(self.chunk_size)
                
                if not text:
                    raise ValueError(f"Empty chunk at index {idx}")
                
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length'
                )
                
                input_ids = tokens[:-1]
                labels = tokens[1:]
                
                return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long)
                }
        except Exception as e:
            print(f"Error reading chunk {idx}: {e}")
            raise

if __name__ == "__main__":
    from config import Config
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = Config()
    config.vocab_size = len(tokenizer)
    
    dataset = TextDataset(
        file_path="data/train.txt",
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        stride=512,
        min_length=32,
        combine_short=True,
        cache_tokens=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.max_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollator(tokenizer.pad_token_id),
        drop_last=True
    )
    
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    sample_input = batch['input_ids'][0]
    sample_text = tokenizer.decode(sample_input[:50])
    print(f"\nSample text: {sample_text}")
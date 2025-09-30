import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Optional, Dict, List
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from pathlib import Path
import sys
class FineWebDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        num_samples: int = 1_000_000,
        cache_dir: str = "./fineweb_cache",
        streaming: bool = True,
        min_length: int = 100,
        language: str = "en",
        subset: str = "sample-10BT",  # or "sample-100BT" for larger
        text_column: str = "text",
        rebuild_cache: bool = False
    ):
        """
        FineWeb dataset for training Nano KIMI K2
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            num_samples: Number of samples to use (1M default)
            cache_dir: Directory to cache processed data
            streaming: Use streaming mode for memory efficiency
            min_length: Minimum text length to keep
            language: Language filter (en for English)
            subset: FineWeb subset to use
            text_column: Column name containing text
            rebuild_cache: Force rebuild cache
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.min_length = min_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache file paths
        self.cache_file = self.cache_dir / f"fineweb_{num_samples}_{max_length}.pkl"
        self.tokens_cache = self.cache_dir / f"fineweb_{num_samples}_tokens.npy"
        
        # Special tokens
        self.bos_token = tokenizer.bos_token_id
        self.eos_token = tokenizer.eos_token_id
        self.pad_token = tokenizer.pad_token_id or 0
        
        if not rebuild_cache and self.cache_file.exists():
            print(f"Loading cached dataset from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print(f"Loading FineWeb dataset (first {num_samples:,} samples)...")
            self.samples = self._prepare_dataset(streaming, subset, text_column)
            
            # Save cache
            print(f"Saving cache to {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
        
        print(f"Dataset ready with {len(self.samples):,} samples")
    
    def _prepare_dataset(self, streaming: bool, subset: str, text_column: str) -> List[Dict]:
        """Load and prepare FineWeb dataset"""
        samples = []
        
        # Load FineWeb dataset
        if streaming:
            # Streaming mode - memory efficient
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name=subset,
                split="train",
                streaming=True,
            )
            
            # Process samples
            pbar = tqdm(total=self.num_samples, desc="Processing FineWeb")
            sample_count = 0
            
            for item in dataset:
                if sample_count >= self.num_samples:
                    break
                
                text = item.get(text_column, "")
                
                # Skip short texts
                if len(text) < self.min_length:
                    continue
                
                # Tokenize text
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.max_length - 2,  # Reserve space for special tokens
                    truncation=True
                )
                
                # Skip if too short after tokenization
                if len(tokens) < self.min_length // 4:  # Rough token estimate
                    continue
                
                # Create sample
                samples.extend(self._create_sequences(tokens))
                
                sample_count += 1
                pbar.update(1)
            
            pbar.close()
        
        else:
            # Non-streaming mode - loads into memory
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name=subset,
                split=f"train[:{self.num_samples}]",
            )
            
            print(f"Tokenizing {len(dataset):,} texts...")
            for item in tqdm(dataset):
                text = item.get(text_column, "")
                
                if len(text) < self.min_length:
                    continue
                
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.max_length * 10,  # Get more tokens for chunking
                    truncation=True
                )
                
                samples.extend(self._create_sequences(tokens))
        
        return samples
    
    def _create_sequences(self, tokens: List[int]) -> List[Dict]:
        """Create training sequences from token list"""
        sequences = []
        
        # Sliding window approach
        stride = self.max_length // 2  # 50% overlap
        
        for i in range(0, len(tokens) - self.min_length, stride):
            chunk = tokens[i:i + self.max_length - 2]
            
            # Add special tokens
            if self.bos_token is not None:
                chunk = [self.bos_token] + chunk
            if self.eos_token is not None:
                chunk = chunk + [self.eos_token]
            
            # Pad if necessary
            if len(chunk) < self.max_length:
                padding_length = self.max_length - len(chunk)
                attention_mask = [1] * len(chunk) + [0] * padding_length
                chunk = chunk + [self.pad_token] * padding_length
            else:
                chunk = chunk[:self.max_length]
                attention_mask = [1] * self.max_length
            
            sequences.append({
                'input_ids': chunk[:-1],
                'labels': chunk[1:],
                'attention_mask': attention_mask[:-1]
            })
            
            # Break if we only want one sequence per text
            if len(tokens) < self.max_length * 2:
                break
        
        return sequences
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long)
        }


class FineWebStreamingDataset(torch.utils.data.IterableDataset):
    """Streaming version for continuous training without storing all samples"""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        num_samples: Optional[int] = 1_000_000,
        buffer_size: int = 10000,
        subset: str = "sample-10BT"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.buffer_size = buffer_size
        self.subset = subset
        
        # Special tokens
        self.bos_token = tokenizer.bos_token_id
        self.eos_token = tokenizer.eos_token_id
        self.pad_token = tokenizer.pad_token_id or 0
    
    def __iter__(self):
        # Load streaming dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=self.subset,
            split="train",
            streaming=True,
        )
        
        samples_yielded = 0
        token_buffer = []
        
        for item in dataset:
            if self.num_samples and samples_yielded >= self.num_samples:
                break
            
            # Get text
            text = item.get("text", "")
            if len(text) < 100:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text, truncation=False)
            token_buffer.extend(tokens)
            
            # Create samples from buffer
            while len(token_buffer) >= self.max_length:
                chunk = token_buffer[:self.max_length]
                token_buffer = token_buffer[self.max_length // 2:]  # 50% overlap
                
                # Prepare input/label
                input_ids = chunk[:-1]
                labels = chunk[1:]
                
                # Pad if needed
                if len(input_ids) < self.max_length - 1:
                    pad_len = (self.max_length - 1) - len(input_ids)
                    attention_mask = torch.ones(len(input_ids), dtype=torch.long)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_len, dtype=torch.long)
                    ])
                    input_ids = input_ids + [self.pad_token] * pad_len
                    labels = labels + [self.pad_token] * pad_len
                else:
                    attention_mask = torch.ones(self.max_length - 1, dtype=torch.long)
                
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'attention_mask': attention_mask
                }
                
                samples_yielded += 1
                
                if self.num_samples and samples_yielded >= self.num_samples:
                    break


# Usage example
if __name__ == "__main__":
    from config import Config
    config = Config()
    # Initialize tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = Config()
    config.vocab_size = len(tokenizer)
    
    # Option 1: Regular dataset (caches everything)
    # print("Loading FineWeb dataset...")
    # dataset = FineWebDataset(
    #     tokenizer=tokenizer,
    #     max_length=config.max_seq_length,
    #     num_samples=1_000_000,  # First 1M samples
    #     cache_dir="./fineweb_cache",
    #     streaming=True,  # Memory efficient loading
    #     min_length=100,
    #     rebuild_cache=False  # Set True to rebuild cache
    # )
    
    # # Create dataloader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=config.max_batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=torch.cuda.is_available(),
    #     drop_last=True
    # )
    
    # print(f"\nDataset stats:")
    # print(f"  Total samples: {len(dataset):,}")
    # print(f"  Batches: {len(dataloader):,}")
    # print(f"  Tokens per epoch: ~{len(dataset) * config.max_seq_length:,}")
    
    # Option 2: Streaming dataset (no storage, continuous)
    streaming_dataset = FineWebStreamingDataset(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        num_samples=1_000_000
    )
    
    streaming_loader = DataLoader(
        streaming_dataset,
        batch_size=config.max_batch_size,
        num_workers=0,  # Must be 0 for IterableDataset
        pin_memory=torch.cuda.is_available()
    )
    
    # Test batch
    batch = next(iter(streaming_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    # Decode sample
    sample_text = tokenizer.decode(batch['input_ids'][0][:50])
    print(f"\nSample text (first 50 tokens):")
    print(sample_text)
    sys.exit(0)
    # Training loop example
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Transformer(config).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # print("\nStarting training on FineWeb...")
    # for epoch in range(1):
    #     total_loss = 0
    #     progress = tqdm(dataloader, desc=f"Epoch {epoch}")
        
    #     for batch_idx, batch in enumerate(progress):
    #         input_ids = batch['input_ids'].to(device)
    #         labels = batch['labels'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
            
    #         # Reset cache every N batches to prevent memory issues
    #         if batch_idx % 100 == 0:
    #             for layer in model.layers:
    #                 layer.attn.k_cache.zero_()
    #                 layer.attn.v_cache.zero_()
            
    #         # Forward
    #         logits = model(input_ids, start_pos=0)
            
    #         # Loss (ignore padding)
    #         loss = F.cross_entropy(
    #             logits.view(-1, config.vocab_size),
    #             labels.view(-1),
    #             ignore_index=tokenizer.pad_token_id
    #         )
            
    #         # Backward
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    #         optimizer.zero_grad()
            
    #         total_loss += loss.item()
    #         progress.set_postfix({'loss': loss.item()})
            
    #         if batch_idx >= 100:  # Quick test
    #             break
        
    #     print(f"Epoch {epoch} - Avg Loss: {total_loss / (batch_idx + 1):.4f}")
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
        """
        Args:
            file_path: Path to txt file(s)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (matches config.max_seq_length)
            stride: Sliding window stride for creating chunks
            min_length: Minimum sequence length to keep
            combine_short: Combine short sequences to reduce padding
            cache_tokens: Cache tokenized data for faster loading
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        self.combine_short = combine_short
        
        # Handle single or multiple files
        if isinstance(file_path, (str, Path)):
            file_paths = [Path(file_path)]
        else:
            file_paths = [Path(p) for p in file_path]
        
        # Load and tokenize all texts
        self.samples = []
        
        for path in file_paths:
            if cache_tokens and Path(f"{path}.tokens.npy").exists():
                # Load cached tokens
                print(f"Loading cached tokens from {path}.tokens.npy")
                tokens = np.load(f"{path}.tokens.npy")
                self._create_samples(tokens.tolist())
            else:
                # Read and tokenize text
                print(f"Processing {path}")
                text = path.read_text(encoding='utf-8', errors='ignore')
                
                # Tokenize entire text
                tokens = self.tokenizer.encode(text)
                
                # Cache tokens for next time
                if cache_tokens:
                    np.save(f"{path}.tokens.npy", np.array(tokens))
                
                self._create_samples(tokens)
        
        print(f"Created {len(self.samples)} training samples")
    
    def _create_samples(self, tokens: List[int]):
        """Create overlapping samples from token list"""
        # Special tokens
        bos_token = self.tokenizer.bos_token_id
        eos_token = self.tokenizer.eos_token_id
        pad_token = self.tokenizer.pad_token_id or 0
        
        # Sliding window to create samples
        pos = 0
        short_buffer = []
        
        while pos < len(tokens):
            # Get chunk
            chunk = tokens[pos:pos + self.max_length - 2]  # -2 for BOS/EOS
            
            # Handle short sequences
            if len(chunk) < self.min_length:
                if self.combine_short:
                    short_buffer.extend(chunk)
                    if len(short_buffer) >= self.min_length:
                        chunk = short_buffer[:self.max_length - 2]
                        short_buffer = []
                    else:
                        pos += self.stride
                        continue
                else:
                    pos += self.stride
                    continue
            
            # Add special tokens
            if bos_token is not None:
                chunk = [bos_token] + chunk
            if eos_token is not None:
                chunk = chunk + [eos_token]
            
            # Pad if needed
            if len(chunk) < self.max_length:
                padding_length = self.max_length - len(chunk)
                chunk = chunk + [pad_token] * padding_length
                attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
            else:
                chunk = chunk[:self.max_length]
                attention_mask = [1] * self.max_length
            
            self.samples.append({
                'input_ids': chunk[:-1],  # Everything except last token
                'labels': chunk[1:],       # Everything except first token  
                'attention_mask': attention_mask[:-1]
            })
            
            # Move position
            pos += self.stride
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long)
        }

class DataCollator:
    """Custom collator for batching"""
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

# Streaming dataset for very large files
class StreamingTextDataset(Dataset):
    """Memory-efficient dataset that reads chunks on-the-fly"""
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 1024,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        
        # Calculate file chunks
        self.file_size = self.file_path.stat().st_size
        self.n_chunks = (self.file_size // chunk_size) + 1
        
    def __len__(self):
        return self.n_chunks
    
    def __getitem__(self, idx):
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(idx * self.chunk_size)
            text = f.read(self.chunk_size)
            
            # Tokenize
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

# Example usage
if __name__ == "__main__":
    from config import Config
    
    # Initialize tokenizer (using ungated options)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Set special tokens if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = Config()
    config.vocab_size = len(tokenizer)
    
    # Create dataset
    dataset = TextDataset(
        file_path="data/train.txt",  # Your text file
        # file_path=["data/file1.txt", "data/file2.txt"],  # Multiple files
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        stride=512,  # Overlap between sequences
        min_length=32,
        combine_short=True,
        cache_tokens=True  # Cache for faster reloading
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.max_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollator(tokenizer.pad_token_id),
        drop_last=True  # Drop incomplete batches
    )
    
    # Test loading
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a sample batch
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    # Decode sample
    sample_input = batch['input_ids'][0]
    sample_text = tokenizer.decode(sample_input[:50])  # First 50 tokens
    print(f"\nSample text: {sample_text}")
    
    # For streaming very large files
    # stream_dataset = StreamingTextDataset(
    #     file_path="data/huge_file.txt",
    #     tokenizer=tokenizer,
    #     max_length=config.max_seq_length
    # )
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import random
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import mmap
import io

class OptimizedTextDataset(Dataset):
    """Memory-efficient dataset with lazy loading and mmap support"""
    
    def __init__(
        self,
        file_path: Union[str, Path, List[str]],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
        min_length: int = 32,
        use_mmap: bool = True,
        num_workers: int = 4
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        self.use_mmap = use_mmap
        
        if isinstance(file_path, (str, Path)):
            file_paths = [Path(file_path)]
        else:
            file_paths = [Path(p) for p in file_path]
        
        # Store token file paths instead of loading all tokens
        self.token_files = []
        self.sample_offsets = []  # (file_idx, start, length)
        
        # Parallel tokenization and caching
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_file, path, i) 
                      for i, path in enumerate(file_paths)]
            
            for future in futures:
                file_idx, token_path, offsets = future.result()
                if token_path and offsets:
                    self.token_files.append(token_path)
                    self.sample_offsets.extend(offsets)
        
        if len(self.sample_offsets) == 0:
            raise ValueError("No valid samples created from input files")
        
        print(f"Created {len(self.sample_offsets)} samples from {len(self.token_files)} files")
    
    def _process_file(self, path: Path, file_idx: int):
        """Process single file and return offsets"""
        if not path.exists():
            print(f"File not found: {path}")
            return file_idx, None, []
        
        token_cache = Path(f"{path}.tokens.npy")
        
        # Load or create tokens
        if token_cache.exists():
            try:
                tokens = np.load(token_cache, mmap_mode='r' if self.use_mmap else None)
                if len(tokens) == 0:
                    raise ValueError("Empty cache")
            except Exception as e:
                print(f"Re-tokenizing {path}: {e}")
                tokens = self._tokenize_and_cache(path, token_cache)
        else:
            tokens = self._tokenize_and_cache(path, token_cache)
        
        # Calculate sample offsets without loading all data
        offsets = []
        tokens_len = len(tokens)
        
        if tokens_len < self.min_length:
            return file_idx, None, []
        
        s = 0
        while s < tokens_len:
            end = min(s + self.max_length, tokens_len)
            chunk_len = end - s
            
            if chunk_len >= self.min_length:
                offsets.append((file_idx, s, chunk_len))
            
            s += self.stride
        
        return file_idx, token_cache, offsets
    
    def _tokenize_and_cache(self, path: Path, cache_path: Path):
        """Tokenize file and cache tokens"""
        text = path.read_text(encoding='utf-8', errors='ignore')
        
        if not text.strip():
            return np.array([], dtype=np.int32)
        
        # Tokenize in chunks for large files
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens_array = np.array(tokens, dtype=np.int32)
        
        np.save(cache_path, tokens_array)
        return tokens_array
    
    def __len__(self):
        return len(self.sample_offsets)
    
    def __getitem__(self, idx):
        """Lazy load tokens only when needed"""
        file_idx, start, length = self.sample_offsets[idx]
        
        # Load tokens with mmap
        tokens = np.load(
            self.token_files[file_idx],
            mmap_mode='r' if self.use_mmap else None
        )
        
        chunk = tokens[start:start + length]
        
        # Create input/label pairs
        if len(chunk) > 1:
            input_ids = chunk[:-1]
            labels = chunk[1:]
        else:
            input_ids = chunk
            labels = chunk
        
        return {
            'input_ids': torch.from_numpy(input_ids.copy()).long(),
            'labels': torch.from_numpy(labels.copy()).long(),
        }


class DynamicBatchCollator:
    """Efficient collator with dynamic padding"""
    
    def __init__(self, pad_token_id: int = 0, max_length: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch):
        if not batch:
            raise ValueError("Empty batch")
        
        # Find max length in batch (dynamic padding)
        max_len = max(len(item['input_ids']) for item in batch)
        
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        input_ids = []
        labels = []
        attention_mask = []
        
        for item in batch:
            inp = item['input_ids'][:max_len]
            lab = item['labels'][:max_len]
            
            # Pad to max_len
            pad_len = max_len - len(inp)
            
            if pad_len > 0:
                inp = torch.cat([inp, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
                mask = torch.cat([torch.ones(len(item['input_ids']), dtype=torch.long),
                                 torch.zeros(pad_len, dtype=torch.long)])
            else:
                mask = torch.ones(max_len, dtype=torch.long)
            
            input_ids.append(inp)
            labels.append(lab)
            attention_mask.append(mask)
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_mask)
        }


class StreamingIterableDataset(IterableDataset):
    """True streaming dataset for very large files"""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 1024,
        buffer_size: int = 10000,
        shuffle_buffer: int = 1000,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = []
            
            for line in f:
                if not line.strip():
                    continue
                
                # Multi-worker support
                if worker_info is not None:
                    if random.randint(0, worker_info.num_workers - 1) != worker_info.id:
                        continue
                
                tokens = self.tokenizer.encode(line.strip(), add_special_tokens=False)
                buffer.extend(tokens)
                
                # Yield samples when buffer is full
                while len(buffer) >= self.max_length:
                    chunk = buffer[:self.max_length]
                    buffer = buffer[self.max_length // 2:]  # Overlap
                    
                    if len(chunk) > 1:
                        yield {
                            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                            'labels': torch.tensor(chunk[1:], dtype=torch.long),
                        }


class PreloadedDataset(Dataset):
    """Pre-load entire dataset in RAM for maximum speed (small datasets only)"""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
    ):
        print("Loading entire dataset into memory...")
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize everything at once
        path = Path(file_path)
        text = path.read_text(encoding='utf-8', errors='ignore')
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Pre-create all samples
        self.samples = []
        s = 0
        while s < len(tokens):
            end = min(s + max_length, len(tokens))
            chunk = tokens[s:end]
            
            if len(chunk) > 1:
                self.samples.append({
                    'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                    'labels': torch.tensor(chunk[1:], dtype=torch.long),
                })
            
            s += stride
        
        print(f"Loaded {len(self.samples)} samples into memory")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_optimized_dataloader(
    file_path: Union[str, Path],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4,
    use_streaming: bool = False,
    preload_small: bool = False,
    **kwargs
):
    """Factory function to create optimized dataloader"""
    
    # Choose dataset type based on requirements
    if preload_small:
        dataset = PreloadedDataset(file_path, tokenizer, max_length)
        shuffle = True
    elif use_streaming:
        dataset = StreamingIterableDataset(file_path, tokenizer, max_length)
        shuffle = False
    else:
        dataset = OptimizedTextDataset(
            file_path, tokenizer, max_length, 
            num_workers=num_workers, **kwargs
        )
        shuffle = True
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not use_streaming else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DynamicBatchCollator(
            tokenizer.pad_token_id,
            max_length=max_length
        ),
        drop_last=True,
        persistent_workers=num_workers > 0,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For small datasets (fits in RAM)
    # print("\n=== Preloaded Dataset (fastest for small data) ===")
    # dataloader_preload = create_optimized_dataloader(
    #     "data/train.txt",
    #     tokenizer,
    #     batch_size=8,
    #     max_length=1024,
    #     num_workers=4,
    #     preload_small=True
    # )
    
    # For large datasets (memory-efficient)
    # print("\n=== Memory-Mapped Dataset (best for large data) ===")
    # dataloader_mmap = create_optimized_dataloader(
    #     "data/train.txt",
    #     tokenizer,
    #     batch_size=8,
    #     max_length=1024,
    #     num_workers=4,
    #     use_mmap=True,
    #     stride=512
    # )
    
    # For very large datasets (streaming)
    print("\n=== Streaming Dataset (for huge files) ===")
    dataloader_stream = create_optimized_dataloader(
        "data/train.txt",
        tokenizer,
        batch_size=8,
        max_length=1024,
        num_workers=2,
        use_streaming=True
    )
    
    # Test one batch
    print("\n=== Testing batch ===")
    batch = next(iter(dataloader_stream))
    print(f"input_ids: {batch['input_ids'].shape}")
    print(f"labels: {batch['labels'].shape}")
    print(f"attention_mask: {batch['attention_mask'].shape}")
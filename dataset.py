import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import random
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import os
from datasets import load_dataset, Dataset as HFDataset, get_dataset_config_names
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    """Memory-efficient dataset with lazy loading and mmap support"""
    
    def __init__(
        self,
        file_path: Union[str, Path, List[str]],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
        min_length: int = 32,
        use_mmap: bool = True,
        num_workers: int = 4,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        self.use_mmap = use_mmap
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if isinstance(file_path, (str, Path)):
            file_paths = [Path(file_path)]
        else:
            file_paths = [Path(p) for p in file_path]
        
        # Validate files exist
        file_paths = [p for p in file_paths if p.exists()]
        if not file_paths:
            raise ValueError(f"No valid files found in: {file_path}")
        
        self.token_files = []
        self.sample_offsets = []  # (file_idx, start, length)
        
        # Parallel tokenization and caching
        print(f"Processing {len(file_paths)} file(s)...")
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
            warnings.warn(f"File not found: {path}")
            return file_idx, None, []
        
        # Use cache_dir if provided
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            token_cache = self.cache_dir / f"{path.stem}_{hash(str(path))}.tokens.npy"
        else:
            token_cache = Path(f"{path}.tokens.npy")
        
        # Load or create tokens
        if token_cache.exists():
            try:
                tokens = np.load(token_cache, mmap_mode='r' if self.use_mmap else None)
                if len(tokens) == 0:
                    raise ValueError("Empty cache")
            except Exception as e:
                warnings.warn(f"Re-tokenizing {path.name}: {e}")
                tokens = self._tokenize_and_cache(path, token_cache)
        else:
            tokens = self._tokenize_and_cache(path, token_cache)
        
        # Calculate sample offsets
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
            
            if end == tokens_len:
                break
            s += self.stride
        
        return file_idx, token_cache, offsets
    
    def _tokenize_and_cache(self, path: Path, cache_path: Path):
        """Tokenize file and cache tokens"""
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            warnings.warn(f"Error reading {path}: {e}")
            return np.array([], dtype=np.int32)
        
        if not text.strip():
            return np.array([], dtype=np.int32)
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens_array = np.array(tokens, dtype=np.int32)
        
        # Save cache
        try:
            np.save(cache_path, tokens_array)
        except Exception as e:
            warnings.warn(f"Could not cache tokens to {cache_path}: {e}")
        
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
        
        # Create input/label pairs (next token prediction)
        if len(chunk) > 1:
            input_ids = chunk[:-1]
            labels = chunk[1:]
        else:
            input_ids = chunk
            labels = chunk
        
        return {
            'input_ids': torch.from_numpy(np.array(input_ids)).long(),
            'labels': torch.from_numpy(np.array(labels)).long(),
        }


class DirectStreamingDataset(IterableDataset):
    """
    Streams directly from a Hugging Face dataset, tokenizes on the fly,
    and creates samples without saving to an intermediate file.
    Automatically handles dataset configs.
    """
    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        split: str = "train",
        tokenizer = None,
        max_length: int = 1024,
        shuffle_buffer_size: int = 10000,
        text_column: str = 'text',
        stride: Optional[int] = None,
        hf_shuffle: bool = False,
        seed: int = 42,
        trust_remote_code: bool = False,
        auto_detect_config: bool = True,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.text_column = text_column
        self.stride = stride if stride is not None else max_length
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        
        # Auto-detect config if needed
        if dataset_config_name is None and auto_detect_config:
            try:
                available_configs = get_dataset_config_names(dataset_name)
                if available_configs:
                    # Use first config or a smart default
                    self.dataset_config_name = self._select_best_config(available_configs)
                    print(f"Auto-detected dataset config: '{self.dataset_config_name}'")
                    print(f"Available configs: {available_configs}")
            except Exception as e:
                # If config detection fails, try without config
                warnings.warn(f"Could not detect configs for {dataset_name}: {e}")
        
        # Load the dataset
        print(f"Loading dataset: {dataset_name}" + 
              (f" (config: {self.dataset_config_name})" if self.dataset_config_name else "") + 
              f" (split: {split})")
        
        try:
            self.dataset = load_dataset(
                dataset_name,
                name=self.dataset_config_name,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code
            )
            
            if hf_shuffle:
                self.dataset = self.dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
            
            # Auto-detect text column if needed
            if self.text_column not in self._get_column_names():
                detected_column = self._detect_text_column()
                if detected_column:
                    print(f"Text column '{self.text_column}' not found. Using '{detected_column}' instead.")
                    self.text_column = detected_column
                
            print(f"Dataset loaded successfully. Text column: '{self.text_column}'")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")
    
    def _select_best_config(self, configs: List[str]) -> str:
        """Select the best config from available configs"""
        # Priority order for common patterns
        priority_patterns = [
            'default',
            'train',
            'all',
            '100k',  # cosmopedia-100k
            'en',
            '4plus',  # finemath-4plus (highest quality)
            '3plus',
            'auto_math_text' # hugginfaceTB Cosmopedia ->math
        ]
        
        # Try to find a config matching priority patterns
        for pattern in priority_patterns:
            for config in configs:
                if pattern in config.lower():
                    return config
        
        # If no pattern matches, return first config
        return configs[0]
    
    def _get_column_names(self) -> List[str]:
        """Get column names from the dataset"""
        try:
            # Get first example to inspect columns
            first_example = next(iter(self.dataset.take(1)))
            return list(first_example.keys())
        except:
            return []
    
    def _detect_text_column(self) -> Optional[str]:
        """Auto-detect the text column name"""
        common_text_columns = ['text', 'content',  'document', 'article']
        
        try:
            first_example = next(iter(self.dataset.take(1)))
            columns = list(first_example.keys())
            
            # Try common column names first
            for col in common_text_columns:
                if col in columns:
                    return col
            
            # Find columns with string content
            for col in columns:
                value = first_example[col]
                if isinstance(value, str) and len(value) > 50:
                    return col
            
            # Default to first string column
            for col in columns:
                if isinstance(first_example[col], str):
                    return col
        except:
            pass
        
        return None

    def __iter__(self):
        """Iterator that yields tokenized samples"""
        worker_info = torch.utils.data.get_worker_info()
        
        # Set different seed per worker
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
            random.seed(worker_seed)
        
        token_buffer = []
        sample_shuffle_buffer = []

        for example in self.dataset:
            text = example.get(self.text_column, '')
            
            if not text or not isinstance(text, str):
                continue

            # Tokenize
            try:
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if self.tokenizer.eos_token_id is not None:
                    text_tokens.append(self.tokenizer.eos_token_id)
            except Exception:
                continue
                
            token_buffer.extend(text_tokens)

            # Create samples from buffer
            while len(token_buffer) >= self.max_length:
                chunk = token_buffer[:self.max_length]
                token_buffer = token_buffer[self.stride:]

                if len(chunk) > 1:
                    sample = {
                        'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                        'labels': torch.tensor(chunk[1:], dtype=torch.long),
                    }
                    sample_shuffle_buffer.append(sample)

                # Yield shuffled samples when buffer is full
                if len(sample_shuffle_buffer) >= self.shuffle_buffer_size:
                    random.shuffle(sample_shuffle_buffer)
                    yield from sample_shuffle_buffer
                    sample_shuffle_buffer = []
        
        # Yield remaining samples
        if sample_shuffle_buffer:
            random.shuffle(sample_shuffle_buffer)
            yield from sample_shuffle_buffer


class DataCollator:
    """Efficient collator with dynamic padding"""
    
    def __init__(
        self, 
        pad_token_id: int = 0, 
        max_length: Optional[int] = None,
        return_attention_mask: bool = True
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch")
        
        # Find max length in batch (dynamic padding)
        max_len = max(len(item['input_ids']) for item in batch)
        
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for item in batch:
            inp = item['input_ids'][:max_len]
            lab = item['labels'][:max_len]
            seq_len = len(inp)
            
            # Calculate padding
            pad_len = max_len - seq_len
            
            if pad_len > 0:
                # Pad input_ids
                inp = torch.cat([
                    inp, 
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                ])
                
                # Pad labels with -100 (ignore index)
                lab = torch.cat([
                    lab, 
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])
                
                # Create attention mask
                if self.return_attention_mask:
                    mask = torch.cat([
                        torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long)
                    ])
            else:
                if self.return_attention_mask:
                    mask = torch.ones(max_len, dtype=torch.long)
            
            input_ids_list.append(inp)
            labels_list.append(lab)
            if self.return_attention_mask:
                attention_mask_list.append(mask)
        
        result = {
            'input_ids': torch.stack(input_ids_list),
            'labels': torch.stack(labels_list),
        }
        
        if self.return_attention_mask:
            result['attention_mask'] = torch.stack(attention_mask_list)
        
        return result


class FileStreamingIterableDataset(IterableDataset):
    def __init__(
            self,
            file_path: Union[str, Path],
            tokenizer,
            max_length: int = 1024,
            buffer_size: int = 10000,
            stride: Optional[int] = None,
            seed: int = 42,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # We need 1 extra token for next-token prediction (input + target)
        self.req_len = max_length + 1
        self.buffer_size = buffer_size
        self.stride = stride if stride is not None else max_length
        self.seed = seed

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def __iter__(self):
        worker_info =torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_seed = self.seed + worker_id
        else:
            worker_id = 0
            num_workers = 1
            worker_seed = self.seed

        random.seed(worker_seed)

        shuffle_buf = []
        token_buffer = []

        file_size = os.path.getsize(self.file_path)
        bytes_per_worker = file_size // num_workers
        start_byte = worker_id * bytes_per_worker
        end_byte = start_byte + bytes_per_worker if worker_id != num_workers - 1 else file_size

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_byte)

            # If not the first worker, skip the first partial line (it belongs to prev worker)
            if start_byte != 0:
                f.readline()

            while True:
                # FIX: Read strictly until we pass our chunk,
                # UNLESS we are the last worker who must read to EOF.
                # Ideally, we read until the line *finishes* even if it crosses end_byte.
                current_pos = f.tell()

                if worker_id != num_workers - 1 and current_pos >= end_byte:
                    break

                line = f.readline()
                if not line:
                    break

                if not line.strip():
                    continue

                try:
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    if self.tokenizer.eos_token_id is not None:
                        tokens.append(self.tokenizer.eos_token_id)
                except Exception:
                    continue

                token_buffer.extend(tokens)

                # FIX: Check against req_len (max_length + 1)
                while len(token_buffer) >= self.req_len:
                    window = token_buffer[:self.req_len]
                    token_buffer = token_buffer[self.stride:]

                    # Create exactly max_length inputs
                    if len(window) == self.req_len:
                        sample = {
                            'input_ids': torch.tensor(window[:-1], dtype=torch.long),
                            'labels': torch.tensor(window[1:], dtype=torch.long),
                        }

                        if len(shuffle_buf) < self.buffer_size:
                            shuffle_buf.append(sample)
                        else:
                            idx = random.randint(0, len(shuffle_buf) - 1)
                            yield shuffle_buf[idx]
                            shuffle_buf[idx] = sample

        # Flush remaining buffer
        random.shuffle(shuffle_buf)
        yield from shuffle_buf

class ParquetDataset(Dataset):
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
        num_proc: int = 4
    ):
        try:
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.stride = stride

            if tokenizer is None:
                raise ValueError("Tokenizer cannot be None")

            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Path does not exist: {file_path}")

            parquet_files = list(file_path.glob("*.parquet"))
            if len(parquet_files) == 0:
                raise FileNotFoundError(f"No .parquet files found in: {file_path}")

            try:
                ds = load_dataset(
                    "parquet",
                    data_files=str(file_path / "*.parquet"),
                    split="train"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Parquet files: {e}")

            if "text" not in ds.column_names:
                raise KeyError(f"'text' column not found in dataset. Columns: {ds.column_names}")

            try:
                tokenized = ds.map(
                    lambda x: self.tokenizer(x["text"], add_special_tokens=False),
                    batched=True,
                    num_proc=num_proc,
                    remove_columns=ds.column_names
                )
            except Exception as e:
                raise RuntimeError(f"Tokenization failed: {e}")

            samples = []
            for tokens in tokenized["input_ids"]:
                if not isinstance(tokens, list):
                    continue

                L = len(tokens)
                if L < 2:
                    continue

                i = 0
                while i + 1 < L:
                    end = min(i + max_length + 1, L)
                    chunk = tokens[i:end]

                    if len(chunk) > 1:
                        samples.append(chunk)

                    if end == L:
                        break

                    i += stride

            if len(samples) == 0:
                raise RuntimeError("No valid samples were generated from the Parquet dataset.")

            self.samples = samples

        except Exception as e:
            raise RuntimeError(f"ParquetDataset initialization failed: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            chunk = self.samples[idx]
            chunk = torch.tensor(chunk, dtype=torch.long)
            return {
                "input_ids": chunk[:-1],
                "labels": chunk[1:]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load sample {idx}: {e}")

class PreloadedDataset(Dataset):
    """Pre-load entire dataset in RAM for maximum speed (small datasets only)"""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512,
        min_length: int = 32,
    ):
        print("Loading entire dataset into memory...")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        # Load and tokenize everything at once
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        text = path.read_text(encoding='utf-8', errors='ignore')
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Pre-create all samples
        self.samples = []
        s = 0
        while s < len(tokens):
            end = min(s + max_length, len(tokens))
            chunk = tokens[s:end]
            
            if len(chunk) >= min_length and len(chunk) > 1:
                self.samples.append({
                    'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                    'labels': torch.tensor(chunk[1:], dtype=torch.long),
                })
            
            if end == len(tokens):
                break
            s += stride
        
        if not self.samples:
            raise ValueError("No valid samples created")
        
        print(f"Loaded {len(self.samples)} samples into memory")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_optimized_dataloader(
    file_path: Optional[Union[str, Path]] = None,
    tokenizer = None,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4,
    use_streaming: bool = False,
    preload_small: bool = False,
    dataset: Optional[Union[Dataset, IterableDataset]] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    **kwargs
) -> DataLoader:
    """
    Factory function to create optimized dataloader.
    
    Args:
        file_path: Path to text file(s)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        use_streaming: Use streaming dataset for large files
        preload_small: Preload entire dataset into RAM (small datasets only)
        dataset: Pre-created dataset (if provided, file_path is ignored)
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        **kwargs: Additional arguments passed to dataset constructor
    """
    
    # If dataset is provided, use it directly
    if dataset is not None:
        is_iterable = isinstance(dataset, IterableDataset)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if is_iterable else kwargs.get('shuffle', True),
            num_workers=num_workers,
            pin_memory=pin_memory if pin_memory is not None else torch.cuda.is_available(),
            collate_fn=DataCollator(
                pad_token_id=tokenizer.pad_token_id if tokenizer else 0,
                max_length=max_length
            ),
            drop_last=True,
            persistent_workers=persistent_workers if persistent_workers is not None else (num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        return dataloader
    
    # Validate inputs
    if file_path is None:
        raise ValueError("Either file_path or dataset must be provided")
    
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    
    # Choose dataset type based on requirements
    if preload_small:
        dataset = PreloadedDataset(
            file_path, 
            tokenizer, 
            max_length,
            stride=kwargs.get('stride', max_length // 2)
        )
        shuffle = True
    elif use_streaming:
        dataset = FileStreamingIterableDataset(
            file_path, 
            tokenizer, 
            max_length,
            **kwargs
        )
        shuffle = False
    else:
        dataset = TextDataset(
            file_path, 
            tokenizer, 
            max_length, 
            num_workers=num_workers, 
            **kwargs
        )
        shuffle = kwargs.get('shuffle', True)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
        num_workers=num_workers,
        pin_memory=pin_memory if pin_memory is not None else torch.cuda.is_available(),
        collate_fn=DataCollator(
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_length
        ),
        drop_last=True,
        persistent_workers=persistent_workers if persistent_workers is not None else (num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    return dataloader


if __name__ == '__main__':
    print("=== Testing DirectStreamingDataset ===\n")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test 1: Dataset with config auto-detection
    print("\n--- Test 1: Auto-detect config (finemath) ---")
    streaming_dataset = DirectStreamingDataset(
        dataset_name="HuggingFaceTB/finemath",
        tokenizer=tokenizer,
        max_length=512,
        stride=256,
        shuffle_buffer_size=1000,
        hf_shuffle=True,
        seed=42,
        auto_detect_config=True  # Will auto-select best config
    )
    
    # Create dataloader with the streaming dataset
    dataloader = create_optimized_dataloader(
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512,
        num_workers=0,  # Use 0 for testing with streaming datasets
        dataset=streaming_dataset
    )
    
    # Test batches
    print("\nTesting dataloader batches...\n")
    num_batches_to_test = 3
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_test:
            break
            
        print(f"--- Batch {i+1} ---")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")
        
        # Decode first sequence
        decoded = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        print(f"Sample text: {decoded[:200]}...")
        print()
    
    # Test 2: Dataset without config (should work without auto-detect)
    print("\n--- Test 2: Dataset without config (cosmopedia) ---")
    streaming_dataset2 = DirectStreamingDataset(
        dataset_name="HuggingFaceTB/cosmopedia-100k",
        tokenizer=tokenizer,
        max_length=512,
        stride=256,
        shuffle_buffer_size=1000,
        hf_shuffle=True,
        seed=42
    )
    
    dataloader2 = create_optimized_dataloader(
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_workers=0,
        dataset=streaming_dataset2
    )
    
    print("\nTesting batch from cosmopedia...\n")
    for i, batch in enumerate(dataloader2):
        if i >= 1:
            break
        print(f"Batch shape: {batch['input_ids'].shape}")
        decoded = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        print(f"Sample: {decoded[:150]}...")
    
    print("\n=== All tests completed successfully! ===")

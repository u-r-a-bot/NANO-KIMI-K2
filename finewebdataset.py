import sys
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, List, Dict
import numpy as np
import gc
from tqdm import tqdm

class FineWebDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        max_length: int = 512,
        tokenizer: AutoTokenizer = None,
        cache_dir: str = "./cache",
        streaming: bool = False
    ):
        self.num_samples = num_samples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        
        if streaming:
            print("Note: For streaming use FineWebStreamingDataset class")
        
        print(f"Loading FineWeb dataset with {num_samples} samples...")
        
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split=f"train[:{num_samples}]",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.tokenized_samples = []
        self._tokenize_dataset()
    
    def _tokenize_dataset(self):
        print("Tokenizing dataset...")
        for idx, example in enumerate(tqdm(self.dataset, desc="Tokenizing")):
            text = example.get('text', '')
            
            if not text:
                continue
            
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.tokenized_samples.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'labels': tokens['input_ids'].squeeze(0).clone()
            })
            
            if len(self.tokenized_samples) >= self.num_samples:
                break
        
        print(f"Tokenized {len(self.tokenized_samples)} samples")
    
    def __len__(self):
        return len(self.tokenized_samples)
    
    def __getitem__(self, idx):
        return self.tokenized_samples[idx]

class FineWebStreamingDataset(IterableDataset):
    def __init__(
        self,
        num_samples: Optional[int] = None,
        max_length: int = 512,
        tokenizer: AutoTokenizer = None,
        batch_size: int = 1,
        buffer_size: int = 1000
    ):
        self.num_samples = num_samples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def __iter__(self):
        samples_processed = 0  # ← FIXED: Local variable, resets each epoch
        
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        buffer = []
        
        for example in dataset:
            if self.num_samples and samples_processed >= self.num_samples:
                break
            
            text = example.get('text', '')
            
            if not text:
                continue
            
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            sample = {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'labels': tokens['input_ids'].squeeze(0).clone()
            }
            
            buffer.append(sample)
            samples_processed += 1  # ← Now increments local variable
            
            if len(buffer) >= self.buffer_size:
                np.random.shuffle(buffer)
                for item in buffer[:self.batch_size]:
                    yield item
                buffer = buffer[self.batch_size:]
        
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item

class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def test_fineweb():
    from config import Config
    from transformers import AutoTokenizer
    
    print("Testing FineWeb Dataset Integration...")
    
    config = Config()
    config.max_batch_size = 2
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n1. Testing Standard Dataset...")
    dataset = FineWebDataset(
        num_samples=100,
        max_length=config.max_seq_length,
        tokenizer=tokenizer,
        cache_dir="./cache"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=config.max_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    sample_text = tokenizer.decode(batch['input_ids'][0][:50])
    print(f"\nSample text (first 50 tokens):")
    print(sample_text)
    
    print("\n2. Testing Streaming Dataset...")
    streaming_dataset = FineWebStreamingDataset(
        num_samples=50,
        max_length=config.max_seq_length,
        tokenizer=tokenizer,
        batch_size=1,
        buffer_size=10
    )
    
    streaming_loader = DataLoader(
        streaming_dataset,
        batch_size=config.max_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    batch = next(iter(streaming_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    sample_text = tokenizer.decode(batch['input_ids'][0][:50])
    print(f"\nSample text (first 50 tokens):")
    print(sample_text)
    
    print("\n✅ FineWeb dataset test completed successfully!")
    
    del dataset
    del streaming_dataset
    del dataloader
    del streaming_loader
    gc.collect()
    
    return True

if __name__ == "__main__":
    success = test_fineweb()
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nTests failed!")
        sys.exit(1)
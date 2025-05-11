# lmtraining/data/dataset.py
import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import gc
from typing import Optional, Dict, List, Tuple, Union, Any

logger = logging.getLogger(__name__)

class ChunkingTextDataset(IterableDataset):
    """
    Memory-efficient dataset that processes text files in chunks.
    Dynamically loads and tokenizes data without holding everything in memory.
    """
    def __init__(
        self,
        tokenizer,
        file_path: str,
        max_seq_length: int,
        stride: int = None,
        block_size: int = 1024 * 128,  # Read ~128KB at a time
        is_train: bool = True,
        min_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.stride = stride if stride is not None else max_seq_length // 2
        self.block_size = block_size
        self.is_train = is_train
        self.min_length = min_length if min_length is not None else max_seq_length // 2
        
        # Verify the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file {file_path} not found")
        
        # Get approximate size for progress reporting
        self.file_size = os.path.getsize(file_path)
        logger.info(f"Initializing chunking dataset from {file_path} ({self.file_size / 1024 / 1024:.2f} MB)")
    
    def __iter__(self):
        """Iterate through the file in chunks, yielding processed examples."""
        buffer = []  # Accumulator for tokens across chunks
        bytes_read = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            while True:
                # Read a chunk of text
                text_chunk = f.read(self.block_size)
                if not text_chunk:
                    break  # End of file
                
                bytes_read += len(text_chunk.encode('utf-8'))
                
                # Tokenize this chunk
                chunk_tokens = self.tokenizer.encode(text_chunk, add_special_tokens=False)
                buffer.extend(chunk_tokens)
                
                # Process complete sequences from buffer
                while len(buffer) >= self.max_seq_length:
                    # Ensure we only take max_seq_length tokens (fixes truncation issue)
                    input_ids = buffer[:self.max_seq_length]
                    attention_mask = [1] * self.max_seq_length
                    
                    # Create a training example
                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(input_ids, dtype=torch.long)
                    }
                    
                    yield example
                    
                    # Move buffer forward by stride (allows for overlapping contexts)
                    buffer = buffer[self.stride:]
                
                # Periodically force garbage collection
                if bytes_read % (10 * self.block_size) == 0:
                    gc.collect()
        
        # Process any remaining buffer if it's long enough
        if buffer and len(buffer) >= self.min_length:
            # Pad if necessary
            if len(buffer) < self.max_seq_length:
                padding_length = self.max_seq_length - len(buffer)
                attention_mask = [1] * len(buffer) + [0] * padding_length
                buffer = buffer + [self.tokenizer.pad_token_id] * padding_length
            else:
                # IMPORTANT: Make sure we truncate to max_seq_length
                buffer = buffer[:self.max_seq_length]
                attention_mask = [1] * self.max_seq_length
            
            # Create a final example
            example = {
                "input_ids": torch.tensor(buffer, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(buffer, dtype=torch.long)
            }
            
            yield example

# Custom collate function to ensure all tensors are properly sized
def collate_fn(batch):
    """Custom collate function to ensure all tensors have the right shape."""
    # Get the max sequence length from the batch
    max_len = max(x["input_ids"].size(0) for x in batch)
    
    # Initialize tensors
    input_ids = []
    attention_mask = []
    labels = []
    
    # Process each example
    for example in batch:
        # Get current lengths
        cur_len = example["input_ids"].size(0)
        
        # Make sure they're all the same length
        if cur_len < max_len:
            # Pad if needed
            padding_length = max_len - cur_len
            input_ids.append(F.pad(example["input_ids"], (0, padding_length), value=example["input_ids"][-1]))
            attention_mask.append(F.pad(example["attention_mask"], (0, padding_length), value=0))
            labels.append(F.pad(example["labels"], (0, padding_length), value=-100))  # -100 is ignored in loss
        else:
            # Truncate if needed (this should be rare with properly implemented dataset)
            input_ids.append(example["input_ids"][:max_len])
            attention_mask.append(example["attention_mask"][:max_len])
            labels.append(example["labels"][:max_len])
    
    # Stack into batches
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels)
    }

class CachedChunkDataset(Dataset):
    """Dataset that caches a limited number of processed examples in memory."""
    def __init__(
        self,
        tokenizer,
        file_path: str,
        max_seq_length: int,
        max_samples: int = 10000,
        min_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.max_samples = max_samples
        self.min_length = min_length if min_length is not None else max_seq_length // 2
        
        # Load and preprocess a limited number of examples
        self.examples = self._load_and_preprocess()
    
    def _load_and_preprocess(self):
        """Load and preprocess a limited number of examples from the file."""
        logger.info(f"Loading up to {self.max_samples} examples from {self.file_path}")
        
        examples = []
        chunking_dataset = ChunkingTextDataset(
            self.tokenizer,
            self.file_path,
            self.max_seq_length,
            min_length=self.min_length
        )
        
        for i, example in enumerate(chunking_dataset):
            # Ensure examples are properly sized
            if len(example["input_ids"]) > self.max_seq_length:
                example["input_ids"] = example["input_ids"][:self.max_seq_length]
                example["attention_mask"] = example["attention_mask"][:self.max_seq_length]
                example["labels"] = example["labels"][:self.max_seq_length]
            
            # Add example after ensuring correct size
            examples.append(example)
            
            if i + 1 >= self.max_samples:
                break
        
        logger.info(f"Loaded {len(examples)} examples")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_dataloaders(tokenizer, config, streaming=False, max_samples=None):
    """
    Create dataloaders for training and evaluation.
    
    Args:
        tokenizer: Tokenizer to use for processing text
        config: Configuration object
        streaming: Whether to use streaming dataset (memory-efficient) or cached dataset
        max_samples: Maximum number of samples to load (only for cached dataset)
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    if streaming:
        # Use streaming dataset for training
        train_dataset = ChunkingTextDataset(
            tokenizer,
            config.data.train_file,
            config.data.max_seq_length
        )
        
        # For evaluation, we use a cached dataset since it's smaller
        eval_dataset = CachedChunkDataset(
            tokenizer,
            config.data.validation_file,
            config.data.max_seq_length,
            max_samples=1000  # Limit eval samples
        )
        
        # Create dataloaders with custom collate function for proper tensor sizes
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.train_batch_size,
            num_workers=0,  # Streaming dataset handles its own parallelism
            collate_fn=collate_fn  # Use custom collate function
        )
        
    else:
        # Use cached datasets with limited samples
        train_max_samples = max_samples if max_samples is not None else 10000
        
        train_dataset = CachedChunkDataset(
            tokenizer,
            config.data.train_file,
            config.data.max_seq_length,
            max_samples=train_max_samples
        )
        
        eval_dataset = CachedChunkDataset(
            tokenizer,
            config.data.validation_file,
            config.data.max_seq_length,
            max_samples=min(train_max_samples // 5, 1000)  # Smaller eval set
        )
        
        # Create dataloaders with custom collate function
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.train_batch_size,
            shuffle=True,
            num_workers=config.data.preprocessing_num_workers if hasattr(config.data, 'preprocessing_num_workers') else 0,
            collate_fn=collate_fn  # Use custom collate function
        )
    
    # Eval dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.preprocessing_num_workers if hasattr(config.data, 'preprocessing_num_workers') else 0,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    return train_dataloader, eval_dataloader
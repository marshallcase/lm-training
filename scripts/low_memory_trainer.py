# low_memory_trainer.py
"""
Modified version of train_small_lm.py that uses less memory
for processing datasets and incremental loading.
"""

import os
import sys
import argparse
import json
import logging
import torch
import gc
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Add path for local imports
sys.path.insert(0, ".")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def create_small_dataset(tokenizer, file_path, max_seq_length, max_samples=1000):
    """Create a small dataset without loading the entire file into memory."""
    logger.info(f"Creating small dataset from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found")
    
    # Process data with low memory usage
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read file in chunks
        text_chunk = ""
        line_count = 0
        sample_count = 0
        
        for line in f:
            text_chunk += line
            line_count += 1
            
            # Process chunk when it's big enough or at a reasonable interval
            if line_count >= 50 or len(text_chunk) > max_seq_length * 4:
                # Tokenize text chunk
                input_ids = tokenizer.encode(text_chunk, truncation=False)
                
                # Create examples from this chunk
                for i in range(0, len(input_ids), max_seq_length):
                    # Break if we have enough samples
                    if sample_count >= max_samples:
                        break
                    
                    # Get a segment of appropriate length
                    chunk = input_ids[i:i + max_seq_length]
                    
                    # Skip if chunk is too small
                    if len(chunk) < max_seq_length // 2:
                        continue
                    
                    # Pad if necessary
                    if len(chunk) < max_seq_length:
                        # Pad with padding token
                        chunk = chunk + [tokenizer.pad_token_id] * (max_seq_length - len(chunk))
                    
                    # Create tensor
                    input_tensor = torch.tensor(chunk, dtype=torch.long)
                    
                    # Add to examples
                    examples.append({
                        "input_ids": input_tensor,
                        "attention_mask": torch.ones_like(input_tensor),
                        "labels": input_tensor.clone()
                    })
                    
                    sample_count += 1
                
                # Reset chunk and force garbage collection
                text_chunk = ""
                line_count = 0
                gc.collect()
                
                # Break if we have enough samples
                if sample_count >= max_samples:
                    logger.info(f"Reached max samples ({max_samples})")
                    break
        
        # Process any remaining text
        if text_chunk and sample_count < max_samples:
            input_ids = tokenizer.encode(text_chunk, truncation=False)
            
            # Create examples from this chunk
            for i in range(0, len(input_ids), max_seq_length):
                # Break if we have enough samples
                if sample_count >= max_samples:
                    break
                
                # Get a segment of appropriate length
                chunk = input_ids[i:i + max_seq_length]
                
                # Skip if chunk is too small
                if len(chunk) < max_seq_length // 2:
                    continue
                
                # Pad if necessary
                if len(chunk) < max_seq_length:
                    # Pad with padding token
                    chunk = chunk + [tokenizer.pad_token_id] * (max_seq_length - len(chunk))
                
                # Create tensor
                input_tensor = torch.tensor(chunk, dtype=torch.long)
                
                # Add to examples
                examples.append({
                    "input_ids": input_tensor,
                    "attention_mask": torch.ones_like(input_tensor),
                    "labels": input_tensor.clone()
                })
                
                sample_count += 1
    
    logger.info(f"Created {len(examples)} examples from {file_path}")
    return examples

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset from pre-loaded examples."""
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_simple_dataloaders(tokenizer, config, max_samples=1000):
    """Create dataloaders with lower memory usage."""
    logger.info("Creating simplified dataloaders with lower memory usage")
    
    # Create training examples
    train_examples = create_small_dataset(
        tokenizer, 
        config.data.train_file, 
        config.data.max_seq_length,
        max_samples=max_samples
    )
    
    # Create validation examples
    val_examples = create_small_dataset(
        tokenizer, 
        config.data.validation_file, 
        config.data.max_seq_length,
        max_samples=max_samples // 2
    )
    
    # Create datasets
    train_dataset = SimpleDataset(train_examples)
    val_dataset = SimpleDataset(val_examples)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.train_batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to reduce memory usage
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing to reduce memory usage
    )
    
    return train_dataloader, val_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory in config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of training samples to use",
    )
    args = parser.parse_args()
    
    # Display GPU info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available - training on CPU")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_path}")
    
    # Import config class here to avoid memory issues
    from lmtraining.config import Config
    config = Config.from_json(args.config_path)
    
    # Override output dir if provided
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Reduce model size for low memory usage
    logger.info("Reducing model size for low memory usage")
    config.model.hidden_size = 128  # Original: 256
    config.model.num_hidden_layers = 3  # Original: 6
    config.model.num_attention_heads = 4  # Original: 8
    config.model.intermediate_size = 512  # Original: 1024
    config.data.max_seq_length = 256  # Original: 1024
    
    # Print memory usage
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.data.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    
    # Make sure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create simple dataloaders with fewer samples
    logger.info("Creating simplified dataloaders")
    train_dataloader, eval_dataloader = create_simple_dataloaders(
        tokenizer, config, max_samples=args.max_samples
    )
    
    logger.info(f"Created dataloaders: {len(train_dataloader)} training batches, {len(eval_dataloader)} validation batches")
    
    # Update vocabulary size in config if needed
    if tokenizer.vocab_size != config.model.vocab_size:
        logger.info(f"Updating vocabulary size from {config.model.vocab_size} to {tokenizer.vocab_size}")
        config.model.vocab_size = tokenizer.vocab_size
    
    # Print memory usage after dataloader creation
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Import model class here to avoid memory issues
    logger.info("Creating model")
    from lmtraining.models.transformer import TransformerModel
    model = TransformerModel(config.model)
    
    # Move model to GPU
    if torch.cuda.is_available():
        logger.info("Moving model to GPU")
        model = model.to("cuda")
    
    # Verify model is on the correct device
    device = next(model.parameters()).device
    logger.info(f"Model is on device: {device}")
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {param_count:,} parameters")
    
    # Print memory usage after model creation
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Import trainer class here to avoid memory issues
    logger.info("Creating trainer")
    from lmtraining.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    # Print memory usage before training
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
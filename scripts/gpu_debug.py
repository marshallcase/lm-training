# low_memory_gpu_debug.py
"""
Low memory version of GPU diagnostics, designed to avoid out-of-memory errors
while still verifying GPU functionality.
"""

import os
import sys
import time
import torch
import argparse
import logging
import gc
from transformers import AutoTokenizer

# Add the parent directory to the path
sys.path.append(".")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_cuda_basic():
    """Basic CUDA availability and functionality test with small tensors."""
    logger.info("\n===== BASIC CUDA TEST =====")
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available in PyTorch.")
        return False
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Test with much smaller tensor to avoid memory issues
    try:
        logger.info("Creating small test tensor on CUDA...")
        test_tensor = torch.rand(100, 100, device="cuda")
        result = (test_tensor @ test_tensor).sum().item()
        logger.info(f"Tensor operation result: {result}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info("CUDA tensor operations successful!")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"Error in CUDA tensor operations: {e}")
        return False

def test_model_loading(config_path):
    """Test loading just model architecture without forward pass."""
    logger.info("\n===== MODEL LOADING TEST =====")
    
    try:
        # Import here to avoid loading unnecessary modules if CUDA test fails
        from lmtraining.config import Config
        from lmtraining.models.transformer import TransformerModel
        
        # Load configuration
        config = Config.from_json(config_path)
        
        # Reduce model size for testing
        config.model.hidden_size = 128
        config.model.num_hidden_layers = 2
        config.model.num_attention_heads = 4
        config.model.intermediate_size = 512
        
        # Create model
        logger.info("Creating small test model...")
        model = TransformerModel(config.model)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Test model has {param_count:,} parameters")
        
        # Move to GPU and check device
        logger.info("Moving model to CUDA...")
        model = model.to("cuda")
        model_device = next(model.parameters()).device
        logger.info(f"Model is on device: {model_device}")
        logger.info(f"Memory used by model: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error in model loading: {e}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        return False

def test_small_batch_processing():
    """Test processing a small batch of data on GPU."""
    logger.info("\n===== SMALL BATCH TEST =====")
    
    try:
        # Create small random batch
        logger.info("Creating small test batch...")
        batch_size = 2
        seq_len = 32
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        logger.info(f"Initial batch devices: input_ids={input_ids.device}, mask={attention_mask.device}")
        
        # Move to GPU
        logger.info("Moving batch to GPU...")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        
        logger.info(f"After moving: input_ids={input_ids.device}, mask={attention_mask.device}")
        logger.info(f"Memory used: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Create small classifier
        logger.info("Testing with a small classifier...")
        hidden_size = 64
        small_model = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, hidden_size),
            torch.nn.Linear(hidden_size, 10)
        ).to("cuda")
        
        # Forward pass
        with torch.no_grad():
            embeddings = small_model[0](input_ids)
            output = small_model[1](embeddings.mean(dim=1))
        
        logger.info(f"Output shape: {output.shape}, device: {output.device}")
        logger.info(f"Memory after forward pass: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Clean up
        del input_ids, attention_mask, small_model, embeddings, output
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return False

def test_data_path_exists(config_path):
    """Test if the data files specified in config exist."""
    logger.info("\n===== DATA PATH TEST =====")
    
    try:
        from lmtraining.config import Config
        
        # Load configuration
        config = Config.from_json(config_path)
        
        # Check train file
        train_file = config.data.train_file
        logger.info(f"Train file path: {train_file}")
        if os.path.exists(train_file):
            logger.info(f"Train file exists")
            # Check file size
            size_mb = os.path.getsize(train_file) / (1024 * 1024)
            logger.info(f"Train file size: {size_mb:.2f} MB")
        else:
            logger.error(f"Train file does not exist!")
        
        # Check validation file
        val_file = config.data.validation_file
        logger.info(f"Validation file path: {val_file}")
        if os.path.exists(val_file):
            logger.info(f"Validation file exists")
            # Check file size
            size_mb = os.path.getsize(val_file) / (1024 * 1024)
            logger.info(f"Validation file size: {size_mb:.2f} MB")
        else:
            logger.error(f"Validation file does not exist!")
            
        # Check if tokenizer exists
        tokenizer_name = config.data.tokenizer_name
        logger.info(f"Tokenizer name: {tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Tokenizer loaded successfully")
            logger.info(f"Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            
        return True
    except Exception as e:
        logger.error(f"Error checking data paths: {e}")
        return False

def check_memory_leak():
    """Test for CUDA memory leaks."""
    logger.info("\n===== MEMORY LEAK TEST =====")
    
    try:
        # Get initial memory usage
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated(0) / 1024**2
        logger.info(f"Initial memory: {initial_memory:.2f} MB")
        
        # Create and delete tensors in a loop
        for i in range(5):
            # Create a tensor
            tensor = torch.rand(1000, 1000, device="cuda")
            current_memory = torch.cuda.memory_allocated(0) / 1024**2
            logger.info(f"Iteration {i+1}: memory after allocation: {current_memory:.2f} MB")
            
            # Delete tensor
            del tensor
            torch.cuda.empty_cache()
            gc.collect()
            
            after_del_memory = torch.cuda.memory_allocated(0) / 1024**2
            logger.info(f"Iteration {i+1}: memory after deletion: {after_del_memory:.2f} MB")
        
        # Check for memory leak
        final_memory = torch.cuda.memory_allocated(0) / 1024**2
        memory_diff = final_memory - initial_memory
        
        logger.info(f"Final memory: {final_memory:.2f} MB")
        logger.info(f"Memory difference: {memory_diff:.2f} MB")
        
        if memory_diff > 5.0:  # If more than 5MB difference, might be a leak
            logger.warning(f"Possible memory leak detected: {memory_diff:.2f} MB")
        else:
            logger.info("No significant memory leak detected")
            
        return True
    except Exception as e:
        logger.error(f"Error in memory leak test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Low memory GPU diagnostics")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to model configuration",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "basic", "model", "batch", "data", "memory"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()
    
    # Basic CUDA test is always run
    test_cuda_basic()
    
    if args.test in ["all", "model"] and args.config_path:
        test_model_loading(args.config_path)
    
    if args.test in ["all", "batch"]:
        test_small_batch_processing()
    
    if args.test in ["all", "data"] and args.config_path:
        test_data_path_exists(args.config_path)
    
    if args.test in ["all", "memory"]:
        check_memory_leak()
    
    logger.info("\n===== ALL TESTS COMPLETED =====")
    logger.info("If you see this message, the script completed without being killed.")

if __name__ == "__main__":
    main()
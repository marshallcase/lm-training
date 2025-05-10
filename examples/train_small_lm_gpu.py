# train_small_lm_gpu.py
# Modified version of examples/train_small_lm.py with explicit GPU handling
import os
import argparse
import json
import logging
import torch
import gc
from transformers import AutoTokenizer

# Add parent directory to path for imports
import sys
sys.path.append(".")

from lmtraining.config import Config
from lmtraining.models.transformer import TransformerModel
from lmtraining.data.dataset import create_dataloaders
from lmtraining.training.trainer import Trainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
        "--force_gpu",
        action="store_true",
        help="Force using GPU and exit if not available",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use",
    )
    args = parser.parse_args()
    
    # Check GPU availability if force_gpu is set
    if args.force_gpu and not torch.cuda.is_available():
        logger.error("GPU not available but --force_gpu was set. Exiting.")
        sys.exit(1)
    
    # Select specific GPU if multiple are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Multiple GPUs available. Using GPU {args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    
    # Enhanced GPU info display
    if torch.cuda.is_available():
        logger.info("===== GPU Information =====")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDA: {torch.version.cuda}")
        logger.info(f"Device Count: {torch.cuda.device_count()}")
        logger.info(f"Current Device: {torch.cuda.current_device()}")
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device Properties: {torch.cuda.get_device_properties(0)}")
        logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        logger.info("=========================")
    else:
        logger.warning("CUDA is not available. Training will run on CPU.")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = Config.from_json(args.config_path)
    
    # Override output dir if provided
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.data.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    
    # Create dataloaders
    logger.info("Creating dataloaders")
    train_dataloader, eval_dataloader = create_dataloaders(tokenizer, config)
    
    # Update vocabulary size in config if needed
    if tokenizer.vocab_size != config.model.vocab_size:
        logger.info(f"Updating vocabulary size in config from {config.model.vocab_size} to {tokenizer.vocab_size}")
        config.model.vocab_size = tokenizer.vocab_size
    
    # Create model
    logger.info("Creating model")
    model = TransformerModel(config.model)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {param_count:,} parameters")
    
    # Force model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    
    # Verify model is on correct device
    model_device = next(model.parameters()).device
    logger.info(f"Model is on device: {model_device}")
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    # Clean up before training
    if torch.cuda.is_available():
        # Explicitly run garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
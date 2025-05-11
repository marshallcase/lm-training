# examples/train_small_lm.py
import os
import argparse
import json
import logging
import torch
from transformers import AutoTokenizer
import sys
from datetime import datetime

from lmtraining.config import Config
from lmtraining.models.transformer import TransformerModel
from lmtraining.data.dataset import create_dataloaders
from lmtraining.training.trainer import Trainer


# Set up logging to both console and file
def setup_logging(output_dir):
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
    
    # Configure logging
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    
    return logger


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
        default=None,
        help="Maximum number of samples to load (None for streaming mode)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset (lowest memory usage but may be slower)",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more logging",
    )
    args = parser.parse_args()
    
    # Load configuration first to get output_dir for logging setup
    config = Config.from_json(args.config_path)
    
    # Override output dir if provided
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Setup logging to both console and file
    logger = setup_logging(config.training.output_dir)
    
    # Save command line arguments
    args_file = os.path.join(config.training.output_dir, "run_args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Command line arguments saved to {args_file}")
    
    # Print run information
    logger.info("=" * 50)
    logger.info(f"Starting training run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config file: {args.config_path}")
    logger.info(f"Output directory: {config.training.output_dir}")
    logger.info("=" * 50)
    
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
    logger.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Save expanded configuration
    config_file = os.path.join(config.training.output_dir, "full_config.json")
    with open(config_file, "w") as f:
        json.dump({
            "model": {k: getattr(config.model, k) for k in config.model.__dataclass_fields__},
            "training": {k: getattr(config.training, k) for k in config.training.__dataclass_fields__},
            "data": {k: getattr(config.data, k) for k in config.data.__dataclass_fields__}
        }, f, indent=2)
    logger.info(f"Full configuration saved to {config_file}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.data.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    
    # Make sure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders with memory-efficient options
    logger.info("Creating dataloaders")
    if args.streaming:
        logger.info("Using streaming dataset for lowest memory usage")
    elif args.max_samples:
        logger.info(f"Using cached dataset with max_samples={args.max_samples}")
    else:
        logger.info("Using default cached dataset with 10000 samples")
        
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenizer, 
        config,
        streaming=args.streaming,
        max_samples=args.max_samples
    )
    
    # Log dataloader information
    if not args.streaming and not isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
        logger.info(f"Train dataset size: {len(train_dataloader.dataset)}")
        logger.info(f"Train batches: {len(train_dataloader)}")
    if eval_dataloader and not isinstance(eval_dataloader.dataset, torch.utils.data.IterableDataset):
        logger.info(f"Eval dataset size: {len(eval_dataloader.dataset)}")
        logger.info(f"Eval batches: {len(eval_dataloader)}")
    
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
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    
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
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Train model
    logger.info("=" * 30)
    logger.info("Starting training")
    logger.info("=" * 30)
    try:
        train_loss = trainer.train()
        logger.info(f"Training complete. Final loss: {train_loss:.4f}")
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise
    
    logger.info("=" * 30)
    logger.info("Training complete")
    logger.info("=" * 30)
    
    # Print final summary
    if torch.cuda.is_available():
        logger.info(f"Final GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    logger.info(f"Output directory: {config.training.output_dir}")
    logger.info(f"Best model saved at: {os.path.join(config.training.output_dir, 'best_model')}")
    logger.info(f"Training metrics saved at: {os.path.join(config.training.output_dir, 'training_metrics.csv')}")


if __name__ == "__main__":
    main()
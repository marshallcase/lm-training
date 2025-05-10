# examples/train_small_lm.py
import os
import argparse
import json
import logging
import torch
from transformers import AutoTokenizer

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
    args = parser.parse_args()
    
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
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
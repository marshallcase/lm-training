# lmtraining/training/trainer.py
import os
import time
import math
import json
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from lmtraining.models.transformer import TransformerModel


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: TransformerModel,
        config: Any,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Create optimizer if not provided
        self.optimizer = optimizer or self._create_optimizer()
        
        # Create learning rate scheduler if not provided
        self.lr_scheduler = lr_scheduler or self._create_lr_scheduler()
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            param_device = next(self.model.parameters()).device
            logger.info(f"Model parameters are on: {param_device}")
        
        # Set up automatic mixed precision if enabled
        self.use_amp = config.training.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Create output directory if it doesn't exist
        os.makedirs(config.training.output_dir, exist_ok=True)
        
    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_params,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
        )
    
    def _create_lr_scheduler(self):
        if self.config.training.warmup_steps > 0 or self.config.training.warmup_ratio > 0:
            if self.config.training.max_steps > 0:
                t_total = self.config.training.max_steps
            else:
                # Calculate total steps based on epochs
                t_total = len(self.train_dataloader) * self.config.training.num_train_epochs
                t_total = math.ceil(t_total / self.config.training.gradient_accumulation_steps)
            
            warmup_steps = self.config.training.warmup_steps
            if warmup_steps == 0 and self.config.training.warmup_ratio > 0:
                warmup_steps = int(t_total * self.config.training.warmup_ratio)
            
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(t_total - current_step) / float(max(1, t_total - warmup_steps))
                )
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        return None
    
    def train(self):
        """Main training loop."""
        # Calculate total number of training steps
        if self.config.training.max_steps > 0:
            t_total = self.config.training.max_steps
            num_train_epochs = math.ceil(
                self.config.training.max_steps / len(self.train_dataloader) * 
                self.config.training.gradient_accumulation_steps
            )
        else:
            t_total = len(self.train_dataloader) * self.config.training.num_train_epochs
            t_total = math.ceil(t_total / self.config.training.gradient_accumulation_steps)
            num_train_epochs = self.config.training.num_train_epochs
        
        logger.info(f"***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Total train batch size = {self.config.training.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {t_total}")
        logger.info(f"  Using automatic mixed precision: {self.use_amp}")
        
        # Initialize progress tracking
        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        tr_loss = 0.0
        
        # Set model to training mode
        self.model.train()
        self.optimizer.zero_grad()
        
        train_iterator = range(int(num_train_epochs))
        
        for _ in train_iterator:
            self.epoch = epochs_trained + _ + 1
            epoch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
            
            for step, batch in enumerate(epoch_iterator):
                # Skip steps if resuming from checkpoint
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with or without mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                        loss = loss / self.config.training.gradient_accumulation_steps
                    
                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / self.config.training.gradient_accumulation_steps
                    loss.backward()
                
                tr_loss += loss.item()
                
                # Update weights if gradient accumulation is complete
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        # Unscale gradients and clip
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.max_grad_norm
                        )
                        
                        # Update weights with scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.max_grad_norm
                        )
                        
                        # Update weights
                        self.optimizer.step()
                    
                    # Update learning rate
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Update progress bar
                    epoch_iterator.set_postfix(loss=tr_loss / self.global_step)
                    
                    # Log progress
                    if self.global_step % self.config.training.logging_steps == 0:
                        logs = {
                            "loss": tr_loss / self.global_step,
                            "learning_rate": self.get_lr(),
                            "epoch": self.epoch,
                            "step": self.global_step,
                        }
                        logger.info(f"Training: {logs}")
                    
                    # Evaluate if needed
                    if (self.config.training.evaluation_strategy == "steps" and 
                        self.global_step % self.config.training.eval_steps == 0):
                        self.evaluate()
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.training.save_steps == 0:
                        self.save_model()
                
                # Break if max steps reached
                if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                    epoch_iterator.close()
                    break
            
            # Evaluate at end of epoch if configured
            if self.config.training.evaluation_strategy == "epoch":
                self.evaluate()
                self.model.train()
            
            # Break if max steps reached
            if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                break
        
        # Save final model
        self.save_model()
        
        return tr_loss / self.global_step
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided, skipping evaluation.")
            return None
        
        logger.info(f"***** Running evaluation *****")
        logger.info(f"  Num examples = {len(self.eval_dataloader.dataset)}")
        logger.info(f"  Batch size = {self.config.training.eval_batch_size}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        eval_loss = 0.0
        eval_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                eval_loss += loss.mean().item()
            
            eval_steps += 1
        
        eval_loss = eval_loss / eval_steps
        perplexity = math.exp(eval_loss)
        
        # Log evaluation results
        result = {"loss": eval_loss, "perplexity": perplexity}
        logger.info(f"***** Eval results *****")
        for key, value in result.items():
            logger.info(f"  {key} = {value}")
        
        # Save best model
        if eval_loss < self.best_metric:
            self.best_metric = eval_loss
            self.save_model(path=os.path.join(self.config.training.output_dir, "best_model"))
        
        return result
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def save_model(self, path=None):
        """Save model and training state."""
        if path is None:
            path = os.path.join(self.config.training.output_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        
        # Save config
        self.config.model.save_to_json(os.path.join(path, "config.json"))
        
        # Save training arguments
        with open(os.path.join(path, "training_args.json"), "w") as f:
            json.dump({k: getattr(self.config.training, k) for k in self.config.training.__dataclass_fields__}, f, indent=2)
        
        # Save optimizer and scheduler state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            },
            os.path.join(path, "optimizer.pt")
        )
        
        logger.info(f"Model saved to {path}")
        
        # Delete old checkpoints if save_total_limit is set
        if self.config.training.save_total_limit > 0:
            self._rotate_checkpoints()
    
    def _rotate_checkpoints(self):
        """Delete old checkpoints when the number of saved checkpoints exceeds save_total_limit."""
        checkpoints = [
            path for path in os.listdir(self.config.training.output_dir)
            if path.startswith("checkpoint-")
        ]
        
        if len(checkpoints) <= self.config.training.save_total_limit:
            return
        
        # Sort checkpoints by step
        checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        
        # Remove oldest checkpoints
        checkpoints_to_delete = checkpoints_sorted[:-self.config.training.save_total_limit]
        for checkpoint in checkpoints_to_delete:
            path = os.path.join(self.config.training.output_dir, checkpoint)
            logger.info(f"Deleting older checkpoint: {path}")
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            os.rmdir(path)
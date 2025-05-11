# lmtraining/training/trainer.py
import os
import time
import math
import json
import logging
import csv
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
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
        
        # Setup training metrics tracking
        self.metrics_file = os.path.join(config.training.output_dir, "training_metrics.csv")
        self.training_metrics = []
        self.setup_metrics_file()
        
        # Save initial configuration
        self.save_config()
        
    def setup_metrics_file(self):
        """Create metrics file with header row if it doesn't exist."""
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'step', 'learning_rate', 'train_loss', 
                    'eval_loss', 'eval_perplexity', 'memory_used_mb'
                ])
            logger.info(f"Created metrics file at {self.metrics_file}")
        
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
                # Check if dataset is IterableDataset - it has no __len__
                if isinstance(self.train_dataloader.dataset, IterableDataset):
                    # For streaming datasets, we estimate using a default size
                    # This is just for the scheduler, so precision isn't critical
                    logger.info("Using estimated dataset size for scheduler as dataloader uses IterableDataset")
                    estimated_dataset_size = 100000  # A reasonable estimate for WikiText
                    steps_per_epoch = estimated_dataset_size // self.config.training.train_batch_size
                    t_total = steps_per_epoch * self.config.training.num_train_epochs
                else:
                    # For regular datasets, use the length
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
    
    def log_metrics(self, metrics):
        """Log metrics to CSV file."""
        # Add timestamp and memory usage to metrics
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if torch.cuda.is_available():
            metrics['memory_used_mb'] = torch.cuda.memory_allocated(0) / (1024 * 1024)
        else:
            metrics['memory_used_mb'] = 0
            
        # Store in memory
        self.training_metrics.append(metrics)
        
        # Write to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                metrics.get('timestamp', ''),
                metrics.get('epoch', ''),
                metrics.get('step', ''),
                metrics.get('learning_rate', ''),
                metrics.get('train_loss', ''),
                metrics.get('eval_loss', ''),
                metrics.get('eval_perplexity', ''),
                metrics.get('memory_used_mb', '')
            ]
            writer.writerow(row)
            
        # Also save as JSON for easy parsing
        json_file = os.path.join(os.path.dirname(self.metrics_file), "training_metrics.json")
        with open(json_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
    
    def save_config(self):
        """Save full config as JSON for reference."""
        config_path = os.path.join(self.config.training.output_dir, "full_config.json")
        config_dict = {
            "model": {k: getattr(self.config.model, k) for k in self.config.model.__dataclass_fields__},
            "training": {k: getattr(self.config.training, k) for k in self.config.training.__dataclass_fields__},
            "data": {k: getattr(self.config.data, k) for k in self.config.data.__dataclass_fields__},
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Full configuration saved to {config_path}")
    
    def train(self):
        """Main training loop."""
        # Calculate total number of training steps
        if self.config.training.max_steps > 0:
            t_total = self.config.training.max_steps
            
            # For streaming datasets with no len() support
            if isinstance(self.train_dataloader.dataset, IterableDataset):
                num_train_epochs = None  # We'll just track steps, not epochs
            else:
                num_train_epochs = math.ceil(
                    self.config.training.max_steps / len(self.train_dataloader) * 
                    self.config.training.gradient_accumulation_steps
                )
        else:
            # For streaming datasets with no len() support
            if isinstance(self.train_dataloader.dataset, IterableDataset):
                logger.info("Using streaming dataset - will train for the specified number of epochs")
                # We'll define a step limit per epoch to avoid infinite iterations
                steps_per_epoch = 5000  # A reasonable default for one epoch through WikiText
                t_total = steps_per_epoch * self.config.training.num_train_epochs
            else:
                t_total = len(self.train_dataloader) * self.config.training.num_train_epochs
                t_total = math.ceil(t_total / self.config.training.gradient_accumulation_steps)
                
            num_train_epochs = self.config.training.num_train_epochs
        
        logger.info(f"***** Running training *****")
        if isinstance(self.train_dataloader.dataset, IterableDataset):
            logger.info(f"  Using streaming dataset (no fixed size)")
        else:
            logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
            
        logger.info(f"  Num Epochs = {num_train_epochs if num_train_epochs else 'tracking by steps'}")
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
        
        # Log initial metrics
        self.log_metrics({
            'epoch': 0, 
            'step': 0,
            'learning_rate': self.get_lr(),
            'train_loss': 0.0,
        })
        
        train_iterator = range(int(num_train_epochs)) if num_train_epochs else range(1000)  # Large range for step-based stopping
        
        for _ in train_iterator:
            self.epoch = epochs_trained + _ + 1
            
            # For streaming datasets, we need to handle epoch differently
            if isinstance(self.train_dataloader.dataset, IterableDataset):
                logger.info(f"Starting epoch {self.epoch} (streaming dataset)")
                steps_in_epoch = 0
                max_steps_in_epoch = 5000 if self.config.training.max_steps <= 0 else t_total // len(train_iterator)
                
                # Use tqdm with an estimated range
                epoch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}", total=max_steps_in_epoch)
                
                # We'll break manually after max_steps_in_epoch for streaming datasets
                epoch_done = False
                
                for batch in epoch_iterator:
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
                    if (steps_in_epoch + 1) % self.config.training.gradient_accumulation_steps == 0:
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
                                "epoch": self.epoch,
                                "step": self.global_step,
                                "learning_rate": self.get_lr(),
                                "train_loss": tr_loss / self.global_step,
                            }
                            logger.info(f"Training: {logs}")
                            self.log_metrics(logs)
                        
                        # Evaluate if needed
                        if (self.config.training.evaluation_strategy == "steps" and 
                            self.global_step % self.config.training.eval_steps == 0):
                            eval_results = self.evaluate()
                            if eval_results:
                                self.log_metrics({
                                    'epoch': self.epoch,
                                    'step': self.global_step,
                                    'learning_rate': self.get_lr(),
                                    'train_loss': tr_loss / self.global_step,
                                    'eval_loss': eval_results['loss'],
                                    'eval_perplexity': eval_results['perplexity'],
                                })
                            self.model.train()
                        
                        # Save checkpoint
                        if self.global_step % self.config.training.save_steps == 0:
                            self.save_model()
                    
                    # Increment step counter
                    steps_in_epoch += 1
                    
                    # Break if max steps reached
                    if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                        epoch_done = True
                        break
                    
                    # Break if we've done enough steps for this epoch
                    if steps_in_epoch >= max_steps_in_epoch:
                        break
                
                # Skip regular epoch handling if we're done
                if epoch_done:
                    break
            else:
                # Regular dataset with __len__ support
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
                                "epoch": self.epoch,
                                "step": self.global_step,
                                "learning_rate": self.get_lr(),
                                "train_loss": tr_loss / self.global_step,
                            }
                            logger.info(f"Training: {logs}")
                            self.log_metrics(logs)
                        
                        # Evaluate if needed
                        if (self.config.training.evaluation_strategy == "steps" and 
                            self.global_step % self.config.training.eval_steps == 0):
                            eval_results = self.evaluate()
                            if eval_results:
                                self.log_metrics({
                                    'epoch': self.epoch,
                                    'step': self.global_step,
                                    'learning_rate': self.get_lr(),
                                    'train_loss': tr_loss / self.global_step,
                                    'eval_loss': eval_results['loss'],
                                    'eval_perplexity': eval_results['perplexity'],
                                })
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
                eval_results = self.evaluate()
                if eval_results:
                    self.log_metrics({
                        'epoch': self.epoch,
                        'step': self.global_step,
                        'learning_rate': self.get_lr(),
                        'train_loss': tr_loss / self.global_step,
                        'eval_loss': eval_results['loss'],
                        'eval_perplexity': eval_results['perplexity'],
                    })
                self.model.train()
            
            # Break if max steps reached
            if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                break
        
        # Save final model
        self.save_model()
        
        # Log final metrics
        final_logs = {
            'epoch': self.epoch,
            'step': self.global_step,
            'learning_rate': self.get_lr(),
            'train_loss': tr_loss / self.global_step if self.global_step > 0 else 0,
        }
        logger.info(f"Final training metrics: {final_logs}")
        self.log_metrics(final_logs)
        
        return tr_loss / self.global_step if self.global_step > 0 else 0
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided, skipping evaluation.")
            return None
        
        logger.info(f"***** Running evaluation *****")
        if isinstance(self.eval_dataloader.dataset, IterableDataset):
            logger.info(f"  Using streaming dataset for evaluation")
            # Set a reasonable maximum number of steps for evaluation
            max_eval_steps = 500  # Limit evaluation on streaming datasets
        else:
            logger.info(f"  Num examples = {len(self.eval_dataloader.dataset)}")
            max_eval_steps = None
        
        logger.info(f"  Batch size = {self.config.training.eval_batch_size}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        eval_loss = 0.0
        eval_steps = 0
        
        eval_iterator = tqdm(self.eval_dataloader, desc="Evaluating")
        for batch in eval_iterator:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                eval_loss += loss.mean().item()
            
            eval_steps += 1
            
            # Limit evaluation steps for streaming datasets
            if max_eval_steps is not None and eval_steps >= max_eval_steps:
                break
        
        if eval_steps == 0:
            logger.warning("No evaluation steps performed! Check your eval dataset.")
            return None
            
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
        
        # Save training metrics
        if self.training_metrics:
            with open(os.path.join(path, "metrics.json"), "w") as f:
                json.dump(self.training_metrics, f, indent=2)
        
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
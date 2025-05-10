# lmtraining/config.py
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
import json
import os


@dataclass
class ModelConfig:
    """Configuration for the transformer model architecture."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    layer_norm_eps: float = 1e-12
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_file: str) -> "ModelConfig":
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    def save_to_json(self, json_file: str) -> None:
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    output_dir: str = "outputs"
    seed: int = 42
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1  # Override num_train_epochs if > 0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    logging_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 5
    evaluation_strategy: str = "epoch"  # "steps" or "epoch" or "no"
    eval_steps: int = 500
    use_amp: bool = True  # Automatic Mixed Precision
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_file: str) -> "TrainingConfig":
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 1024
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    pad_to_max_length: bool = True
    mlm: bool = False  # Masked language modeling objective
    mlm_probability: float = 0.15
    clm: bool = True  # Causal language modeling
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    """Main configuration combining model, training, and data configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def from_json(cls, json_file: str) -> "Config":
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig.from_dict(config_dict.get("model", {}))
        training_config = TrainingConfig.from_dict(config_dict.get("training", {}))
        data_config = DataConfig.from_dict(config_dict.get("data", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config
        )
    
    def save_to_json(self, json_file: str) -> None:
        config_dict = {
            "model": self.model.to_dict(),
            "training": {k: getattr(self.training, k) for k in self.training.__dataclass_fields__},
            "data": {k: getattr(self.data, k) for k in self.data.__dataclass_fields__}
        }
        
        with open(json_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
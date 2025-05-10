# lmtraining/modeling/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .embedding import TransformerEmbeddings


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout_prob
        )
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.feedforward = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_dropout_prob,
            config.hidden_act
        )
        self.feedforward_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        if output_attentions:
            attention_output, attention_probs = attention_output
        
        # First residual connection
        hidden_states = self.attention_layernorm(hidden_states + attention_output)
        
        # Feed-forward network with residual connection and layer norm
        ff_output = self.feedforward(hidden_states)
        hidden_states = self.feedforward_layernorm(hidden_states + ff_output)
        
        if output_attentions:
            return hidden_states, attention_probs
        return hidden_states


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = TransformerEmbeddings(config)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.word_embeddings.embedding.weight
            
        self._init_weights()
            
    def _init_weights(self):
        # Initialize non-embedding weights that haven't been initialized elsewhere
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        
        # Prepare attention mask
        if attention_mask is not None:
            # Extend attention mask [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert mask to float and apply scaling to make 0 = "not masked" and -inf = "masked"
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Initialize outputs
        hidden_states = embedding_output
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Forward pass through transformer blocks
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            if output_attentions:
                layer_output, attention_probs = block(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    output_attentions=True
                )
                all_attentions.append(attention_probs)
                hidden_states = layer_output
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=extended_attention_mask
                )
        
        # Add last hidden state to list if needed
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Apply LM head for prediction
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (lm_logits, hidden_states)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
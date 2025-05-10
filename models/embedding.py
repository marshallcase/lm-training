# lmtraining/modeling/embedding.py
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        
        # Initialize weights to approximate normal distribution
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = self.dropout(position_embeddings)
        return position_embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (from the original Transformer paper)."""
    def __init__(self, max_position_embeddings, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Create sinusoidal position embeddings
        position = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_position_embeddings, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        pos_embeddings = self.pe[:seq_len, :]
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)
        x = x + pos_embeddings
        return self.dropout(x)


class TransformerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = TokenEmbedding(config.vocab_size, config.hidden_size)
        
        # Choose between learned or sinusoidal positional embeddings
        use_sinusoidal = getattr(config, 'use_sinusoidal_pos_emb', False)
        if use_sinusoidal:
            self.position_embeddings = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, 
                config.hidden_size, 
                config.hidden_dropout_prob
            )
        else:
            self.position_embeddings = PositionalEmbedding(
                config.max_position_embeddings, 
                config.hidden_size, 
                config.hidden_dropout_prob
            )
            
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register buffer for position ids
        position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.register_buffer('position_ids', position_ids)
        
    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if isinstance(self.position_embeddings, SinusoidalPositionalEmbedding):
            embeddings = self.position_embeddings(inputs_embeds)
        else:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = inputs_embeds + position_embeddings
        
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
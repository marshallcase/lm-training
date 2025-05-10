# lmtraining/modeling/feedforward.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation_string):
    """Get activation function by name."""
    if activation_string == "relu":
        return F.relu
    elif activation_string == "gelu":
        return F.gelu
    elif activation_string == "swish" or activation_string == "silu":
        return F.silu
    elif activation_string == "glu":
        return F.glu
    elif activation_string == "leaky_relu":
        return F.leaky_relu
    else:
        raise ValueError(f"Unsupported activation: {activation_string}")


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1, hidden_act="gelu"):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = get_activation(hidden_act)
        
        # Initialize weights
        nn.init.normal_(self.dense1.weight, std=0.02)
        nn.init.normal_(self.dense2.weight, std=0.02)
    
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
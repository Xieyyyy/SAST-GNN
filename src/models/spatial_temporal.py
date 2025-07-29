"""
Spatial-Temporal models for traffic prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


from .gat_dgl import GraphAttentionNetwork as DGLGraphAttentionNetwork


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network (GAT) implementation using PyTorch Geometric."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.5,
        alpha: float = 0.2,
        n_heads: int = 8,
        n_hidden: int = 64
    ):
        """
        Initialize GAT using DGL-based implementation.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            n_heads: Number of attention heads
            n_hidden: Hidden layer dimension
        """
        super().__init__()
        self.dgl_gat = DGLGraphAttentionNetwork(
            in_features=in_features,
            out_features=out_features,
            dropout=dropout,
            alpha=alpha,
            n_heads=n_heads,
            n_hidden=n_hidden
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using DGL implementation.
        
        Args:
            x: Input features [batch_size, n_nodes, n_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Output features [batch_size, n_nodes, out_features]
        """
        return self.dgl_gat(x, adj)




class TransformerBlock(nn.Module):
    """Transformer block for temporal modeling."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence [batch_size, src_len, d_model]
            tgt: Target sequence [batch_size, tgt_len, d_model]
            
        Returns:
            Output sequence [batch_size, tgt_len, d_model]
        """
        return self.transformer(src, tgt)


class LSTMBlock(nn.Module):
    """LSTM block for temporal modeling."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM block.
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            
        Returns:
            output: Output sequence [batch_size, seq_len, hidden_size * num_directions]
            hidden: Hidden state tuple
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        
        if x.is_cuda:
            h0 = h0.to(x.device)
            c0 = c0.to(x.device)
        
        output, hidden = self.lstm(x, (h0, c0))
        return output, hidden
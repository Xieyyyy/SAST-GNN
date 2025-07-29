"""
GAT implementation using PyTorch Geometric for improved efficiency.
Maintains the same interface as the original implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from typing import Optional


class GATLayer(nn.Module):
    """
    Graph Attention Layer using PyTorch Geometric.
    
    This implementation uses GATConv from PyTorch Geometric to provide
    the same functionality as the original implementation while being
    more efficient and well-tested.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.5,
        alpha: float = 0.2,
        concat: bool = True,
        heads: int = 1
    ):
        """
        Initialize GAT layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
            alpha: LeakyReLU negative slope (used as negative_slope in GATConv)
            concat: Whether to concatenate heads (for multi-head attention)
            heads: Number of attention heads
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Use PyTorch Geometric's GATConv
        self.gat_conv = pyg_nn.GATConv(
            in_channels=in_features,
            out_channels=out_features,
            heads=heads,
            dropout=dropout,
            negative_slope=alpha,
            concat=concat,
            bias=True
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, n_nodes, n_features] or [n_nodes, n_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Output features [batch_size, n_nodes, out_features] or [n_nodes, out_features]
        """
        # Handle both batch and single graph cases
        if len(x.shape) == 3:
            # Batch processing: [batch_size, n_nodes, n_features]
            batch_size, n_nodes, features = x.shape
            
            # Create edge indices from adjacency matrix
            edge_index = adj.nonzero().t().contiguous()
            
            # Process each sample in the batch
            outputs = []
            for i in range(batch_size):
                sample_x = x[i]  # [n_nodes, n_features]
                out = self.gat_conv(sample_x, edge_index)
                outputs.append(out)
            
            return torch.stack(outputs, dim=0)
        else:
            # Single graph: [n_nodes, n_features]
            edge_index = adj.nonzero().t().contiguous()
            return self.gat_conv(x, edge_index)


class GraphAttentionNetwork(nn.Module):
    """
    Multi-head Graph Attention Network using PyTorch Geometric.
    
    This replaces the original GAT implementation with PyTorch Geometric
    while maintaining the same interface.
    """
    
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
        Initialize GAT.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            n_heads: Number of attention heads
            n_hidden: Hidden layer dimension
        """
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        
        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GATLayer(in_features, n_hidden, dropout=dropout, alpha=alpha, heads=1, concat=True)
            for _ in range(n_heads)
        ])
        
        # Output attention layer
        self.out_attn = GATLayer(
            n_hidden * n_heads, 
            out_features, 
            dropout=dropout, 
            alpha=alpha, 
            concat=False,
            heads=1
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, n_nodes, n_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Output features [batch_size, n_nodes, out_features]
        """
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Apply multi-head attention
        attention_outputs = []
        for attention_head in self.attentions:
            attention_outputs.append(attention_head(x, adj))
        
        x = torch.cat(attention_outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Output attention
        x = self.out_attn(x, adj)
        x = F.elu(x)
        
        return F.log_softmax(x, dim=-1)
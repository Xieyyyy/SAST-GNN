"""
SAST-GNN: Spatial-Temporal Graph Neural Network for traffic prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from .spatial_temporal import GraphAttentionNetwork, TransformerBlock, LSTMBlock
from ..utils.config import Config

logger = logging.getLogger(__name__)


class SASTGNN(nn.Module):
    """SAST-GNN model for traffic prediction."""
    
    def __init__(self, config: Config):
        """
        Initialize SAST-GNN model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(
            config.data.input_dim, 
            config.model.upper_temporal['hidden_dim']
        )
        
        # Model selection based on configuration
        temporal_module = config.model.temporal_module.lower()
        spatial_module = config.model.spatial_module.lower()
        
        if temporal_module == "lstm" and spatial_module == "gat":
            self.block = GATLSTMBlock(config)
        elif temporal_module == "transformer" and spatial_module == "gat":
            self.block = GATTransformerBlock(config)
            self.target_projection = nn.Linear(
                config.data.input_dim,
                config.model.upper_temporal['hidden_dim']
            )
        else:
            raise ValueError(
                f"Unsupported model combination: {temporal_module} + {spatial_module}"
            )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.model.lower_temporal['features'],
            1
        )
        
        self.dropout = nn.Dropout(config.model.dropout)
        
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None, 
                adjacency: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n_history, n_nodes, input_dim]
            target: Target tensor for transformer [batch_size, n_prediction, n_nodes, input_dim]
            adjacency: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Output prediction [batch_size, n_prediction, n_nodes, 1]
        """
        batch_size, n_history, n_nodes, _ = x.shape
        
        # Project input to higher dimension
        x = self.input_projection(x)  # [B, T, N, D]
        
        # Apply spatial-temporal block
        if self.config.model.temporal_module.lower() == "transformer":
            if target is None:
                raise ValueError("Target is required for transformer model")
            target = self.target_projection(target)
            output = self.block(x, target, adjacency)
        else:
            output = self.block(x, adjacency)
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output


class GATLSTMBlock(nn.Module):
    """GAT-LSTM spatial-temporal block."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Upper LSTM
        self.upper_lstm = LSTMBlock(
            input_size=config.model.upper_temporal['hidden_dim'],
            hidden_size=config.model.upper_temporal['features'],
            num_layers=config.model.upper_temporal['num_layers'],
            dropout=config.model.dropout
        )
        
        # Residual connection 1
        self.residual_conv = nn.Conv2d(
            in_channels=config.data.n_history,
            out_channels=config.data.n_prediction,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )
        
        # GAT layer
        self.gat = GraphAttentionNetwork(
            in_features=config.model.spatial['in_features'],
            out_features=config.model.spatial['out_features'],
            dropout=config.model.dropout,
            alpha=config.model.spatial['alpha'],
            n_heads=config.model.spatial['heads'],
            n_hidden=config.model.spatial['hidden_dim']
        )
        
        # Residual connection 2
        self.residual_linear = nn.Linear(
            config.model.spatial['in_features'],
            config.model.spatial['out_features']
        )
        
        # Lower LSTM
        self.lower_lstm = nn.LSTM(
            input_size=config.model.lower_temporal['features'],
            hidden_size=config.model.lower_temporal['hidden_dim'],
            num_layers=config.model.lower_temporal['num_layers'],
            dropout=config.model.dropout,
            batch_first=True
        )
        
        # Initialize LSTM hidden states
        self.register_parameter(
            'hidden_state_lower',
            nn.Parameter(torch.zeros(
                config.model.lower_temporal['num_layers'],
                config.model.spatial['out_features'],
                config.model.lower_temporal['hidden_dim']
            ))
        )
        
        self.register_parameter(
            'cell_state_lower',
            nn.Parameter(torch.zeros(
                config.model.lower_temporal['num_layers'],
                config.model.spatial['out_features'],
                config.model.lower_temporal['hidden_dim']
            ))
        )
        
        nn.init.xavier_uniform_(self.hidden_state_lower.data, gain=1.414)
        nn.init.xavier_uniform_(self.cell_state_lower.data, gain=1.414)
    
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT-LSTM block."""
        residual = x
        
        # Upper LSTM
        batch_size, seq_len, n_nodes, features = x.shape
        x_lstm = x.reshape(batch_size * n_nodes, seq_len, features)
        lstm_out, _ = self.upper_lstm(x_lstm)
        lstm_out = lstm_out.reshape(batch_size, seq_len, n_nodes, -1)
        
        lstm_out = lstm_out + residual
        lstm_out = self.residual_conv(lstm_out)
        lstm_out = F.dropout(lstm_out, self.config.model.dropout, training=self.training)
        
        # GAT layer
        residual_gat = lstm_out
        batch_size, seq_len, n_nodes, features = lstm_out.shape
        
        gat_outputs = []
        for seq_idx in range(seq_len):
            seq_data = lstm_out[:, seq_idx, :, :]
            gat_out = self.gat(seq_data, adjacency)
            gat_outputs.append(gat_out)
        
        gat_out = torch.stack(gat_outputs, dim=1)
        gat_out = gat_out + self.residual_linear(residual_gat)
        gat_out = F.dropout(gat_out, self.config.model.dropout, training=self.training)
        
        # Lower LSTM
        batch_size, seq_len, n_nodes, features = gat_out.shape
        x_lstm_lower = gat_out.reshape(batch_size * n_nodes, seq_len, features)
        
        h0 = self.hidden_state_lower.unsqueeze(1).repeat(1, batch_size * n_nodes, 1)
        c0 = self.cell_state_lower.unsqueeze(1).repeat(1, batch_size * n_nodes, 1)
        
        lstm_lower_out, _ = self.lower_lstm(x_lstm_lower, (h0, c0))
        lstm_lower_out = lstm_lower_out.reshape(batch_size, seq_len, n_nodes, -1)
        
        lstm_lower_out = lstm_lower_out + gat_out
        lstm_lower_out = F.dropout(lstm_lower_out, self.config.model.dropout, training=self.training)
        
        return lstm_lower_out


class GATTransformerBlock(nn.Module):
    """GAT-Transformer spatial-temporal block."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Upper Transformer
        self.upper_transformer = TransformerBlock(
            d_model=config.model.upper_temporal['features'],
            n_heads=config.model.upper_temporal['heads'],
            n_encoder_layers=config.model.upper_temporal['enc_layers'],
            n_decoder_layers=config.model.upper_temporal['dec_layers'],
            d_ff=config.model.upper_temporal['ff_dim'],
            dropout=config.model.dropout
        )
        
        # Residual connection 1
        self.residual_conv = nn.Conv2d(
            in_channels=config.data.n_history,
            out_channels=config.data.n_prediction,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )
        
        # Target projection
        self.target_projection = nn.Linear(
            config.model.upper_temporal['features'],
            config.model.spatial['out_features']
        )
        
        # GAT layer
        self.gat = GraphAttentionNetwork(
            in_features=config.model.spatial['in_features'],
            out_features=config.model.spatial['out_features'],
            dropout=config.model.dropout,
            alpha=config.model.spatial['alpha'],
            n_heads=config.model.spatial['heads'],
            n_hidden=config.model.spatial['hidden_dim']
        )
        
        # Residual connection 2
        self.residual_linear = nn.Linear(
            config.model.spatial['in_features'],
            config.model.spatial['out_features']
        )
        
        # Lower Transformer
        self.lower_transformer = TransformerBlock(
            d_model=config.model.lower_temporal['features'],
            n_heads=config.model.lower_temporal['heads'],
            n_encoder_layers=config.model.lower_temporal['enc_layers'],
            n_decoder_layers=config.model.lower_temporal['dec_layers'],
            d_ff=config.model.lower_temporal['ff_dim'],
            dropout=config.model.dropout
        )
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, 
                adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT-Transformer block."""
        residual = x
        
        # Upper Transformer
        batch_size, seq_len, n_nodes, features = x.shape
        
        # Reshape for transformer processing
        x_transformer = x.reshape(batch_size * n_nodes, seq_len, features)
        target_transformer = target.reshape(batch_size * n_nodes, target.size(1), features)
        
        transformer_out = self.upper_transformer(x_transformer, target_transformer)
        transformer_out = transformer_out.reshape(batch_size, seq_len, n_nodes, -1)
        
        transformer_out = transformer_out + self.residual_conv(residual)
        
        # GAT layer
        residual_gat = transformer_out
        batch_size, seq_len, n_nodes, features = transformer_out.shape
        
        gat_outputs = []
        for seq_idx in range(seq_len):
            seq_data = transformer_out[:, seq_idx, :, :]
            gat_out = self.gat(seq_data, adjacency)
            gat_outputs.append(gat_out)
        
        gat_out = torch.stack(gat_outputs, dim=1)
        gat_out = gat_out + self.residual_linear(residual_gat)
        gat_out = F.dropout(gat_out, self.config.model.dropout, training=self.training)
        
        # Lower Transformer
        target_lower = self.target_projection(target)
        
        # Reshape for lower transformer
        gat_out_reshaped = gat_out.reshape(batch_size * n_nodes, seq_len, -1)
        target_lower_reshaped = target_lower.reshape(batch_size * n_nodes, target.size(1), -1)
        
        transformer_lower_out = self.lower_transformer(gat_out_reshaped, target_lower_reshaped)
        transformer_lower_out = transformer_lower_out.reshape(batch_size, seq_len, n_nodes, -1)
        
        transformer_lower_out = transformer_lower_out + gat_out
        transformer_lower_out = F.dropout(transformer_lower_out, self.config.model.dropout, training=self.training)
        
        return transformer_lower_out
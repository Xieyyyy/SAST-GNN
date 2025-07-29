"""
Configuration management for SAST-GNN.
"""

import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    node_num: int
    data_path: str
    adj_path: str


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int
    learning_rate: float
    device: str
    batch_size: Dict[str, int]


@dataclass
class DataConfig:
    """Data configuration."""
    n_train: int
    n_val: int
    n_test: int
    n_history: int
    n_prediction: int
    input_dim: int
    day_slot: int


@dataclass
class ModelConfig:
    """Model configuration."""
    temporal_module: str
    spatial_module: str
    dropout: float
    upper_temporal: Dict[str, Any]
    spatial: Dict[str, Any]
    lower_temporal: Dict[str, Any]


@dataclass
class PathConfig:
    """Path configuration."""
    model_save_path: str
    log_path: str
    record_path: str


class Config:
    """Configuration manager for SAST-GNN."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "config", "config.yaml"
            )
        
        self.config_path = Path(config_path)
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        self.dataset = DatasetConfig(**config_data['dataset'])
        self.training = TrainingConfig(**config_data['training'])
        self.data = DataConfig(**config_data['data'])
        self.model = ModelConfig(**config_data['model'])
        self.paths = PathConfig(**config_data['paths'])
        
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.paths.model_save_path,
            self.paths.log_path,
            os.path.dirname(self.paths.record_path)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'dataset': self.dataset.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'paths': self.paths.__dict__
        }
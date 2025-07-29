"""
Training utilities for SAST-GNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
import json

from ..models.sast_gnn import SASTGNN
from ..data.dataset import TrafficDataset
from ..utils.config import Config
from ..utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict()
    
    def load_best_weights(self, model: nn.Module) -> None:
        """Load best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class SASTGNNTrainer:
    """Trainer for SAST-GNN model."""
    
    def __init__(self, config: Config, model: SASTGNN, 
                 device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: SAST-GNN model
            device: Device to use for training
        """
        self.config = config
        self.model = model
        self.device = device or torch.device(config.training.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and loss functions
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate
        )
        
        self.criterion_l1 = nn.L1Loss(reduction='mean')
        self.criterion_l2 = nn.MSELoss(reduction='mean')
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    def train_epoch(self, dataset: TrafficDataset, 
                   adjacency: torch.Tensor) -> float:
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset
            adjacency: Adjacency matrix
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        train_data = dataset.get_data('train')
        batches = torch.from_numpy(train_data).float().to(self.device)
        
        # Create batches
        batch_size = self.config.training.batch_size['train']
        n_samples = batches.size(0)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = batches[start_idx:end_idx]
            
            self.optimizer.zero_grad()
            
            # Prepare inputs
            x = batch[:, :self.config.data.n_history, :, :]
            target = batch[:, self.config.data.n_history-self.config.data.n_prediction:self.config.data.n_history, :, :]
            y = batch[:, self.config.data.n_history:, :, :]
            
            # Forward pass
            output = self.model(x, target, adjacency)
            
            # Calculate loss
            loss = self.criterion_l1(output, y) + torch.sqrt(self.criterion_l2(output, y))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, dataset: TrafficDataset, 
                adjacency: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            dataset: Validation dataset
            adjacency: Adjacency matrix
            
        Returns:
            Validation loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        val_data = dataset.get_data('val')
        batches = torch.from_numpy(val_data).float().to(self.device)
        
        batch_size = self.config.training.batch_size['val']
        n_samples = batches.size(0)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = batches[start_idx:end_idx]
                
                # Prepare inputs
                x = batch[:, :self.config.data.n_history, :, :]
                target = batch[:, self.config.data.n_history-self.config.data.n_prediction:self.config.data.n_history, :, :]
                y = batch[:, self.config.data.n_history:, :, :]
                
                # Forward pass
                output = self.model(x, target, adjacency)
                
                # Calculate loss
                loss = self.criterion_l1(output, y) + torch.sqrt(self.criterion_l2(output, y))
                
                total_loss += loss.item()
                n_batches += 1
                
                # Collect predictions and targets for metrics
                predictions.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Denormalize
        stats = dataset.get_stats()
        predictions_denorm = predictions * stats['std'] + stats['mean']
        targets_denorm = targets * stats['std'] + stats['mean']
        
        metrics = calculate_metrics(predictions_denorm, targets_denorm)
        
        return total_loss / n_batches, metrics
    
    def train(self, dataset: TrafficDataset, adjacency: torch.Tensor,
              save_path: Optional[str] = None) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            dataset: Complete dataset
            adjacency: Adjacency matrix
            save_path: Path to save the best model
            
        Returns:
            Training history
        """
        logger.info("Starting training...")
        logger.info(f"Model: {self.config.model.temporal_module} + {self.config.model.spatial_module}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {dataset.get_length('train')}")
        logger.info(f"Validation samples: {dataset.get_length('val')}")
        
        # Move adjacency to device
        adjacency = adjacency.to(self.device)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(dataset, adjacency)
            
            # Validate
            val_loss, val_metrics = self.validate(dataset, adjacency)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"MAPE: {val_metrics['mape']:.4f} - "
                f"MAE: {val_metrics['mae']:.4f} - "
                f"RMSE: {val_metrics['rmse']:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                self.early_stopping.load_best_weights(self.model)
                break
            
            # Save best model
            if save_path and val_loss == min(self.history['val_loss']):
                self.save_model(save_path)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self, dataset: TrafficDataset, 
                adjacency: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            dataset: Dataset containing test data
            adjacency: Adjacency matrix
            
        Returns:
            Test metrics
        """
        logger.info("Evaluating model on test data...")
        
        self.model.eval()
        test_data = dataset.get_data('test')
        batches = torch.from_numpy(test_data).float().to(self.device)
        
        predictions = []
        targets = []
        
        adjacency = adjacency.to(self.device)
        
        with torch.no_grad():
            x = batches[:, :self.config.data.n_history, :, :]
            target = batches[:, self.config.data.n_history-self.config.data.n_prediction:self.config.data.n_history, :, :]
            y = batches[:, self.config.data.n_history:, :, :]
            
            output = self.model(x, target, adjacency)
            
            predictions.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Denormalize
        stats = dataset.get_stats()
        predictions_denorm = predictions * stats['std'] + stats['mean']
        targets_denorm = targets * stats['std'] + stats['mean']
        
        metrics = calculate_metrics(predictions_denorm, targets_denorm)
        
        logger.info("Test Results:")
        logger.info(f"MAPE: {metrics['mape']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        
        return metrics
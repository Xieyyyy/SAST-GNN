"""
Evaluation metrics for traffic prediction.
"""

import numpy as np
from typing import Dict


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Dictionary containing MAPE, MAE, and RMSE
    """
    # Ensure arrays have the same shape
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    # Flatten arrays for calculation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Avoid division by zero in MAPE
    mask = target_flat != 0
    if not np.any(mask):
        mape = np.inf
    else:
        mape = np.mean(np.abs((pred_flat[mask] - target_flat[mask]) / target_flat[mask])) * 100
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
    
    return {
        'mape': float(mape),
        'mae': float(mae),
        'rmse': float(rmse)
    }


def masked_mae_loss(predictions: np.ndarray, targets: np.ndarray, null_val: float = 0.0) -> float:
    """
    Calculate masked MAE loss.
    
    Args:
        predictions: Predicted values
        targets: True values
        null_val: Value to mask
        
    Returns:
        Masked MAE
    """
    mask = targets != null_val
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs(predictions[mask] - targets[mask]))


def masked_rmse_loss(predictions: np.ndarray, targets: np.ndarray, null_val: float = 0.0) -> float:
    """
    Calculate masked RMSE loss.
    
    Args:
        predictions: Predicted values
        targets: True values
        null_val: Value to mask
        
    Returns:
        Masked RMSE
    """
    mask = targets != null_val
    if not np.any(mask):
        return 0.0
    
    return np.sqrt(np.mean((predictions[mask] - targets[mask]) ** 2))


def masked_mape_loss(predictions: np.ndarray, targets: np.ndarray, null_val: float = 0.0) -> float:
    """
    Calculate masked MAPE loss.
    
    Args:
        predictions: Predicted values
        targets: True values
        null_val: Value to mask
        
    Returns:
        Masked MAPE
    """
    mask = targets != null_val
    if not np.any(mask):
        return 0.0
    
    mask &= targets != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100
"""
Data utility functions for loading and processing traffic data.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def weight_matrix(file_path: str, sigma2: float = 0.1, epsilon: float = 0.5, 
                  scaling: bool = True) -> np.ndarray:
    """
    Load weight matrix from CSV file.
    
    Args:
        file_path: Path to CSV file containing distance matrix
        sigma2: Variance parameter for Gaussian kernel
        epsilon: Threshold parameter
        scaling: Whether to apply scaling
        
    Returns:
        Weight matrix as numpy array
    """
    try:
        dist_matrix = pd.read_csv(file_path, header=None).values
        
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        
        # Apply Gaussian kernel
        n = dist_matrix.shape[0]
        W = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                if dist_matrix[i, j] > epsilon:
                    W[i, j] = np.exp(-dist_matrix[i, j] ** 2 / sigma2)
                else:
                    W[i, j] = 0.0
        
        # Apply scaling
        if scaling:
            row_sums = np.sum(W, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            W = W / row_sums
        
        return W
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading weight matrix: {str(e)}")


def load_bay_graph(adj_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Bay dataset adjacency matrix.
    
    Args:
        adj_path: Path to adjacency matrix file
        
    Returns:
        Tuple of (sensor_ids, sensor_id_to_ind, adjacency_matrix)
    """
    import pickle
    
    try:
        with open(adj_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adjacency_matrix = pickle.load(f, encoding='latin1')
        return sensor_ids, sensor_id_to_ind, adjacency_matrix
    except Exception as e:
        raise RuntimeError(f"Error loading Bay graph: {str(e)}")


def validate_data_shape(data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
    """
    Validate data shape.
    
    Args:
        data: Input data array
        expected_shape: Expected shape tuple (None for any dimension)
        
    Returns:
        True if shape is valid
    """
    if len(data.shape) != len(expected_shape):
        return False
    
    for actual, expected in zip(data.shape, expected_shape):
        if expected is not None and actual != expected:
            return False
    
    return True
"""
Data utilities for SAST-GNN.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Iterator, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics for the dataset."""
    mean: float
    std: float


class TrafficDataset:
    """Traffic dataset class for managing training, validation, and test data."""
    
    def __init__(self, data: Dict[str, np.ndarray], stats: DatasetStats):
        """
        Initialize the dataset.
        
        Args:
            data: Dictionary containing 'train', 'val', and 'test' data
            stats: Dataset statistics (mean, std)
        """
        self._data = data
        self.stats = stats
        
    def get_data(self, data_type: str) -> np.ndarray:
        """
        Get data for a specific type.
        
        Args:
            data_type: One of 'train', 'val', or 'test'
            
        Returns:
            numpy array of data
        """
        if data_type not in self._data:
            raise ValueError(f"Invalid data type: {data_type}")
        return self._data[data_type]
    
    def get_stats(self) -> DatasetStats:
        """Get dataset statistics."""
        return self.stats
    
    def get_length(self, data_type: str) -> int:
        """
        Get length of dataset for a specific type.
        
        Args:
            data_type: One of 'train', 'val', or 'test'
            
        Returns:
            Length of dataset
        """
        return len(self._data[data_type])
    
    def inverse_z_score(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse Z-score normalization.
        
        Args:
            data: Normalized data
            
        Returns:
            Denormalized data
        """
        return data * self.stats.std + self.stats.mean


def z_score_normalization(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Apply Z-score normalization to data.
    
    Args:
        data: Input data
        mean: Mean value for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized data
    """
    return (data - mean) / std


def generate_sequences(
    length: int,
    data_sequence: np.ndarray,
    offset: int,
    n_frames: int,
    n_routes: int,
    day_slots: int,
    channel_size: int = 1
) -> np.ndarray:
    """
    Generate sequences from time series data.
    
    Args:
        length: Length of target date sequence
        data_sequence: Source time series data
        offset: Starting index for different dataset types
        n_frames: Number of frames in a sequence unit
        n_routes: Number of routes in the graph
        day_slots: Number of time slots per day
        channel_size: Size of input channel
        
    Returns:
        numpy array of shape [length * slots, n_frames, n_routes, channel_size]
    """
    n_slots = day_slots - n_frames + 1
    
    sequences = np.zeros((length * n_slots, n_frames, n_routes, channel_size))
    
    for i in range(length):
        for j in range(n_slots):
            start_idx = (i + offset) * day_slots + j
            end_idx = start_idx + n_frames
            sequences[i * n_slots + j, :, :, :] = data_sequence[start_idx:end_idx, :].reshape(
                n_frames, n_routes, channel_size
            )
    
    return sequences


def load_dataset(
    file_path: str,
    data_config: Tuple[int, int, int],
    n_routes: int,
    n_frames: int = 21,
    day_slots: int = 288,
    is_csv: bool = True
) -> TrafficDataset:
    """
    Load dataset from file.
    
    Args:
        file_path: Path to the data file
        data_config: Tuple of (n_train, n_val, n_test)
        n_routes: Number of routes in the graph
        n_frames: Number of frames in a sequence unit
        day_slots: Number of time slots per day
        is_csv: Whether the input file is CSV format
        
    Returns:
        TrafficDataset instance
    """
    n_train, n_val, n_test = data_config
    
    try:
        if is_csv:
            data_sequence = pd.read_csv(file_path, header=None).values
        else:
            # Handle other formats like HDF5
            import h5py
            with h5py.File(file_path, 'r') as f:
                data_sequence = f['data'][:]
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")
    
    # Generate sequences for each dataset
    train_sequences = generate_sequences(n_train, data_sequence, 0, n_frames, n_routes, day_slots)
    val_sequences = generate_sequences(n_val, data_sequence, n_train, n_frames, n_routes, day_slots)
    test_sequences = generate_sequences(n_test, data_sequence, n_train + n_val, n_frames, n_routes, day_slots)
    
    # Calculate statistics from training data
    train_mean = np.mean(train_sequences)
    train_std = np.std(train_sequences)
    
    # Normalize data
    train_normalized = z_score_normalization(train_sequences, train_mean, train_std)
    val_normalized = z_score_normalization(val_sequences, train_mean, train_std)
    test_normalized = z_score_normalization(test_sequences, train_mean, train_std)
    
    data = {
        'train': train_normalized,
        'val': val_normalized,
        'test': test_normalized
    }
    
    stats = DatasetStats(mean=train_mean, std=train_std)
    
    logger.info(f"Dataset loaded: train={len(train_sequences)}, val={len(val_sequences)}, test={len(test_sequences)}")
    logger.info(f"Data statistics: mean={train_mean:.4f}, std={train_std:.4f}")
    
    return TrafficDataset(data, stats)


def create_batches(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False
) -> Iterator[np.ndarray]:
    """
    Create batches from data.
    
    Args:
        data: Input data array
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        
    Yields:
        Batches of data
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        
        if end_idx > n_samples:
            if drop_last:
                break
            end_idx = n_samples
        
        batch_indices = indices[start_idx:end_idx]
        yield data[batch_indices]
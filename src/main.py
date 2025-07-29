"""
Main entry point for SAST-GNN training and evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

from .utils.config import Config
from .data.dataset import load_dataset
from .models.sast_gnn import SASTGNN
from .training.trainer import SASTGNNTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/sast_gnn.log')
        ]
    )


def load_adjacency_matrix(config: Config) -> torch.Tensor:
    """Load adjacency matrix based on dataset."""
    if config.dataset.name == "PEMS":
        adj_path = Path(config.dataset.adj_path)
        if not adj_path.exists():
            raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")
        
        # Load and process adjacency matrix
        from .utils.data_utils import weight_matrix
        adj = weight_matrix(str(adj_path))
        return torch.tensor(adj, dtype=torch.float32)
    
    elif config.dataset.name == "Bay":
        adj_path = Path(config.dataset.adj_path)
        if not adj_path.exists():
            raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")
        
        # Load Bay dataset adjacency matrix
        import pickle
        with open(adj_path, 'rb') as f:
            adj = pickle.load(f)[2]  # Assuming this is the correct index
        return torch.tensor(adj, dtype=torch.float32)
    
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='SAST-GNN Training and Evaluation')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'],
                        default='both', help='Mode to run')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save/load model')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config(args.config)
        config.create_directories()
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Dataset: {config.dataset.name}")
        logger.info(f"Model: {config.model.temporal_module} + {config.model.spatial_module}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(
            file_path=config.dataset.data_path,
            data_config=(config.data.n_train, config.data.n_val, config.data.n_test),
            n_routes=config.dataset.node_num,
            n_frames=config.data.n_history + config.data.n_prediction,
            day_slots=config.data.day_slot,
            is_csv=True
        )
        
        # Load adjacency matrix
        logger.info("Loading adjacency matrix...")
        adjacency = load_adjacency_matrix(config)
        
        # Create model
        logger.info("Creating model...")
        model = SASTGNN(config)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        trainer = SASTGNNTrainer(config, model)
        
        # Determine model save path
        if args.model_path is None:
            model_name = f"sast_gnn_{config.model.temporal_module}_{config.model.spatial_module}.pth"
            model_path = Path(config.paths.model_save_path) / model_name
        else:
            model_path = Path(args.model_path)
        
        # Training
        if args.mode in ['train', 'both']:
            logger.info("Starting training...")
            history = trainer.train(dataset, adjacency, str(model_path))
            
            # Save training history
            history_path = model_path.parent / f"{model_path.stem}_history.json"
            with open(history_path, 'w') as f:
                import json
                json.dump(history, f, indent=2)
            
            logger.info(f"Training completed. Model saved to {model_path}")
        
        # Evaluation
        if args.mode in ['eval', 'both']:
            if not model_path.exists() and args.mode == 'eval':
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            if model_path.exists():
                logger.info("Loading trained model...")
                trainer.load_model(str(model_path))
            
            logger.info("Evaluating on test data...")
            test_metrics = trainer.evaluate(dataset, adjacency)
            
            # Save evaluation results
            eval_path = model_path.parent / f"{model_path.stem}_evaluation.json"
            with open(eval_path, 'w') as f:
                import json
                json.dump(test_metrics, f, indent=2)
            
            logger.info("Evaluation completed")
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
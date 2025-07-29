# SAST-GNN: Self-Attention based Spatio-Temporal Graph Neural Network for Traffic Flow Prediction (DASFAA 2020)

## Paper Link: https://dl.acm.org/doi/abs/10.1007/978-3-030-59410-7_4

## 🚦 Project Overview

SAST-GNN is a high-precision traffic flow prediction system based on spatial-temporal graph neural networks. This project combines the advantages of Transformer and Graph Attention Network (GAT) to accurately predict future traffic flow in urban transportation networks.

### Core Features
- **Spatial Modeling**: Uses GAT to capture spatial correlations in road networks
- **Temporal Modeling**: Uses Transformer to capture temporal dependencies in traffic patterns
- **End-to-End**: Complete solution from data preprocessing to model deployment
- **High Performance**: Supports GPU acceleration, scalable to large-scale traffic networks

## 📁 Project Structure

```
SAST-GNN/
├── src/                          # Source code directory
│   ├── __init__.py
│   ├── main.py                  # Main training script
│   ├── data/                    # Data processing module
│   │   ├── __init__.py
│   │   └── dataset.py          # Dataset loading and preprocessing
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   ├── sast_gnn.py        # SAST-GNN main model
│   │   ├── gat_dgl.py         # Graph attention network implementation
│   │   ├── spatial_temporal.py # Spatial-temporal module
│   │   └── singleTransformerBlock.py # Transformer block
│   ├── training/               # Training module
│   │   ├── __init__.py
│   │   └── trainer.py        # Trainer
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── data_utils.py     # Data utilities
│       └── metrics.py        # Evaluation metrics
├── config/                     # Configuration files
│   └── config.yaml           # Main configuration file
├── data/                      # Data directory
│   ├── V_228.csv            # Traffic flow data
│   ├── W_228.csv            # Adjacency matrix weights
│   └── PeMSD7_Full.zip      # Complete dataset
├── models/                    # Model save directory
├── logs/                      # Log directory
├── record/                    # Training records
├── requirements.txt          # Dependencies list
└── README.md                # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The project includes sample data:
- `V_228.csv`: Speed data from 228 traffic detectors
- `W_228.csv`: Adjacency matrix of traffic network
- `PeMSD7_Full.zip`: Complete PeMS dataset

For custom data, ensure:
- Flow data dimensions: [time_steps, nodes, features]
- Adjacency matrix dimensions: [nodes, nodes]
- Data format matches sample files

### 3. Model Training

#### Basic Training
```bash
# Train with default configuration
python src/main.py
```

#### Advanced Training Options
```bash
# Specify configuration file
python src/main.py --config config/custom_config.yaml

# Specify device
python src/main.py --device cuda

# Specify training/validation/test days
python src/main.py --n_train 31 --n_val 9 --n_test 4
```

### 4. Model Evaluation

After training, results will automatically display:
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **Training Time**: Total training duration

## ⚙️ Configuration Guide

### Configuration File (config/config.yaml)

```yaml
# Dataset Configuration
dataset:
  name: "PEMS"
  node_num: 228
  data_path: "../data/V_228.csv"
  adj_path: "../data/W_228.csv"

# Training Configuration
training:
  epochs: 50
  batch_size:
    train: 32
    val: 32
  learning_rate: 0.002
  device: "mps"

# Data Configuration
data:
  n_train: 31      # Training days
  n_val: 9         # Validation days
  n_test: 4        # Test days
  n_history: 12    # Historical time steps
  n_prediction: 12 # Prediction time steps
  input_dim: 1     # Input feature dimension
  day_slot: 288    # Daily time slots

# Model Architecture
model:
  temporal_module: "Transformer"
  spatial_module: "GAT"
  dropout: 0.5
  
  upper_temporal:
    hidden_dim: 512
    num_layers: 3
    features: 512
    heads: 8
    
  spatial:
    in_features: 512
    out_features: 64
    heads: 8
    
  lower_temporal:
    features: 64
    hidden_dim: 64
    num_layers: 3
```

## 📊 Data Format

### Input Data
- **Flow Data**: CSV format, each row represents a time step, each column represents a detector
- **Adjacency Matrix**: CSV format, represents connection relationships and weights in traffic network

### Output Format
Model outputs include:
- Traffic flow predictions for next 12 time steps
- Individual predictions for each detector
- Confidence intervals (optional)

## 🔧 Custom Usage

### 1. Custom Dataset

```python
from src.data.dataset import load_dataset

# Load custom data
dataset = load_dataset(
    data_path="path/to/your/data.csv",
    data_split=(train_days, val_days, test_days),
    node_num=num_sensors,
    seq_len=history_steps + prediction_steps,
    day_slot=time_slots_per_day,
    normalize=True
)
```

### 2. Model Fine-tuning

```python
from src.models.sast_gnn import SASTGNN
from src.utils.config import Config

config = Config()
config.model.temporal_module = "LSTM"  # Change to LSTM
config.model.spatial_module = "GAT"
config.training.epochs = 100

model = SASTGNN(config)
```

### 3. Predict New Data

```python
import torch
from src.models.sast_gnn import SASTGNN
from src.utils.config import Config

# Load trained model
config = Config()
model = SASTGNN(config)
model.load_state_dict(torch.load('models/sast_gnn_final.pth')['model_state_dict'])

# Prepare input data
input_data = ...  # [batch, history, nodes, features]
prediction = model(input_data, target_data, adjacency_matrix)
```


## 🎛️ Advanced Features

### 1. Multi-GPU Training
```bash
# Auto-select available GPU
python src/main.py --device auto

# Specify multiple GPUs
python src/main.py --device cuda:0,1,2,3
```

### 2. Model Checkpoints
```python
# Save checkpoints
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoints/checkpoint_epoch_{}.pth'.format(epoch))
```

### 3. Hyperparameter Tuning
```python
# Use optuna for hyperparameter optimization
python scripts/hyperparameter_tuning.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

#!/usr/bin/env python3
"""
Quick run script for SAST-GNN refactored project.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import main

if __name__ == "__main__":
    # Add src to Python path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    # Run main function
    main()
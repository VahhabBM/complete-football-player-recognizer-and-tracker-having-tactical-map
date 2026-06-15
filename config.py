"""
Configuration module for the Football Video Analysis system.

This module defines global configuration settings including:
- Device type (CUDA GPU or CPU)
- Project root directory
- Dataset location paths

These settings are used across all modules for consistent path and device management.
"""

from pathlib import Path

from ultralytics import YOLO

# GPU device for model inference (CUDA-enabled GPU)
DEVICE = "cuda"

# Project root directory for resolving relative paths
ROOT = Path(".").resolve()

# Directory containing training datasets (players, ball, pitch)
DATASET_DIR = ROOT / "dataset"

print("Project root:", ROOT)
print("Device:", DEVICE)
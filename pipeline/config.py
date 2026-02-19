"""Configuration management for TSRM imputation pipeline.

Loads configuration from .env and YAML files, merging them into a single
config dictionary for easy access throughout the pipeline.
"""

import os
from typing import Dict

import yaml
from dotenv import load_dotenv


def load_config(yaml_path: str) -> Dict:
    """Load configuration from .env and YAML file.

    Args:
        yaml_path: Path to YAML config file.

    Returns:
        Dictionary containing merged configuration from .env and YAML.
    """
    load_dotenv()

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def build_tsrm_config(yaml_cfg: Dict) -> Dict:
    """Build TSRM config dict from YAML configuration.

    Translates YAML config to the format expected by TSRM model.
    All keys are required, especially 'task' and 'phase' for Transformations.

    Args:
        yaml_cfg: Configuration dictionary loaded from YAML.

    Returns:
        Dictionary containing TSRM configuration with all required keys.
    """
    config = {
        # Data dimensions
        "feature_dimension": yaml_cfg.get("feature_dimension", 7),
        "seq_len": yaml_cfg.get("seq_len", 96),
        "pred_len": 0,  # HARDCODED - imputation, not forecasting

        # Model architecture
        "encoding_size": yaml_cfg.get("encoding_size", 16),
        "h": yaml_cfg.get("h", 4),
        "N": yaml_cfg.get("N", 3),
        "conv_dims": yaml_cfg.get("conv_dims", [[0.1, 1, -1], [0.2, 1, -1], [0.6, 1, -1]]),
        "attention_func": yaml_cfg.get("attention_func", "classic"),
        "batch_size": yaml_cfg.get("batch_size", 8),
        "dropout": yaml_cfg.get("dropout", 0.25),

        # Normalization - HARDCODED to False for external normalization only
        "revin": False,

        # Loss configuration
        "loss_function_imputation": yaml_cfg.get("loss_function_imputation", "mse+mae"),
        "loss_imputation_mode": yaml_cfg.get("loss_imputation_mode", "weighted_imputation"),
        "loss_weight_alpha": yaml_cfg.get("loss_weight_alpha", 10.0),

        # Masking - HARDCODED to 0 for external masking
        "missing_ratio": 0.0,

        # Time features
        "embed": "timeF",
        "freq": "t",

        # Mask configuration
        "mask_size": yaml_cfg.get("mask_size", 10),
        "mask_count": yaml_cfg.get("mask_count", 4),

        # REQUIRED by Transformations.__init__ at model.py:224
        "task": "imputation",
        "phase": "downstream",
    }

    return config

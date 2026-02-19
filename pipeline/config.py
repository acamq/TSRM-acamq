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
    data_cfg = yaml_cfg.get('data', {})
    tsrm_cfg = yaml_cfg.get('tsrm', {})

    config = {
        "feature_dimension": len(data_cfg.get('variables', [])),
        "seq_len": data_cfg.get('window_size', 30),
        "pred_len": 0,

        "encoding_size": tsrm_cfg.get("encoding_size", 16),
        "h": tsrm_cfg.get("h", 4),
        "N": tsrm_cfg.get("N", 3),
        "conv_dims": tsrm_cfg.get("conv_dims", [[0.1, 1, -1], [0.2, 1, -1], [0.6, 1, -1]]),
        "attention_func": tsrm_cfg.get("attention_func", "classic"),
        "batch_size": tsrm_cfg.get("batch_size", 8),
        "dropout": tsrm_cfg.get("dropout", 0.25),

        "revin": False,

        "loss_function_imputation": tsrm_cfg.get("loss_function_imputation", "mse+mae"),
        "loss_imputation_mode": tsrm_cfg.get("loss_imputation_mode", "weighted_imputation"),
        "loss_weight_alpha": tsrm_cfg.get("loss_weight_alpha", 10.0),

        "missing_ratio": 0.0,

        "embed": tsrm_cfg.get("embed", "timeF"),
        "freq": tsrm_cfg.get("freq", "t"),

        "mask_size": tsrm_cfg.get("mask_size", 10),
        "mask_count": tsrm_cfg.get("mask_count", 4),

        "task": "imputation",
        "phase": "downstream",
    }

    return config

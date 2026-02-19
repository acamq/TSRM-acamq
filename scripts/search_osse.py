"""Hyperparameter search script using ParameterGrid.

Usage:
    python scripts/search_osse.py --config configs/osse_default.yaml --folds 0,1 --epochs 10 --quick
    python scripts/search_osse.py --config configs/osse_default.yaml --folds all --epochs 50
"""

import argparse
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import yaml


def build_search_grid(quick: bool = False) -> list:
    """Build hyperparameter search grid.
    
    Args:
        quick: If True, use smaller grid (~32 configs)
               If False, use larger grid (~128 configs)
    
    Returns:
        List of config dicts to try
    """
    if quick:
        # Smaller grid for quick testing: 2^5 = 32 configs
        search_grid = {
            "attention_func": ["entmax15", "classic"],  # 2
            "N": [1, 2],                                # 2
            "h": [4, 8],                                # 2
            "encoding_size": [64, 128],                 # 2
            "dropout": [0.0, 0.1],                      # 2
            # Use defaults for: learning_rate, batch_size, conv_dims
        }
    else:
        # Full grid: 2^7 = 128 configs
        search_grid = {
            "attention_func": ["entmax15", "classic"],
            "N": [1, 2],
            "h": [4, 8],
            "encoding_size": [64, 128],
            "dropout": [0.0, 0.1],
            "learning_rate": [0.0005, 0.001],
            "batch_size": [32, 64],
            # conv_dims: use default from YAML
        }
    
    return list(ParameterGrid(search_grid))


def parse_folds(folds_str: str) -> list:
    """Parse folds argument into list of fold indices.
    
    Args:
        folds_str: Either 'all' or comma-separated fold indices (e.g., '0,1,2')
    
    Returns:
        List of fold indices
    """
    if folds_str == "all":
        return list(range(4))
    else:
        return [int(f.strip()) for f in folds_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for TSRM OSSE imputation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to base config YAML"
    )
    parser.add_argument(
        "--folds", 
        type=str, 
        default="0", 
        help="Folds to run (comma-separated or 'all')"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of epochs per config"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Use smaller search grid (~32 configs)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="search_results.csv", 
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    print(f"Loaded base config: {list(base_config.keys())[:5]}...")
    
    # Build search grid
    grid = build_search_grid(quick=args.quick)
    print(f"Search grid size: {len(grid)} configs")
    
    # Parse folds
    folds = parse_folds(args.folds)
    print(f"Running on folds: {folds}")
    print(f"Epochs per config: {args.epochs}")
    print(f"Output path: {args.output}")
    
    # Print sample configs
    print("\nSample configs (first 5):")
    for i, cfg in enumerate(grid[:5]):
        print(f"  Config {i+1}: {cfg}")
    
    print(f"\nTotal configs to evaluate: {len(grid)}")
    print(f"Total training runs: {len(grid) * len(folds)}")
    print("\nImplementation note: Training loop will be added by Task 13")


if __name__ == "__main__":
    main()

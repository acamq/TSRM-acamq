"""Hyperparameter search script using ParameterGrid.

Usage:
    python scripts/search_osse.py --config configs/osse_default.yaml --folds 0,1 --epochs 10 --quick
    python scripts/search_osse.py --config configs/osse_default.yaml --folds all --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
try:
    import torch  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    torch = None

from sklearn.model_selection import ParameterGrid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pipeline.config import load_config  # noqa: E402
from pipeline.data.loader import OSSEDataLoader  # noqa: E402
from pipeline.tracking.experiment import ExperimentTracker  # noqa: E402
from scripts.train_osse import (  # noqa: E402
    FOLD_DEFINITIONS,
    resolve_missing_settings,
    resolve_path,
    train_fold,
)


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


def extract_val_loss(fold_result: dict[str, Any]) -> float:
    """Extract validation loss from fold training result."""
    best_score = fold_result.get("best_score")
    if isinstance(best_score, (int, float)) and math.isfinite(float(best_score)):
        return float(best_score)

    metrics = fold_result.get("metrics", {})
    for key in ("loss", "val_loss"):
        metric_value = metrics.get(key)
        if isinstance(metric_value, (int, float)) and math.isfinite(float(metric_value)):
            return float(metric_value)

    raise ValueError("Unable to extract validation loss from fold result")


def build_result_row(
    config_id: int,
    run_config: dict[str, Any],
    fold_losses: list[float],
) -> dict[str, Any]:
    """Build a result row for CSV output."""
    tsrm_cfg = run_config.get("tsrm", {})
    mean_val_loss = float(np.mean(fold_losses))
    std_val_loss = float(np.std(fold_losses, ddof=0))

    return {
        "config_id": config_id,
        "attention_func": tsrm_cfg.get("attention_func"),
        "N": int(tsrm_cfg.get("N")),
        "h": int(tsrm_cfg.get("h")),
        "encoding_size": int(tsrm_cfg.get("encoding_size")),
        "dropout": float(tsrm_cfg.get("dropout")),
        "learning_rate": float(tsrm_cfg.get("learning_rate")),
        "batch_size": int(tsrm_cfg.get("batch_size")),
        "mean_val_loss": mean_val_loss,
        "std_val_loss": std_val_loss,
    }


def write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write results to CSV file."""
    fieldnames = [
        "config_id",
        "attention_func",
        "N",
        "h",
        "encoding_size",
        "dropout",
        "learning_rate",
        "batch_size",
        "mean_val_loss",
        "std_val_loss",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError(
            "torch is required for hyperparameter search. Install training dependencies first."
        )

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = load_config(str(config_path))
    print(f"Loaded base config: {list(base_config.keys())[:5]}...")
    
    # Build search grid
    grid = build_search_grid(quick=args.quick)
    print(f"Search grid size: {len(grid)} configs")
    
    # Parse folds
    folds = parse_folds(args.folds)
    invalid_folds = sorted(set(folds) - set(FOLD_DEFINITIONS.keys()))
    if invalid_folds:
        raise ValueError(f"Invalid folds requested: {invalid_folds}")

    missing_pattern, missing_rate = resolve_missing_settings(
        base_config,
        argparse.Namespace(missing_pattern=None, missing_rate=None),
    )

    paths_cfg = base_config.get("paths", {})
    data_cfg = base_config.get("data", {})
    variable_names = list(data_cfg.get("variables", OSSEDataLoader.DEFAULT_VARIABLES))
    data_dir = resolve_path(str(paths_cfg.get("data_dir", "")), fallback=".")
    loader = OSSEDataLoader(data_dir=str(data_dir), variables=variable_names)
    raw_data = loader.to_multivariate_array()

    boundaries_raw = data_cfg.get("block_boundaries", loader.get_block_boundaries())
    block_boundaries = [int(v) for v in boundaries_raw]

    scratch_dir = resolve_path(str(paths_cfg.get("scratch_dir", "./outputs")))

    print(f"Running on folds: {folds}")
    print(f"Epochs per config: {args.epochs}")
    print(f"Output path: {args.output}")
    
    # Print sample configs
    print("\nSample configs (first 5):")
    for i, cfg in enumerate(grid[:5]):
        print(f"  Config {i+1}: {cfg}")

    print(f"\nTotal configs to evaluate: {len(grid)}")
    print(f"Total training runs: {len(grid) * len(folds)}")

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    total_configs = len(grid)

    for idx, override_cfg in enumerate(grid, start=1):
        print(f"\n=== Config {idx}/{total_configs} ===")
        print(f"Overrides: {override_cfg}")

        run_config = copy.deepcopy(base_config)
        run_config.setdefault("tsrm", {})
        run_config["tsrm"].update(override_cfg)

        tracker = ExperimentTracker(
            str(scratch_dir / "experiments"),
            f"search_osse_cfg_{idx:03d}",
        )
        tracker.save_config(run_config)
        tracker.save_metadata(
            seed=int(run_config.get("masking", {}).get("seed", 42)),
            folds=folds,
            missing_pattern=missing_pattern,
            missing_rate=missing_rate,
            epochs_override=args.epochs,
            search_config_id=idx,
        )

        fold_losses: list[float] = []
        fold_summaries: list[dict[str, Any]] = []

        for fold_idx in folds:
            print(f"  Fold {fold_idx}...", end=" ", flush=True)
            try:
                fold_result = train_fold(
                    config=run_config,
                    fold_idx=fold_idx,
                    data=raw_data,
                    block_boundaries=block_boundaries,
                    variable_names=variable_names,
                    missing_pattern=missing_pattern,
                    missing_rate=missing_rate,
                    epochs_override=args.epochs,
                    tracker=tracker,
                )
                val_loss = extract_val_loss(fold_result)
                fold_losses.append(val_loss)
                fold_summaries.append({
                    "fold": fold_idx,
                    "val_loss": val_loss,
                    "status": "success",
                    "best_checkpoint": fold_result.get("best_checkpoint"),
                })
                print(f"val_loss={val_loss:.6f}")
            except Exception as e:
                print(f"FAILED: {e}")
                fold_summaries.append({
                    "fold": fold_idx,
                    "status": "failed",
                    "error": str(e),
                })

        if fold_losses:
            row = build_result_row(idx, run_config, fold_losses)
            rows.append(row)
            print(f"  Mean val_loss: {row['mean_val_loss']:.6f} ± {row['std_val_loss']:.6f}")

        details.append({
            "config_id": idx,
            "overrides": override_cfg,
            "folds": fold_summaries,
        })

    # Write CSV results
    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"\nResults written to: {output_path}")

    # Write detailed JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, default=str)
    print(f"Detailed results written to: {json_path}")

    # Print best config
    if rows:
        best = min(rows, key=lambda r: r["mean_val_loss"])
        print(f"\n=== BEST CONFIG ===")
        print(f"Config ID: {best['config_id']}")
        print(f"Mean val_loss: {best['mean_val_loss']:.6f} ± {best['std_val_loss']:.6f}")
        for key in ["attention_func", "N", "h", "encoding_size", "dropout", "learning_rate", "batch_size"]:
            if key in best:
                print(f"  {key}: {best[key]}")


if __name__ == "__main__":
    main()

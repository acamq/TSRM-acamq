#!/usr/bin/env python
"""Evaluation script for TSRM OSSE imputation.

Usage:
    python scripts/evaluate_osse.py --config configs/osse_default.yaml --experiment-dir <dir> --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import load_config, build_tsrm_config
from pipeline.data.loader import OSSEDataLoader
from pipeline.data.preprocessor import Preprocessor
import pipeline.data.preprocessor as preprocessor_module
from pipeline.data.masking import apply_missing_pattern
from pipeline.evaluation.metrics import (
    compute_metrics,
    compute_shared_eval_mask,
    compute_skill_scores_vs_baselines,
    compute_metrics_per_variable,
)
from pipeline.evaluation.baselines import locf_impute, linear_interp_impute
from architecture.tsrm_external import TSRMImputationExternal

# Module-level function references for optional dynamic dispatch
prepare_splits_block_level = getattr(preprocessor_module, "prepare_splits_block_level", None)
create_windows_3d = getattr(preprocessor_module, "create_windows_3d", None)

# Fold definitions (same as train)
FOLD_DEFINITIONS: Dict[int, Dict[str, List[int]]] = {
    0: {"train": [2, 3], "val": [1], "test": [0]},
    1: {"train": [0, 3], "val": [2], "test": [1]},
    2: {"train": [0, 1], "val": [3], "test": [2]},
    3: {"train": [1, 2], "val": [0], "test": [3]},
}


def to_builtin(value: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable builtins."""
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value.item()) if np.isfinite(value) else str(value)
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            v = float(value.item())
            return v if np.isfinite(v) else str(v)
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def split_blocks(
    preprocessor: Preprocessor,
    data: np.ndarray,
    block_boundaries: List[int],
    train_blocks: List[int],
    val_blocks: List[int],
    test_blocks: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]]:
    """Split data by blocks using preprocessor or module-level function."""
    if callable(prepare_splits_block_level):
        return cast(
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]],
            prepare_splits_block_level(
                data,
                block_boundaries,
                train_blocks,
                val_blocks,
                test_blocks,
            ),
        )
    return preprocessor.prepare_splits_block_level(
        data=data,
        block_boundaries=block_boundaries,
        train_blocks=train_blocks,
        val_blocks=val_blocks,
        test_blocks=test_blocks,
    )


def make_windows(
    preprocessor: Preprocessor,
    data_list: List[np.ndarray],
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Create windows from data list."""
    if callable(create_windows_3d):
        return np.asarray(create_windows_3d(data_list, window_size, stride))
    return np.asarray(
        preprocessor.create_windows_3d(
            data_list=data_list,
            window_size=window_size,
            stride=stride,
        )
    )


def evaluate_fold(
    config: Dict[str, Any],
    fold_idx: int,
    data: np.ndarray,
    block_boundaries: List[int],
    variable_names: List[str],
    missing_pattern: str,
    missing_rate: float,
    checkpoint_path: Path,
    scaler_path: Path,
) -> Dict[str, Any]:
    """Evaluate a single fold.

    Args:
        config: Configuration dictionary
        fold_idx: Fold index (0-3)
        data: Raw data array [N_sat, T, F]
        block_boundaries: Block boundary indices
        variable_names: List of variable names
        missing_pattern: Missing pattern ('point', 'subseq', 'block')
        missing_rate: Missing rate (0.0-1.0)
        checkpoint_path: Path to model checkpoint
        scaler_path: Path to saved scaler

    Returns:
        Dictionary containing evaluation results
    """
    print(f"\nEvaluating Fold {fold_idx}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Scaler: {scaler_path}")

    split = FOLD_DEFINITIONS[fold_idx]
    data_cfg = config.get("data", {})
    mask_cfg = config.get("masking", {})

    window_size = int(data_cfg.get("window_size", 30))
    eval_stride = int(data_cfg.get("window_stride_eval", window_size))
    fold_seed = int(mask_cfg.get("seed", 42)) + (fold_idx * 1000)

    # Initialize preprocessor and load scaler
    preprocessor = Preprocessor(config)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    preprocessor.load_scalers(scaler_path)

    # Set variable names for density var identification
    preprocessor.set_variable_names(variable_names)

    # Split data by blocks
    split_result = split_blocks(
        preprocessor=preprocessor,
        data=data,
        block_boundaries=block_boundaries,
        train_blocks=split["train"],
        val_blocks=split["val"],
        test_blocks=split["test"],
    )
    test_blocks_data, _, _, _, _, _ = cast(
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]],
        (split_result[2], split_result[0], split_result[1], split_result[3], split_result[4], split_result[5]),
    )
    test_blocks_data = split_result[2]  # test is at index 2

    if not test_blocks_data:
        raise ValueError(f"Fold {fold_idx} produced empty test split")

    # Transform test data
    test_scaled = [preprocessor.transform(block) for block in test_blocks_data]

    # Create windows
    test_windows = make_windows(preprocessor, test_scaled, window_size, eval_stride)

    if test_windows.size == 0:
        raise ValueError(f"Fold {fold_idx} produced empty test windows")

    print(f"  Test windows shape: {test_windows.shape}")

    # Store original (ground truth)
    test_original = test_windows.copy()

    # Apply masking
    test_masked, art_mask, realized_rate = apply_missing_pattern(
        data=test_windows,
        pattern=missing_pattern,
        missing_rate=missing_rate,
        seed=fold_seed + 900,
    )

    print(f"  Realized missing rate: {realized_rate:.4f}")
    print(f"  Artificially masked positions: {art_mask.sum()}")

    # Load model
    tsrm_cfg = build_tsrm_config(config)

    if not checkpoint_path.exists():
        # Try to find best.ckpt or last.ckpt
        parent_dir = checkpoint_path.parent
        if (parent_dir / "best.ckpt").exists():
            checkpoint_path = parent_dir / "best.ckpt"
        elif (parent_dir / "last.ckpt").exists():
            checkpoint_path = parent_dir / "last.ckpt"
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = TSRMImputationExternal.load_from_checkpoint(
        str(checkpoint_path),
        config=tsrm_cfg,
    )
    model.eval()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Using device: {device}")

    # Run TSRM inference
    tsrm_imputed_list = []
    with torch.no_grad():
        for i in range(len(test_masked)):
            masked = torch.from_numpy(test_masked[i : i + 1]).float().to(device)
            original = torch.from_numpy(test_original[i : i + 1]).float().to(device)
            # Create synthetic time marks (zeros - model expects 5 time features)
            time_marks = torch.zeros(1, window_size, 5).float().to(device)
            result = model.impute(masked, original, time_marks, time_marks)
            tsrm_imputed_list.append(result.cpu().numpy())

    tsrm_imputed = np.concatenate(tsrm_imputed_list, axis=0)
    print(f"  TSRM inference complete: shape={tsrm_imputed.shape}")

    # Run baselines
    locf_result = locf_impute(test_masked)
    linear_result = linear_interp_impute(test_masked)
    print("  Baseline inference complete (LOCF, Linear)")

    # Compute shared eval mask (excludes NaNs from any method)
    shared_mask, n_excl, excl_reason = compute_shared_eval_mask(
        artificial_mask=art_mask,
        truth=test_original,
        pred_saits=tsrm_imputed,  # Using TSRM as the main model
        pred_locf=locf_result,
        pred_linear=linear_result,
    )

    print(f"  Shared eval mask: {shared_mask.sum()} positions")
    if n_excl > 0:
        print(f"  Excluded {n_excl} positions: {', '.join(excl_reason)}")

    # Compute metrics
    tsrm_metrics = compute_metrics(tsrm_imputed, test_original, shared_mask)
    locf_metrics = compute_metrics(locf_result, test_original, shared_mask)
    linear_metrics = compute_metrics(linear_result, test_original, shared_mask)

    # Compute skill scores
    skill_vs_locf = compute_skill_scores_vs_baselines(tsrm_metrics, {"locf": locf_metrics})
    skill_vs_linear = compute_skill_scores_vs_baselines(tsrm_metrics, {"linear": linear_metrics})

    # Per-variable metrics
    per_var = compute_metrics_per_variable(
        tsrm_imputed,
        test_original,
        shared_mask,
        variable_names,
        baselines={"locf": locf_result, "linear": linear_result},
    )

    results = {
        "fold": fold_idx,
        "split": split,
        "missing_pattern": missing_pattern,
        "missing_rate": float(missing_rate),
        "realized_rate": float(realized_rate),
        "n_test_windows": int(test_windows.shape[0]),
        "n_artificial_masked": int(art_mask.sum()),
        "n_shared_eval": int(shared_mask.sum()),
        "n_excluded": n_excl,
        "exclusion_reasons": excl_reason,
        "tsrm": tsrm_metrics,
        "locf": locf_metrics,
        "linear": linear_metrics,
        "skill_vs_locf": skill_vs_locf.get("locf", {}),
        "skill_vs_linear": skill_vs_linear.get("linear", {}),
        "per_variable": per_var,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSRM OSSE imputation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to experiment directory containing models/",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Fold to evaluate (default: 0)",
    )
    parser.add_argument(
        "--missing-pattern",
        choices=["point", "subseq", "block"],
        default="point",
        help="Missing pattern to apply (default: point)",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.2,
        help="Missing rate (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <experiment-dir>/evaluation_fold<N>.json)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve paths
    exp_dir = Path(args.experiment_dir)
    model_dir = exp_dir / "models" / f"fold_{args.fold}"
    checkpoint_path = model_dir / "best.ckpt"
    scaler_path = model_dir / "scaler.pkl"

    # Load data
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})

    variable_names = list(data_cfg.get("variables", OSSEDataLoader.DEFAULT_VARIABLES))
    data_dir = Path(paths_cfg.get("data_dir", ".")).expanduser()

    loader = OSSEDataLoader(data_dir=str(data_dir), variables=variable_names)
    raw_data = loader.to_multivariate_array()

    boundaries_raw = data_cfg.get("block_boundaries", loader.get_block_boundaries())
    block_boundaries = [int(v) for v in boundaries_raw]

    # Run evaluation
    results = evaluate_fold(
        config=config,
        fold_idx=args.fold,
        data=raw_data,
        block_boundaries=block_boundaries,
        variable_names=variable_names,
        missing_pattern=args.missing_pattern,
        missing_rate=args.missing_rate,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
    )

    # Determine output path
    output_path = Path(args.output) if args.output else exp_dir / f"evaluation_fold{args.fold}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_path, "w") as f:
        json.dump(to_builtin(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"TSRM MSE: {results['tsrm']['mse']:.6f}, MAE: {results['tsrm']['mae']:.6f}")

    skill_locf = results["skill_vs_locf"]
    if "msess" in skill_locf and isinstance(skill_locf["msess"], (int, float)):
        print(f"Skill vs LOCF: MSESS={skill_locf['msess']:.4f}")
    else:
        print(f"Skill vs LOCF: MSESS={skill_locf.get('msess', 'N/A')}")

    skill_linear = results["skill_vs_linear"]
    if "msess" in skill_linear and isinstance(skill_linear["msess"], (int, float)):
        print(f"Skill vs Linear: MSESS={skill_linear['msess']:.4f}")
    else:
        print(f"Skill vs Linear: MSESS={skill_linear.get('msess', 'N/A')}")


if __name__ == "__main__":
    main()

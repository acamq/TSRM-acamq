#!/usr/bin/env python
"""Training script for TSRM OSSE imputation with 4-fold CV.

Usage:
    python scripts/train_osse.py --config configs/osse_default.yaml --fold 0
    python scripts/train_osse.py --config configs/osse_default.yaml --all-folds
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
try:
    import torch  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pipeline.config import load_config, build_tsrm_config  # noqa: E402
from pipeline.data.loader import OSSEDataLoader  # noqa: E402
from pipeline.data.preprocessor import Preprocessor  # noqa: E402
import pipeline.data.preprocessor as preprocessor_module  # noqa: E402
from pipeline.data.masking import apply_missing_pattern  # noqa: E402
from pipeline.tracking.experiment import ExperimentTracker  # noqa: E402


prepare_splits_block_level = getattr(preprocessor_module, "prepare_splits_block_level", None)
create_windows_3d = getattr(preprocessor_module, "create_windows_3d", None)


if torch is not None:
    torch.set_float32_matmul_precision("medium")


FOLD_DEFINITIONS: Dict[int, Dict[str, List[int]]] = {
    0: {"train": [2, 3], "val": [1], "test": [0]},
    1: {"train": [0, 3], "val": [2], "test": [1]},
    2: {"train": [0, 1], "val": [3], "test": [2]},
    3: {"train": [1, 2], "val": [0], "test": [3]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TSRM on OSSE with 4-fold CV")
    parser.add_argument("--config", required=True)

    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument("--fold", type=int, choices=[0, 1, 2, 3], default=None)
    fold_group.add_argument("--all-folds", action="store_true")

    parser.add_argument("--missing-pattern", choices=["point", "subseq", "block"], default=None)
    parser.add_argument("--missing-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--experiment-name", default="tsrm_osse")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    return value


def resolve_path(path_value: str, fallback: str = ".") -> Path:
    raw = path_value if path_value else fallback
    return Path(os.path.expandvars(raw)).expanduser()


def load_lightning() -> Tuple[Any, Any, Any]:
    try:
        import lightning.pytorch as pl_mod  # pyright: ignore[reportMissingImports]
        from lightning.pytorch.callbacks import (  # pyright: ignore[reportMissingImports]
            EarlyStopping as early_stopping_cls,
            ModelCheckpoint as checkpoint_cls,
        )
    except ModuleNotFoundError:
        import pytorch_lightning as pl_mod  # pyright: ignore[reportMissingImports]
        from pytorch_lightning.callbacks import (  # pyright: ignore[reportMissingImports]
            EarlyStopping as early_stopping_cls,
            ModelCheckpoint as checkpoint_cls,
        )

    return pl_mod, early_stopping_cls, checkpoint_cls


def select_folds(args: argparse.Namespace) -> List[int]:
    if args.all_folds:
        return [0, 1, 2, 3]
    if args.fold is not None:
        return [args.fold]
    return [0]


def resolve_missing_settings(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, float]:
    masking_cfg = config.get("masking", {})
    patterns = masking_cfg.get("patterns", ["point"])
    rates = masking_cfg.get("missing_rates", [0.1])

    pattern = args.missing_pattern or (patterns[0] if patterns else "point")
    rate = float(args.missing_rate) if args.missing_rate is not None else float(rates[0])

    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {rate}")

    return pattern, rate


def split_blocks(
    preprocessor: Preprocessor,
    data: np.ndarray,
    block_boundaries: List[int],
    train_blocks: List[int],
    val_blocks: List[int],
    test_blocks: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]]:
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
    if callable(create_windows_3d):
        return np.asarray(create_windows_3d(data_list, window_size, stride))

    return np.asarray(
        preprocessor.create_windows_3d(
            data_list=data_list,
            window_size=window_size,
            stride=stride,
        )
    )


def pick_batch_size(requested: int, train_samples: int, val_samples: int) -> int:
    if train_samples <= 0 or val_samples <= 0:
        raise ValueError(
            f"Both train and val must be non-empty: train={train_samples}, val={val_samples}"
        )

    return max(1, min(requested, train_samples))


def resolve_num_workers(training_cfg: Dict[str, Any]) -> int:
    configured = training_cfg.get("num_workers")
    if configured is not None:
        return max(0, int(configured))

    cpu_count = os.cpu_count() or 4
    return min(12, max(4, cpu_count // 4))


def resolve_trainer_settings(training_cfg: Dict[str, Any]) -> Tuple[str, str, float]:
    requested_accelerator = str(training_cfg.get("accelerator", "auto"))
    requested_precision = str(training_cfg.get("precision", "16-mixed"))
    gradient_clip_val = float(training_cfg.get("gradient_clip_val", 1.0))

    if requested_accelerator == "auto":
        accelerator = "gpu" if torch is not None and torch.cuda.is_available() else "cpu"
    else:
        accelerator = requested_accelerator

    precision = requested_precision
    if accelerator == "cpu" and requested_precision in {"16-mixed", "bf16-mixed"}:
        precision = "32-true"

    return accelerator, precision, gradient_clip_val


def train_fold(
    config: Dict[str, Any],
    fold_idx: int,
    data: np.ndarray,
    block_boundaries: List[int],
    variable_names: List[str],
    missing_pattern: str,
    missing_rate: float,
    epochs_override: int | None,
    tracker: ExperimentTracker,
) -> Dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to train TSRM. Install dependencies first.")

    from pipeline.data.dataset import create_dataloaders
    from architecture.tsrm_external import TSRMImputationExternal

    pl_mod, early_stopping_cls, checkpoint_cls = load_lightning()

    split = FOLD_DEFINITIONS[fold_idx]
    data_cfg = config.get("data", {})
    tsrm_cfg = config.get("tsrm", {})
    train_cfg = config.get("training", {})
    mask_cfg = config.get("masking", {})

    window_size = int(data_cfg.get("window_size", 30))
    train_stride = int(data_cfg.get("window_stride_train", 5))
    eval_stride = int(data_cfg.get("window_stride_eval", 30))
    augmentation_factor = max(1, int(mask_cfg.get("augmentation_factor", 1)))
    fold_seed = int(mask_cfg.get("seed", 42)) + (fold_idx * 1000)

    pl_mod.seed_everything(fold_seed, workers=True)

    preprocessor = Preprocessor(config)
    split_result = split_blocks(
        preprocessor=preprocessor,
        data=data,
        block_boundaries=block_boundaries,
        train_blocks=split["train"],
        val_blocks=split["val"],
        test_blocks=split["test"],
    )
    train_blocks, val_blocks, test_blocks, _, _, _ = cast(
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]],
        split_result,
    )

    if not train_blocks or not val_blocks or not test_blocks:
        raise ValueError(f"Fold {fold_idx} produced empty split(s): {split}")

    train_fit_data = np.concatenate(train_blocks, axis=1)
    preprocessor.fit(train_fit_data, variable_names=variable_names)

    train_scaled = [preprocessor.transform(block) for block in train_blocks]
    val_scaled = [preprocessor.transform(block) for block in val_blocks]
    test_scaled = [preprocessor.transform(block) for block in test_blocks]

    train_windows = make_windows(preprocessor, train_scaled, window_size, train_stride)
    val_windows = make_windows(preprocessor, val_scaled, window_size, eval_stride)
    test_windows = make_windows(preprocessor, test_scaled, window_size, eval_stride)

    if train_windows.size == 0 or val_windows.size == 0 or test_windows.size == 0:
        raise ValueError(
            f"Fold {fold_idx} produced empty windows: "
            f"train={train_windows.shape}, val={val_windows.shape}, test={test_windows.shape}"
        )

    train_masked_batches: List[np.ndarray] = []
    train_original_batches: List[np.ndarray] = []
    train_rates: List[float] = []
    for aug_idx in range(augmentation_factor):
        masked, _, realized_rate = apply_missing_pattern(
            data=train_windows,
            pattern=missing_pattern,
            missing_rate=missing_rate,
            seed=fold_seed + aug_idx,
        )
        train_masked_batches.append(masked)
        train_original_batches.append(train_windows.copy())
        train_rates.append(realized_rate)

    train_masked = np.concatenate(train_masked_batches, axis=0)
    train_original = np.concatenate(train_original_batches, axis=0)
    val_masked, _, val_rate = apply_missing_pattern(
        data=val_windows,
        pattern=missing_pattern,
        missing_rate=missing_rate,
        seed=fold_seed + 500,
    )

    model_cfg = build_tsrm_config(config)
    requested_batch_size = int(model_cfg.get("batch_size", 32))
    batch_size = pick_batch_size(requested_batch_size, train_masked.shape[0], val_masked.shape[0])
    model_cfg["batch_size"] = batch_size

    if batch_size != requested_batch_size:
        print(
            f"Fold {fold_idx}: adjusted batch size {requested_batch_size} -> {batch_size} "
            "to fit available samples"
        )

    num_workers = resolve_num_workers(train_cfg)
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", True))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))

    train_dl, val_dl = create_dataloaders(
        train_masked=train_masked,
        train_original=train_original,
        train_timestamps=None,
        val_masked=val_masked,
        val_original=val_windows,
        val_timestamps=None,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last_train=True,
        drop_last_val=False,
        freq=str(model_cfg.get("freq", "t")),
    )

    model = TSRMImputationExternal(config=model_cfg)
    model.lr = float(tsrm_cfg.get("learning_rate", 0.001))

    max_epochs = int(epochs_override if epochs_override is not None else tsrm_cfg.get("epochs", 100))
    early_stopping = early_stopping_cls(
        monitor="loss",
        mode="min",
        patience=int(tsrm_cfg.get("patience", 15)),
        min_delta=float(tsrm_cfg.get("earlystopping_min_delta", 0.0)),
        verbose=True,
    )

    fold_model_dir = tracker.get_model_dir() / f"fold_{fold_idx}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = checkpoint_cls(
        dirpath=str(fold_model_dir),
        filename="best-{epoch:02d}-{loss:.6f}",
        monitor="loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    accelerator, precision, gradient_clip_val = resolve_trainer_settings(train_cfg)
    trainer = pl_mod.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=str(tracker.experiment_dir),
        logger=[],
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    scaler_path = fold_model_dir / "scaler.pkl"
    preprocessor.save_scalers(scaler_path)

    summary = {
        "fold": fold_idx,
        "split": split,
        "train_samples": int(train_masked.shape[0]),
        "val_samples": int(val_masked.shape[0]),
        "test_samples": int(test_windows.shape[0]),
        "missing_pattern": missing_pattern,
        "missing_rate": float(missing_rate),
        "realized_train_rate_mean": float(np.mean(train_rates)) if train_rates else 0.0,
        "realized_val_rate": float(val_rate),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "best_checkpoint": checkpoint_callback.best_model_path,
        "best_score": to_builtin(checkpoint_callback.best_model_score),
        "scaler_path": str(scaler_path),
        "metrics": to_builtin(dict(trainer.callback_metrics)),
    }

    with open(fold_model_dir / "fold_summary.json", "w") as f:
        json.dump(to_builtin(summary), f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    folds = select_folds(args)
    missing_pattern, missing_rate = resolve_missing_settings(config, args)

    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})

    variable_names = list(data_cfg.get("variables", OSSEDataLoader.DEFAULT_VARIABLES))
    data_dir = resolve_path(str(paths_cfg.get("data_dir", "")), fallback=".")
    loader = OSSEDataLoader(data_dir=str(data_dir), variables=variable_names)
    raw_data = loader.to_multivariate_array()

    boundaries_raw = data_cfg.get("block_boundaries", loader.get_block_boundaries())
    block_boundaries = [int(v) for v in boundaries_raw]

    scratch_dir = resolve_path(str(paths_cfg.get("scratch_dir", "./outputs")))
    tracker = ExperimentTracker(str(scratch_dir / "experiments"), args.experiment_name)

    run_config = copy.deepcopy(config)
    run_config["run"] = {
        "folds": folds,
        "missing_pattern": missing_pattern,
        "missing_rate": missing_rate,
        "epochs_override": args.epochs,
    }
    tracker.save_config(run_config)
    tracker.save_metadata(
        seed=int(config.get("masking", {}).get("seed", 42)),
        folds=folds,
        missing_pattern=missing_pattern,
        missing_rate=missing_rate,
        epochs_override=args.epochs,
    )

    print(f"Experiment ID: {tracker.experiment_id}")
    print(f"Experiment directory: {tracker.experiment_dir}")

    fold_results = []
    for fold_idx in folds:
        print(f"\n=== Training fold {fold_idx} ===")
        result = train_fold(
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
        fold_results.append(result)
        print(f"Fold {fold_idx} complete. Best checkpoint: {result['best_checkpoint']}")

    summary = {
        "experiment_id": tracker.experiment_id,
        "experiment_dir": str(tracker.experiment_dir),
        "config_path": args.config,
        "folds_run": folds,
        "missing_pattern": missing_pattern,
        "missing_rate": float(missing_rate),
        "results": fold_results,
    }
    summary_path = tracker.experiment_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(to_builtin(summary), f, indent=2)

    print(f"\nTraining complete. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

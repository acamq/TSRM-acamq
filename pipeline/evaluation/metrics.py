"""Evaluation metrics for imputation quality assessment.

Implements:
- MSE, MAE on masked positions only
- Skill scores (MSESS, MAESS) with eps-based division-by-zero handling
- Per-variable and aggregate metric computation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_shared_eval_mask(
    artificial_mask: np.ndarray,
    truth: np.ndarray,
    pred_saits: np.ndarray,
    pred_locf: np.ndarray,
    pred_linear: np.ndarray
) -> Tuple[np.ndarray, int, List[str]]:
    """Create shared evaluation mask that excludes NaNs from ANY method.

    This ensures fair comparison: all methods scored on same points.
    If a method produces NaN where others produce values, those points
    are excluded for ALL methods (not just the NaN-producing method).

    Args:
        artificial_mask: Boolean array of artificially masked positions
        truth: Ground truth array
        pred_saits: SAITS predictions
        pred_locf: LOCF baseline predictions
        pred_linear: Linear interpolation predictions

    Returns:
        tuple: (shared_mask, n_excluded, exclusion_reason)
            - shared_mask: Boolean array of valid evaluation points
            - n_excluded: Number of points excluded from evaluation
            - exclusion_reason: List of strings describing why points were excluded
    """
    valid = (
        artificial_mask &
        np.isfinite(truth) &
        np.isfinite(pred_saits) &
        np.isfinite(pred_locf) &
        np.isfinite(pred_linear)
    )

    n_excluded = int(artificial_mask.sum() - valid.sum())
    exclusion_reason: List[str] = []
    if n_excluded > 0:
        # Identify which methods caused exclusions
        saits_nan = artificial_mask & ~np.isfinite(pred_saits)
        locf_nan = artificial_mask & ~np.isfinite(pred_locf)
        linear_nan = artificial_mask & ~np.isfinite(pred_linear)
        truth_nan = artificial_mask & ~np.isfinite(truth)

        if saits_nan.any():
            exclusion_reason.append(f'saits:{int(saits_nan.sum())}')
        if locf_nan.any():
            exclusion_reason.append(f'locf:{int(locf_nan.sum())}')
        if linear_nan.any():
            exclusion_reason.append(f'linear:{int(linear_nan.sum())}')
        if truth_nan.any():
            exclusion_reason.append(f'truth:{int(truth_nan.sum())}')

    return valid, n_excluded, exclusion_reason


def compute_mse(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray
) -> float:
    """Compute Mean Squared Error on masked positions only.

    Args:
        imputed: Imputed values [n_samples, n_steps, n_features] or flattened
        truth: Ground truth values
        mask: Boolean mask, True where values were artificially masked

    Returns:
        MSE value (float)
    """
    # Flatten if needed
    mask_flat = mask.flatten() if mask.ndim > 1 else mask

    if not mask_flat.any():
        return np.nan

    imp_flat = imputed.flatten() if imputed.ndim > 1 else imputed
    tru_flat = truth.flatten() if truth.ndim > 1 else truth

    # Get masked values
    imp_masked = imp_flat[mask_flat]
    tru_masked = tru_flat[mask_flat]

    # Filter out any remaining NaN
    valid = np.isfinite(imp_masked) & np.isfinite(tru_masked)
    if not valid.any():
        return np.nan

    return float(np.mean((imp_masked[valid] - tru_masked[valid]) ** 2))


def compute_mae(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray
) -> float:
    """Compute Mean Absolute Error on masked positions only.

    Args:
        imputed: Imputed values [n_samples, n_steps, n_features] or flattened
        truth: Ground truth values
        mask: Boolean mask, True where values were artificially masked

    Returns:
        MAE value (float)
    """
    mask_flat = mask.flatten() if mask.ndim > 1 else mask

    if not mask_flat.any():
        return np.nan

    imp_flat = imputed.flatten() if imputed.ndim > 1 else imputed
    tru_flat = truth.flatten() if truth.ndim > 1 else truth

    imp_masked = imp_flat[mask_flat]
    tru_masked = tru_flat[mask_flat]

    valid = np.isfinite(imp_masked) & np.isfinite(tru_masked)
    if not valid.any():
        return np.nan

    return float(np.mean(np.abs(imp_masked[valid] - tru_masked[valid])))


def compute_skill_score(
    model_error: float,
    baseline_error: float,
    eps: float = 1e-10
) -> float:
    """Compute skill score: 1 - model_error / baseline_error.

    A skill score > 0 means the model outperforms the baseline.
    A skill score < 0 means the model underperforms.
    A skill score of 1.0 means perfect model (zero error).

    CRITICAL: Uses eps-based division-by-zero handling:
    - If baseline_error < eps and model_error < eps: return 0.0 (both perfect)
    - If baseline_error < eps and model_error >= eps: return -inf (baseline better)
    - If baseline_error >= eps: normal calculation

    Args:
        model_error: MSE or MAE from the model
        baseline_error: MSE or MAE from the baseline
        eps: Threshold for considering values as zero

    Returns:
        Skill score (float, may be -inf)
    """
    if baseline_error < eps:
        if model_error < eps:
            return 0.0  # Both perfect, no advantage
        else:
            return float('-inf')  # Baseline perfect, model not

    return 1.0 - (model_error / baseline_error)


def compute_metrics(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-10
) -> Dict[str, float]:
    """Compute all metrics for imputation evaluation.

    Args:
        imputed: Imputed values
        truth: Ground truth values
        mask: Boolean mask for artificially masked positions
        eps: Threshold for skill score zero handling

    Returns:
        Dict with 'mse', 'mae' values
    """
    mse = compute_mse(imputed, truth, mask)
    mae = compute_mae(imputed, truth, mask)

    return {
        'mse': mse,
        'mae': mae,
    }


def compute_metrics_per_variable(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    variable_names: Optional[List[str]] = None,
    baselines: Optional[Dict[str, np.ndarray]] = None,
    eps: float = 1e-10
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each variable separately, including skill scores.

    Args:
        imputed: Imputed values [n_samples, n_steps, n_features]
        truth: Ground truth values [n_samples, n_steps, n_features]
        mask: Boolean mask for artificially masked positions [n_samples, n_steps, n_features]
        variable_names: Names for each feature (optional)
        baselines: Optional dict of baseline name -> baseline predictions array
        eps: Threshold for skill score zero handling

    Returns:
        Dict mapping variable name to metrics dict with:
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'msess_{baseline}': MSE skill score vs each baseline (if baselines provided)
            - 'maess_{baseline}': MAE skill score vs each baseline (if baselines provided)
    """
    n_features = imputed.shape[-1]

    if variable_names is None:
        variable_names = [f'var_{i}' for i in range(n_features)]

    results = {}

    for i, var_name in enumerate(variable_names[:n_features]):
        var_mask = mask[..., i]
        var_imp = imputed[..., i]
        var_tru = truth[..., i]

        # Get masked values for this variable
        masked_positions = var_mask.flatten() if var_mask.ndim > 1 else var_mask

        if not masked_positions.any():
            results[var_name] = {'mse': None, 'mae': None}
            continue

        imp_masked = var_imp.flatten()[masked_positions]
        tru_masked = var_tru.flatten()[masked_positions]

        # Filter NaN values
        valid = np.isfinite(imp_masked) & np.isfinite(tru_masked)
        if not valid.any():
            results[var_name] = {'mse': None, 'mae': None}
            continue

        imp_valid = imp_masked[valid]
        tru_valid = tru_masked[valid]

        # Compute MSE and MAE
        mse = float(np.mean((imp_valid - tru_valid) ** 2))
        mae = float(np.mean(np.abs(imp_valid - tru_valid)))

        # Compute skill scores vs baselines
        skill_scores = {}
        if baselines:
            for bl_name, bl_pred in baselines.items():
                var_bl = bl_pred[..., i]
                bl_masked = var_bl.flatten()[masked_positions]

                if len(bl_masked) == len(imp_masked):
                    bl_valid = bl_masked[valid]
                    if np.all(np.isfinite(bl_valid)):
                        bl_mse = float(np.mean((bl_valid - tru_valid) ** 2))
                        bl_mae = float(np.mean(np.abs(bl_valid - tru_valid)))
                        skill_scores[f'msess_{bl_name}'] = compute_skill_score(mse, bl_mse, eps)
                        skill_scores[f'maess_{bl_name}'] = compute_skill_score(mae, bl_mae, eps)

        results[var_name] = {
            'mse': mse,
            'mae': mae,
            **skill_scores
        }

    return results


def compute_skill_scores_vs_baselines(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    eps: float = 1e-10
) -> Dict[str, Dict[str, float]]:
    """Compute skill scores comparing model to each baseline.

    Args:
        model_metrics: Dict with 'mse', 'mae' for model
        baseline_metrics: Dict mapping baseline name to its metrics
        eps: Threshold for division-by-zero handling

    Returns:
        Dict mapping baseline name to skill scores {'msess': ..., 'maess': ...}
    """
    results = {}

    model_mse = model_metrics.get('mse', np.nan)
    model_mae = model_metrics.get('mae', np.nan)

    for baseline_name, baseline_m in baseline_metrics.items():
        baseline_mse = baseline_m.get('mse', np.nan)
        baseline_mae = baseline_m.get('mae', np.nan)

        msess = compute_skill_score(model_mse, baseline_mse, eps)
        maess = compute_skill_score(model_mae, baseline_mae, eps)

        results[baseline_name] = {
            'msess': msess,
            'maess': maess,
        }

    return results
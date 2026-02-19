"""Baseline imputation methods for comparison with SAITS.

Implements two classical baselines:
- LOCF (Last Observation Carried Forward)
- Linear interpolation

Both methods handle edge cases appropriately.
"""

import numpy as np
import pandas as pd


def locf_impute(data: np.ndarray) -> np.ndarray:
    """Last Observation Carried Forward imputation.
    
    Fills NaN values by carrying forward the last observed value.
    For NaN at the start of a sequence, uses backward-fill from first observed.
    For entirely missing features, leaves as NaN (excluded from scoring).
    
    Args:
        data: Input array [n_samples, n_steps, n_features] with NaN for missing
    
    Returns:
        Imputed array with NaN only where entire feature is missing
    """
    result = data.copy()
    n_samples, n_steps, n_features = data.shape
    
    for i in range(n_samples):
        for f in range(n_features):
            series = result[i, :, f].copy()
            
            # Check if all NaN
            if np.all(np.isnan(series)):
                continue  # Leave as NaN (will be excluded from scoring)
            
            # Find first and last observed positions
            observed_mask = ~np.isnan(series)
            observed_indices = np.where(observed_mask)[0]
            
            if len(observed_indices) == 0:
                continue
            
            first_obs = observed_indices[0]
            
            # Backward-fill from first observed to start
            if first_obs > 0:
                series[:first_obs] = series[first_obs]
            
            # LOCF for gaps in the middle
            last_value = series[first_obs]
            for t in range(first_obs + 1, n_steps):
                if np.isnan(series[t]):
                    series[t] = last_value
                else:
                    last_value = series[t]
            
            result[i, :, f] = series
    
    return result


def linear_interp_impute(data: np.ndarray) -> np.ndarray:
    """Linear interpolation imputation.
    
    Fills NaN values using linear interpolation between observed values.
    For NaN at the start: uses forward-fill from first observed.
    For NaN at the end: uses backward-fill from last observed.
    For entirely missing features, leaves as NaN.
    
    Args:
        data: Input array [n_samples, n_steps, n_features] with NaN for missing
    
    Returns:
        Imputed array with NaN only where entire feature is missing
    """
    result = data.copy()
    n_samples, n_steps, n_features = data.shape
    
    for i in range(n_samples):
        for f in range(n_features):
            series = result[i, :, f].copy()
            
            # Check if all NaN
            if np.all(np.isnan(series)):
                continue  # Leave as NaN
            
            # Use pandas interpolate with both-direction limit
            # This handles edge cases automatically
            s = pd.Series(series)
            s = s.interpolate(method='linear', limit_direction='both')
            
            result[i, :, f] = s.values
    
    return result


def get_available_baselines() -> list:
    """Get list of available baseline method names.
    
    Returns:
        List of baseline method names: ['locf', 'linear']
    """
    return ['locf', 'linear']


def apply_baseline(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply a baseline imputation method.
    
    Args:
        data: Input array [n_samples, n_steps, n_features] with NaN for missing
        method: Baseline method name ('locf' or 'linear')
    
    Returns:
        Imputed array
    
    Raises:
        ValueError: If method is not recognized
    """
    method = method.lower()
    
    if method == 'locf':
        return locf_impute(data)
    elif method == 'linear':
        return linear_interp_impute(data)
    else:
        raise ValueError(
            f"Unknown baseline method: {method}. "
            f"Available: {get_available_baselines()}"
        )

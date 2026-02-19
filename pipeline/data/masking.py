"""Missing pattern simulation for OSSE data.

Implements three missing patterns based on LEO satellite characteristics:
- Point (MCAR): Random scattered missing values
- Subsequence: Sensor-specific dropout (contiguous per feature)
- Block: Satellite outage (time interval across all features)

All patterns enforce EXACT missing rates on observed values.
"""

from typing import Tuple
import numpy as np


def apply_point_mcar(
    data: np.ndarray,
    missing_rate: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply point (scattered) missing completely at random.

    Randomly selects exactly k = round(rate * n_observed) observed positions.
    Each observed position has equal probability.

    Args:
        data: Input array [n_samples, n_steps, n_features]
        missing_rate: Target fraction of observed entries to mask
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - masked: Copy of data with NaN at masked positions
        - artificial_mask: Boolean array True where artificially masked
        - realized_rate: Actual achieved rate (should equal target)
    """
    rng = np.random.default_rng(seed)

    # Find observed positions (not already NaN)
    observed_mask = ~np.isnan(data)
    n_observed = observed_mask.sum()

    if n_observed == 0:
        return data.copy(), np.zeros_like(data, dtype=bool), 0.0

    # Calculate exact number to mask
    k = round(n_observed * missing_rate)

    if k == 0:
        return data.copy(), np.zeros_like(data, dtype=bool), 0.0

    # Get indices of observed positions
    observed_indices = np.argwhere(observed_mask)

    # Randomly select k positions without replacement
    selected = rng.choice(len(observed_indices), k, replace=False)

    # Create artificial mask
    artificial_mask = np.zeros_like(data, dtype=bool)
    for idx in observed_indices[selected]:
        artificial_mask[tuple(idx)] = True

    # Apply mask to copy of data
    masked = data.copy()
    masked[artificial_mask] = np.nan

    # Calculate realized rate
    realized_rate = k / n_observed

    return masked, artificial_mask, realized_rate


def apply_subseq_missing(
    data: np.ndarray,
    missing_rate: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply subsequence missing (sensor-specific dropout).

    For each feature independently, masks contiguous segments.
    Simulates sensor-specific dropout where one instrument fails.

    Args:
        data: Input array [n_samples, n_steps, n_features]
        missing_rate: Target fraction of observed entries to mask
        seed: Random seed for reproducibility

    Returns:
        Tuple of (masked, artificial_mask, realized_rate)
    """
    rng = np.random.default_rng(seed)
    masked = data.copy()
    artificial_mask = np.zeros_like(data, dtype=bool)

    n_samples, n_steps, n_features = data.shape
    n_total_observed = (~np.isnan(data)).sum()

    if n_total_observed == 0:
        return masked, artificial_mask, 0.0

    for f in range(n_features):
        for s in range(n_samples):
            # Find observed positions in this sample/feature
            observed = ~np.isnan(data[s, :, f])
            if not observed.any():
                continue

            obs_indices = np.where(observed)[0]
            n_obs = len(obs_indices)

            # How many to mask in this sample/feature
            k_f = round(n_obs * missing_rate)
            if k_f == 0 or k_f >= n_obs:
                continue

            # Select contiguous segment
            max_start = n_obs - k_f
            if max_start <= 0:
                continue

            start_i = rng.integers(0, max_start)
            for i in range(start_i, start_i + k_f):
                t = obs_indices[i]
                artificial_mask[s, t, f] = True

    # Apply mask
    masked[artificial_mask] = np.nan

    # Calculate realized rate
    n_masked = artificial_mask.sum()
    realized_rate = n_masked / n_total_observed if n_total_observed > 0 else 0.0

    return masked, artificial_mask, realized_rate


def apply_block_missing(
    data: np.ndarray,
    missing_rate: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply block missing (satellite outage).

    Masks a time interval that affects ALL features simultaneously.
    Simulates satellite outage (comms loss, power cycle).

    Args:
        data: Input array [n_samples, n_steps, n_features]
        missing_rate: Target fraction of observed entries to mask
        seed: Random seed for reproducibility

    Returns:
        Tuple of (masked, artificial_mask, realized_rate)
    """
    rng = np.random.default_rng(seed)
    masked = data.copy()
    artificial_mask = np.zeros_like(data, dtype=bool)

    n_samples, n_steps, n_features = data.shape
    n_total_observed = (~np.isnan(data)).sum()

    if n_total_observed == 0 or n_steps == 0:
        return masked, artificial_mask, 0.0

    # Calculate block length to achieve target rate
    # block affects n_samples * n_features entries per timestep
    k_total = round(n_total_observed * missing_rate)
    entries_per_timestep = n_samples * n_features
    block_len = max(1, k_total // entries_per_timestep)

    if block_len >= n_steps:
        block_len = n_steps - 1

    for s in range(n_samples):
        # Random start position
        max_start = n_steps - block_len
        if max_start <= 0:
            continue

        start_t = rng.integers(0, max_start)

        # Mask this time interval for all features (if observed)
        for t in range(start_t, start_t + block_len):
            for f in range(n_features):
                if not np.isnan(data[s, t, f]):
                    artificial_mask[s, t, f] = True

    # Apply mask
    masked[artificial_mask] = np.nan

    # Calculate realized rate
    n_masked = artificial_mask.sum()
    realized_rate = n_masked / n_total_observed if n_total_observed > 0 else 0.0

    return masked, artificial_mask, realized_rate


def apply_missing_pattern(
    data: np.ndarray,
    pattern: str,
    missing_rate: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply a specific missing pattern to data.

    Args:
        data: Input array [n_samples, n_steps, n_features]
        pattern: One of 'point', 'subseq', 'block'
        missing_rate: Target fraction of observed entries to mask
        seed: Random seed for reproducibility

    Returns:
        Tuple of (masked_data, artificial_mask, realized_rate)

    Raises:
        ValueError: If pattern is not recognized
    """
    pattern = pattern.lower()

    if pattern == 'point':
        return apply_point_mcar(data, missing_rate, seed)
    elif pattern == 'subseq':
        return apply_subseq_missing(data, missing_rate, seed)
    elif pattern == 'block':
        return apply_block_missing(data, missing_rate, seed)
    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'point', 'subseq', or 'block'.")

"""Normalization and windowing for spacew package."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class NaNSafeScaler:
    """StandardScaler that handles NaN values correctly.
    
    CRITICAL: Computes per-feature statistics over ALL satellites and timesteps.
    For data shaped [N_sat, T, F], mean/std are computed over axes (0, 1)
    to produce per-feature scalars, NOT time-varying statistics.
    """
    
    def __init__(self):
        self.mean_ = None  # Shape: [1, 1, F] for broadcasting
        self.std_ = None   # Shape: [1, 1, F] for broadcasting
    
    def fit(self, X: np.ndarray) -> 'NaNSafeScaler':
        """Fit on non-NaN values only, computing per-feature stats.
        
        Args:
            X: ndarray [N_satellites, T_timesteps, F_features]
        
        Returns:
            self
        """
        # CRITICAL: axis=(0, 1) to reduce over satellites AND time
        # This produces per-feature statistics, not time-varying
        self.mean_ = np.nanmean(X, axis=(0, 1), keepdims=True)  # [1, 1, F]
        self.std_ = np.nanstd(X, axis=(0, 1), keepdims=True)   # [1, 1, F]
        # Guard against zero std (constant features)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform, preserving NaN positions.
        
        Args:
            X: ndarray [N_sat, T, F] or [n_samples, L, F]
        
        Returns:
            Normalized array with same shape, NaN positions preserved
        """
        return (X - self.mean_) / self.std_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform.
        
        Args:
            X: Normalized data [N_sat, T, F] or [n_samples, L, F]
        
        Returns:
            Data in original scale
        """
        return X * self.std_ + self.mean_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Preprocessor:
    """Preprocessor for OSSE data with block-aware splitting and windowing.
    
    Physics-informed normalization:
    - Density variables: log1p transform + standard scaling
    - Other variables: standard scaling only
    """
    
    def __init__(self, config: Dict):
        """Initialize with config dict.
        
        Args:
            config: Configuration dict with normalization/windowing params
        """
        self.config = config
        self.scaler: Optional[NaNSafeScaler] = None
        
        # Physics-informed normalization settings
        norm_cfg = config.get('normalization', {})
        self.density_var_names: List[str] = norm_cfg.get('density_vars', [])
        self.log_epsilon: float = norm_cfg.get('log_epsilon', 1e-30)
        self.scaler_type: str = norm_cfg.get('scaler_type', 'nan_safe_standard')
        if self.scaler_type != 'nan_safe_standard':
            raise ValueError(
                "Unsupported normalization.scaler_type='{}'; only 'nan_safe_standard' is supported".format(
                    self.scaler_type
                )
            )
        
        # Set during fit
        self.density_var_indices: List[int] = []
        self.variable_names: List[str] = []

    def set_variable_names(self, variable_names: List[str]) -> None:
        self.variable_names = variable_names
        self.density_var_indices = [
            i for i, name in enumerate(self.variable_names)
            if name in self.density_var_names
        ]
    
    def _apply_log_transform(self, data: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Apply log1p transform to density variables.
        
        Args:
            data: Input array [..., F] with F features in last dimension
            inverse: If True, apply expm1 (inverse of log1p)
        
        Returns:
            Transformed array with same shape
        """
        if not self.density_var_indices:
            return data
        
        result = data.copy()
        for idx in self.density_var_indices:
            if inverse:
                # Reverse: expm1 to undo log1p
                result[..., idx] = np.expm1(result[..., idx])
            else:
                # Forward: log1p(x + epsilon) to handle zeros and small values
                values = np.maximum(result[..., idx], 0)
                result[..., idx] = np.log1p(values + self.log_epsilon)
        return result
    
    def fit(self, data: np.ndarray, variable_names: Optional[List[str]] = None) -> 'Preprocessor':
        """Fit NaN-safe scalers on training data with physics-informed transforms.
        
        Args:
            data: Training data array [N_sat, T, F]
            variable_names: List of variable names for identifying density vars
        
        Returns:
            self
        """
        self.set_variable_names(variable_names or [])
        
        # Apply log transform to density vars first
        data_logged = self._apply_log_transform(data, inverse=False)
        
        # Fit standard scaler on logged data
        self.scaler = NaNSafeScaler()
        self.scaler.fit(data_logged)
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted normalization: log transform + standard scaling.
        
        Args:
            data: Data to transform [N_sat, T, F]
        
        Returns:
            Normalized data
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        # Apply log transform to density vars, then standard scaling
        data_logged = self._apply_log_transform(data, inverse=False)
        return self.scaler.transform(data_logged)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization: unstandardize + exp transform for density vars.
        
        Args:
            data: Normalized data [N_sat, T, F] or [n_samples, L, F]
        
        Returns:
            Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        # Reverse: standard scaling first, then expm1 for density vars
        data_unscaled = self.scaler.inverse_transform(data)
        return self._apply_log_transform(data_unscaled, inverse=True)
    
    def fit_transform(self, data: np.ndarray, variable_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data, variable_names).transform(data)
    
    def prepare_splits_block_level(
        self,
        data: np.ndarray,
        block_boundaries: List[int],
        train_blocks: List[int],
        val_blocks: List[int],
        test_blocks: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], 
               List[int], List[int], List[int]]:
        """Split by BLOCKS (not within blocks) for clean separation.
        
        Args:
            data: ndarray [N_satellites, T_total, F_features]
            block_boundaries: [0, 60, 120, 180, 240]
            train_blocks: [0, 1] - use blocks 0 and 1 for training
            val_blocks: [2] - use block 2 for validation
            test_blocks: [3] - use block 3 for testing
        
        Returns:
            Tuple of (train_data_list, val_data_list, test_data_list,
                      train_bounds, val_bounds, test_bounds)
            Each data list contains arrays of shape [N_sat, block_len, F]
            Each bounds list contains cumulative timestep counts
        """
        train_data, val_data, test_data = [], [], []
        train_bounds, val_bounds, test_bounds = [0], [0], [0]
        
        for block_idx in range(len(block_boundaries) - 1):
            start = block_boundaries[block_idx]
            end = block_boundaries[block_idx + 1]
            block_data = data[:, start:end, :]  # [N_sat, block_len, F]
            
            if block_idx in train_blocks:
                train_data.append(block_data)
                train_bounds.append(train_bounds[-1] + (end - start))
            elif block_idx in val_blocks:
                val_data.append(block_data)
                val_bounds.append(val_bounds[-1] + (end - start))
            elif block_idx in test_blocks:
                test_data.append(block_data)
                test_bounds.append(test_bounds[-1] + (end - start))
        
        return train_data, val_data, test_data, train_bounds, val_bounds, test_bounds
    
    def create_windows_3d(
        self,
        data_list: List[np.ndarray],
        window_size: int,
        stride: int,
        block_boundaries: Optional[List[int]] = None
    ) -> np.ndarray:
        """Create windows within each block, then concatenate samples.
        
        CRITICAL: Output is 3D [n_samples, window_size, n_features]
        Each satellite-window is a SEPARATE sample for PyPOTS.
        
        Args:
            data_list: list of [N_sat, block_len, F] arrays (from block-level split)
            window_size: length of each window (n_steps for SAITS)
            stride: step between windows
            block_boundaries: boundaries within this split (for logging)
        
        Returns:
            windows: ndarray [n_samples, window_size, F_features] - 3D!
        """
        samples = []
        N_sat = data_list[0].shape[0]
        
        for block_idx, block_data in enumerate(data_list):
            block_len = block_data.shape[1]
            
            # Create windows WITHIN this block only
            n_windows = (block_len - window_size) // stride + 1
            if n_windows <= 0:
                print(f"Warning: block {block_idx} too small for window_size={window_size}")
                continue
            
            for i in range(n_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                window = block_data[:, start_idx:end_idx, :]  # [N_sat, L, F]
                
                # CRITICAL: Each satellite is a separate sample!
                for sat_idx in range(N_sat):
                    sample = window[sat_idx, :, :]  # [L, F]
                    samples.append(sample)
        
        return np.array(samples)  # [n_samples, L, F] - 3D!
    
    def save_scalers(self, path: Union[str, Path]) -> None:
        """Save fitted scalers to file.
        
        Args:
            path: Path to save scaler pickle file
        """
        if self.scaler is None:
            raise ValueError("No scaler to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scalers(self, path: Union[str, Path]) -> None:
        """Load scalers from file.
        
        Args:
            path: Path to scaler pickle file
        """
        path = Path(path)
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)

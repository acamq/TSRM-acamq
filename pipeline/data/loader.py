"""OSSE Data Loader for GDC synthetic observations."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class OSSEDataLoader:
    """Load GDC synthetic observations from OSSE output.

    The OSSE data consists of 4 separate 1-hour observation blocks with
    3-4 hour gaps between them:
    - Block 0: indices [0:60] - 09:01-10:00
    - Block 1: indices [60:120] - 13:01-14:00 (3hr gap)
    - Block 2: indices [120:180] - 18:01-19:00 (4hr gap)
    - Block 3: indices [180:240] - 22:01-23:00 (3hr gap)

    Total: 240 timesteps in 4 discontinuous blocks.

    This is part of the TSRM imputation pipeline for processing GDC
    synthetic OSSE observations.
    """

    # Canonical variable ordering
    DEFAULT_VARIABLES = [
        'GDC_TEMPERATURE',
        'GDC_TEMPERATURE_ION',
        'GDC_TEMPERATURE_ELEC',
        'GDC_VELOCITY_U',
        'GDC_VELOCITY_V',
        'GDC_DENSITY_ION_OP',
        'GDC_DENSITY_NEUTRAL_O',
        'GDC_DENSITY_NEUTRAL_O2',
    ]

    def __init__(self, data_dir: str, variables: Optional[List[str]] = None):
        """Initialize the data loader.

        Args:
            data_dir: Path to OSSE output directory containing pickle files
            variables: List of variables to load. If None, uses DEFAULT_VARIABLES.
        """
        self.data_dir = Path(data_dir)
        self.variables = variables or self.DEFAULT_VARIABLES.copy()
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._metadata: Optional[Dict] = None

        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"OSSE data directory not found: {data_dir}\n"
                f"Please run GDC_osse.py to generate synthetic observations."
            )

    def load_pickle(self) -> Dict[str, np.ndarray]:
        """Load the main pickle file with all observations.

        Returns:
            Dictionary mapping variable names to arrays of shape (6, T)
            where 6 is the number of satellites and T is the number of timesteps.
        """
        pkl_path = self.data_dir / "GDC_synthetic_observations.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"OSSE observations file not found: {pkl_path}\n"
                f"Please run GDC_osse.py to generate synthetic observations."
            )

        with open(pkl_path, 'rb') as f:
            self._data = pickle.load(f)

        return self._data

    def load_csv(self, variable_name: str) -> np.ndarray:
        """Load individual CSV file for one variable.

        The OSSE output CSV files have a 3-line header comment starting with '#'.

        Args:
            variable_name: Name of the variable (e.g., 'GDC_TEMPERATURE')

        Returns:
            ndarray of the CSV data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        import pandas as pd

        csv_path = self.data_dir / f"{variable_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for variable '{variable_name}': {csv_path}")

        df = pd.read_csv(csv_path, comment='#')
        return df.values

    def get_metadata(self) -> Dict:
        """Load metadata from pickle file.

        Returns:
            Dictionary containing:
            - variables: Dict of variable metadata
            - num_satellites: Number of satellites (6)
            - n_timesteps: Total timesteps (240)
            - block_boundaries: List of block boundary indices [0, 60, 120, 180, 240]
            - block_times: List of (start_time, end_time) tuples for each block
            - timestamps: List of all timestamp strings
        """
        if self._metadata is not None:
            return self._metadata

        meta_path = self.data_dir / "observation_metadata.pkl"

        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                self._metadata = pickle.load(f)
        else:
            # Create default metadata if file doesn't exist
            self._metadata = {
                'block_boundaries': [0, 60, 120, 180, 240],
                'num_satellites': 6,
            }

        return self._metadata

    def get_block_boundaries(self) -> List[int]:
        """Get block boundary indices for discontinuous data.

        Returns:
            List of indices where each block starts/ends: [0, 60, 120, 180, 240]
        """
        metadata = self.get_metadata()
        return metadata.get('block_boundaries', [0, 60, 120, 180, 240])

    def to_multivariate_array(self) -> np.ndarray:
        """Convert dictionary of per-variable arrays to a single 3D array.

        Returns:
            Array of shape (N_satellites, T_timesteps, F_features)
            where N=6, T=240 (or actual timesteps), F=len(variables)
        """
        if self._data is None:
            self.load_pickle()

        # Filter to available variables
        available_vars = [v for v in self.variables if v in self._data]
        missing_vars = [v for v in self.variables if v not in self._data]

        if missing_vars:
            print(f"Warning: Variables not available in data: {missing_vars}")

        if not available_vars:
            raise ValueError("No requested variables found in OSSE data")

        # Get dimensions from first variable
        first_var = available_vars[0]
        n_satellites, n_timesteps = self._data[first_var].shape
        n_features = len(available_vars)

        # Stack into 3D array: (satellites, timesteps, features)
        result = np.zeros((n_satellites, n_timesteps, n_features))

        for i, var in enumerate(available_vars):
            result[:, :, i] = self._data[var]

        return result

    def validate_no_cross_block_windows(
        self,
        window_indices: List[Tuple[int, int]]
    ) -> bool:
        """Verify that no window spans across block boundaries.

        Args:
            window_indices: List of (start, end) tuples for windows

        Returns:
            True if all windows are within blocks, False if any spans boundary
        """
        block_boundaries = self.get_block_boundaries()

        for start, end in window_indices:
            # Find which block this window starts in
            for i in range(len(block_boundaries) - 1):
                block_start = block_boundaries[i]
                block_end = block_boundaries[i + 1]

                if block_start <= start < block_end:
                    # Window starts in this block, check if it ends in same block
                    if end > block_end:
                        return False
                    break

        return True

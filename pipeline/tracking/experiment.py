from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re
import yaml


class ExperimentTracker:
    """Create and manage a unique experiment directory.

    The tracker creates an experiment directory using a hybrid identifier
    (`{timestamp}_{name}`) and ensures atomic creation with collision handling.
    It also initializes the expected 4-fold model directory structure.
    """

    def __init__(self, experiments_dir: str, experiment_name: str) -> None:
        """Initialize tracker and create experiment directory structure.

        Args:
            experiments_dir: Root directory that stores all experiments.
            experiment_name: Human-readable experiment name used in the ID.
        """
        self._experiments_root = Path(experiments_dir)
        self._experiments_root.mkdir(parents=True, exist_ok=True)

        normalized_name = self._normalize_name(experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_experiment_id = f"{timestamp}_{normalized_name}"

        self._experiment_id, self._experiment_dir = self._create_unique_experiment_dir(base_experiment_id)
        self._create_fold_directories()

    @property
    def experiment_id(self) -> str:
        """Return the unique experiment identifier."""
        return self._experiment_id

    @property
    def experiment_dir(self) -> Path:
        """Return the created experiment directory path."""
        return self._experiment_dir

    def _create_unique_experiment_dir(self, base_experiment_id: str) -> tuple[str, Path]:
        """Create a unique experiment directory using atomic mkdir.

        If a collision is detected, numeric suffixes are added as `_2`, `_3`,
        and so on until a unique directory is created.
        """
        suffix = 1
        while True:
            candidate_id = base_experiment_id if suffix == 1 else f"{base_experiment_id}_{suffix}"
            candidate_dir = self._experiments_root / candidate_id
            try:
                candidate_dir.mkdir(parents=False, exist_ok=False)
                return candidate_id, candidate_dir
            except FileExistsError:
                suffix += 1

    def _create_fold_directories(self) -> None:
        """Create the expected 4-fold model subdirectories."""
        models_dir = self._experiment_dir / "models"
        for fold_index in range(4):
            (models_dir / f"fold_{fold_index}").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_name(experiment_name: str) -> str:
        """Normalize experiment names for filesystem-safe experiment IDs."""
        normalized = re.sub(r"\s+", "_", experiment_name.strip().lower())
        normalized = re.sub(r"[^a-z0-9_-]", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            raise ValueError("experiment_name must contain at least one alphanumeric character")
        return normalized

    def save_config(self, config: dict) -> None:
        """Save configuration as YAML to experiment directory.

        The config is saved to {experiment_dir}/config.yaml using safe_dump,
        preserving all values including CLI overrides that were applied
        before calling this method.

        Args:
            config: Configuration dictionary to snapshot.
        """
        config_path = self._experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    def load_config(self) -> dict:
        """Load configuration from experiment directory.

        Reads the config.yaml file from the experiment directory.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config.yaml does not exist in experiment_dir.
        """
        config_path = self._experiment_dir / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_model_dir(self) -> Path:
        """Return the models directory path.

        Returns:
            Path to the models directory within the experiment directory.
        """
        return self._experiment_dir / "models"

    def get_eval_dir(self) -> Path:
        """Return the evaluation directory path.

        Returns:
            Path to the evaluation directory within the experiment directory.
            Creates the directory if it does not exist.
        """
        eval_dir = self._experiment_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

    def _capture_git_info(self) -> dict:
        """Capture git commit hash and dirty state.

        Uses subprocess to run git commands. Handles gracefully if not in a git repo
        or git is not installed.

        Returns:
            Dictionary with keys 'git_commit' (commit hash or 'not_in_repo') and
            'git_dirty' (boolean indicating if working tree has changes).
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            git_commit = result.stdout.strip()

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5
            )
            git_dirty = len(result.stdout.strip()) > 0

            return {"git_commit": git_commit, "git_dirty": git_dirty}
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return {"git_commit": "not_in_repo", "git_dirty": False}

    def _capture_environment(self) -> dict:
        """Capture Python version and installed package versions.

        Uses importlib.metadata for faster package version retrieval than pip freeze.

        Returns:
            Dictionary with keys 'python_version' (string) and 'packages' (dict mapping
            package names to version strings).
        """
        python_version = sys.version.split()[0]

        packages = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            version = dist.version
            packages[name] = version

        return {"python_version": python_version, "packages": packages}

    def save_metadata(self, seed: int, **kwargs) -> None:
        """Save experiment metadata including git info, environment, and seed.

        Combines git information, environment details, and the random seed into a
        metadata.json file in the experiment directory.

        Args:
            seed: Random seed used for reproducibility.
            **kwargs: Additional metadata fields to include in the saved file.
        """
        git_info = self._capture_git_info()
        env_info = self._capture_environment()

        metadata = {
            **git_info,
            **env_info,
            "seed": seed,
            **kwargs
        }

        metadata_path = self._experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

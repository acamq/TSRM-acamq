from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class OSSEDataset(Dataset):

    def __init__(
        self,
        masked_windows: np.ndarray,
        original_windows: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        freq: str = "t",
    ) -> None:
        if masked_windows.shape != original_windows.shape:
            raise ValueError("masked_windows and original_windows must have identical shape")

        if freq != "t":
            raise ValueError("OSSEDataset currently supports only freq='t'")

        self.masked_windows = masked_windows.astype(np.float32, copy=False)
        self.original_windows = original_windows.astype(np.float32, copy=False)
        self.timestamps = timestamps
        self.freq = freq
        self.time_marks = self._generate_time_features()

    def _generate_time_features(self) -> np.ndarray:
        n_samples, window_size = self.masked_windows.shape[:2]
        time_marks = np.zeros((n_samples, window_size, 5), dtype=np.float32)

        if self.timestamps is None:
            base = datetime(2025, 1, 1, 9, 0)
            for i in range(n_samples):
                for j in range(window_size):
                    ts = base + timedelta(minutes=i * window_size + j)
                    time_marks[i, j] = self._extract_time_features(ts)
            return time_marks

        if self.timestamps.shape[:2] != (n_samples, window_size):
            raise ValueError(
                "timestamps must match first two dimensions of window arrays: "
                "[n_samples, window_size]"
            )

        for i in range(n_samples):
            for j in range(window_size):
                ts = self._coerce_datetime(self.timestamps[i, j])
                time_marks[i, j] = self._extract_time_features(ts)

        return time_marks

    @staticmethod
    def _coerce_datetime(ts: object) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, np.datetime64):
            return ts.astype("datetime64[us]").tolist()
        raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")

    @staticmethod
    def _extract_time_features(ts: datetime) -> np.ndarray:
        return np.asarray(
            [
                (ts.month - 6.5) / 12.0,
                (ts.day - 15.5) / 31.0,
                (ts.weekday() - 3.0) / 7.0,
                (ts.hour - 11.5) / 24.0,
                (ts.minute - 30.0) / 60.0,
            ],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return self.masked_windows.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked = torch.from_numpy(self.masked_windows[idx])
        original = torch.from_numpy(self.original_windows[idx])
        time_marks_x = torch.from_numpy(self.time_marks[idx])
        time_marks_y = time_marks_x.clone()
        return masked, original, time_marks_x, time_marks_y


def create_dataloaders(
    train_masked: np.ndarray,
    train_original: np.ndarray,
    train_timestamps: Optional[np.ndarray],
    val_masked: np.ndarray,
    val_original: np.ndarray,
    val_timestamps: Optional[np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last_train: bool = False,
    drop_last_val: bool = False,
    freq: str = "t",
) -> Tuple[DataLoader, DataLoader]:
    train_ds = OSSEDataset(train_masked, train_original, train_timestamps, freq=freq)
    val_ds = OSSEDataset(val_masked, val_original, val_timestamps, freq=freq)

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=drop_last_train,
        **common_loader_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=drop_last_val,
        **common_loader_kwargs,
    )
    return train_dl, val_dl

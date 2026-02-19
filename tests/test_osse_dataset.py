"""Smoke tests for OSSEDataset."""

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta

from pipeline.data.dataset import OSSEDataset, create_dataloaders


def test_dataset_length():
    """Dataset returns correct number of samples."""
    n_samples, window_size, n_features = 20, 30, 8
    masked = np.random.randn(n_samples, window_size, n_features)
    masked[:, 5:10, :] = np.nan
    original = np.random.randn(n_samples, window_size, n_features)

    ds = OSSEDataset(masked, original, timestamps=None, freq='t')
    assert len(ds) == n_samples


def test_getitem_format():
    """__getitem__ returns 4-tuple with correct shapes."""
    n_samples, window_size, n_features = 10, 30, 8
    masked = np.random.randn(n_samples, window_size, n_features)
    original = np.random.randn(n_samples, window_size, n_features)

    ds = OSSEDataset(masked, original, timestamps=None, freq='t')
    item = ds[0]

    assert len(item) == 4, f"Expected 4-tuple, got {len(item)}"
    masked_t, orig_t, tm_x, tm_y = item
    assert masked_t.shape == (window_size, n_features)
    assert orig_t.shape == (window_size, n_features)
    assert tm_x.shape[0] == window_size
    assert tm_x.shape[1] >= 4  # At least 4 time features


def test_nan_preservation():
    """NaN in masked windows is preserved in output tensor."""
    n_samples, window_size, n_features = 10, 30, 8
    masked = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
    masked[:, 5:10, :] = np.nan
    original = np.random.randn(n_samples, window_size, n_features).astype(np.float32)

    ds = OSSEDataset(masked, original, timestamps=None, freq='t')
    item = ds[0]
    masked_t = item[0]

    assert torch.isnan(masked_t[5, 0]), "NaN should be preserved in masked data"


def test_dataloader_batching():
    """DataLoader produces correctly batched outputs."""
    n_samples, window_size, n_features = 20, 30, 8
    masked = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
    original = np.random.randn(n_samples, window_size, n_features).astype(np.float32)

    train_dl, val_dl = create_dataloaders(
        masked[:15], original[:15], None,
        masked[15:], original[15:], None,
        batch_size=4
    )

    batch = next(iter(train_dl))
    assert len(batch) == 4, f"Batch tuple length: {len(batch)}"
    assert batch[0].shape[0] == 4, f"Batch size: {batch[0].shape[0]}"

"""Smoke tests for TSRMImputationExternal."""

import pytest
import torch
import numpy as np

from architecture.tsrm_external import TSRMImputationExternal


def get_test_config():
    return {
        'feature_dimension': 8,
        'seq_len': 30,
        'pred_len': 0,
        'encoding_size': 32,
        'h': 2,
        'N': 1,
        'conv_dims': [[3, 1, 1]],
        'attention_func': 'classic',
        'batch_size': 4,
        'dropout': 0.1,
        'revin': False,
        'missing_ratio': 0.0,
        'loss_function_imputation': 'mse+mae',
        'loss_imputation_mode': 'imputation',
        'loss_weight_alpha': 0.5,
        'embed': 'timeF',
        'freq': 't',
        'mask_size': 3,
        'mask_count': 1,
        'task': 'imputation',
        'phase': 'downstream',
    }


def test_instantiation():
    """TSRMImputationExternal instantiates with valid config."""
    config = get_test_config()
    model = TSRMImputationExternal(config)
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0


def test_forward_shape():
    """Forward pass produces correct output shape."""
    config = get_test_config()
    model = TSRMImputationExternal(config)
    model.eval()

    B, T, F = 4, 30, 8
    masked_data = torch.randn(B, T, F)
    masked_data[:, 5:10, :] = float('nan')
    original_data = torch.randn(B, T, F)
    time_marks = torch.zeros(B, T, 5)

    with torch.no_grad():
        result = model.impute(masked_data, original_data, time_marks, time_marks)

    assert result.shape == (B, T, F), f"Expected shape {(B, T, F)}, got {result.shape}"


def test_nan_handling():
    """NaN in masked data is handled correctly."""
    config = get_test_config()
    model = TSRMImputationExternal(config)
    model.eval()

    B, T, F = 4, 30, 8
    masked_data = torch.randn(B, T, F)
    masked_data[:, 5:10, :] = float('nan')
    original_data = torch.randn(B, T, F)
    time_marks = torch.zeros(B, T, 5)

    with torch.no_grad():
        result = model.impute(masked_data, original_data, time_marks, time_marks)

    assert not torch.isnan(result).any(), "Output should have no NaN"

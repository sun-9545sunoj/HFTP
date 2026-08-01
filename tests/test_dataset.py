"""
Unit tests for PyTorch LOB Sequence Dataset & DataLoader utilities.
"""

import numpy as np
import polars as pl
import pytest
import torch

from src.data.dataset import (
    LOBSequenceDataset,
    create_chronological_dataloaders,
    map_indices_to_signals,
    map_labels_to_indices,
)


def test_label_mapping_utilities():
    """Verify bidirectional mapping between y_t in {-1, 0, 1} and PyTorch indices {0, 1, 2}."""
    original_labels = np.array([-1, 0, 1, 0, -1, 1])
    
    indices = map_labels_to_indices(original_labels)
    expected_indices = np.array([0, 1, 2, 1, 0, 2])
    np.testing.assert_array_equal(indices, expected_indices)

    recovered_signals = map_indices_to_signals(indices)
    np.testing.assert_array_equal(recovered_signals, original_labels)


def test_lob_sequence_dataset():
    """Verify sequence sliding window dataset shapes and target indexing."""
    n_ticks = 150
    n_features = 40
    seq_len = 100

    features = np.random.randn(n_ticks, n_features)
    labels = np.random.choice([-1, 0, 1], size=n_ticks)

    ds = LOBSequenceDataset(features, labels, seq_len=seq_len)
    
    assert len(ds) == n_ticks - seq_len + 1  # 51 samples
    
    x, y = ds[0]
    assert x.shape == (seq_len, n_features)
    assert isinstance(y, torch.Tensor)
    assert y.item() in [0, 1, 2]


def test_create_chronological_dataloaders():
    """Verify chronological dataset splitting and class weighting."""
    n_ticks = 500
    dict_data = {"label": np.random.choice([-1, 0, 1], size=n_ticks)}
    for i in range(1, 11):
        dict_data[f"ask_price_{i}"] = np.random.randn(n_ticks)
        dict_data[f"ask_size_{i}"] = np.random.randn(n_ticks)
        dict_data[f"bid_price_{i}"] = np.random.randn(n_ticks)
        dict_data[f"bid_size_{i}"] = np.random.randn(n_ticks)

    df = pl.DataFrame(dict_data)
    feature_cols = [c for c in df.columns if c != "label"]

    train_loader, val_loader, test_loader, weights = create_chronological_dataloaders(
        df,
        feature_cols=feature_cols,
        label_col="label",
        seq_len=20,
        batch_size=16,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert weights.shape[0] <= 3

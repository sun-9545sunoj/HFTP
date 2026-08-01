"""
PyTorch Dataset and DataLoader Utilities for LOB Sequence Modeling.

Implements sliding-window dataset creation over sequence length T (default T=100)
with chronological train/validation/test splitting for time-series LOB features.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


def map_labels_to_indices(labels: np.ndarray) -> np.ndarray:
    """
    Map canonical LOB directional labels y_t in {-1, 0, 1} to PyTorch class indices {0, 1, 2}.

    Mapping:
        -1 (DOWN)       -> 0
         0 (STATIONARY) -> 1
         1 (UP)         -> 2
    """
    labels_mapped = np.copy(labels)
    if set(np.unique(labels_mapped)).issubset({-1, 0, 1}):
        mapped = np.zeros_like(labels_mapped, dtype=np.int64)
        mapped[labels_mapped == -1] = 0
        mapped[labels_mapped == 0] = 1
        mapped[labels_mapped == 1] = 2
        return mapped
    return labels_mapped.astype(np.int64)


def map_indices_to_signals(indices: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Map PyTorch model predicted class indices {0, 1, 2} back to canonical trading signals y_t in {-1, 0, 1}.

    Mapping:
        0 (DOWN)       -> -1 (SELL)
        1 (STATIONARY) ->  0 (HOLD)
        2 (UP)         ->  1 (BUY)
    """
    if isinstance(indices, torch.Tensor):
        indices_arr = indices.detach().cpu().numpy()
    else:
        indices_arr = np.copy(indices)

    signals = np.zeros_like(indices_arr, dtype=np.int8)
    signals[indices_arr == 0] = -1
    signals[indices_arr == 1] = 0
    signals[indices_arr == 2] = 1
    return signals


class LOBSequenceDataset(Dataset):
    """
    Sliding-window PyTorch Dataset for Limit Order Book sequence inputs.
    
    Transforms LOB feature matrices into sequence tensors of shape (T, F)
    paired with target labels y_t mapped from {-1, 0, 1} to PyTorch class indices {0, 1, 2}.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 100,
    ) -> None:
        """
        Args:
            features: 2D numpy array of shape (N_ticks, F_features).
            labels: 1D numpy array of canonical categorical labels y_t in {-1, 0, 1}.
            seq_len: Temporal window sequence length T (default: 100).
        """
        if len(features) != len(labels):
            raise ValueError(f"Features length ({len(features)}) and labels length ({len(labels)}) mismatch.")
        if len(features) < seq_len:
            raise ValueError(f"Total ticks ({len(features)}) is smaller than sequence length T ({seq_len}).")

        self.features = torch.tensor(features, dtype=torch.float32)
        labels_mapped = map_labels_to_indices(labels)
        self.labels = torch.tensor(labels_mapped, dtype=torch.long)
        self.seq_len = seq_len
        self.num_samples = len(features) - seq_len + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Feature sequence tensor of shape (seq_len, F_features).
            y: Target label class index (0, 1, or 2) at index idx + seq_len - 1.
        """
        x_seq = self.features[idx : idx + self.seq_len]
        y_target = self.labels[idx + self.seq_len - 1]
        return x_seq, y_target


def create_chronological_dataloaders(
    df: pl.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    seq_len: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 0,
    num_classes: int = 3,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Split Polars DataFrame chronologically into Train, Validation, and Test DataLoaders.
    """
    df_clean = df.drop_nulls(subset=[label_col] + feature_cols)
    
    features_arr = df_clean.select(feature_cols).to_numpy()
    labels_arr = df_clean[label_col].to_numpy()

    n_total = len(df_clean)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_feat, train_lbl = features_arr[:n_train], labels_arr[:n_train]
    val_feat, val_lbl = features_arr[n_train : n_train + n_val], labels_arr[n_train : n_train + n_val]
    test_feat, test_lbl = features_arr[n_train + n_val :], labels_arr[n_train + n_val :]

    train_ds = LOBSequenceDataset(train_feat, train_lbl, seq_len=seq_len)
    val_ds = LOBSequenceDataset(val_feat, val_lbl, seq_len=seq_len)
    test_ds = LOBSequenceDataset(test_feat, test_lbl, seq_len=seq_len)

    # Compute inverse class frequencies for weighted loss function across all 3 classes
    weights = np.ones(num_classes, dtype=np.float32)
    unique_classes, counts = np.unique(train_ds.labels.numpy(), return_counts=True)
    total_samples = len(train_ds.labels)

    for cls_idx, count in zip(unique_classes, counts):
        if 0 <= cls_idx < num_classes:
            weights[cls_idx] = total_samples / (float(len(unique_classes)) * float(count))

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights_tensor

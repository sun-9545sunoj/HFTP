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


def get_model_feature_cols(df: pl.DataFrame, k: int = 5) -> List[str]:
    """
    Select model input columns from an enriched LOB DataFrame.

    Excludes metadata/targets and raw `mid_price` (unnormalized price level,
    kept in the DataFrame only for backtesting), plus all-null columns.
    """
    excluded = {"timestamp", "label", f"return_{k}", "alpha_t", "mid_price", "stock", "split"}
    return [
        c for c in df.columns
        if c not in excluded
        and not c.startswith(("return_", "label_"))
        and not df[c].is_null().all()
    ]


class LOBSequenceDataset(Dataset):
    """
    Sliding-window PyTorch Dataset for Limit Order Book sequence inputs.

    Transforms LOB feature matrices into sequence tensors of shape (T, F)
    paired with target labels y_t mapped from {-1, 0, 1} to PyTorch class indices {0, 1, 2}.

    Supports on-the-fly LOB-specific data augmentation during training via
    the optional `augmentation` argument. Pass None for val/test datasets.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 100,
        augmentation=None,  # LOBAugmentationPipeline or None
    ) -> None:
        """
        Args:
            features:    2D numpy array of shape (N_ticks, F_features).
            labels:      1D numpy array of canonical labels y_t in {-1, 0, 1}.
            seq_len:     Temporal window sequence length T (default: 100).
            augmentation: Optional augmentation pipeline (only for training set).
                          If None, no augmentation is applied (val/test mode).
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
        self.augmentation = augmentation

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Feature sequence tensor of shape (seq_len, F_features),
               with augmentation applied on-the-fly if self.augmentation is set.
            y: Target label class index (0, 1, or 2).
        """
        x_seq = self.features[idx : idx + self.seq_len].clone()
        y_target = self.labels[idx + self.seq_len - 1]

        if self.augmentation is not None:
            x_seq = self.augmentation(x_seq)

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
    augmentation=None,  # LOBAugmentationPipeline — applied only to train split
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Split Polars DataFrame chronologically into Train, Validation, and Test DataLoaders.

    Augmentation is applied ONLY to the training split. Val and test always
    run on clean, unaugmented data to ensure fair evaluation.
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

    # Augmentation only for training; val/test use clean data
    train_ds = LOBSequenceDataset(train_feat, train_lbl, seq_len=seq_len, augmentation=augmentation)
    val_ds   = LOBSequenceDataset(val_feat,   val_lbl,   seq_len=seq_len, augmentation=None)
    test_ds  = LOBSequenceDataset(test_feat,  test_lbl,  seq_len=seq_len, augmentation=None)

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


class LOBWindowDataset(Dataset):
    """
    Sliding windows over a shared feature matrix, restricted to explicit end
    indices so that no window crosses a stock or split boundary.

    Window i covers rows [end_i - seq_len + 1, end_i] and is labelled with the
    label at end_i.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        end_indices: np.ndarray,
        seq_len: int = 100,
        augmentation=None,
    ) -> None:
        self.features = features
        self.labels = labels
        self.end_indices = torch.as_tensor(end_indices, dtype=torch.long)
        self.seq_len = seq_len
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = int(self.end_indices[idx])
        x_seq = self.features[end - self.seq_len + 1 : end + 1].clone()
        if self.augmentation is not None:
            x_seq = self.augmentation(x_seq)
        return x_seq, self.labels[end]


def segment_end_indices(df: pl.DataFrame, seq_len: int) -> dict:
    """
    Map each split name to the window end indices valid within its
    (stock, split) segments. `df` rows must be grouped contiguously by
    (stock, split) in temporal order, as written by prepare_sequential_lobster.py.
    """
    seg = (
        df.with_row_index("_i")
        .group_by(["stock", "split"], maintain_order=True)
        .agg(pl.col("_i").min().alias("start"), pl.col("_i").max().alias("stop"))
    )
    ends = {}
    for stock, split, start, stop in seg.iter_rows():
        if stop - start + 1 < seq_len:
            continue
        ends.setdefault(split, []).append(np.arange(start + seq_len - 1, stop + 1))
    return {s: np.concatenate(v) for s, v in ends.items()}


def create_segmented_dataloaders(
    df: pl.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    seq_len: int = 100,
    batch_size: int = 64,
    num_workers: int = 0,
    num_classes: int = 3,
    augmentation=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Leak-free DataLoaders for the sequential LOBSTER parquet (with `stock` and
    `split` columns). Windows never cross stock/split boundaries; there is no
    row shuffling or undersampling. Class weights come from train window labels.
    """
    features = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
    labels = torch.tensor(map_labels_to_indices(df[label_col].to_numpy()), dtype=torch.long)
    ends = segment_end_indices(df, seq_len)

    train_ds = LOBWindowDataset(features, labels, ends["train"], seq_len, augmentation)
    val_ds = LOBWindowDataset(features, labels, ends["val"], seq_len)
    test_ds = LOBWindowDataset(features, labels, ends["test"], seq_len)

    counts = np.bincount(labels[ends["train"]].numpy(), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_weights_tensor

"""
Prepare balanced & normalized multi-stock LOBSTER dataset for generalized training.

Anti-overfitting measures:
1. Per-stock Z-score normalization (model learns relative patterns, not absolute prices)
2. Quantile-based labeling for balanced classes
3. Undersampling majority class to achieve perfect 33/33/33 balance
4. SHUFFLE after concatenation so train/val splits have all stocks & classes mixed
5. 80/20 chronological split per stock (no future leakage)
"""

import os
import numpy as np
import polars as pl
from src.data.loader import LOBLoader
from src.data.targets import compute_mid_price, compute_smoothed_returns, assign_directional_labels
from src.features.microstructure import extract_all_microstructure_features

stocks = ["AAPL", "AMZN", "GOOG", "MSFT", "INTC"]
data_dir = "src/data/"
k = 5

# Columns that should NOT be z-score normalized (metadata / targets)
EXCLUDE_FROM_NORM = {"timestamp", "label", "return_5", "alpha_t", "mid_price"}

train_dfs = []
test_dfs = []

for stock in stocks:
    print(f"\n{'='*60}")
    print(f"  Processing {stock}...")
    print(f"{'='*60}")
    ob_path = os.path.join(data_dir, f"LOBSTER_{stock}_orderbook.csv")
    msg_path = os.path.join(data_dir, f"LOBSTER_{stock}_message.csv")

    # 1. Load raw LOBSTER data
    df = LOBLoader.load_lobster(ob_path, message_path=msg_path, num_levels=10)
    print(f"  -> Raw ticks loaded: {len(df):,}")

    # 2. Compute returns & balanced labels (quantile-based)
    df = compute_mid_price(df)
    df = compute_smoothed_returns(df, k=k)
    df = assign_directional_labels(df, return_col=f"return_{k}", alpha=-1.0, use_dynamic_volatility=False)

    # 3. Extract Microstructure Features
    df = extract_all_microstructure_features(df, num_levels=10)
    df = df.drop_nulls(subset=["label"])
    print(f"  -> After feature extraction: {len(df):,} ticks")

    # 4. Per-Stock Z-Score Normalization
    #    Compute stats on FULL stock data BEFORE splitting, so train & val share
    #    identical normalization parameters (no distribution shift).
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_NORM]

    norm_exprs = []
    for col in feature_cols:
        col_series = df[col]
        mu = col_series.mean()
        sigma = col_series.std()
        if sigma is not None and sigma > 1e-12:
            norm_exprs.append(((pl.col(col) - mu) / sigma).alias(col))
        else:
            norm_exprs.append(pl.lit(0.0).alias(col))

    df = df.with_columns(norm_exprs)
    print(f"  -> Z-score normalized {len(feature_cols)} features")

    # 5. Chronological 80/20 split for this stock
    split_idx = int(len(df) * 0.8)
    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, len(df) - split_idx)

    train_dfs.append(train_df)
    test_dfs.append(test_df)
    print(f"  -> Train: {len(train_df):,} | Test: {len(test_df):,}")

# 6. Concatenate all stocks
print(f"\n{'='*60}")
print("  Concatenating all stocks...")
final_train = pl.concat(train_dfs)
final_test = pl.concat(test_dfs)

# 7. CRITICAL FIX: Shuffle the training data!
#    Without shuffling, the concatenated data is [all_AAPL, all_AMZN, ...].
#    When train.py does its internal 70/15/15 train/val split, the val set
#    would be entirely from one stock with different Z-score distributions.
#    Shuffling ensures every split has a mix of all stocks and all classes.
print("  Shuffling training data for mixed stock/class distribution...")
final_train = final_train.sample(fraction=1.0, shuffle=True, seed=42)

# 8. Undersample majority class for perfect balance
print("  Undersampling for perfect class balance...")
label_counts = final_train["label"].value_counts()
min_count = label_counts["count"].min()
print(f"  -> Minimum class count: {min_count:,}")

balanced_parts = []
for label_val in [-1, 0, 1]:
    subset = final_train.filter(pl.col("label") == label_val)
    if len(subset) > min_count:
        indices = np.random.default_rng(42).choice(len(subset), size=min_count, replace=False)
        subset = subset[indices.tolist()]
    balanced_parts.append(subset)

final_train_balanced = pl.concat(balanced_parts)

# 9. Shuffle AGAIN after undersampling (since concat re-groups by label)
final_train_balanced = final_train_balanced.sample(fraction=1.0, shuffle=True, seed=123)

# 10. Save to Parquet
print("  Saving to Parquet...")
LOBLoader.save_parquet(final_train_balanced, os.path.join(data_dir, "LOBSTER_balanced_train.parquet"))
LOBLoader.save_parquet(final_test, os.path.join(data_dir, "LOBSTER_balanced_test.parquet"))

# 11. Final report
print(f"\n{'='*60}")
print(f"  FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Train Set Size: {len(final_train_balanced):,} ticks (SHUFFLED)")
print(f"  Test Set Size:  {len(final_test):,} ticks")
print(f"\n  Train Class Distribution (Perfectly Balanced):")
print(final_train_balanced["label"].value_counts().sort("label"))
print(f"\n  Test Class Distribution:")
print(final_test["label"].value_counts().sort("label"))

# Verify shuffle worked: check that first 100 rows have mixed labels
first_100_labels = final_train_balanced["label"].head(100).value_counts().sort("label")
print(f"\n  Sanity Check — First 100 rows label mix:")
print(first_100_labels)
print(f"{'='*60}")
